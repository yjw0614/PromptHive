"""Optimize cached prompts, run SAM, and report initial versus optimized metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
import torch

from .core import (
    PromptPilotConfig,
    QNetwork,
    discover_samples,
    greedy_optimize_prompts,
    load_graph_sample,
    optimize_prompts,
    seed_everything,
)
from .improved import CandidateQNetwork, optimize_prompts_improved
from .release_qtable import optimize_prompts_release_qtable
from .reference_selector import (
    load_reference_bank,
    load_reference_classifier,
    select_reference_prompts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--checkpoint", help="Required for network optimizers")
    parser.add_argument(
        "--optimizer",
        choices=(
            "network",
            "improved-network",
            "greedy",
            "reference-prototype",
            "release-qtable-diagnostic",
        ),
        default="network",
    )
    parser.add_argument(
        "--greedy-counts",
        help="Optional fixed positive,negative counts, for example 6,20",
    )
    parser.add_argument("--positive-count", type=int, default=6)
    parser.add_argument("--negative-count", type=int, default=12)
    parser.add_argument("--prototype-neighbors", type=int, default=5)
    parser.add_argument("--prototype-smoothing", type=int, default=3)
    parser.add_argument("--prompt-min-separation", type=float, default=4.0)
    parser.add_argument(
        "--prototype-candidate-source", choices=("all", "initial"), default="all"
    )
    parser.add_argument("--reference-bank")
    parser.add_argument("--reference-classifier")
    parser.add_argument("--reference-topk", type=int, default=0)
    parser.add_argument(
        "--prototype-score-mode",
        choices=("margin", "vote", "hybrid", "linear", "hybrid-linear"),
        default="margin",
    )
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-source", default=str(Path(__file__).resolve().parents[1] / "segmenter"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument(
        "--sample-list",
        help="Optional newline list or CSV with a name column for a stable evaluation subset.",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--policy-rollouts",
        type=int,
        default=1,
        help="Repeat stochastic node/direction trajectories and keep the best score.",
    )
    parser.add_argument(
        "--inference-epsilon",
        type=float,
        default=0.0,
        help="Epsilon-greedy exploration retained during inference.",
    )
    parser.add_argument(
        "--mask-refinement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply connected-component cleanup before metric evaluation.",
    )
    parser.add_argument("--image-format", choices=("rgb", "official-bgr"), default="official-bgr")
    parser.add_argument(
        "--selection-gate",
        choices=("none", "sam-predicted-iou"),
        default="none",
        help="Optionally retain the initial prompts unless SAM predicts a better candidate mask.",
    )
    parser.add_argument(
        "--gate-margin",
        type=float,
        default=0.0,
        help="Minimum predicted-IoU improvement required by the SAM selection gate.",
    )
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--multimask-output", action="store_true")
    parser.add_argument("--largest-component", action="store_true")
    parser.add_argument("--per-positive-union", action="store_true")
    parser.add_argument("--prototype-box", action="store_true")
    parser.add_argument("--prototype-mask-input", action="store_true")
    parser.add_argument("--prototype-mask-quantile", type=float, default=0.7)
    parser.add_argument("--prototype-mask-scale", type=float, default=2.0)
    parser.add_argument(
        "--candidate-mask-selection",
        choices=("sam-quality", "dino-contrast", "dino-iou", "hybrid"),
        default="sam-quality",
    )
    return parser.parse_args()


def indices_to_points(indices: torch.Tensor, grid_size: int, image_size: int) -> np.ndarray:
    values = indices.detach().cpu().numpy().astype(np.int64)
    patch = image_size / grid_size
    x = (values % grid_size + 0.5) * patch
    y = (values // grid_size + 0.5) * patch
    return np.stack((x, y), axis=1).astype(np.float32)


def predict(
    predictor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    grid_size: int,
    image_size: int,
    multimask_output: bool = False,
    largest_component: bool = False,
    box: np.ndarray | None = None,
    mask_input: np.ndarray | None = None,
    selection_scores: torch.Tensor | None = None,
    mask_selection: str = "sam-quality",
) -> tuple[np.ndarray, float]:
    points = np.concatenate(
        (indices_to_points(pos, grid_size, image_size), indices_to_points(neg, grid_size, image_size)), axis=0
    )
    labels = np.concatenate((np.ones(len(pos)), np.zeros(len(neg)))).astype(np.int64)
    masks, quality, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=box,
        mask_input=mask_input,
        multimask_output=multimask_output,
    )
    selected = select_mask_index(
        masks,
        quality,
        mask_selection,
        selection_scores,
    )
    mask = masks[selected].astype(np.uint8)
    if largest_component and mask.any():
        count, labels_map, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count > 1:
            component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (labels_map == component).astype(np.uint8)
    return mask.astype(bool), float(quality[selected])


def predict_per_positive_union(
    predictor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    grid_size: int,
    image_size: int,
    multimask_output: bool,
) -> tuple[np.ndarray, float]:
    """Decode one instance per positive prompt and return their semantic union."""
    if pos.numel() == 0:
        return predict(
            predictor,
            pos,
            neg,
            grid_size,
            image_size,
            multimask_output=multimask_output,
        )
    negative_points = indices_to_points(neg, grid_size, image_size)
    union: np.ndarray | None = None
    qualities: list[float] = []
    for positive_index in pos:
        positive_point = indices_to_points(
            positive_index.reshape(1), grid_size, image_size
        )
        points = np.concatenate((positive_point, negative_points), axis=0)
        labels = np.concatenate(
            (np.ones(1), np.zeros(len(negative_points)))
        ).astype(np.int64)
        masks, quality, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=multimask_output,
        )
        selected = int(np.argmax(quality))
        current = masks[selected].astype(bool)
        union = current if union is None else np.logical_or(union, current)
        qualities.append(float(quality[selected]))
    assert union is not None
    return union, float(np.mean(qualities))


def select_mask_index(
    masks: np.ndarray,
    quality: np.ndarray,
    mode: str,
    patch_scores: torch.Tensor | None = None,
) -> int:
    if mode == "sam-quality" or len(masks) == 1:
        return int(np.argmax(quality))
    if patch_scores is None:
        raise ValueError(f"{mode} mask selection requires DINO patch scores")
    grid = int(round(patch_scores.numel() ** 0.5))
    if grid * grid != patch_scores.numel():
        raise ValueError("DINO patch scores do not form a square grid")
    dense = cv2.resize(
        patch_scores.float().reshape(grid, grid).numpy(),
        (masks.shape[2], masks.shape[1]),
        interpolation=cv2.INTER_CUBIC,
    )
    threshold = float(np.quantile(dense, 0.7))
    pseudo = dense >= threshold
    dino_values: list[float] = []
    for mask in masks.astype(bool):
        if mode in {"dino-contrast", "hybrid"}:
            inside = float(dense[mask].mean()) if mask.any() else float("-inf")
            outside = float(dense[~mask].mean()) if (~mask).any() else float("inf")
            dino_values.append(inside - outside)
        elif mode == "dino-iou":
            union = np.logical_or(mask, pseudo).sum()
            dino_values.append(
                float(np.logical_and(mask, pseudo).sum() / union) if union else 0.0
            )
        else:
            raise ValueError(f"Unknown candidate mask selection mode: {mode}")
    if mode == "hybrid":
        def standardize(values: np.ndarray) -> np.ndarray:
            return (values - values.mean()) / max(float(values.std()), 1e-6)
        combined = standardize(np.asarray(quality)) + standardize(np.asarray(dino_values))
        return int(np.argmax(combined))
    return int(np.argmax(dino_values))


def score_guidance(
    scores: torch.Tensor,
    positive_indices: torch.Tensor,
    grid_size: int,
    image_size: int,
    quantile: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a patch confidence map into SAM box and low-resolution logits."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("prototype mask quantile must be between zero and one")
    score_grid = scores.float().reshape(grid_size, grid_size)
    threshold = torch.quantile(score_grid, quantile)
    selected = (score_grid >= threshold).numpy().astype(np.uint8)
    count, components, _, _ = cv2.connectedComponentsWithStats(selected, connectivity=8)
    anchor = int(positive_indices[0])
    anchor_component = int(components[anchor // grid_size, anchor % grid_size])
    if anchor_component == 0 and count > 1:
        anchor_component = int(components.reshape(-1)[int(torch.argmax(scores))])
    component = components == anchor_component if anchor_component else selected.astype(bool)
    rows, columns = np.nonzero(component)
    if len(rows) == 0:
        rows = np.asarray([anchor // grid_size])
        columns = np.asarray([anchor % grid_size])
    patch = image_size / grid_size
    padding = patch
    box = np.asarray(
        [
            max(0.0, columns.min() * patch - padding),
            max(0.0, rows.min() * patch - padding),
            min(float(image_size - 1), (columns.max() + 1) * patch + padding),
            min(float(image_size - 1), (rows.max() + 1) * patch + padding),
        ],
        dtype=np.float32,
    )
    logits = (score_grid - threshold) / score_grid.std().clamp_min(1e-6)
    logits = torch.nn.functional.interpolate(
        logits.reshape(1, 1, grid_size, grid_size),
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    )[0].numpy().astype(np.float32) * scale
    return box, logits


def use_candidate(
    selection_gate: str,
    initial_quality: float,
    candidate_quality: float,
    margin: float,
) -> bool:
    if selection_gate == "none":
        return True
    if selection_gate != "sam-predicted-iou":
        raise ValueError(f"Unknown selection gate: {selection_gate}")
    return bool(np.isfinite(candidate_quality) and candidate_quality >= initial_quality + margin)


def dice(pred: np.ndarray, target: np.ndarray) -> float:
    denominator = int(pred.sum()) + int(target.sum())
    return 1.0 if denominator == 0 else 2.0 * float(np.logical_and(pred, target).sum()) / denominator


def hd95(pred: np.ndarray, target: np.ndarray) -> float:
    if not pred.any() or not target.any():
        return float("nan")
    footprint = generate_binary_structure(pred.ndim, 1)
    pred_surface = np.logical_xor(pred, binary_erosion(pred, structure=footprint, border_value=0))
    target_surface = np.logical_xor(target, binary_erosion(target, structure=footprint, border_value=0))
    dt_target = distance_transform_edt(~target_surface)
    dt_pred = distance_transform_edt(~pred_surface)
    distances = np.concatenate((dt_target[pred_surface], dt_pred[target_surface]))
    return float(np.percentile(distances, 95))


def refine_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Apply the component filter used by the public evaluator."""
    binary = mask.astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary.astype(bool)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    minimum_area = threshold * cv2.contourArea(contours[0])
    selected = [contour for contour in contours if cv2.contourArea(contour) >= minimum_area]
    refined = np.zeros_like(binary)
    cv2.drawContours(refined, selected, -1, 1, -1)
    return refined.astype(bool)


def find_file(directory: Path, name: str) -> Path:
    for suffix in (".jpg", ".png", ".jpeg"):
        candidate = directory / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image for {name} in {directory}")


def select_sample_paths(bundle_dir: str, sample_list: str | None) -> list[Path]:
    paths = discover_samples(bundle_dir)
    if not sample_list:
        return paths
    list_path = Path(sample_list)
    if list_path.suffix.lower() == ".csv":
        with list_path.open(newline="", encoding="utf-8") as handle:
            names = [str(row["name"]) for row in csv.DictReader(handle)]
    else:
        names = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    by_name = {path.name: path for path in paths}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} listed samples have no complete bundle; first missing: {missing[0]}"
        )
    return [by_name[name] for name in names]


def main() -> None:
    args = parse_args()
    if args.policy_rollouts <= 0:
        raise ValueError("--policy-rollouts must be positive")
    if not 0.0 <= args.inference_epsilon <= 1.0:
        raise ValueError("--inference-epsilon must be between zero and one")
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint = None
    release_qtable = None
    reference_bank = (
        load_reference_bank(args.reference_bank, device) if args.reference_bank else None
    )
    reference_classifier = (
        load_reference_classifier(args.reference_classifier, device)
        if args.reference_classifier
        else None
    )
    network: QNetwork | CandidateQNetwork | None = None
    if args.optimizer in {"network", "improved-network"}:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for network optimizers")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config_values = dict(checkpoint["config"])
        encoder = str(config_values.get("state_encoder", "legacy-eight"))
        if encoder.endswith("-five"):
            config_values["state_encoder"] = "strict-five"
        elif encoder.endswith("-graph"):
            config_values["state_encoder"] = "graph-moments"
        config = PromptPilotConfig(**config_values)
        if args.optimizer == "improved-network":
            if checkpoint.get("format_version") != 2:
                raise ValueError("improved-network requires a format_version=2 checkpoint")
            network = CandidateQNetwork(checkpoint.get("candidate_dim", 22)).to(device)
        else:
            first_layer = checkpoint["model_state_dict"]["layers.0.weight"]
            state_dim = int(checkpoint.get("state_dim", first_layer.shape[1]))
            network = QNetwork(state_dim=state_dim).to(device)
        network.load_state_dict(checkpoint["model_state_dict"])
        network.eval()
    elif args.optimizer == "release-qtable-diagnostic":
        if not args.checkpoint:
            raise ValueError(
                "--checkpoint must point to the released best-policy-table.pkl"
            )
        config = PromptPilotConfig()
    else:
        config = PromptPilotConfig()
    greedy_counts = None
    if args.greedy_counts:
        values = tuple(int(value) for value in args.greedy_counts.split(","))
        if len(values) != 2:
            raise ValueError("--greedy-counts must contain positive,negative counts")
        greedy_counts = values

    sys.path.insert(0, str(Path(args.sam_source).resolve()))
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry["vit_l"](checkpoint=args.sam_checkpoint).to(device)
    predictor = SamPredictor(sam)
    if args.optimizer == "release-qtable-diagnostic":
        # Loading the object-heavy table before SAM causes a large transient CPU peak.
        release_qtable = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    paths = select_sample_paths(args.bundle_dir, args.sample_list)[args.skip :]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError("No samples selected")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        for variant in ("initial", "optimized"):
            (output_dir / variant).mkdir(exist_ok=True)
    image_dir, mask_dir = Path(args.image_dir), Path(args.mask_dir)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    started = time.time()

    for offset, path in enumerate(paths):
        sample_index = args.skip + offset
        try:
            sample = load_graph_sample(path, device=device, require_both_categories=False)
            optimizable = bool(
                args.optimizer == "reference-prototype"
                or (sample.positive_indices.numel() > 0 and sample.negative_indices.numel() > 0)
            )
            if not optimizable:
                pos, neg = sample.positive_indices, sample.negative_indices
                trace = {
                    "total_reward": 0.0,
                    "action_counts": {},
                    "best_potential": float("nan"),
                }
            elif args.optimizer == "network":
                assert network is not None
                pos, neg, trace = optimize_prompts(
                    network,
                    sample,
                    config,
                    seed=args.seed + sample_index * args.policy_rollouts,
                    rollouts=args.policy_rollouts,
                    inference_epsilon=args.inference_epsilon,
                )
            elif args.optimizer == "improved-network":
                assert isinstance(network, CandidateQNetwork)
                pos, neg, trace = optimize_prompts_improved(
                    network,
                    sample,
                    config,
                    seed=args.seed + sample_index,
                    start_from_initial=True,
                )
            elif args.optimizer == "reference-prototype":
                pos, neg, trace = select_reference_prompts(
                    sample,
                    positive_count=args.positive_count,
                    negative_count=args.negative_count,
                    neighbors=args.prototype_neighbors,
                    smoothing=args.prototype_smoothing,
                    min_separation=args.prompt_min_separation,
                    candidate_source=args.prototype_candidate_source,
                    reference_bank=reference_bank,
                    score_mode=args.prototype_score_mode,
                    reference_classifier=reference_classifier,
                    reference_topk=args.reference_topk,
                )
            elif args.optimizer == "release-qtable-diagnostic":
                assert release_qtable is not None
                pos, neg, trace = optimize_prompts_release_qtable(
                    release_qtable,
                    sample,
                    seed=args.seed + sample_index,
                    epsilon=args.inference_epsilon,
                    max_steps=100,
                )
            else:
                pos, neg, trace = greedy_optimize_prompts(sample, config, target_counts=greedy_counts)
            image_path = find_file(image_dir, sample.name)
            target_path = find_file(mask_dir, sample.name)
            rgb = np.asarray(Image.open(image_path).convert("RGB").resize((sample.image_size, sample.image_size)))
            sam_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if args.image_format == "official-bgr" else rgb
            predictor.set_image(sam_image)
            initial, initial_quality = predict(
                predictor,
                sample.positive_indices.cpu(),
                sample.negative_indices.cpu(),
                sample.grid_size,
                sample.image_size,
                multimask_output=args.multimask_output,
                largest_component=args.largest_component,
            )
            candidate_box = (
                    score_guidance(
                        trace["_score_map"],
                        pos,
                        sample.grid_size,
                        sample.image_size,
                        args.prototype_mask_quantile,
                        args.prototype_mask_scale,
                    )[0]
                    if args.optimizer == "reference-prototype" and args.prototype_box
                    else None
                )
            candidate_mask_input = (
                    score_guidance(
                        trace["_score_map"],
                        pos,
                        sample.grid_size,
                        sample.image_size,
                        args.prototype_mask_quantile,
                        args.prototype_mask_scale,
                    )[1]
                    if args.optimizer == "reference-prototype" and args.prototype_mask_input
                    else None
                )
            candidate_selection_scores = (
                    trace["_score_map"]
                    if args.optimizer == "reference-prototype"
                    and args.candidate_mask_selection != "sam-quality"
                    else None
                )
            if args.per_positive_union:
                if candidate_box is not None or candidate_mask_input is not None:
                    raise ValueError("per-positive union cannot be combined with box or mask input")
                candidate, candidate_quality = predict_per_positive_union(
                    predictor,
                    pos,
                    neg,
                    sample.grid_size,
                    sample.image_size,
                    multimask_output=args.multimask_output,
                )
            else:
                candidate, candidate_quality = predict(
                    predictor,
                    pos,
                    neg,
                    sample.grid_size,
                    sample.image_size,
                    multimask_output=args.multimask_output,
                    largest_component=args.largest_component,
                    box=candidate_box,
                    mask_input=candidate_mask_input,
                    selection_scores=candidate_selection_scores,
                    mask_selection=args.candidate_mask_selection,
                )
            selected_candidate = optimizable and use_candidate(
                args.selection_gate,
                initial_quality,
                candidate_quality,
                args.gate_margin,
            )
            optimized = candidate if selected_candidate else initial
            selected_pos = pos if selected_candidate else sample.positive_indices
            selected_neg = neg if selected_candidate else sample.negative_indices
            target_raw = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
            target = cv2.resize(target_raw, (sample.image_size, sample.image_size), interpolation=cv2.INTER_LINEAR) > 127
            initial_metric = refine_mask(initial) if args.mask_refinement else initial
            optimized_metric = refine_mask(optimized) if args.mask_refinement else optimized
            target_metric = refine_mask(target) if args.mask_refinement else target
            row = {
                "name": sample.name,
                "initial_dice": dice(initial_metric, target_metric),
                "optimized_dice": dice(optimized_metric, target_metric),
                "initial_hd95": hd95(initial_metric, target_metric),
                "optimized_hd95": hd95(optimized_metric, target_metric),
                "initial_pos": int(sample.positive_indices.numel()),
                "initial_neg": int(sample.negative_indices.numel()),
                "candidate_pos": int(pos.numel()),
                "candidate_neg": int(neg.numel()),
                "optimized_pos": int(selected_pos.numel()),
                "optimized_neg": int(selected_neg.numel()),
                "initial_sam_quality": initial_quality,
                "candidate_sam_quality": candidate_quality,
                "selected_variant": "candidate" if selected_candidate else "initial",
                "optimizable": optimizable,
                "reward": trace["total_reward"],
                "restore_actions": trace["action_counts"].get("restore", 0),
                "delete_actions": trace["action_counts"].get("delete", 0),
                "move_actions": trace["action_counts"].get("move", 0),
                "forced_delete_actions": trace["action_counts"].get("forced_delete", 0),
                "improving_delete_actions": trace["action_counts"].get("improving_delete", 0),
                "swap_actions": trace["action_counts"].get("swap", 0),
                "stop_actions": trace["action_counts"].get("stop", 0),
                "best_graph_potential": trace.get("best_potential", float("nan")),
                "pretrained_qtable_key_hits": trace.get("pretrained_key_hits", 0),
                "qtable_candidate_lookups": trace.get("candidate_lookups", 0),
            }
            rows.append(row)
            if args.save_masks:
                cv2.imwrite(str(output_dir / "initial" / f"{sample.name}.png"), initial.astype(np.uint8) * 255)
                cv2.imwrite(str(output_dir / "optimized" / f"{sample.name}.png"), optimized.astype(np.uint8) * 255)
            if len(rows) % 10 == 0 or len(rows) == len(paths):
                print(f"evaluated={len(rows)}/{len(paths)} name={sample.name}", flush=True)
        except Exception as exc:
            failures.append({"sample": str(path), "error": str(exc)})
            print(f"ERROR sample={path}: {exc}", flush=True)

    if not rows:
        raise RuntimeError("Every sample failed")
    csv_path = output_dir / "per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def mean(key: str) -> float:
        return float(np.nanmean([float(row[key]) for row in rows]))

    dice_deltas = np.asarray(
        [float(row["optimized_dice"]) - float(row["initial_dice"]) for row in rows], dtype=np.float64
    )
    rewards = np.asarray([float(row["reward"]) for row in rows], dtype=np.float64)
    reward_dice_correlation = (
        float(np.corrcoef(rewards, dice_deltas)[0, 1])
        if len(rows) > 1 and rewards.std() > 0 and dice_deltas.std() > 0
        else None
    )
    summary = {
        "samples": len(rows),
        "selected_samples": len(paths),
        "sample_list": str(Path(args.sample_list).resolve()) if args.sample_list else None,
        "failed_samples": len(failures),
        "failures": failures,
        "initial": {
            "mDSC_percent": 100.0 * mean("initial_dice"),
            "mHD": mean("initial_hd95"),
            "mHD_public_implementation": "HD95 at 560-pixel evaluation size",
        },
        "optimized": {
            "mDSC_percent": 100.0 * mean("optimized_dice"),
            "mHD": mean("optimized_hd95"),
            "mHD_public_implementation": "HD95 at 560-pixel evaluation size",
        },
        "delta": {
            "mDSC_points": 100.0 * (mean("optimized_dice") - mean("initial_dice")),
            "mHD": mean("optimized_hd95") - mean("initial_hd95"),
        },
        "paired_analysis": {
            "improved": int(np.count_nonzero(dice_deltas > 0)),
            "degraded": int(np.count_nonzero(dice_deltas < 0)),
            "tied": int(np.count_nonzero(dice_deltas == 0)),
            "median_mDSC_change_points": 100.0 * float(np.median(dice_deltas)),
            "reward_dice_change_correlation": reward_dice_correlation,
        },
        "metric_note": "HD95 is evaluated after resizing masks to 560x560.",
        "optimizer": args.optimizer,
        "selection_gate": args.selection_gate,
        "gate_margin": args.gate_margin,
        "gate_selected_candidate": sum(row["selected_variant"] == "candidate" for row in rows),
        "gate_selected_initial": sum(row["selected_variant"] == "initial" for row in rows),
        "optimizable_samples": sum(bool(row["optimizable"]) for row in rows),
        "initial_only_samples": sum(not bool(row["optimizable"]) for row in rows),
        "greedy_counts": list(greedy_counts) if greedy_counts else None,
        "pretrained_qtable_key_hits": int(
            sum(int(row["pretrained_qtable_key_hits"]) for row in rows)
        ),
        "qtable_candidate_lookups": int(
            sum(int(row["qtable_candidate_lookups"]) for row in rows)
        ),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "checkpoint_format_version": checkpoint.get("format_version", 1) if checkpoint else None,
        "algorithm": checkpoint.get("algorithm", args.optimizer) if checkpoint else args.optimizer,
        "training_domain": checkpoint.get("training_domain", "unknown") if checkpoint else "not-applicable",
        "policy_rollouts": args.policy_rollouts,
        "inference_epsilon": args.inference_epsilon,
        "mask_refinement": args.mask_refinement,
        "image_format": args.image_format,
        "multimask_output": args.multimask_output,
        "largest_component": args.largest_component,
        "per_positive_union": args.per_positive_union,
        "reference_bank": str(Path(args.reference_bank).resolve()) if args.reference_bank else None,
        "reference_classifier": (
            str(Path(args.reference_classifier).resolve()) if args.reference_classifier else None
        ),
        "prototype_score_mode": args.prototype_score_mode,
        "reference_topk": args.reference_topk,
        "prototype_box": args.prototype_box,
        "prototype_mask_input": args.prototype_mask_input,
        "prototype_mask_quantile": args.prototype_mask_quantile,
        "prototype_mask_scale": args.prototype_mask_scale,
        "candidate_mask_selection": args.candidate_mask_selection,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
