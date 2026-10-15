"""Reference-conditioned prompt selection without target-mask access.

The released feature matcher projects every labelled reference patch to its
nearest target patch.  That produces hundreds of duplicated and conflicting
prompts on some domains.  This selector instead classifies each target patch
against foreground/background reference features, smooths the confidence map,
and keeps a small spatially diverse set of high-margin prompts.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .core import GraphSample


@dataclass
class ReferenceBank:
    features: torch.Tensor
    mask_paths: list[str]
    image_size: int
    patch_size: int
    foreground_features: torch.Tensor
    background_features: torch.Tensor
    group_features: dict[str, tuple[torch.Tensor, torch.Tensor]]
    group_entries: dict[str, list[tuple[torch.Tensor, torch.Tensor]]]


class ReferenceFeatureClassifier(torch.nn.Module):
    """Small patch classifier trained only from annotated reference images."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(F.normalize(features.float(), dim=-1)).squeeze(-1)


def load_reference_bank(path: str | Path, device: str | torch.device) -> ReferenceBank:
    payload = torch.load(path, map_location=device, weights_only=False)
    features = payload["features"].float().to(device)
    mask_paths = [str(value) for value in payload["mask_paths"]]
    image_size = int(payload["image_size"])
    patch_size = int(payload["patch_size"])
    grid = image_size // patch_size
    positive: list[torch.Tensor] = []
    negative: list[torch.Tensor] = []
    group_labels = [str(value) for value in payload.get("group_labels", ["__all__"] * len(mask_paths))]
    grouped_positive: dict[str, list[torch.Tensor]] = {}
    grouped_negative: dict[str, list[torch.Tensor]] = {}
    group_entries: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for reference_features, mask_path, group_label in zip(features, mask_paths, group_labels):
        foreground, background = _labels_from_mask(Path(mask_path), image_size, grid)
        foreground_features = reference_features[foreground.to(reference_features.device)]
        background_features = reference_features[background.to(reference_features.device)]
        positive.append(foreground_features)
        negative.append(background_features)
        grouped_positive.setdefault(group_label, []).append(foreground_features)
        grouped_negative.setdefault(group_label, []).append(background_features)
        group_entries.setdefault(group_label, []).append(
            (foreground_features, background_features)
        )
    group_features = {
        label: (torch.cat(values), torch.cat(grouped_negative[label]))
        for label, values in grouped_positive.items()
    }
    return ReferenceBank(
        features=features,
        mask_paths=mask_paths,
        image_size=image_size,
        patch_size=patch_size,
        foreground_features=torch.cat(positive),
        background_features=torch.cat(negative),
        group_features=group_features,
        group_entries=group_entries,
    )


def load_reference_classifier(
    path: str | Path, device: str | torch.device
) -> ReferenceFeatureClassifier:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = ReferenceFeatureClassifier(
        feature_dim=int(payload["feature_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _labels_from_mask(
    mask_path: Path,
    image_size: int,
    grid: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not mask_path.exists():
        raise FileNotFoundError(f"Reference mask does not exist: {mask_path}")
    mask = Image.open(mask_path).convert("L").resize(
        (image_size, image_size), Image.Resampling.NEAREST
    )
    values = torch.from_numpy(np.asarray(mask, dtype=np.float32).copy() / 255.0)
    patch = image_size // grid
    if patch * grid != image_size:
        raise ValueError("Image size is not divisible by feature grid")
    fractions = values.reshape(grid, patch, grid, patch).mean(dim=(1, 3)).flatten()
    foreground = torch.nonzero(fractions >= 0.8, as_tuple=False).flatten()
    background = torch.nonzero(fractions <= 0.2, as_tuple=False).flatten()
    if foreground.numel() == 0 or background.numel() == 0:
        raise ValueError(f"{mask_path}: no confident patch labels")
    return foreground, background


def _reference_sets(
    sample: GraphSample,
    bank: ReferenceBank | None,
    reference_topk: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = sample.features.device
    if bank is not None:
        if bank.image_size != sample.image_size or bank.patch_size != sample.patch_size:
            raise ValueError("Reference bank geometry differs from target bundle")
        class_name = str(sample.metadata.get("class_name", "__all__"))
        entries = bank.group_entries.get(class_name, bank.group_entries.get("__all__", []))
        if reference_topk > 0 and len(entries) > reference_topk:
            target = F.normalize(sample.features, dim=1)
            similarities: list[torch.Tensor] = []
            for foreground_features, _ in entries:
                centroid = F.normalize(foreground_features.mean(dim=0), dim=0)
                patch_similarity = target @ centroid.to(target.device)
                tail = max(1, patch_similarity.numel() // 10)
                similarities.append(patch_similarity.topk(tail).values.mean())
            selected = torch.topk(torch.stack(similarities), reference_topk).indices.tolist()
            return (
                torch.cat([entries[index][0] for index in selected]).to(device),
                torch.cat([entries[index][1] for index in selected]).to(device),
            )
        foreground, background = bank.group_features.get(
            class_name, (bank.foreground_features, bank.background_features)
        )
        return foreground.to(device), background.to(device)
    if sample.reference_features is None:
        raise ValueError(f"{sample.name}: bundle does not contain reference features")
    mask_value = sample.metadata.get("reference_mask")
    if not mask_value:
        raise ValueError(f"{sample.name}: reference_mask is missing from bundle metadata")
    foreground, background = _labels_from_mask(
        Path(str(mask_value)), sample.image_size, sample.grid_size
    )
    reference = sample.reference_features.to(device)
    return reference[foreground.to(device)], reference[background.to(device)]


def reference_margin(
    sample: GraphSample,
    neighbors: int = 5,
    smoothing: int = 3,
    reference_bank: ReferenceBank | None = None,
    reference_topk: int = 0,
) -> torch.Tensor:
    """Return positive-vs-negative cosine margin for every target patch."""
    device = sample.features.device
    foreground_features, background_features = _reference_sets(
        sample, reference_bank, reference_topk
    )
    foreground_features = F.normalize(foreground_features, dim=1)
    background_features = F.normalize(background_features, dim=1)
    target = F.normalize(sample.features, dim=1)
    pos_k = min(neighbors, foreground_features.shape[0])
    neg_k = min(neighbors, background_features.shape[0])
    pos = (target @ foreground_features.T).topk(pos_k, dim=1).values.mean(dim=1)
    neg = (target @ background_features.T).topk(neg_k, dim=1).values.mean(dim=1)
    margin = pos - neg
    if smoothing > 1:
        grid = sample.grid_size
        margin = F.avg_pool2d(
            margin.reshape(1, 1, grid, grid),
            kernel_size=smoothing,
            stride=1,
            padding=smoothing // 2,
            count_include_pad=False,
        ).flatten()
    return margin


def reference_vote_margin(
    sample: GraphSample,
    smoothing: int = 3,
    reference_bank: ReferenceBank | None = None,
    reference_topk: int = 0,
) -> torch.Tensor:
    """Score patches by normalized forward-matching foreground/background votes."""
    foreground_features, background_features = _reference_sets(
        sample, reference_bank, reference_topk
    )
    target = F.normalize(sample.features, dim=1)
    foreground_features = F.normalize(foreground_features, dim=1)
    background_features = F.normalize(background_features, dim=1)
    positive_matches = (foreground_features @ target.T).argmax(dim=1)
    negative_matches = (background_features @ target.T).argmax(dim=1)
    positive_votes = torch.bincount(
        positive_matches, minlength=target.shape[0]
    ).float() / max(foreground_features.shape[0], 1)
    negative_votes = torch.bincount(
        negative_matches, minlength=target.shape[0]
    ).float() / max(background_features.shape[0], 1)
    score = positive_votes - negative_votes
    if smoothing > 1:
        grid = sample.grid_size
        score = F.avg_pool2d(
            score.reshape(1, 1, grid, grid),
            kernel_size=smoothing,
            stride=1,
            padding=smoothing // 2,
            count_include_pad=False,
        ).flatten()
    return score


def classifier_margin(
    sample: GraphSample,
    classifier: ReferenceFeatureClassifier,
    smoothing: int = 3,
) -> torch.Tensor:
    """Return a learned foreground logit for every target patch."""
    with torch.no_grad():
        score = classifier(sample.features)
    if smoothing > 1:
        grid = sample.grid_size
        score = F.avg_pool2d(
            score.reshape(1, 1, grid, grid),
            kernel_size=smoothing,
            stride=1,
            padding=smoothing // 2,
            count_include_pad=False,
        ).flatten()
    return score


def _diverse_topk(
    scores: torch.Tensor,
    count: int,
    grid_size: int,
    largest: bool,
    min_separation: float,
    allowed: torch.Tensor | None = None,
) -> torch.Tensor:
    if allowed is None:
        allowed = torch.arange(scores.numel(), device=scores.device)
    order_within_allowed = torch.argsort(scores[allowed], descending=largest)
    order = allowed[order_within_allowed].detach().cpu().tolist()
    selected: list[int] = []
    minimum_squared = min_separation * min_separation
    for index in order:
        row, col = divmod(index, grid_size)
        if all((row - other // grid_size) ** 2 + (col - other % grid_size) ** 2 >= minimum_squared for other in selected):
            selected.append(index)
            if len(selected) == count:
                break
    if len(selected) < count:
        selected_set = set(selected)
        selected.extend(index for index in order if index not in selected_set)
    return torch.tensor(selected[:count], dtype=torch.long, device=scores.device)


def select_reference_prompts(
    sample: GraphSample,
    positive_count: int = 6,
    negative_count: int = 12,
    neighbors: int = 5,
    smoothing: int = 3,
    min_separation: float = 4.0,
    candidate_source: str = "all",
    reference_bank: ReferenceBank | None = None,
    score_mode: str = "margin",
    reference_classifier: ReferenceFeatureClassifier | None = None,
    reference_topk: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Select confident reference-conditioned prompts with spatial coverage."""
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("Prompt counts must be positive")
    margin_score = reference_margin(
        sample,
        neighbors=neighbors,
        smoothing=smoothing,
        reference_bank=reference_bank,
        reference_topk=reference_topk,
    )
    vote_score = reference_vote_margin(
        sample,
        smoothing=smoothing,
        reference_bank=reference_bank,
        reference_topk=reference_topk,
    )
    linear_score = (
        classifier_margin(sample, reference_classifier, smoothing=smoothing)
        if reference_classifier is not None
        else None
    )
    if score_mode == "margin":
        margin = margin_score
    elif score_mode == "vote":
        margin = vote_score
    elif score_mode == "hybrid":
        def standardize(values: torch.Tensor) -> torch.Tensor:
            return (values - values.mean()) / values.std().clamp_min(1e-6)
        margin = standardize(margin_score) + standardize(vote_score)
    elif score_mode == "linear":
        if linear_score is None:
            raise ValueError("linear score mode requires a reference classifier")
        margin = linear_score
    elif score_mode == "hybrid-linear":
        if linear_score is None:
            raise ValueError("hybrid-linear score mode requires a reference classifier")
        def standardize(values: torch.Tensor) -> torch.Tensor:
            return (values - values.mean()) / values.std().clamp_min(1e-6)
        margin = standardize(margin_score) + standardize(linear_score)
    else:
        raise ValueError(f"Unknown score mode: {score_mode}")
    if candidate_source == "all":
        positive_allowed = torch.arange(margin.numel(), device=margin.device)
        negative_allowed = positive_allowed
    elif candidate_source == "initial":
        positive_allowed = sample.positive_indices.to(margin.device)
        negative_allowed = sample.negative_indices.to(margin.device)
    else:
        raise ValueError(f"Unknown candidate source: {candidate_source}")
    positive_count = min(positive_count, positive_allowed.numel())
    negative_count = min(negative_count, negative_allowed.numel())
    positive = _diverse_topk(
        margin,
        positive_count,
        sample.grid_size,
        largest=True,
        min_separation=min_separation,
        allowed=positive_allowed,
    )
    negative = _diverse_topk(
        margin,
        negative_count,
        sample.grid_size,
        largest=False,
        min_separation=min_separation,
        allowed=negative_allowed[~torch.isin(negative_allowed, positive)],
    )
    trace = {
        "total_reward": 0.0,
        "action_counts": {"reference_select": positive_count + negative_count},
        "best_potential": float("nan"),
        "margin_mean": float(margin.mean().item()),
        "margin_std": float(margin.std().item()),
        "counts": [positive_count, negative_count],
        "candidate_source": candidate_source,
        "score_mode": score_mode,
        "reference_topk": reference_topk,
        "_score_map": margin.detach().cpu(),
    }
    return positive.cpu(), negative.cpu(), trace
