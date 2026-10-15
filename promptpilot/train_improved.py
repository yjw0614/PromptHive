"""Train the node-aware PromptPilot-v2 policy with replay and Double DQN targets."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .core import PromptPilotConfig, config_dict, discover_samples, load_graph_sample, seed_everything
from .improved import (
    CANDIDATE_DIM,
    CandidateActionEnv,
    CandidateQNetwork,
    choose_candidate,
)


@dataclass
class Transition:
    descriptor: torch.Tensor
    reward: float
    next_descriptors: torch.Tensor
    done: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--target-update", type=int, default=250)
    parser.add_argument("--max-next-candidates", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--training-domain", default="unspecified")
    return parser.parse_args()


def sample_descriptors(
    descriptors: torch.Tensor,
    limit: int,
    rng: random.Random,
    network: CandidateQNetwork | None = None,
) -> torch.Tensor:
    detached = descriptors.detach().cpu()
    if limit <= 0 or len(detached) <= limit:
        return detached
    best = None
    if network is not None:
        with torch.no_grad():
            best = int(network(descriptors).argmax().item())
    pool = [index for index in range(len(detached)) if index != best]
    indices = rng.sample(pool, limit - int(best is not None))
    if best is not None:
        indices.append(best)
    return detached[indices]


def double_dqn_loss(
    online: CandidateQNetwork,
    target: CandidateQNetwork,
    transitions: list[Transition],
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    descriptors = torch.stack([item.descriptor for item in transitions]).to(device)
    predictions = online(descriptors)
    rewards = torch.tensor([item.reward for item in transitions], dtype=torch.float32, device=device)
    targets = rewards.clone()

    with torch.no_grad():
        active = [
            (index, item.next_descriptors)
            for index, item in enumerate(transitions)
            if not item.done and item.next_descriptors.numel() > 0
        ]
        if active:
            lengths = [len(descriptors) for _, descriptors in active]
            candidates = torch.cat([descriptors for _, descriptors in active]).to(device)
            online_chunks = torch.split(online(candidates), lengths)
            target_chunks = torch.split(target(candidates), lengths)
            for (index, _), online_values, target_values in zip(active, online_chunks, target_chunks):
                best = int(online_values.argmax().item())
                targets[index] += gamma * target_values[best]
    return F.smooth_l1_loss(predictions, targets)


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "replay_size": args.replay_size,
        "train_every": args.train_every,
        "target_update": args.target_update,
        "max_next_candidates": args.max_next_candidates,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"Arguments must be positive: {', '.join(invalid)}")
    if args.warmup_steps < 0:
        raise ValueError("warmup-steps cannot be negative")
    if args.replay_size < args.batch_size:
        raise ValueError("replay-size must be at least batch-size")


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    paths = discover_samples(args.bundle_dir)
    if args.limit_samples:
        paths = paths[: args.limit_samples]
    if not paths:
        raise FileNotFoundError(f"No feature bundles found in {args.bundle_dir}")

    config = PromptPilotConfig(
        max_steps=args.max_steps,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
    )
    online = CandidateQNetwork().to(device)
    target = CandidateQNetwork().to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.AdamW(online.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    replay: deque[Transition] = deque(maxlen=args.replay_size)
    rng = random.Random(args.seed)
    epsilon = config.epsilon_start
    history: list[dict[str, float]] = []
    global_step = 0
    update_count = 0
    started = time.time()

    for epoch in range(args.epochs):
        sample_path = paths[rng.randrange(len(paths))]
        sample = load_graph_sample(sample_path, device=device)
        env = CandidateActionEnv(sample, config, seed=args.seed + epoch)
        env.reset_to_initial_prompts(prune_to_bounds=True)
        total_reward = 0.0
        losses: list[float] = []
        done = False
        online.train()

        while not done:
            actions = env.candidate_actions()
            descriptors = env.candidate_descriptors(actions)
            selected = choose_candidate(online, descriptors, actions, epsilon, rng)
            action = actions[selected]
            _, reward, done, _ = env.step_candidate(action)
            next_descriptors = torch.empty((0, CANDIDATE_DIM))
            if not done:
                next_actions = env.candidate_actions()
                next_descriptors = sample_descriptors(
                    env.candidate_descriptors(next_actions),
                    args.max_next_candidates,
                    rng,
                    online,
                )
            replay.append(
                Transition(
                    descriptor=descriptors[selected].detach().cpu(),
                    reward=reward,
                    next_descriptors=next_descriptors,
                    done=done,
                )
            )
            total_reward += reward
            global_step += 1

            ready = global_step >= args.warmup_steps and len(replay) >= args.batch_size
            if ready and global_step % args.train_every == 0:
                batch = rng.sample(list(replay), args.batch_size)
                loss = double_dqn_loss(online, target, batch, config.gamma, device)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.item()))
                update_count += 1
            if global_step % args.target_update == 0:
                target.load_state_dict(online.state_dict())

        history.append(
            {
                "epoch": float(epoch + 1),
                "reward": total_reward,
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "epsilon": epsilon,
                "steps": float(env.steps),
            }
        )
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        if (epoch + 1) % args.log_every == 0 or epoch == 0 or epoch + 1 == args.epochs:
            window = history[-min(args.log_every, len(history)) :]
            mean_reward = float(np.mean([row["reward"] for row in window]))
            finite_losses = [row["loss"] for row in window if np.isfinite(row["loss"])]
            mean_loss = float(np.mean(finite_losses)) if finite_losses else float("nan")
            print(
                f"epoch={epoch + 1}/{args.epochs} epsilon={epsilon:.4f} "
                f"reward={mean_reward:.5f} loss={mean_loss:.6f} "
                f"replay={len(replay)} updates={update_count}",
                flush=True,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "algorithm": "node-aware-double-dqn",
        "model_state_dict": online.state_dict(),
        "target_state_dict": target.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config_dict(config),
        "candidate_dim": CANDIDATE_DIM,
        "training_domain": args.training_domain,
        "source_bundle_dir": str(Path(args.bundle_dir).resolve()),
        "epochs": args.epochs,
        "seed": args.seed,
        "training": {
            "optimizer": "AdamW",
            "batch_size": args.batch_size,
            "replay_size": args.replay_size,
            "warmup_steps": args.warmup_steps,
            "train_every": args.train_every,
            "target_update": args.target_update,
            "max_next_candidates": args.max_next_candidates,
            "loss": "Huber",
            "target": "Double DQN",
        },
        "differences_from_release": [
            "scores concrete node-level actions instead of random nodes",
            "uses the same bounded feature-matching initialization in training and inference",
            "adds an explicit stop action",
            "uses replay memory and a target network",
            "uses Double DQN targets and Huber loss",
            "uses AdamW and gradient clipping",
        ],
        "history": history,
        "global_steps": global_step,
        "updates": update_count,
        "elapsed_seconds": time.time() - started,
    }
    torch.save(payload, output)
    json_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"model_state_dict", "target_state_dict", "optimizer_state_dict"}
    }
    output.with_suffix(".json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
