"""Train the PromptPilot neural Q policy."""

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

from .core import (
    STRICT_STATE_DIM,
    GRAPH_STATE_DIM,
    STATE_DIM,
    PromptPilotConfig,
    QNetwork,
    DistanceGraphEnv,
    choose_action,
    config_dict,
    discover_samples,
    load_graph_sample,
    masked_q_values,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, help="Flat official cache or nested feature bundles")
    parser.add_argument("--output", required=True, help="Output .pt checkpoint")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--state-encoder",
        choices=("strict-five", "graph-moments", "legacy-eight"),
        default="graph-moments",
        help="graph-moments encodes the published heterogeneous state using relation moments",
    )
    parser.add_argument(
        "--distance-normalization",
        choices=("official-minmax", "fixed-scale", "none"),
        default="official-minmax",
        help="official-minmax follows the normalization in the released preprocessing code",
    )
    parser.add_argument(
        "--epsilon-decay-interval",
        choices=("step", "epoch"),
        default="epoch",
        help="Choose whether epsilon decays after each step or epoch",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--target-update", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument(
        "--preload-samples",
        action="store_true",
        help="Keep all feature bundles and normalization ranges in memory",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--training-domain",
        default="unspecified",
        help="Optional training-domain label stored in checkpoint metadata",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class ReplayTransition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    next_valid: torch.Tensor
    done: bool


def dqn_loss(
    online: QNetwork,
    target: QNetwork,
    transitions: list[ReplayTransition],
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    """Squared TD error from Equation (9), stabilized by a target network."""
    states = torch.stack([item.state for item in transitions]).to(device)
    actions = torch.tensor([item.action for item in transitions], device=device)
    predictions = online(states).gather(1, actions[:, None]).squeeze(1)
    rewards = torch.tensor(
        [item.reward for item in transitions], dtype=torch.float32, device=device
    )
    done = torch.tensor([item.done for item in transitions], dtype=torch.bool, device=device)
    next_states = torch.stack([item.next_state for item in transitions]).to(device)
    next_valid = torch.stack([item.next_valid for item in transitions]).to(device)
    with torch.no_grad():
        next_values = target(next_states).masked_fill(~next_valid, -torch.inf).max(dim=1).values
        targets = rewards + (~done).float() * gamma * next_values
    return F.mse_loss(predictions, targets)


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "replay_size": args.replay_size,
        "train_every": args.train_every,
        "target_update": args.target_update,
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
    samples = [load_graph_sample(path, device=device) for path in paths] if args.preload_samples else None

    config = PromptPilotConfig(
        max_steps=args.max_steps,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        state_encoder=args.state_encoder,
        distance_normalization=args.distance_normalization,
    )
    state_dims = {
        "strict-five": STRICT_STATE_DIM,
        "graph-moments": GRAPH_STATE_DIM,
        "legacy-eight": STATE_DIM,
    }
    state_dim = state_dims[args.state_encoder]
    network = QNetwork(state_dim=state_dim).to(device)
    target_network = QNetwork(state_dim=state_dim).to(device)
    target_network.load_state_dict(network.state_dict())
    target_network.eval()
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    replay: deque[ReplayTransition] = deque(maxlen=args.replay_size)
    rng = random.Random(args.seed)
    epsilon = config.epsilon_start
    history: list[dict[str, float | None]] = []
    global_step = 0
    update_count = 0
    started = time.time()

    for epoch in range(args.epochs):
        sample_index = rng.randrange(len(paths))
        sample = samples[sample_index] if samples is not None else load_graph_sample(paths[sample_index], device=device)
        env = DistanceGraphEnv(sample, config, seed=args.seed + epoch)
        state = env.reset()
        total_reward = 0.0
        losses: list[float] = []
        done = False
        network.train()

        while not done:
            valid = env.valid_actions()
            action = choose_action(network, state, valid, epsilon, rng)
            next_state, reward, done, _ = env.step(action)
            replay.append(
                ReplayTransition(
                    state=state.detach().cpu(),
                    action=action,
                    reward=reward,
                    next_state=next_state.detach().cpu(),
                    next_valid=env.valid_actions().detach().cpu(),
                    done=done,
                )
            )
            state = next_state
            total_reward += reward
            global_step += 1
            ready = global_step >= args.warmup_steps and len(replay) >= args.batch_size
            if ready and global_step % args.train_every == 0:
                loss = dqn_loss(
                    network,
                    target_network,
                    rng.sample(list(replay), args.batch_size),
                    config.gamma,
                    device,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 10.0)
                optimizer.step()
                losses.append(float(loss.item()))
                update_count += 1
            if global_step % args.target_update == 0:
                target_network.load_state_dict(network.state_dict())
            if args.epsilon_decay_interval == "step":
                epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

        history.append(
            {
                "epoch": float(epoch + 1),
                "reward": total_reward,
                "loss": float(np.mean(losses)) if losses else None,
                "epsilon": epsilon,
            }
        )
        if args.epsilon_decay_interval == "epoch":
            epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        if (epoch + 1) % args.log_every == 0 or epoch == 0 or epoch + 1 == args.epochs:
            window = history[-min(args.log_every, len(history)) :]
            mean_reward = float(np.mean([row["reward"] for row in window]))
            window_losses = [float(row["loss"]) for row in window if row["loss"] is not None]
            mean_loss = float(np.mean(window_losses)) if window_losses else float("nan")
            print(
                f"epoch={epoch + 1}/{args.epochs} epsilon={epsilon:.4f} "
                f"reward={mean_reward:.5f} loss={mean_loss:.6f}",
                flush=True,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": network.state_dict(),
        "target_state_dict": target_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "format_version": 3,
        "algorithm": "promptpilot-three-action-dqn",
        "config": config_dict(config),
        "state_dim": state_dim,
        "state_encoder": (
            "[Df(p,p), Df(p,n), Dp(p,p), Dp(n,n), Dp(p,n)]"
            if args.state_encoder == "strict-five"
            else (
                "five edge relations x [mean,std,min,max] plus normalized positive/negative counts"
                if args.state_encoder == "graph-moments"
                else "[Df(p,p), Df(p,n), Dp(p,p), Dp(n,n), Dp(p,n), Np/Nmax, Nn/Nmax, step/max_steps]"
            )
        ),
        "training_domain": args.training_domain,
        "source_bundle_dir": str(Path(args.bundle_dir).resolve()),
        "epochs": args.epochs,
        "seed": args.seed,
        "policy_config": {
            "max_steps": 100,
            "epochs": 1000,
            "gamma": 0.1,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay": 0.995,
            "hidden_layers": [1024, 64],
            "actions": ["restore", "delete", "move"],
            "loss": "squared TD error",
        },
        "engineering_assumptions": {
            "state_encoder": (
                "the five published graph means in Equations (3)-(7)"
                if args.state_encoder == "strict-five"
                else (
                    "moments of the five published heterogeneous edge relations plus node counts"
                    if args.state_encoder == "graph-moments"
                    else "five equation means plus counts and step"
                )
            ),
            "optimizer": "Adam",
            "learning_rate": config.learning_rate,
            "distance_normalization": args.distance_normalization,
            "move_neighborhood": "8-connected patch grid",
            "epsilon_decay_interval": args.epsilon_decay_interval,
            "training_sample_selection": "one uniformly sampled graph per epoch",
            "preloaded_samples": args.preload_samples,
            "replay_memory": args.replay_size,
            "batch_size": args.batch_size,
            "warmup_steps": args.warmup_steps,
            "train_every": args.train_every,
            "target_update_steps": args.target_update,
            "target_rule": "Equation (9) max over target Q-network",
            "method_invariants": (
                "five graph statistics; three operation outputs; random node/direction; potential-difference reward"
            ),
        },
        "history": history,
        "global_steps": global_step,
        "updates": update_count,
        "elapsed_seconds": time.time() - started,
    }
    torch.save(payload, output)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key not in {"model_state_dict", "target_state_dict", "optimizer_state_dict"}
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved={output}")


if __name__ == "__main__":
    main()
