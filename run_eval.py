#!/usr/bin/env python3
"""Run one dataset with a packaged prompt-optimization checkpoint or policy table."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PACKAGE_ROOT = Path(__file__).resolve().parent
MODES = {
    "pt-strict": {
        "optimizer": "network",
        "checkpoint": PACKAGE_ROOT
        / "weights"
        / "best-strict.pt",
        "epsilon": 0.0,
        "rollouts": 8,
    },
    "pt-graph": {
        "optimizer": "network",
        "checkpoint": PACKAGE_ROOT
        / "weights"
        / "best-graph.pt",
        "epsilon": 0.1,
        "rollouts": 8,
    },
    "pkl": {
        "optimizer": "release-qtable-diagnostic",
        "checkpoint": PACKAGE_ROOT / "weights" / "best-policy-table.pkl",
        "epsilon": 1.0,
        "rollouts": 1,
    },
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--inference-epsilon", type=float)
    parser.add_argument("--policy-rollouts", type=int)
    return parser.parse_known_args()


def main() -> None:
    args, extra = parse_args()
    selected = MODES[args.mode]
    checkpoint = Path(selected["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing packaged checkpoint: {checkpoint}")

    command = [
        sys.executable,
        "-m",
        "promptpilot.evaluate",
        "--optimizer",
        str(selected["optimizer"]),
        "--bundle-dir",
        args.bundle_dir,
        "--image-dir",
        args.image_dir,
        "--mask-dir",
        args.mask_dir,
        "--checkpoint",
        str(checkpoint),
        "--sam-checkpoint",
        args.sam_checkpoint,
        "--sam-source",
        str(PACKAGE_ROOT / "segmenter"),
        "--output-dir",
        args.output_dir,
        "--device",
        args.device,
        "--limit",
        str(args.limit),
        "--skip",
        str(args.skip),
        "--seed",
        str(args.seed),
        "--inference-epsilon",
        str(
            selected["epsilon"]
            if args.inference_epsilon is None
            else args.inference_epsilon
        ),
        "--policy-rollouts",
        str(
            selected["rollouts"]
            if args.policy_rollouts is None
            else args.policy_rollouts
        ),
        "--mask-refinement",
        *extra,
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PACKAGE_ROOT, check=True)


if __name__ == "__main__":
    main()
