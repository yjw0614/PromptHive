"""Core graph environment and Q-network for PromptPilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math
import random
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from torch import nn


ACTIONS = ("restore", "delete", "move")
RESTORE, DELETE, MOVE = range(3)
STATE_DIM = 8
STRICT_STATE_DIM = 5
GRAPH_STATE_DIM = 22


@dataclass(frozen=True)
class PromptPilotConfig:
    image_size: int = 560
    patch_size: int = 14
    min_nodes: int = 5
    max_nodes: int = 20
    max_steps: int = 100
    alpha: float = 5.0
    beta: float = 1.0
    gamma: float = 0.1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995
    learning_rate: float = 1e-3
    state_encoder: str = "legacy-eight"
    distance_normalization: str = "fixed-scale"


@dataclass
class GraphSample:
    name: str
    features: torch.Tensor
    positive_indices: torch.Tensor
    negative_indices: torch.Tensor
    reference_features: torch.Tensor | None = None
    metadata: dict[str, object] = field(default_factory=dict, repr=False)
    image_size: int = 560
    patch_size: int = 14
    distance_ranges: dict[str, tuple[float, float]] = field(default_factory=dict, repr=False)

    @property
    def grid_size(self) -> int:
        n = int(round(math.sqrt(self.features.shape[0])))
        if n * n != self.features.shape[0]:
            raise ValueError(f"{self.name}: feature count is not a square grid")
        return n


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, got {type(value)!r}")
    return value


def load_graph_sample(
    path: str | Path,
    device: str | torch.device = "cpu",
    require_both_categories: bool = True,
) -> GraphSample:
    """Load either the official flat cache or a nested feature bundle."""
    path = Path(path)
    if path.is_dir():
        metadata_path = path / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        feature_path = path / "features.pt"
        pos_path = path / "initial_indices_pos.pt"
        neg_path = path / "initial_indices_neg.pt"
        name = path.name
        image_size = int(metadata.get("image_size", 560))
        patch_size = int(metadata.get("patch_size", image_size // 40))
    else:
        suffix = "_features.pt"
        if not path.name.endswith(suffix):
            raise ValueError(f"Flat feature file must end in {suffix}: {path}")
        prefix = path.name[: -len(suffix)]
        feature_path = path
        pos_path = path.with_name(prefix + "_initial_indices_pos.pt")
        neg_path = path.with_name(prefix + "_initial_indices_neg.pt")
        name = Path(prefix).stem
        image_size = 560
        patch_size = 14

    features = _safe_load_tensor(feature_path)
    reference_features = None
    if features.ndim == 3:
        if features.shape[0] >= 2:
            reference_features = features[0]
        features = features[-1]
    if features.ndim != 2:
        raise ValueError(f"{feature_path}: expected [N,D] or [2,N,D], got {tuple(features.shape)}")

    pos = torch.unique(_safe_load_tensor(pos_path).long().flatten())
    neg = torch.unique(_safe_load_tensor(neg_path).long().flatten())
    pos_overlap = torch.isin(pos, neg)
    neg_overlap = torch.isin(neg, pos)
    pos = pos[~pos_overlap]
    neg = neg[~neg_overlap]
    n = features.shape[0]
    pos = pos[(pos >= 0) & (pos < n)]
    neg = neg[(neg >= 0) & (neg < n)]
    if pos.numel() == 0 and neg.numel() == 0:
        raise ValueError(f"{name}: no prompt candidates are available")
    if require_both_categories and (pos.numel() == 0 or neg.numel() == 0):
        raise ValueError(f"{name}: both positive and negative candidates are required")

    return GraphSample(
        name=name,
        features=features.float().to(device),
        positive_indices=pos.to(device),
        negative_indices=neg.to(device),
        reference_features=(
            reference_features.float().to(device) if reference_features is not None else None
        ),
        metadata=metadata if path.is_dir() else {},
        image_size=image_size,
        patch_size=patch_size,
    )


def discover_samples(bundle_dir: str | Path) -> list[Path]:
    root = Path(bundle_dir)
    nested = sorted(p for p in root.iterdir() if p.is_dir() and (p / "features.pt").exists())
    if nested:
        return nested
    return sorted(root.glob("*_features.pt"))


class DistanceGraphEnv:
    """Variable-node graph environment combining feature and spatial distances."""

    def __init__(self, sample: GraphSample, config: PromptPilotConfig, seed: int = 0):
        self.sample = sample
        self.config = config
        self.rng = random.Random(seed)
        self.features = sample.features
        self.grid_size = sample.grid_size
        self._original_index: dict[int, int] = {}
        self._category: dict[int, int] = {}
        uid = 0
        for category, indices in ((1, sample.positive_indices), (0, sample.negative_indices)):
            for index in indices.detach().cpu().tolist():
                self._original_index[uid] = int(index)
                self._category[uid] = category
                uid += 1
        self._feature_scale = max(2.0 * float(self.features.norm(dim=1).mean().item()), 1e-6)
        self._physical_scale = math.sqrt(2.0) * max(self.grid_size - 1, 1)
        self._distance_ranges = self._build_distance_ranges()
        self.active: dict[int, int] = {}
        self.removed: set[int] = set()
        self.steps = 0
        self._stats = np.zeros(5, dtype=np.float32)
        self.reset()

    def reset(self) -> torch.Tensor:
        self.active = {}
        self.removed = set(self._original_index)
        self.steps = 0
        self._stats = self.graph_statistics()
        return self.state_vector()

    def _uids(self, category: int, active: bool = True) -> list[int]:
        source: Iterable[int] = self.active if active else self.removed
        return [uid for uid in source if self._category[uid] == category]

    @property
    def positive_indices(self) -> list[int]:
        return sorted(self.active[uid] for uid in self._uids(1))

    @property
    def negative_indices(self) -> list[int]:
        return sorted(self.active[uid] for uid in self._uids(0))

    def counts(self) -> tuple[int, int]:
        return len(self._uids(1)), len(self._uids(0))

    def valid_actions(self) -> torch.Tensor:
        n_pos, n_neg = self.counts()
        can_restore = bool(self.removed)
        can_delete = bool(self.active)
        can_move = bool(self.active)
        if n_pos < self.config.min_nodes or n_neg < self.config.min_nodes:
            mask = [can_restore, False, False]
        elif n_pos > self.config.max_nodes or n_neg > self.config.max_nodes:
            mask = [False, can_delete, False]
        else:
            mask = [can_restore, can_delete, can_move]
        if not any(mask):
            mask[MOVE] = can_move
        return torch.tensor(mask, dtype=torch.bool, device=self.features.device)

    def _distance_range(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        physical: bool,
    ) -> tuple[float, float]:
        if left.numel() == 0 or right.numel() == 0:
            return 0.0, 1.0
        if physical:
            values = torch.cdist(self._points(left.tolist()), self._points(right.tolist()))
        else:
            values = torch.cdist(self.features[left], self.features[right])
        minimum = float(values.min().item())
        maximum = float(values.max().item())
        return minimum, max(maximum - minimum, 1e-6)

    def _build_distance_ranges(self) -> dict[str, tuple[float, float]]:
        if self.sample.distance_ranges:
            return self.sample.distance_ranges
        pos = self.sample.positive_indices
        neg = self.sample.negative_indices
        ranges = {
            "f_pp": self._distance_range(pos, pos, physical=False),
            "f_pn": self._distance_range(pos, neg, physical=False),
            "p_pp": self._distance_range(pos, pos, physical=True),
            "p_nn": self._distance_range(neg, neg, physical=True),
            "p_pn": self._distance_range(pos, neg, physical=True),
        }
        self.sample.distance_ranges = ranges
        return ranges

    def _normalize_mean(self, value: torch.Tensor, relation: str, scale: float) -> float:
        mean = float(value.mean().item())
        if self.config.distance_normalization == "official-minmax":
            minimum, denominator = self._distance_ranges[relation]
            return (mean - minimum) / denominator
        if self.config.distance_normalization == "fixed-scale":
            return mean / scale
        if self.config.distance_normalization == "none":
            return mean
        raise ValueError(f"Unknown distance normalization: {self.config.distance_normalization}")

    def _normalize_values(
        self, values: torch.Tensor, relation: str, scale: float
    ) -> torch.Tensor:
        if self.config.distance_normalization == "official-minmax":
            minimum, denominator = self._distance_ranges[relation]
            return (values - minimum) / denominator
        if self.config.distance_normalization == "fixed-scale":
            return values / scale
        if self.config.distance_normalization == "none":
            return values
        raise ValueError(f"Unknown distance normalization: {self.config.distance_normalization}")

    def _mean_feature_distance(
        self,
        left: Sequence[int],
        right: Sequence[int],
        relation: str,
    ) -> float:
        if not left or not right:
            return 0.0
        a = self.features[torch.as_tensor(left, device=self.features.device)]
        b = self.features[torch.as_tensor(right, device=self.features.device)]
        return self._normalize_mean(torch.cdist(a, b), relation, self._feature_scale)

    def _points(self, indices: Sequence[int]) -> torch.Tensor:
        values = torch.as_tensor(indices, dtype=torch.long, device=self.features.device)
        rows = torch.div(values, self.grid_size, rounding_mode="floor")
        cols = values.remainder(self.grid_size)
        return torch.stack((cols, rows), dim=1).float()

    def _mean_physical_distance(
        self,
        left: Sequence[int],
        right: Sequence[int],
        relation: str,
    ) -> float:
        if not left or not right:
            return 0.0
        return self._normalize_mean(
            torch.cdist(self._points(left), self._points(right)),
            relation,
            self._physical_scale,
        )

    def graph_statistics(self) -> np.ndarray:
        pos, neg = self.positive_indices, self.negative_indices
        return np.asarray(
            [
                self._mean_feature_distance(pos, pos, "f_pp"),
                self._mean_feature_distance(pos, neg, "f_pn"),
                self._mean_physical_distance(pos, pos, "p_pp"),
                self._mean_physical_distance(neg, neg, "p_nn"),
                self._mean_physical_distance(pos, neg, "p_pn"),
            ],
            dtype=np.float32,
        )

    def graph_distribution_state(self) -> torch.Tensor:
        """Encode the current dual-space graph without changing its reward.

        Relation moments retain five distance means and add distribution
        information from the same graph edges plus the two node counts.
        """
        pos, neg = self.positive_indices, self.negative_indices

        def values(
            left: Sequence[int],
            right: Sequence[int],
            relation: str,
            physical: bool,
        ) -> torch.Tensor:
            if not left or not right:
                return torch.zeros(1, device=self.features.device)
            if physical:
                raw = torch.cdist(self._points(left), self._points(right))
                scale = self._physical_scale
            else:
                left_features = self.features[
                    torch.as_tensor(left, device=self.features.device)
                ]
                right_features = self.features[
                    torch.as_tensor(right, device=self.features.device)
                ]
                raw = torch.cdist(left_features, right_features)
                scale = self._feature_scale
            return self._normalize_values(raw, relation, scale).flatten()

        relations = (
            values(pos, pos, "f_pp", False),
            values(pos, neg, "f_pn", False),
            values(pos, pos, "p_pp", True),
            values(neg, neg, "p_nn", True),
            values(pos, neg, "p_pn", True),
        )
        moments: list[torch.Tensor] = []
        for relation in relations:
            moments.extend(
                (
                    relation.mean(),
                    relation.std(unbiased=False),
                    relation.min(),
                    relation.max(),
                )
            )
        n_pos, n_neg = self.counts()
        counts = torch.tensor(
            [n_pos / self.config.max_nodes, n_neg / self.config.max_nodes],
            dtype=torch.float32,
            device=self.features.device,
        )
        return torch.cat((torch.stack(moments).float(), counts))

    def state_vector(self) -> torch.Tensor:
        if self.config.state_encoder == "strict-five":
            return torch.from_numpy(self._stats.copy()).to(self.features.device)
        if self.config.state_encoder == "graph-moments":
            return self.graph_distribution_state()
        if self.config.state_encoder != "legacy-eight":
            raise ValueError(f"Unknown state encoder: {self.config.state_encoder}")
        n_pos, n_neg = self.counts()
        extra = np.asarray(
            [
                n_pos / self.config.max_nodes,
                n_neg / self.config.max_nodes,
                self.steps / self.config.max_steps,
            ],
            dtype=np.float32,
        )
        return torch.from_numpy(np.concatenate((self._stats, extra))).to(self.features.device)

    def _restore(self) -> None:
        if not self.removed:
            return
        n_pos, n_neg = self.counts()
        if n_pos < self.config.min_nodes:
            choices = self._uids(1, active=False)
        elif n_neg < self.config.min_nodes:
            choices = self._uids(0, active=False)
        else:
            choices = list(self.removed)
        if not choices:
            choices = list(self.removed)
        uid = self.rng.choice(choices)
        self.removed.remove(uid)
        self.active[uid] = self._original_index[uid]

    def _delete(self) -> None:
        if not self.active:
            return
        n_pos, n_neg = self.counts()
        if n_pos > self.config.max_nodes:
            choices = self._uids(1)
        elif n_neg > self.config.max_nodes:
            choices = self._uids(0)
        else:
            choices = list(self.active)
        uid = self.rng.choice(choices)
        del self.active[uid]
        self.removed.add(uid)

    def _move(self) -> None:
        if not self.active:
            return
        uid = self.rng.choice(list(self.active))
        index = self.active[uid]
        row, col = divmod(index, self.grid_size)
        occupied = set(self.active.values())
        neighbors = []
        for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            nr, nc = row + dr, col + dc
            ni = nr * self.grid_size + nc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size and ni not in occupied:
                neighbors.append(ni)
        if neighbors:
            self.active[uid] = self.rng.choice(neighbors)

    def potential(self, stats: np.ndarray) -> float:
        f_pp, f_pn, p_pp, p_nn, p_pn = stats
        return float(
            -self.config.alpha * f_pp
            + self.config.alpha * f_pn
            + self.config.beta * p_pp
            + self.config.beta * p_nn
            - self.config.beta * p_pn
        )

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, dict[str, object]]:
        if not 0 <= action < len(ACTIONS):
            raise ValueError(f"Unknown action index: {action}")
        if not bool(self.valid_actions()[action]):
            raise ValueError(f"Action {ACTIONS[action]} is invalid for counts={self.counts()}")
        previous = self._stats.copy()
        (self._restore, self._delete, self._move)[action]()
        self.steps += 1
        self._stats = self.graph_statistics()
        reward = self.potential(self._stats) - self.potential(previous)
        done = self.steps >= self.config.max_steps
        return self.state_vector(), reward, done, {"statistics": self._stats.copy(), "counts": self.counts()}


class QNetwork(nn.Module):
    """A 1024/64 hidden-layer Q-network with three action outputs."""

    def __init__(self, state_dim: int = STATE_DIM):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, len(ACTIONS)),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


def masked_q_values(q_values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return q_values.masked_fill(~valid, -torch.inf)


def choose_action(
    network: QNetwork,
    state: torch.Tensor,
    valid: torch.Tensor,
    epsilon: float,
    rng: random.Random,
) -> int:
    choices = torch.nonzero(valid, as_tuple=False).flatten().detach().cpu().tolist()
    if not choices:
        raise RuntimeError("Environment has no valid action")
    if rng.random() < epsilon:
        # Section 4.1 sets move probability to half that of restore/delete.
        weights = [0.5 if action == MOVE else 1.0 for action in choices]
        return int(rng.choices(choices, weights=weights, k=1)[0])
    with torch.no_grad():
        return int(masked_q_values(network(state), valid).argmax().item())


def optimize_prompts(
    network: QNetwork,
    sample: GraphSample,
    config: PromptPilotConfig,
    seed: int = 0,
    rollouts: int = 1,
    inference_epsilon: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Run the policy and retain the best stochastic graph trajectory."""
    if rollouts <= 0:
        raise ValueError("rollouts must be positive")
    if not 0.0 <= inference_epsilon <= 1.0:
        raise ValueError("inference_epsilon must be between zero and one")
    network.eval()
    candidates: list[tuple[float, torch.Tensor, torch.Tensor, dict[str, object]]] = []
    for rollout in range(rollouts):
        rollout_seed = seed + rollout
        env = DistanceGraphEnv(sample, config, seed=rollout_seed)
        state = env.reset()
        rng = random.Random(rollout_seed)
        total_reward = 0.0
        action_counts = {name: 0 for name in ACTIONS}
        done = False
        while not done:
            action = choose_action(
                network, state, env.valid_actions(), inference_epsilon, rng
            )
            state, reward, done, _ = env.step(action)
            total_reward += reward
            action_counts[ACTIONS[action]] += 1
        final_potential = env.potential(env._stats)
        trace: dict[str, object] = {
            "total_reward": total_reward,
            "final_potential": final_potential,
            "action_counts": action_counts,
            "statistics": env._stats.tolist(),
            "counts": list(env.counts()),
            "rollout_seed": rollout_seed,
        }
        candidates.append(
            (
                final_potential,
                torch.tensor(env.positive_indices, dtype=torch.long),
                torch.tensor(env.negative_indices, dtype=torch.long),
                trace,
            )
        )
    feasible = [
        index
        for index, (_, positive, negative, _) in enumerate(candidates)
        if config.min_nodes <= len(positive) <= config.max_nodes
        and config.min_nodes <= len(negative) <= config.max_nodes
    ]
    selection_pool = feasible if feasible else list(range(len(candidates)))
    selected = max(selection_pool, key=lambda index: candidates[index][0])
    _, positive, negative, trace = candidates[selected]
    trace["rollouts"] = rollouts
    trace["inference_epsilon"] = inference_epsilon
    trace["selected_rollout"] = selected
    trace["feasible_rollouts"] = len(feasible)
    trace["rollout_potentials"] = [item[0] for item in candidates]
    trace["rollout_action_counts"] = [item[3]["action_counts"] for item in candidates]
    return positive, negative, trace


def greedy_optimize_prompts(
    sample: GraphSample,
    config: PromptPilotConfig,
    max_passes: int = 4,
    target_counts: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Directly optimize the potential whose change is Equation (8).

    The released table cannot address a newly constructed graph and the DQN is
    not public. This deterministic diagnostic starts with all matched nodes,
    enforces the public implementation's node bounds, then performs improving
    deletions and one-for-one swaps under the disclosed objective.
    """
    env = DistanceGraphEnv(sample, config, seed=0)
    env.active = dict(env._original_index)
    env.removed = set()
    env._stats = env.graph_statistics()
    initial_potential = env.potential(env._stats)

    ordered_uids = sorted(env._original_index)
    uid_to_row = {uid: row for row, uid in enumerate(ordered_uids)}
    indices = torch.tensor(
        [env._original_index[uid] for uid in ordered_uids], dtype=torch.long, device=env.features.device
    )
    candidate_features = env.features[indices]
    feature_distances = (
        torch.cdist(candidate_features, candidate_features) / env._feature_scale
    ).detach().cpu().numpy()
    physical_distances = (
        torch.cdist(env._points(indices.tolist()), env._points(indices.tolist())) / env._physical_scale
    ).detach().cpu().numpy()

    def score(active: dict[int, int]) -> float:
        pos = [uid for uid in active if env._category[uid] == 1]
        neg = [uid for uid in active if env._category[uid] == 0]

        def mean(matrix: np.ndarray, left: list[int], right: list[int]) -> float:
            if not left or not right:
                return 0.0
            left_rows = [uid_to_row[uid] for uid in left]
            right_rows = [uid_to_row[uid] for uid in right]
            return float(matrix[np.ix_(left_rows, right_rows)].mean())

        stats = np.asarray(
            [
                mean(feature_distances, pos, pos),
                mean(feature_distances, pos, neg),
                mean(physical_distances, pos, pos),
                mean(physical_distances, neg, neg),
                mean(physical_distances, pos, neg),
            ],
            dtype=np.float32,
        )
        return env.potential(stats)

    def best_deletion(category: int) -> int:
        """Score every deletion from sums of precomputed distance blocks."""
        pos_uids = env._uids(1)
        neg_uids = env._uids(0)
        pos = np.asarray([uid_to_row[uid] for uid in pos_uids], dtype=np.int64)
        neg = np.asarray([uid_to_row[uid] for uid in neg_uids], dtype=np.int64)
        f_pp = feature_distances[np.ix_(pos, pos)]
        f_pn = feature_distances[np.ix_(pos, neg)]
        p_pp = physical_distances[np.ix_(pos, pos)]
        p_nn = physical_distances[np.ix_(neg, neg)]
        p_pn = physical_distances[np.ix_(pos, neg)]
        n_pos, n_neg = len(pos), len(neg)

        if category == 1:
            f_pp_after = (f_pp.sum() - 2 * f_pp.sum(axis=1) + np.diag(f_pp)) / (n_pos - 1) ** 2
            f_pn_after = (f_pn.sum() - f_pn.sum(axis=1)) / ((n_pos - 1) * n_neg)
            p_pp_after = (p_pp.sum() - 2 * p_pp.sum(axis=1) + np.diag(p_pp)) / (n_pos - 1) ** 2
            p_pn_after = (p_pn.sum() - p_pn.sum(axis=1)) / ((n_pos - 1) * n_neg)
            values = (
                -config.alpha * f_pp_after
                + config.alpha * f_pn_after
                + config.beta * p_pp_after
                + config.beta * p_nn.mean()
                - config.beta * p_pn_after
            )
            return pos_uids[int(np.argmax(values))]

        f_pn_after = (f_pn.sum() - f_pn.sum(axis=0)) / (n_pos * (n_neg - 1))
        p_nn_after = (p_nn.sum() - 2 * p_nn.sum(axis=1) + np.diag(p_nn)) / (n_neg - 1) ** 2
        p_pn_after = (p_pn.sum() - p_pn.sum(axis=0)) / (n_pos * (n_neg - 1))
        values = (
            -config.alpha * f_pp.mean()
            + config.alpha * f_pn_after
            + config.beta * p_pp.mean()
            + config.beta * p_nn_after
            - config.beta * p_pn_after
        )
        return neg_uids[int(np.argmax(values))]

    operations = {"forced_delete": 0, "improving_delete": 0, "swap": 0}

    for category in (1, 0):
        target = target_counts[0 if category == 1 else 1] if target_counts else config.max_nodes
        if target < config.min_nodes:
            raise ValueError(f"target count {target} is below min_nodes={config.min_nodes}")
        while len(env._uids(category)) > target:
            best_uid = best_deletion(category)
            del env.active[best_uid]
            env.removed.add(best_uid)
            operations["forced_delete"] += 1

    while target_counts is None:
        current = score(env.active)
        candidates: list[tuple[float, int]] = []
        for category in (1, 0):
            if len(env._uids(category)) <= config.min_nodes:
                continue
            for uid in env._uids(category):
                proposal = {key: value for key, value in env.active.items() if key != uid}
                candidates.append((score(proposal), uid))
        if not candidates:
            break
        best_score, best_uid = max(candidates)
        if best_score <= current + 1e-9:
            break
        del env.active[best_uid]
        env.removed.add(best_uid)
        operations["improving_delete"] += 1

    for _ in range(max_passes):
        improved = False
        for category in (1, 0):
            active_uids = env._uids(category)
            removed_uids = env._uids(category, active=False)
            if not active_uids or not removed_uids:
                continue
            current = score(env.active)
            best: tuple[float, int, int] | None = None
            for old_uid in active_uids:
                for new_uid in removed_uids:
                    proposal = dict(env.active)
                    del proposal[old_uid]
                    proposal[new_uid] = env._original_index[new_uid]
                    candidate = (score(proposal), old_uid, new_uid)
                    if best is None or candidate[0] > best[0]:
                        best = candidate
            if best is not None and best[0] > current + 1e-9:
                _, old_uid, new_uid = best
                del env.active[old_uid]
                env.removed.add(old_uid)
                env.removed.remove(new_uid)
                env.active[new_uid] = env._original_index[new_uid]
                operations["swap"] += 1
                improved = True
        if not improved:
            break

    env._stats = env.graph_statistics()
    final_potential = env.potential(env._stats)
    return (
        torch.tensor(env.positive_indices, dtype=torch.long),
        torch.tensor(env.negative_indices, dtype=torch.long),
        {
            "total_reward": final_potential - initial_potential,
            "action_counts": operations,
            "statistics": env._stats.tolist(),
            "counts": list(env.counts()),
            "initial_potential": initial_potential,
            "final_potential": final_potential,
        },
    )


def config_dict(config: PromptPilotConfig) -> dict[str, object]:
    return asdict(config)
