"""Node-aware PromptPilot-v2 environment and policy.

The base policy predicts an operation type and samples the affected node. This
module makes the concrete node and destination part of the action and adds an
explicit stop action for converged prompt sets.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn

from .core import DELETE, MOVE, RESTORE, DistanceGraphEnv, GraphSample, PromptPilotConfig


STOP = 3
IMPROVED_ACTIONS = ("restore", "delete", "move", "stop")
CANDIDATE_DIM = 22


@dataclass(frozen=True)
class CandidateAction:
    operation: int
    uid: int = -1
    target_index: int = -1


class CandidateActionEnv(DistanceGraphEnv):
    """Distance graph with explicit, deterministic node-level actions."""

    def _bounded_initial_uids(self) -> list[int]:
        """Select a label-free, diverse subset when FM emits too many points."""
        positive_uids = sorted(uid for uid in self._original_index if self._category[uid] == 1)
        negative_uids = sorted(uid for uid in self._original_index if self._category[uid] == 0)

        def tensors(uids: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            indices = torch.tensor(
                [self._original_index[uid] for uid in uids],
                dtype=torch.long,
                device=self.features.device,
            )
            return indices, self.features[indices], self._points(indices.tolist())

        def select_positive(uids: list[int]) -> list[int]:
            if len(uids) <= self.config.max_nodes:
                return uids
            _, features, points = tensors(uids)
            cohesion = torch.linalg.vector_norm(features - features.mean(dim=0), dim=1)
            cohesion = cohesion / self._feature_scale
            selected = [int(cohesion.argmin().item())]
            min_physical = torch.cdist(points, points[selected]).squeeze(1) / self._physical_scale
            while len(selected) < self.config.max_nodes:
                score = -self.config.alpha * cohesion + self.config.beta * min_physical
                score[selected] = -torch.inf
                chosen = int(score.argmax().item())
                selected.append(chosen)
                distance = torch.cdist(points, points[[chosen]]).squeeze(1) / self._physical_scale
                min_physical = torch.minimum(min_physical, distance)
            return [uids[index] for index in selected]

        selected_positive = select_positive(positive_uids)
        if len(negative_uids) <= self.config.max_nodes:
            return selected_positive + negative_uids

        _, negative_features, negative_points = tensors(negative_uids)
        positive_indices = [self._original_index[uid] for uid in selected_positive]
        positive_features = self.features[
            torch.tensor(positive_indices, dtype=torch.long, device=self.features.device)
        ]
        positive_points = self._points(positive_indices)
        feature_cross = torch.cdist(negative_features, positive_features).mean(dim=1)
        feature_cross = feature_cross / self._feature_scale
        physical_cross = torch.cdist(negative_points, positive_points).mean(dim=1)
        physical_cross = physical_cross / self._physical_scale
        base_score = self.config.alpha * feature_cross - self.config.beta * physical_cross
        selected_negative = [int(base_score.argmax().item())]
        min_physical = (
            torch.cdist(negative_points, negative_points[selected_negative]).squeeze(1)
            / self._physical_scale
        )
        while len(selected_negative) < self.config.max_nodes:
            score = base_score + self.config.beta * min_physical
            score[selected_negative] = -torch.inf
            chosen = int(score.argmax().item())
            selected_negative.append(chosen)
            distance = (
                torch.cdist(negative_points, negative_points[[chosen]]).squeeze(1)
                / self._physical_scale
            )
            min_physical = torch.minimum(min_physical, distance)
        return selected_positive + [negative_uids[index] for index in selected_negative]

    def reset_to_initial_prompts(self, prune_to_bounds: bool = True) -> torch.Tensor:
        """Start from FM prompts, optionally reducing them to the graph bounds."""
        self.active = dict(self._original_index)
        self.removed = set()
        self.steps = 0
        self._stats = self.graph_statistics()
        self.full_initial_active = dict(self.active)
        self.full_initial_statistics = self._stats.copy()
        if prune_to_bounds:
            selected = set(self._bounded_initial_uids())
            self.active = {
                uid: index for uid, index in self._original_index.items() if uid in selected
            }
            self.removed = set(self._original_index) - selected
            self._stats = self.graph_statistics()
        return self.state_vector()

    def candidate_actions(self) -> list[CandidateAction]:
        n_pos, n_neg = self.counts()
        missing_categories = {
            category
            for category, count in ((1, n_pos), (0, n_neg))
            if count < self.config.min_nodes
        }
        excess_categories = {
            category
            for category, count in ((1, n_pos), (0, n_neg))
            if count > self.config.max_nodes
        }

        if missing_categories:
            restore_actions = [
                CandidateAction(RESTORE, uid, self._original_index[uid])
                for uid in sorted(self.removed)
                if self._category[uid] in missing_categories
            ]
            if restore_actions:
                return restore_actions
        if excess_categories:
            return [
                CandidateAction(DELETE, uid, self.active[uid])
                for uid in sorted(self.active)
                if self._category[uid] in excess_categories
            ]

        actions: list[CandidateAction] = []
        for uid in sorted(self.removed):
            category = self._category[uid]
            count = n_pos if category == 1 else n_neg
            if count < self.config.max_nodes:
                actions.append(CandidateAction(RESTORE, uid, self._original_index[uid]))

        occupied = set(self.active.values())
        for uid in sorted(self.active):
            category = self._category[uid]
            count = n_pos if category == 1 else n_neg
            if count > self.config.min_nodes:
                actions.append(CandidateAction(DELETE, uid, self.active[uid]))

            index = self.active[uid]
            row, col = divmod(index, self.grid_size)
            for dr, dc in (
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ):
                nr, nc = row + dr, col + dc
                target = nr * self.grid_size + nc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size and target not in occupied:
                    actions.append(CandidateAction(MOVE, uid, target))

        actions.append(CandidateAction(STOP))
        return actions

    def describe(self, action: CandidateAction) -> torch.Tensor:
        return self.candidate_descriptors([action])[0]

    def candidate_descriptors(self, actions: list[CandidateAction] | None = None) -> torch.Tensor:
        actions = actions if actions is not None else self.candidate_actions()
        if not actions:
            raise RuntimeError("Environment has no candidate actions")
        device = self.features.device
        count = len(actions)
        n_pos, n_neg = self.counts()
        operations = torch.tensor([action.operation for action in actions], device=device)
        is_stop = operations == STOP
        uids = [action.uid for action in actions]
        categories = torch.tensor(
            [self._category[uid] if uid >= 0 else -1 for uid in uids],
            dtype=torch.long,
            device=device,
        )
        target_indices = torch.tensor(
            [max(action.target_index, 0) for action in actions],
            dtype=torch.long,
            device=device,
        )

        operation_encoding = torch.zeros((count, len(IMPROVED_ACTIONS)), device=device)
        operation_encoding.scatter_(1, operations[:, None], 1.0)
        category_encoding = torch.stack(
            ((categories == 1).float(), (categories == 0).float()), dim=1
        )

        rows = torch.div(target_indices, self.grid_size, rounding_mode="floor")
        cols = target_indices.remainder(self.grid_size)
        denominator = max(self.grid_size - 1, 1)
        coordinates = torch.stack((cols, rows), dim=1).float() / denominator

        target_features = self.features[target_indices]
        target_points = self._points(target_indices.tolist())

        def mean_distances(category: int) -> tuple[torch.Tensor, torch.Tensor]:
            active_indices = [self.active[uid] for uid in self._uids(category)]
            if not active_indices:
                zeros = torch.zeros(count, device=device)
                return zeros, zeros
            index_tensor = torch.tensor(active_indices, dtype=torch.long, device=device)
            feature = torch.cdist(target_features, self.features[index_tensor]).mean(dim=1)
            physical = torch.cdist(target_points, self._points(active_indices)).mean(dim=1)
            return feature / self._feature_scale, physical / self._physical_scale

        feature_pos, physical_pos = mean_distances(1)
        feature_neg, physical_neg = mean_distances(0)
        distances = torch.stack(
            (feature_pos, feature_neg, physical_pos, physical_neg), dim=1
        )

        restore_delta = (operations == RESTORE).float()
        delete_delta = (operations == DELETE).float()
        signed_delta = restore_delta - delete_delta
        post_pos = n_pos + signed_delta * (categories == 1).float()
        post_neg = n_neg + signed_delta * (categories == 0).float()
        post_counts = torch.stack((post_pos, post_neg), dim=1) / self.config.max_nodes

        coordinates[is_stop] = 0.0
        distances[is_stop] = 0.0
        global_state = self.state_vector().unsqueeze(0).expand(count, -1)
        descriptors = torch.cat(
            (
                global_state,
                operation_encoding,
                category_encoding,
                coordinates,
                distances,
                post_counts,
            ),
            dim=1,
        )
        if descriptors.shape[1] != CANDIDATE_DIM:
            raise RuntimeError(
                f"Expected {CANDIDATE_DIM} candidate values, got {descriptors.shape[1]}"
            )
        return descriptors

    def step_candidate(
        self, action: CandidateAction
    ) -> tuple[torch.Tensor, float, bool, dict[str, object]]:
        candidates = set(self.candidate_actions())
        if action not in candidates:
            raise ValueError(f"Invalid candidate action: {action}")
        if action.operation == STOP:
            return self.state_vector(), 0.0, True, {
                "statistics": self._stats.copy(),
                "counts": self.counts(),
                "stopped": True,
            }

        previous = self._stats.copy()
        if action.operation == RESTORE:
            self.removed.remove(action.uid)
            self.active[action.uid] = self._original_index[action.uid]
        elif action.operation == DELETE:
            del self.active[action.uid]
            self.removed.add(action.uid)
        elif action.operation == MOVE:
            self.active[action.uid] = action.target_index
        else:
            raise ValueError(f"Unknown operation: {action.operation}")

        self.steps += 1
        self._stats = self.graph_statistics()
        reward = self.potential(self._stats) - self.potential(previous)
        done = self.steps >= self.config.max_steps
        return self.state_vector(), reward, done, {
            "statistics": self._stats.copy(),
            "counts": self.counts(),
            "stopped": False,
        }


class CandidateQNetwork(nn.Module):
    """Score variable-length concrete actions instead of three random types."""

    def __init__(self, input_dim: int = CANDIDATE_DIM):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        return self.layers(descriptors).squeeze(-1)


def choose_candidate(
    network: CandidateQNetwork,
    descriptors: torch.Tensor,
    actions: list[CandidateAction],
    epsilon: float,
    rng: random.Random,
) -> int:
    if len(actions) != len(descriptors):
        raise ValueError("Action and descriptor counts differ")
    if rng.random() < epsilon:
        by_operation: dict[int, list[int]] = {}
        for index, action in enumerate(actions):
            by_operation.setdefault(action.operation, []).append(index)
        operations = sorted(by_operation)
        weights = [0.5 if operation == MOVE else 0.25 if operation == STOP else 1.0 for operation in operations]
        operation = rng.choices(operations, weights=weights, k=1)[0]
        return rng.choice(by_operation[operation])
    with torch.no_grad():
        return int(network(descriptors).argmax().item())


def optimize_prompts_improved(
    network: CandidateQNetwork,
    sample: GraphSample,
    config: PromptPilotConfig,
    seed: int = 0,
    start_from_initial: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Run the node-aware policy and return its best label-free graph state."""
    env = CandidateActionEnv(sample, config, seed=seed)
    if start_from_initial:
        env.reset_to_initial_prompts(prune_to_bounds=True)
    else:
        env.reset()
    network.eval()
    rng = random.Random(seed)
    total_reward = 0.0
    action_counts = {name: 0 for name in IMPROVED_ACTIONS}
    best_active: dict[int, int] | None = dict(env.active)
    best_potential = env.potential(env._stats)
    pruned_initial_potential = best_potential
    full_initial_potential: float | None = None
    if start_from_initial:
        full_initial_potential = env.potential(env.full_initial_statistics)
        if full_initial_potential > best_potential:
            best_potential = full_initial_potential
            best_active = dict(env.full_initial_active)
    first_feasible_potential: float | None = None
    if min(env.counts()) >= config.min_nodes:
        first_feasible_potential = best_potential
    done = False

    while not done:
        actions = env.candidate_actions()
        descriptors = env.candidate_descriptors(actions)
        selected = choose_candidate(network, descriptors, actions, 0.0, rng)
        action = actions[selected]
        _, reward, done, _ = env.step_candidate(action)
        total_reward += reward
        action_counts[IMPROVED_ACTIONS[action.operation]] += 1

        n_pos, n_neg = env.counts()
        if n_pos >= config.min_nodes and n_neg >= config.min_nodes:
            potential = env.potential(env._stats)
            if first_feasible_potential is None:
                first_feasible_potential = potential
            if potential > best_potential:
                best_potential = potential
                best_active = dict(env.active)

    if best_active is not None:
        env.active = best_active
        env._stats = env.graph_statistics()
    return (
        torch.tensor(env.positive_indices, dtype=torch.long),
        torch.tensor(env.negative_indices, dtype=torch.long),
        {
            "total_reward": total_reward,
            "action_counts": action_counts,
            "statistics": env._stats.tolist(),
            "counts": list(env.counts()),
            "first_feasible_potential": first_feasible_potential,
            "full_initial_potential": full_initial_potential,
            "pruned_initial_potential": pruned_initial_potential,
            "best_potential": best_potential,
            "selected_best_graph_state": best_active is not None,
        },
    )
