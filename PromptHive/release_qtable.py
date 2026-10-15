"""Diagnostic adapter for the released object-keyed Q-table."""

from __future__ import annotations

from collections import Counter
import random

import networkx as nx
import torch

from utils_test import (
    GraphOptimizationEnv,
    QLearningAgent,
    calculate_distances,
    convert_to_edges,
)

from .core import GraphSample


def _build_public_graph(sample: GraphSample) -> nx.MultiGraph:
    """Build the graph exactly as the released test path does."""
    positive = sample.positive_indices
    negative = sample.negative_indices
    feature_pair = torch.stack((sample.features, sample.features))
    distances = calculate_distances(
        feature_pair,
        positive,
        negative,
        sample.image_size,
        sample.features.device,
    )
    feature_pos, feature_cross, physical_pos, physical_neg, physical_cross = distances

    graph = nx.MultiGraph()
    graph.add_nodes_from(positive.detach().cpu().numpy(), category="pos")
    graph.add_nodes_from(negative.detach().cpu().numpy(), category="neg")
    graph.add_weighted_edges_from(
        convert_to_edges(positive, positive, feature_pos), weight="feature_pos"
    )
    graph.add_weighted_edges_from(
        convert_to_edges(positive, positive, physical_pos), weight="physical_pos"
    )
    graph.add_weighted_edges_from(
        convert_to_edges(negative, negative, physical_neg), weight="physical_neg"
    )
    graph.add_weighted_edges_from(
        convert_to_edges(positive, negative, feature_cross), weight="feature_cross"
    )
    graph.add_weighted_edges_from(
        convert_to_edges(positive, negative, physical_cross), weight="physical_cross"
    )
    return graph


def optimize_prompts_release_qtable(
    q_table: object,
    sample: GraphSample,
    *,
    seed: int,
    epsilon: float,
    max_steps: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Execute the released agent while measuring whether saved keys are used."""
    random.seed(seed)
    graph = _build_public_graph(sample)
    environment = GraphOptimizationEnv(graph, max_steps=max_steps)
    agent = QLearningAgent(GraphOptimizationEnv, epsilon_start=epsilon)
    agent.env = environment
    agent.epsilon = epsilon
    agent.q_table = q_table

    state = environment.reset()
    total_reward = 0.0
    pretrained_key_hits = 0
    candidate_lookups = 0
    operations: Counter[str] = Counter()
    while not environment.is_done():
        candidates = agent.get_possible_actions(state)
        candidate_lookups += len(candidates)
        pretrained_key_hits += sum((state, action) in q_table for action in candidates)
        action = agent.get_action(state)
        operations[str(action[1])] += 1
        state, reward, _ = environment.step(action)
        total_reward += float(reward)

    device = sample.features.device
    positive = torch.tensor(
        [int(node) for node in environment.pos_nodes], dtype=torch.long, device=device
    )
    negative = torch.tensor(
        [int(node) for node in environment.neg_nodes], dtype=torch.long, device=device
    )
    action_counts = {
        "restore": operations["restore_pos"] + operations["restore_neg"],
        "delete": operations["remove_pos"] + operations["remove_neg"],
        "move": 0,
        "add": operations["add"],
    }
    trace: dict[str, object] = {
        "total_reward": total_reward,
        "action_counts": action_counts,
        "best_potential": float("nan"),
        "pretrained_key_hits": pretrained_key_hits,
        "candidate_lookups": candidate_lookups,
    }
    return positive, negative, trace
