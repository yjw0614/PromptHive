"""Audit the structure and usability of the released Q-table."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("q_table")
    parser.add_argument("--output", help="Optional JSON report path")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    path = Path(args.q_table)
    table = torch.load(path, map_location="cpu", weights_only=False)
    graphs: dict[int, object] = {}
    actions_per_graph: Counter[int] = Counter()
    operation_counts: Counter[str] = Counter()
    values: list[float] = []

    for (graph, action), value in table.items():
        graph_id = id(graph)
        graphs[graph_id] = graph
        actions_per_graph[graph_id] += 1
        operation_counts[str(action[1])] += 1
        values.append(float(value))

    node_pairs: Counter[tuple[int, int]] = Counter()
    edge_attributes: Counter[tuple[str, ...]] = Counter()
    for graph in graphs.values():
        categories = Counter(data.get("category") for _, data in graph.nodes(data=True))
        node_pairs[(categories["pos"], categories["neg"])] += 1
        for _, _, data in graph.edges(data=True):
            edge_attributes[tuple(sorted(data))] += 1

    q_values = np.asarray(values, dtype=np.float64)
    graph_action_counts = np.asarray(list(actions_per_graph.values()), dtype=np.float64)
    quantiles = [0.0, 0.01, 0.5, 0.9, 0.99, 1.0]
    report = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "container_type": type(table).__name__,
        "entries": len(table),
        "unique_graph_objects": len(graphs),
        "operation_counts": dict(operation_counts),
        "q_nonzero": int(np.count_nonzero(q_values)),
        "q_nonzero_fraction": float(np.count_nonzero(q_values) / len(q_values)),
        "q_quantiles": {
            str(q): float(value) for q, value in zip(quantiles, np.quantile(q_values, quantiles))
        },
        "actions_per_graph_quantiles": {
            str(q): float(value)
            for q, value in zip(quantiles, np.quantile(graph_action_counts, quantiles))
        },
        "node_count_pairs_top20": [
            {"positive": pair[0], "negative": pair[1], "graphs": count}
            for pair, count in node_pairs.most_common(20)
        ],
        "edge_attribute_counts": {"+".join(names): count for names, count in edge_attributes.items()},
        "generalization_note": (
            "Keys contain NetworkX graph instances, whose default equality and hash use object identity. "
            "A newly constructed target graph cannot retrieve these entries."
        ),
    }
    output = json.dumps(report, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
