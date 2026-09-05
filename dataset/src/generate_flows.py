from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np


Flow = Dict[str, object]

# ---------------------------------------------------------------------------
# Flow-size CDF distribution (from real network traffic measurements).
# Each entry: (cumulative_upper_bound, size_bytes).
# The outcome is a uniform random number in [0, 1]; the size is chosen from
# whichever bucket the outcome falls into.
#
# Probability of each bucket (equals the width of its outcome interval):
#   [0.00, 0.10) → 180 B       10%   (tiny packets / ACKs)
#   [0.10, 0.20) → 216 B       10%
#   [0.20, 0.30) → 560 B       10%
#   [0.30, 0.40) → 900 B       10%
#   [0.40, 0.50) → 1 100 B     10%
#   [0.50, 0.60) → 1 870 B     10%
#   [0.60, 0.70) → 3 160 B     10%
#   [0.70, 0.80) → 10 000 B    10%   (10 KB)
#   [0.80, 0.90) → 400 000 B   10%   (400 KB)
#   [0.90, 0.95) → 3 160 000 B  5%   (3.16 MB)
#   [0.95, 0.98) → 100 000 000 B 3%  (100 MB)
#   [0.98, 1.00] → 1 000 000 000 B 2% (1 GB  — elephant flow)
# ---------------------------------------------------------------------------
_SIZE_DIST_BOUNDS: List[float] = [
    0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90, 0.95,
    0.98, 1.01,                       # 1.01 acts as a catch-all upper bound
]
_SIZE_DIST_BYTES: List[int] = [
    180, 216, 560, 900, 1_100,
    1_870, 3_160, 10_000, 400_000, 3_160_000,
    100_000_000, 1_000_000_000,
]
# Pre-computed probability weights (bucket widths) for np.random.choice.
_SIZE_DIST_PROBS: np.ndarray = np.array([
    0.10, 0.10, 0.10, 0.10, 0.10,
    0.10, 0.10, 0.10, 0.10, 0.05,
    0.03, 0.02,
], dtype=float)


def sample_flow_sizes_bytes(count: int) -> np.ndarray:
    """Draw ``count`` flow sizes (bytes) from the CDF size distribution."""
    indices = np.random.choice(len(_SIZE_DIST_BYTES), size=count, p=_SIZE_DIST_PROBS)
    return np.array(_SIZE_DIST_BYTES, dtype=float)[indices]


def _gravity_probabilities(graph: nx.Graph, allow_self_flows: bool) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    nodes = list(graph.nodes())
    degrees = dict(graph.degree())
    pairs: List[Tuple[str, str]] = []
    weights: List[float] = []

    for src in nodes:
        for dst in nodes:
            if not allow_self_flows and src == dst:
                continue
            weight = (degrees.get(src, 0) + 1) * (degrees.get(dst, 0) + 1)
            if weight > 0:
                pairs.append((src, dst))
                weights.append(float(weight))

    if not pairs:
        raise ValueError("No valid source/destination pairs found for flow generation.")

    probs = np.array(weights, dtype=float)
    probs /= probs.sum()
    return pairs, probs


def _uniform_pairs(graph: nx.Graph, allow_self_flows: bool) -> List[Tuple[str, str]]:
    """Return all valid (src, dst) pairs with equal probability."""
    nodes = list(graph.nodes())
    pairs = [
        (src, dst)
        for src in nodes
        for dst in nodes
        if allow_self_flows or src != dst
    ]
    if not pairs:
        raise ValueError("No valid source/destination pairs found.")
    return pairs


def generate_flows(
    graph: nx.Graph,
    count: int,
    model: str = "gravity",
    demand_scale: float = 10.0,
    demand_sigma: float = 0.8,
    allow_self_flows: bool = False,
) -> List[Flow]:
    """Generate flows for the given topology.

    Models
    ------
    ``gravity``
        Source-destination pair sampled by gravity (degree-product) weight.
        Flow demand drawn from log-normal(demand_scale, demand_sigma).
    ``size_dist``
        Source-destination pair sampled **uniformly** across all valid pairs.
        Flow demand (bytes) drawn from the real-traffic CDF size distribution.
        Capacity admission in compute_shortest_paths uses this per-flow
        demand_bytes field: only paths whose bottleneck link can carry the
        flow's demand are kept.
    """
    if model == "gravity":
        pairs, probs = _gravity_probabilities(graph, allow_self_flows)
        indices = np.random.choice(len(pairs), size=count, p=probs)
        sizes = np.random.lognormal(mean=np.log(demand_scale), sigma=demand_sigma, size=count)
        flows: List[Flow] = [
            {"id": int(fid), "src": pairs[idx][0], "dst": pairs[idx][1], "demand": float(sizes[fid])}
            for fid, idx in enumerate(indices)
        ]

    elif model == "size_dist":
        # Uniform src-dst selection + CDF-based per-flow size in bytes.
        all_pairs = _uniform_pairs(graph, allow_self_flows)
        pair_indices = np.random.choice(len(all_pairs), size=count)
        sizes_bytes = sample_flow_sizes_bytes(count)
        flows = [
            {
                "id": int(fid),
                "src": all_pairs[idx][0],
                "dst": all_pairs[idx][1],
                "demand": float(sizes_bytes[fid]),
                "demand_bytes": int(sizes_bytes[fid]),
            }
            for fid, idx in enumerate(pair_indices)
        ]

    else:
        raise ValueError(f"Unsupported flow model: {model!r}. Choose 'gravity' or 'size_dist'.")

    return flows


def write_flows_csv(flows: Sequence[Flow], path) -> None:
    import csv
    from pathlib import Path

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "src", "dst", "demand"])
        writer.writeheader()
        for flow in flows:
            writer.writerow(
                {
                    "id": flow["id"],
                    "src": str(flow["src"]),
                    "dst": str(flow["dst"]),
                    "demand": f"{float(flow['demand']):.6f}",
                }
            )
