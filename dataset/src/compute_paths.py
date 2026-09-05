import gzip
import json
import logging
from typing import Dict, Iterable, List, Optional, Sequence

import networkx as nx


PathRecord = Dict[str, object]


def _bottleneck_capacity_gbps(graph: nx.Graph, path_nodes: List) -> float:
    """Return the minimum link capacity (Gbps) along a path.

    This is the bottleneck: the weakest link that limits how much traffic
    the entire path can carry. If any edge is missing capacity_gbps it
    defaults to 1.0 Gbps (the same fallback used in annotate_link_capacity).
    """
    bottleneck = float("inf")
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        edge_data = graph.get_edge_data(u, v, default={})
        cap = float(edge_data.get("capacity_gbps", 1.0))
        if cap < bottleneck:
            bottleneck = cap
    # Single-node path (src == dst) has no edges — treat as unlimited.
    return bottleneck if bottleneck != float("inf") else float("inf")


def compute_shortest_paths(
    graph: nx.Graph,
    flows: Sequence[Dict[str, object]],
    weight_attr: Optional[str] = None,
    flow_size_kb: Optional[float] = None,
) -> List[PathRecord]:
    """Compute shortest paths for every flow and filter by link capacity.

    Parameters
    ----------
    graph:
        The topology graph.  Edges must have ``capacity_gbps`` set
        (done automatically by ``load_topology``).
    flows:
        List of flow dicts with ``src`` / ``dst`` keys.  When the dict
        contains a ``demand_bytes`` field (set by the ``size_dist`` model)
        that per-flow size is used for the admission check.  Otherwise the
        global ``flow_size_kb`` fallback is used.
    weight_attr:
        Edge attribute to use as routing cost (e.g. ``inv_capacity``).
        ``None`` means hop-count (unweighted) shortest path.
    flow_size_kb:
        **Global** fallback flow size in Kbps used when a flow has no
        ``demand_bytes`` field.  ``None`` disables the capacity check for
        those flows.
    """
    # Global fallback demand in Gbps (used when a flow has no demand_bytes).
    global_demand_gbps: Optional[float] = None
    if flow_size_kb is not None and flow_size_kb > 0:
        global_demand_gbps = flow_size_kb * 8 / 1_000_000  # Kbps → Gbps

    paths: List[PathRecord] = []
    skipped_no_path = 0
    skipped_no_capacity = 0

    for flow in flows:
        src = flow["src"]
        dst = flow["dst"]
        try:
            path_nodes = nx.shortest_path(graph, source=src, target=dst, weight=weight_attr)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            skipped_no_path += 1
            continue

        # --- Per-flow capacity admission check ----------------------------------
        # Determine this flow's demand in Gbps:
        #   • size_dist model sets demand_bytes (bytes) → convert to Gbps.
        #   • gravity model / no demand_bytes → use global_demand_gbps fallback.
        # A flow is REJECTED if the path's bottleneck link capacity (Gbps) is
        # strictly less than the flow's demand.  No path = no flow.
        demand_bytes: Optional[float] = flow.get("demand_bytes")  # type: ignore[assignment]
        if demand_bytes is not None:
            flow_demand_gbps: Optional[float] = float(demand_bytes) * 8 / 1_000_000_000  # bytes → Gbps
        else:
            flow_demand_gbps = global_demand_gbps

        bottleneck = _bottleneck_capacity_gbps(graph, path_nodes)
        if flow_demand_gbps is not None and bottleneck < flow_demand_gbps:
            skipped_no_capacity += 1
            continue
        # -------------------------------------------------------------------------

        total_weight = 0.0
        if weight_attr:
            for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                edge_data = graph.get_edge_data(u, v, default={})
                total_weight += float(edge_data.get(weight_attr, 1.0))
        else:
            total_weight = float(len(path_nodes) - 1)

        record: PathRecord = {
            "id": flow["id"],
            "src": str(src),
            "dst": str(dst),
            "path": [str(node) for node in path_nodes],
            "hops": len(path_nodes) - 1,
            "cost": total_weight,
            "bottleneck_gbps": round(bottleneck, 6) if bottleneck != float("inf") else None,
        }
        # Carry demand_bytes forward into the path record for downstream use.
        if demand_bytes is not None:
            record["demand_bytes"] = int(demand_bytes)
            record["demand_gbps"] = round(float(demand_bytes) * 8 / 1_000_000_000, 10)
        paths.append(record)

    if skipped_no_path:
        logging.warning("Skipped %d flows — no path exists in topology", skipped_no_path)
    if skipped_no_capacity:
        logging.warning(
            "Skipped %d flows — flow demand exceeds path bottleneck capacity",
            skipped_no_capacity,
        )
    logging.info(
        "Path computation done | admitted=%d | rejected_no_path=%d | rejected_capacity=%d",
        len(paths), skipped_no_path, skipped_no_capacity,
    )
    return paths


def write_paths_jsonl_gz(paths: Iterable[PathRecord], path) -> None:
    out_path = str(path)
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        for record in paths:
            fh.write(json.dumps(record))
            fh.write("\n")
