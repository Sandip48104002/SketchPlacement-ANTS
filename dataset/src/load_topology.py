from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from .utils import sanitize_attrs, write_json

# Map LinkType strings (from Topology Zoo graphml) to numeric capacity in Gbps.
# Sources: ITU-T G.707 (SDH/SONET OC rates), IEEE 802.3 (Ethernet).
_LINK_TYPE_TO_GBPS: Dict[str, float] = {
    # SONET/SDH optical carrier rates
    "OC-3":      0.155,
    "OC-12":     0.622,
    "OC-48":     2.488,
    "OC-192":    9.953,
    "OC-768":   39.813,
    "STM-1":     0.155,
    "STM-4":     0.622,
    "STM-16":    2.488,
    "STM-64":    9.953,
    "STM-256":  39.813,
    # Ethernet
    "GE":        1.0,
    "GigE":      1.0,
    "1GE":       1.0,
    "10GE":     10.0,
    "100GE":   100.0,
    "400GE":   400.0,
}

# Default capacity (Gbps) used when a link's type is unknown or missing.
_DEFAULT_CAPACITY_GBPS: float = 1.0


def _parse_capacity_gbps(link_type: Optional[str]) -> float:
    """Return numeric Gbps capacity for a LinkType string.

    Matches case-insensitively and strips suffixes like 'c' or 'r'
    (e.g. 'OC-192c' → 'OC-192').
    """
    if not link_type:
        return _DEFAULT_CAPACITY_GBPS
    raw = str(link_type).strip()

    if raw in _LINK_TYPE_TO_GBPS:
        return _LINK_TYPE_TO_GBPS[raw]
    
    stripped = raw.rstrip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if stripped in _LINK_TYPE_TO_GBPS:
        return _LINK_TYPE_TO_GBPS[stripped]

    lower = raw.lower()
    for key, val in _LINK_TYPE_TO_GBPS.items():
        if key.lower() == lower or key.lower() == lower.rstrip("abcdefghijklmnopqrstuvwxyz"):
            return val
    return _DEFAULT_CAPACITY_GBPS


def annotate_link_capacity(graph: nx.Graph) -> nx.Graph:
    """Add two numeric edge attributes derived from LinkType:

    - ``capacity_gbps``  : link bandwidth in Gbps (higher = more capacity).
    - ``inv_capacity``   : 1 / capacity_gbps (lower = more capacity).
                           Use as ``weight_attr`` for capacity-aware shortest
                           paths (flows prefer high-capacity links).
    """
    for u, v, data in graph.edges(data=True):
        link_type = data.get("LinkType") or data.get("link_type")
        cap = _parse_capacity_gbps(link_type)
        data["capacity_gbps"] = cap
        data["inv_capacity"] = round(1.0 / cap, 6)
    return graph


def load_topology(path: Path, directed: bool = False) -> nx.Graph:
    graph = nx.read_graphml(path)
    if not directed:
        graph = graph.to_undirected()

    graph = nx.relabel_nodes(graph, lambda n: str(n))
    graph.remove_edges_from(nx.selfloop_edges(graph))
    annotate_link_capacity(graph)
    return graph


def export_topology(graph: nx.Graph, path: Path, name: str) -> None:
    nodes: List[Dict[str, Any]] = []
    for node_id, attrs in graph.nodes(data=True):
        nodes.append({"id": str(node_id), **sanitize_attrs(attrs)})

    edges: List[Dict[str, Any]] = []
    for u, v, attrs in graph.edges(data=True):
        edges.append({"u": str(u), "v": str(v), **sanitize_attrs(attrs)})

    payload: Dict[str, Any] = {
        "name": name,
        "directed": graph.is_directed(),
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "nodes": nodes,
        "edges": edges,
    }
    write_json(path, payload)
