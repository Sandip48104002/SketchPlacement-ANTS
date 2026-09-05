"""
Phase II — Flow Assignment with Load Balancing
===============================================
Given a fixed set of selected switches (output of Phase I), assign each flow
to exactly one switch on its path.  Three strategies are provided:

  first_match   — assign to the first selected switch on the path (baseline,
                  no balancing).  O(flows × path_length).

  balance_count — assign to the selected switch with the fewest flows assigned
                  so far.  Balances the NUMBER of flows per switch.
                  O(flows × path_length).

  balance_size  — assign to the selected switch carrying the least total bytes
                  so far.  Balances the TOTAL TRAFFIC SIZE per switch.
                  Requires flow_sizes to be available in the dataset; falls
                  back to balance_count if sizes are missing.
                  O(flows × path_length).

All three strategies guarantee:
  • Every flow is assigned to exactly one switch.
  • The assigned switch lies on the flow's path.
  • Only selected switches receive assignments.

None of the strategies require a solver — they are O(n) greedy passes.

Public API
----------
assign_flows(dataset, selected, strategy, flow_sizes) -> Dict[int, int]
    Returns {flow_idx: switch_idx} assignments.

assignment_stats(dataset, assignments, flow_sizes) -> Dict
    Returns per-switch load statistics useful for reporting.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.data import FlowDataset


# ---------------------------------------------------------------------------
# Strategy constants
# ---------------------------------------------------------------------------

STRATEGY_FIRST_MATCH   = "first_match"
STRATEGY_BALANCE_COUNT = "balance_count"
STRATEGY_BALANCE_SIZE  = "balance_size"

ALL_STRATEGIES = (STRATEGY_FIRST_MATCH, STRATEGY_BALANCE_COUNT, STRATEGY_BALANCE_SIZE)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def assign_flows(
    dataset: FlowDataset,
    selected: List[int],
    strategy: str = STRATEGY_BALANCE_SIZE,
    flow_sizes: Optional[np.ndarray] = None,
) -> Dict[int, int]:
    """Assign each flow to one selected switch on its path.

    Parameters
    ----------
    dataset:
        Loaded FlowDataset.  ``dataset.flow_sizes`` is used when strategy is
        ``balance_size`` and ``flow_sizes`` override is not provided.
    selected:
        List of switch indices chosen by Phase I (ILP or greedy).
    strategy:
        One of ``first_match``, ``balance_count``, ``balance_size``.
    flow_sizes:
        Optional override array of per-flow sizes in bytes (length = n_flows).
        When None, ``dataset.flow_sizes`` is used.

    Returns
    -------
    Dict[int, int]
        ``{flow_idx: switch_idx}`` — one entry per flow that has at least one
        selected switch on its path.  Flows with no selected switch on their
        path are omitted (coverage violation, should not happen after Phase I).
    """
    if strategy not in ALL_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Choose from {ALL_STRATEGIES}.")

    if strategy == STRATEGY_FIRST_MATCH:
        return _first_match(dataset, selected)

    if strategy == STRATEGY_BALANCE_COUNT:
        return _balance_count(dataset, selected)

    # balance_size — resolve sizes array
    sizes = flow_sizes if flow_sizes is not None else dataset.flow_sizes
    if sizes is None or not np.any(sizes > 0):
        # No size information available: fall back gracefully.
        return _balance_count(dataset, selected)
    return _balance_size(dataset, selected, sizes)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _first_match(dataset: FlowDataset, selected: List[int]) -> Dict[int, int]:
    """Baseline: assign each flow to the first selected switch on its path."""
    selected_set = set(selected)
    assignments: Dict[int, int] = {}
    for f_idx, switches in enumerate(dataset.P):
        for sid in switches:
            if sid in selected_set:
                assignments[f_idx] = sid
                break
    return assignments


def _balance_count(dataset: FlowDataset, selected: List[int]) -> Dict[int, int]:
    """Assign each flow to the selected switch on its path with the fewest
    flows assigned so far (least-loaded by flow count).

    Flows are processed in descending size order when sizes are available, so
    large flows are distributed first — reducing the chance that all large
    flows land on the same switch.
    """
    selected_set = set(selected)
    # flow_count[sid] = number of flows currently assigned to switch sid
    flow_count: Dict[int, int] = {sid: 0 for sid in selected}
    assignments: Dict[int, int] = {}

    # Process largest flows first if sizes are available (better spread).
    order = _flow_order_by_size_desc(dataset)

    for f_idx in order:
        switches = dataset.P[f_idx]
        candidates = [sid for sid in switches if sid in selected_set]
        if not candidates:
            continue
        # Pick candidate with smallest current flow count.
        best = min(candidates, key=lambda s: flow_count[s])
        assignments[f_idx] = best
        flow_count[best] += 1

    return assignments


def _balance_size(
    dataset: FlowDataset, selected: List[int], sizes: np.ndarray
) -> Dict[int, int]:
    """Assign each flow to the selected switch on its path carrying the least
    total traffic (bytes) so far — equalising total byte load across switches.

    Processing order: largest flows first (Longest Processing Time heuristic).
    This is the same idea used in bin-packing: placing big items early gives
    the algorithm more flexibility to balance them.
    """
    selected_set = set(selected)
    # byte_load[sid] = total bytes currently assigned to switch sid
    byte_load: Dict[int, float] = {sid: 0.0 for sid in selected}
    assignments: Dict[int, int] = {}

    # Process largest flows first (LPT heuristic).
    order = np.argsort(sizes)[::-1]   # descending by size

    for f_idx in order:
        switches = dataset.P[f_idx]
        candidates = [sid for sid in switches if sid in selected_set]
        if not candidates:
            continue
        # Pick candidate with smallest current byte load.
        best = min(candidates, key=lambda s: byte_load[s])
        assignments[f_idx] = best
        byte_load[best] += float(sizes[f_idx])

    return assignments


def _flow_order_by_size_desc(dataset: FlowDataset) -> np.ndarray:
    """Return flow indices sorted largest-first if sizes are available,
    otherwise return natural order."""
    if dataset.flow_sizes is not None and np.any(dataset.flow_sizes > 0):
        return np.argsort(dataset.flow_sizes)[::-1]
    return np.arange(dataset.n_flows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def assignment_stats(
    dataset: FlowDataset,
    assignments: Dict[int, int],
    flow_sizes: Optional[np.ndarray] = None,
) -> Dict:
    """Compute per-switch load statistics for a given assignment.

    Returns a dict with:
      per_switch      — {switch_name: {flow_count, total_bytes, pct_flows, pct_bytes}}
      totals          — {flow_count, total_bytes}
      imbalance_count — max/min flow count ratio across switches (1.0 = perfect)
      imbalance_bytes — max/min byte load ratio across switches (1.0 = perfect)
      std_count       — std-dev of flow counts across switches
      std_bytes       — std-dev of byte loads across switches
    """
    sizes = flow_sizes if flow_sizes is not None else dataset.flow_sizes

    # Aggregate per switch.
    flow_count: Dict[int, int] = {}
    byte_load: Dict[int, float] = {}
    for f_idx, sid in assignments.items():
        flow_count[sid] = flow_count.get(sid, 0) + 1
        sz = float(sizes[f_idx]) if sizes is not None else 0.0
        byte_load[sid] = byte_load.get(sid, 0.0) + sz

    total_flows = sum(flow_count.values())
    total_bytes = sum(byte_load.values())

    per_switch = {}
    for sid, fname in enumerate(dataset.sid_to_name):
        fc = flow_count.get(sid, 0)
        bl = byte_load.get(sid, 0.0)
        per_switch[fname] = {
            "flow_count":  fc,
            "total_bytes": bl,
            "pct_flows":   round(100.0 * fc / total_flows, 2) if total_flows else 0.0,
            "pct_bytes":   round(100.0 * bl / total_bytes, 2) if total_bytes else 0.0,
        }

    counts = np.array(list(flow_count.values()), dtype=float)
    bytes_ = np.array(list(byte_load.values()), dtype=float)

    def _ratio(arr: np.ndarray) -> float:
        mn = arr.min()
        return float(arr.max() / mn) if mn > 0 else float("inf")

    return {
        "per_switch":      per_switch,
        "totals":          {"flow_count": total_flows, "total_bytes": total_bytes},
        "imbalance_count": round(_ratio(counts), 4) if len(counts) > 1 else 1.0,
        "imbalance_bytes": round(_ratio(bytes_),  4) if len(bytes_) > 1 else 1.0,
        "std_count":       round(float(counts.std()), 2) if len(counts) > 1 else 0.0,
        "std_bytes":       round(float(bytes_.std()), 2) if len(bytes_) > 1 else 0.0,
    }