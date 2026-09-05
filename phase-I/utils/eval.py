from typing import Dict, Iterable, List, Tuple

import numpy as np

from .data import FlowDataset


def evaluate_cover(dataset: FlowDataset, selected_switches: Iterable[int]) -> Tuple[bool, List[int]]:
    selected_set = set(selected_switches)
    uncovered: List[int] = []
    for f_idx, switches in enumerate(dataset.P):
        if not selected_set.intersection(switches):
            uncovered.append(f_idx)
    ok = len(uncovered) == 0
    return ok, uncovered


def evaluate_assignment(
    dataset: FlowDataset, assignments: Dict[int, int], capacities: np.ndarray | None = None
) -> Dict[str, object]:
    caps = capacities if capacities is not None else dataset.capacities
    coverage_errors: List[int] = []
    cap_errors: List[Tuple[int, float, float]] = []

    # Coverage check.
    for f_idx, switches in enumerate(dataset.P):
        assigned_sid = assignments.get(f_idx)
        if assigned_sid is None or assigned_sid not in switches:
            coverage_errors.append(f_idx)

    # Capacity check.
    if caps is not None:
        counts = np.zeros(dataset.n_switches, dtype=int)
        for sid in assignments.values():
            counts[sid] += 1
        for sid, count in enumerate(counts):
            cap_val = float(caps[sid])
            if cap_val > 0 and count > cap_val:
                cap_errors.append((sid, count, cap_val))

    return {
        "coverage_ok": len(coverage_errors) == 0,
        "capacity_ok": len(cap_errors) == 0,
        "coverage_errors": coverage_errors,
        "capacity_errors": cap_errors,
    }
