"""
ILP models for sketch/switch placement, implemented with Gurobi.

The Phase I model is selection-only. It decides x[s] variables and leaves all
flow-to-switch assignment and balancing to Phase II.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from utils.data import FlowDataset


def build_core_model(
    dataset: FlowDataset,
    min_selected: int = 0,
    topology_links: Optional[Iterable[Tuple[int, int]]] = None,
) -> Tuple[gp.Model, Dict[int, gp.Var], Dict[Tuple[int, int], gp.Var]]:
    """Build the selection-only Phase I ILP.

    Decision variable:
        x[s] in {0,1}: switch s is selected.

    Objective:
        minimize sum_s x[s]

    Constraints:
        C1 coverage: every flow path has at least one selected switch.
        C2 capacity: if capacities are provided, x[s] * |F_s| <= cap(s).
        C3 minimum selection: if min_selected > 0, sum_s x[s] >= min_selected.
        C4 link coverage: if topology links are provided, x[u] + x[v] >= 1.

    The returned y_vars dict is intentionally empty and kept only so older
    callers that unpack (model, x_vars, y_vars) continue to work.
    """
    model = gp.Model("sketch_placement_selection")
    model.setParam("OutputFlag", 1)

    x_vars: Dict[int, gp.Var] = {
        sid: model.addVar(vtype=GRB.BINARY, name=f"x_{sid}")
        for sid in range(dataset.n_switches)
    }
    y_vars: Dict[Tuple[int, int], gp.Var] = {}
    model.update()

    model.setObjective(gp.quicksum(x_vars.values()), GRB.MINIMIZE)

    # C1 - path coverage.
    for f_idx, switches in enumerate(dataset.P):
        model.addConstr(
            gp.quicksum(x_vars[sid] for sid in switches) >= 1,
            name=f"C1_cover_f{f_idx}",
        )

    # C2 - switch capacity. Non-positive/missing capacities are treated as
    # unspecified, matching the existing evaluation helper semantics.
    if dataset.capacities is not None:
        flow_counts = [0] * dataset.n_switches
        for switches in dataset.P:
            for sid in switches:
                flow_counts[sid] += 1

        for sid, flow_count in enumerate(flow_counts):
            cap_val = float(dataset.capacities[sid])
            if cap_val > 0:
                model.addConstr(
                    x_vars[sid] * flow_count <= cap_val,
                    name=f"C2_capacity_s{sid}",
                )

    # C3 - minimum topology coverage.
    if min_selected > 0:
        model.addConstr(
            gp.quicksum(x_vars.values()) >= min_selected,
            name="C3_min_selected",
        )

    # C4 - link endpoint coverage.
    if topology_links is not None:
        for idx, (u_sid, v_sid) in enumerate(topology_links):
            model.addConstr(
                x_vars[u_sid] + x_vars[v_sid] >= 1,
                name=f"C4_link_{idx}",
            )

    return model, x_vars, y_vars


def build_set_cover_model(dataset):
    """Backward-compatible alias returning (model, x_vars)."""
    model, x_vars, _ = build_core_model(dataset)
    return model, x_vars


def build_assignment_model(dataset, lambda_penalty=0.0, capacities=None):
    """Backward-compatible alias returning an empty y_vars dict."""
    model, x_vars, y_vars = build_core_model(dataset)
    return model, x_vars, y_vars


def build_unified_model(dataset, **kwargs):
    """Backward-compatible alias returning an empty y_vars dict."""
    model, x_vars, y_vars = build_core_model(dataset)
    return model, x_vars, y_vars, None


def solve_model(model: gp.Model, write_lp: Optional[str] = None) -> int:
    """Optimize model and return the Gurobi status code."""
    if write_lp:
        model.write(write_lp)
    model.optimize()
    return model.Status


def extract_switch_selection(
    x_vars: Dict[int, gp.Var],
    sid_to_name: List[str],
) -> List[str]:
    """Return names of selected switches."""
    selected = []
    for sid, var in x_vars.items():
        try:
            if var.X > 0.5:
                selected.append(sid_to_name[sid])
        except gp.GurobiError:
            pass
    return selected


def extract_assignments_idx(
    y_vars: Dict[Tuple[int, int], gp.Var],
) -> Dict[int, int]:
    """Return model assignments. Selection-only models have none."""
    out: Dict[int, int] = {}
    for (f_idx, sid), var in y_vars.items():
        try:
            if var.X > 0.5:
                out[f_idx] = sid
        except gp.GurobiError:
            pass
    return out


def extract_assignments(
    y_vars: Dict[Tuple[int, int], gp.Var],
    fid_to_name: List[str],
    sid_to_name: List[str],
) -> Dict[str, str]:
    """Return model assignments. Selection-only models have none."""
    out: Dict[str, str] = {}
    for (f_idx, sid), var in y_vars.items():
        try:
            if var.X > 0.5:
                out[fid_to_name[f_idx]] = sid_to_name[sid]
        except gp.GurobiError:
            pass
    return out
