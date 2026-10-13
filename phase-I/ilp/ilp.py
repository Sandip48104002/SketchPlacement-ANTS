"""
ILP models for sketch/switch placement, implemented with Gurobi.

Phase I:
    Select switches that will host Count-Min Sketches.

The model supports heterogeneous programmable hardware:
    10%, 25%, 50%, 75%, or 100% of switches can be
    designated as programmable/CMS-capable.

Phase I remains selection-only:
    - x[s] determines whether switch s is selected.
    - Flow-to-switch assignment remains in Phase II.

Constraints:
    C0 - Only programmable switches can be selected.
    C1 - Every flow must have at least one programmable selected
         switch on its path.
    C2 - Optional switch capacity constraint.
    C3 - Optional minimum number of selected switches.
    C4 - Optional link endpoint coverage constraint.
"""

from typing import Dict, Iterable, List, Optional, Tuple
import random

import gurobipy as gp
from gurobipy import GRB

from utils.data import FlowDataset


# ============================================================
# Programmable Switch Selection
# ============================================================

def select_programmable_switches(
    n_switches: int,
    fraction: float,
    seed: int = 42,
) -> List[int]:
    """
    Randomly select the switches that are capable of hosting CMS.

    Parameters
    ----------
    n_switches:
        Total number of switches in the topology.

    fraction:
        Fraction of switches that are programmable/CMS-capable.
        Example:
            0.10 -> 10%
            0.25 -> 25%
            0.50 -> 50%
            0.75 -> 75%
            1.00 -> 100%

    seed:
        Random seed used to make the experiment reproducible.

    Returns
    -------
    List[int]
        Sorted list of programmable switch IDs.
    """

    if not (0.0 < fraction <= 1.0):
        raise ValueError(
            "fraction must be in the range (0, 1]."
        )

    num_programmable = max(
        1,
        int(round(fraction * n_switches))
    )

    num_programmable = min(
        num_programmable,
        n_switches
    )

    rng = random.Random(seed)

    programmable_switches = rng.sample(
        range(n_switches),
        num_programmable
    )

    return sorted(programmable_switches)


# ============================================================
# ILP Model
# ============================================================

def build_core_model(
    dataset: FlowDataset,
    min_selected: int = 0,
    topology_links: Optional[
        Iterable[Tuple[int, int]]
    ] = None,
    programmable_fraction: float = 1.0,
    programmable_switches: Optional[
        Iterable[int]
    ] = None,
) -> Tuple[
    gp.Model,
    Dict[int, gp.Var],
    Dict[Tuple[int, int], gp.Var]
]:
    """
    Build the selection-only Phase I ILP.

    Decision variable
    -----------------
    x[s] in {0,1}:
        1 if switch s is selected for CMS deployment.
        0 otherwise.

    Only programmable/CMS-capable switches are allowed
    to have x[s] = 1.

    Objective
    ---------
        minimize sum_s x[s]

    Constraints
    -----------
    C0:
        Non-programmable switches cannot be selected.

    C1:
        Every flow path must contain at least one selected
        programmable switch.

    C2:
        Optional switch capacity constraint.

    C3:
        Optional minimum number of selected switches.

    C4:
        Optional link endpoint coverage.

    Parameters
    ----------
    dataset:
        FlowDataset containing topology, paths, capacities, etc.

    min_selected:
        Optional lower bound on the number of selected switches.
        Recommended value for the programmable-switch experiment:
        0.

    topology_links:
        Optional list of topology links. If supplied, the current
        link coverage constraint is added.

    programmable_fraction:
        Fraction of switches that are programmable.

        Supported values include:
            0.10
            0.25
            0.50
            0.75
            1.00

    programmable_switches:
        Explicit list/set of programmable switch IDs.

        If supplied, this takes precedence over
        programmable_fraction.

    Returns
    -------
    model:
        Gurobi model.

    x_vars:
        Dictionary mapping switch ID -> binary decision variable.

    y_vars:
        Empty dictionary retained for backward compatibility.
    """

    # --------------------------------------------------------
    # Validate programmable fraction
    # --------------------------------------------------------

    if not (0.0 < programmable_fraction <= 1.0):
        raise ValueError(
            "programmable_fraction must be in the range (0, 1]."
        )

    # --------------------------------------------------------
    # Create Gurobi model
    # --------------------------------------------------------

    model = gp.Model(
        "sketch_placement_selection"
    )

    model.setParam(
        "OutputFlag",
        1
    )

    # --------------------------------------------------------
    # Determine programmable switches
    # --------------------------------------------------------

    if programmable_switches is not None:

        programmable_set = set(
            programmable_switches
        )

        # Check switch IDs
        invalid_switches = [
            sid
            for sid in programmable_set
            if sid < 0 or sid >= dataset.n_switches
        ]

        if invalid_switches:
            raise ValueError(
                "Invalid programmable switch IDs: "
                f"{invalid_switches}"
            )

    else:

        programmable_switches = select_programmable_switches(
            n_switches=dataset.n_switches,
            fraction=programmable_fraction,
            seed=42,
        )

        programmable_set = set(
            programmable_switches
        )

    # --------------------------------------------------------
    # Decision variables
    # --------------------------------------------------------

    x_vars: Dict[int, gp.Var] = {
        sid: model.addVar(
            vtype=GRB.BINARY,
            name=f"x_{sid}"
        )
        for sid in range(dataset.n_switches)
    }

    # Phase I is selection-only.
    # Kept for compatibility with existing code.
    y_vars: Dict[
        Tuple[int, int],
        gp.Var
    ] = {}

    model.update()

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    model.setObjective(
        gp.quicksum(
            x_vars.values()
        ),
        GRB.MINIMIZE
    )

    # ========================================================
    # C0 - Programmable Switch Restriction
    # ========================================================

    for sid in range(dataset.n_switches):

        if sid not in programmable_set:

            model.addConstr(
                x_vars[sid] == 0,
                name=f"C0_non_programmable_s{sid}",
            )

    # ========================================================
    # C1 - Flow / Path Coverage
    # ========================================================

    for f_idx, switches in enumerate(dataset.P):

        # Only programmable switches can satisfy coverage.
        programmable_on_path = [
            sid
            for sid in switches
            if sid in programmable_set
        ]

        # If a flow has no programmable switch on its path,
        # complete coverage is impossible.
        if not programmable_on_path:

            model.addConstr(
                gp.LinExpr() >= 1,
                name=f"C1_infeasible_f{f_idx}",
            )

        else:

            model.addConstr(
                gp.quicksum(
                    x_vars[sid]
                    for sid in programmable_on_path
                ) >= 1,
                name=f"C1_cover_f{f_idx}",
            )

    # ========================================================
    # C2 - Switch Capacity
    # ========================================================

    if dataset.capacities is not None:

        # Number of flows whose paths traverse each switch.
        flow_counts = [
            0
            for _ in range(dataset.n_switches)
        ]

        for switches in dataset.P:

            for sid in switches:

                flow_counts[sid] += 1

        # Apply capacity only to programmable switches.
        for sid in programmable_set:

            flow_count = flow_counts[sid]

            cap_val = float(
                dataset.capacities[sid]
            )

            if cap_val > 0:

                model.addConstr(
                    x_vars[sid] * flow_count
                    <= cap_val,
                    name=f"C2_capacity_s{sid}",
                )

    # ========================================================
    # C3 - Minimum Selected Switches
    # ========================================================

    if min_selected > 0:

        model.addConstr(
            gp.quicksum(
                x_vars[sid]
                for sid in programmable_set
            ) >= min_selected,
            name="C3_min_selected",
        )

    # ========================================================
    # C4 - Link Endpoint Coverage
    # ========================================================

    if topology_links is not None:

        for idx, (
            u_sid,
            v_sid
        ) in enumerate(topology_links):

            model.addConstr(
                x_vars[u_sid]
                + x_vars[v_sid]
                >= 1,
                name=f"C4_link_{idx}",
            )

    return (
        model,
        x_vars,
        y_vars
    )


# ============================================================
# Backward-Compatible Set-Cover Model
# ============================================================

def build_set_cover_model(
    dataset: FlowDataset,
    programmable_fraction: float = 1.0,
    programmable_switches: Optional[
        Iterable[int]
    ] = None,
):
    """
    Backward-compatible alias.

    Returns:
        model, x_vars
    """

    model, x_vars, _ = build_core_model(
        dataset=dataset,
        programmable_fraction=programmable_fraction,
        programmable_switches=programmable_switches,
    )

    return model, x_vars


# ============================================================
# Backward-Compatible Assignment Model
# ============================================================

def build_assignment_model(
    dataset,
    lambda_penalty=0.0,
    capacities=None,
    programmable_fraction: float = 1.0,
    programmable_switches: Optional[
        Iterable[int]
    ] = None,
):
    """
    Backward-compatible alias.

    Phase I remains selection-only.
    """

    model, x_vars, y_vars = build_core_model(
        dataset=dataset,
        programmable_fraction=programmable_fraction,
        programmable_switches=programmable_switches,
    )

    return (
        model,
        x_vars,
        y_vars
    )


# ============================================================
# Unified Model
# ============================================================

def build_unified_model(
    dataset,
    programmable_fraction: float = 1.0,
    programmable_switches: Optional[
        Iterable[int]
    ] = None,
    **kwargs,
):
    """
    Backward-compatible alias returning:

        model, x_vars, y_vars, None
    """

    model, x_vars, y_vars = build_core_model(
        dataset=dataset,
        programmable_fraction=programmable_fraction,
        programmable_switches=programmable_switches,
    )

    return (
        model,
        x_vars,
        y_vars,
        None
    )


# ============================================================
# Solve Model
# ============================================================

def solve_model(
    model: gp.Model,
    write_lp: Optional[str] = None,
) -> int:
    """
    Optimize model and return the Gurobi status code.
    """

    if write_lp:
        model.write(
            write_lp
        )

    model.optimize()

    return model.Status


# ============================================================
# Extract Selected Switches
# ============================================================

def extract_switch_selection(
    x_vars: Dict[int, gp.Var],
    sid_to_name: List[str],
) -> List[str]:
    """
    Return names of selected switches.
    """

    selected = []

    for sid, var in x_vars.items():

        try:

            if var.X > 0.5:

                selected.append(
                    sid_to_name[sid]
                )

        except gp.GurobiError:
            pass

    return selected


# ============================================================
# Extract Assignments - Index Based
# ============================================================

def extract_assignments_idx(
    y_vars: Dict[
        Tuple[int, int],
        gp.Var
    ],
) -> Dict[int, int]:
    """
    Return model assignments.

    Phase-I selection-only models have no assignment variables.
    """

    out: Dict[int, int] = {}

    for (
        f_idx,
        sid
    ), var in y_vars.items():

        try:

            if var.X > 0.5:

                out[f_idx] = sid

        except gp.GurobiError:
            pass

    return out


# ============================================================
# Extract Assignments - Name Based
# ============================================================

def extract_assignments(
    y_vars: Dict[
        Tuple[int, int],
        gp.Var
    ],
    fid_to_name: List[str],
    sid_to_name: List[str],
) -> Dict[str, str]:
    """
    Return model assignments.

    Phase-I selection-only models have no assignment variables.
    """

    out: Dict[str, str] = {}

    for (
        f_idx,
        sid
    ), var in y_vars.items():

        try:

            if var.X > 0.5:

                out[
                    fid_to_name[f_idx]
                ] = sid_to_name[sid]

        except gp.GurobiError:
            pass

    return out
