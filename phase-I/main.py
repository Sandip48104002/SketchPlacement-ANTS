"""
Phase I — Switch Selection ILP  (Gurobi)
=========================================
Reads paths.jsonl.gz and solves the minimal ILP:
  • Select the fewest switches such that every flow's path has at least one.
  • Assign each flow to exactly one selected switch on its path.

Load balancing is NOT done here — it is handled by Phase II.

Output files written to --out-dir
----------------------------------
  model.lp               Gurobi LP file (for inspection)
  selected_switches.json List of selected switch names  → Phase II input
  solution.json          Full solver details
  summary.txt            Human-readable report

Usage
-----
  python main.py --input ../dataset/out/abilene/paths.jsonl.gz --out-dir ../out/abilene
  python main.py --input ../dataset/out/cogent/paths.jsonl.gz  --out-dir ../out/cogent

CLI flags
---------
  --input       paths.jsonl[.gz]          (required)
  --out-dir     Output directory
  --time-limit  Gurobi time limit in sec  (default 600)
  --skip-solve  Write LP only, do not solve
  --mode        solve | preprocess | eval
  --solution    Path to solution.json for eval mode
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from ilp.ilp import (
    build_core_model,
    solve_model,
    extract_switch_selection,
)
from utils.data import FlowDataset, load_paths
from utils.eval import evaluate_cover


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase I ILP: select minimum switches and assign flows."
    )
    parser.add_argument("--input",      required=True,
                        help="Path to paths.jsonl[.gz].")
    parser.add_argument("--out-dir",
                        help="Output directory (default: out/run_<timestamp>).")
    parser.add_argument("--time-limit", type=float, default=600.0,
                        help="Gurobi time limit in seconds (default 600).")
    parser.add_argument("--skip-solve", action="store_true",
                        help="Write LP file but do not solve.")
    parser.add_argument("--mode",       choices=["solve", "preprocess", "eval"],
                        default="solve")
    parser.add_argument("--solution",
                        help="Path to solution.json for eval mode.")
    parser.add_argument("--topology",
                        help="Path to topology.json to enable link coverage (C4).")
    parser.add_argument("--min-coverage-fraction", type=float, default=0.0,
                        help="Minimum fraction of switches to select (C3, e.g. 0.1).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    if args.mode == "preprocess":
        dataset = load_paths(input_path)
        print(_summary(dataset))
        return

    if args.mode == "eval":
        if not args.solution:
            raise ValueError("--solution is required for eval mode.")
        dataset = load_paths(input_path)
        _run_eval(dataset, Path(args.solution))
        return

    dataset = load_paths(input_path)
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    topology_path = Path(args.topology) if args.topology else None

    _run_solve(
        dataset,
        out_dir,
        args.time_limit,
        args.skip_solve,
        topology_path,
        args.min_coverage_fraction,
    )


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def _run_solve(
    dataset:    FlowDataset,
    out_dir:    Path,
    time_limit: float,
    skip_solve: bool,
    topology_path: Optional[Path] = None,
    min_coverage_fraction: float = 0.0,
) -> None:
    import math

    topology_links: Optional[List[Tuple[int, int]]] = None
    if topology_path and topology_path.exists():
        try:
            with topology_path.open() as fh:
                topo_data = json.load(fh)
            # Find the index of switch names to support mapping
            name_to_sid = {name: sid for sid, name in enumerate(dataset.sid_to_name)}
            topology_links = []
            for edge in topo_data.get("edges", []):
                u_name = str(edge.get("u"))
                v_name = str(edge.get("v"))
                if u_name in name_to_sid and v_name in name_to_sid:
                    topology_links.append((name_to_sid[u_name], name_to_sid[v_name]))
        except Exception as e:
            print(f"[Phase-I ILP] Failed to load topology from {topology_path}: {e}")

    min_selected = 0
    if min_coverage_fraction > 0.0:
        min_selected = math.ceil(min_coverage_fraction * dataset.n_switches)

    print(f"[Phase-I ILP] {_summary(dataset)}")
    print(f"[Phase-I ILP] Building model (min_selected={min_selected}, links={len(topology_links) if topology_links else 0}) …")

    model, x_vars, _ = build_core_model(
        dataset,
        min_selected=min_selected,
        topology_links=topology_links
    )
    model.setParam("TimeLimit", time_limit)

    lp_path = str(out_dir / "model.lp")

    if skip_solve:
        model.write(lp_path)
        print(f"[Phase-I ILP] LP written to {lp_path}  (skip-solve)")
        return

    print(f"[Phase-I ILP] Solving (time limit {time_limit}s) …")
    status = solve_model(model, write_lp=lp_path)

    is_feasible = status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0

    status_label = {
        GRB.OPTIMAL:     "Optimal",
        GRB.INFEASIBLE:  "Infeasible",
        GRB.INF_OR_UNBD: "InfOrUnbounded",
        GRB.TIME_LIMIT:  "TimeLimit",
        GRB.UNBOUNDED:   "Unbounded",
    }.get(status, f"GRBStatus{status}")

    if not is_feasible:
        print(f"[Phase-I ILP] No feasible solution — status={status_label}")
        (out_dir / "solution.json").write_text(
            json.dumps({"status": status_label}, indent=2)
        )
        return

    obj_value      = model.ObjVal
    selected_ids   = [sid for sid, v in x_vars.items() if v.X > 0.5]
    selected_names = [dataset.sid_to_name[sid] for sid in selected_ids]

    cover_ok, uncovered = evaluate_cover(dataset, selected_ids)

    # --- write selected_switches.json  (Phase II reads this) -----------------
    sw_path = out_dir / "selected_switches.json"
    with sw_path.open("w") as fh:
        json.dump(selected_names, fh, indent=2)

    # --- write solution.json --------------------------------------------------
    solution: Dict[str, Any] = {
        "status":                status_label,
        "objective":             obj_value,
        "selected_switch_ids":   selected_ids,
        "selected_switch_names": selected_names,
    }
    if not cover_ok:
        solution["uncovered_count"] = len(uncovered)
    (out_dir / "solution.json").write_text(json.dumps(solution, indent=2, sort_keys=True))

    # --- write summary.txt ---------------------------------------------------
    _write_summary(out_dir, dataset, status_label, obj_value,
                   selected_names)

    print(
        f"[Phase-I ILP] Status={status_label} | "
        f"Switches selected={len(selected_ids)} | "
        f"out={out_dir}"
    )
    print(f"\nNext → Phase II (load balancing):")
    print(f"  cd ../phase-II && python main.py "
          f"--switches {sw_path} "
          f"--input ../dataset/out/abilene/paths.jsonl.gz "
          f"--out-dir ../out/abilene_phase2")


# ---------------------------------------------------------------------------
# Eval mode
# ---------------------------------------------------------------------------

def _run_eval(dataset: FlowDataset, solution_path: Path) -> None:
    sol = json.loads(solution_path.read_text())
    selected_names = sol.get("selected_switch_names", [])
    name_to_id   = {name: idx for idx, name in enumerate(dataset.sid_to_name)}
    selected_ids = [name_to_id[n] for n in selected_names if n in name_to_id]

    cover_ok, uncovered = evaluate_cover(dataset, selected_ids)

    print(json.dumps({"cover_ok": cover_ok, "uncovered": len(uncovered)}, indent=2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_summary(
    out_dir:       Path,
    dataset:       FlowDataset,
    status_label:  str,
    obj_value:     Any,
    selected_names: List[str],
) -> None:
    with (out_dir / "summary.txt").open("w") as fh:
        fh.write("Phase I — Switch Selection ILP\n")
        fh.write("=" * 40 + "\n")
        fh.write(_summary(dataset) + "\n\n")
        fh.write(f"Status           : {status_label}\n")
        fh.write(f"Objective        : {obj_value}\n")
        fh.write(f"Switches selected: {len(selected_names)}\n\n")
        fh.write(f"Selected switches:\n")
        for name in selected_names:
            fh.write(f"  {name}\n")
        fh.write(f"\nNext → Phase II for load balancing.\n")


def _summary(dataset: FlowDataset) -> str:
    size_info = "no sizes"
    if dataset.has_flow_sizes():
        total_gb = float(dataset.flow_sizes.sum()) / 1e9
        size_info = f"{total_gb:.3f} GB total"
    return f"Flows={dataset.n_flows} | Switches={dataset.n_switches} | {size_info}"


def _default_out_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("out") / f"run_{ts}"


if __name__ == "__main__":
    main()