"""
Phase II — Flow Assignment
==========================
Reads:
  1. paths.jsonl.gz      — flow paths + sizes (from dataset generation)
  2. solution.json        — selected switches (output of Phase I ILP or greedy)

Assigns every flow to exactly one selected switch on its path using one of
three load-balancing strategies, then writes results to --out-dir.

Usage
-----
  python -m main --paths <paths.jsonl.gz> --solution <solution.json> [options]

  --strategy  first_match | balance_count | balance_size (default: balance_size)
  --out-dir   where to write outputs (default: out/run_<timestamp>)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Allow running as  python -m main  from inside phase-II/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase-I"))

from utils.data import FlowDataset, load_paths
from utils.eval import evaluate_assignment
from assignment.assign import (
    ALL_STRATEGIES,
    STRATEGY_BALANCE_SIZE,
    assign_flows,
    assignment_stats,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase II: assign flows to Phase-I-selected switches with load balancing."
    )
    parser.add_argument(
        "--paths", required=True,
        help="Path to paths.jsonl[.gz] (dataset output).",
    )
    parser.add_argument(
        "--solution", required=True,
        help="Path to solution.json produced by Phase I (ILP or greedy).",
    )
    parser.add_argument(
        "--strategy",
        choices=list(ALL_STRATEGIES),
        default=STRATEGY_BALANCE_SIZE,
        help=(
            "Assignment strategy. "
            "'first_match': first selected switch on path, no balancing. "
            "'balance_count': minimise max flow count per switch. "
            "'balance_size': minimise max byte load per switch (default)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        help="Output directory (default: out/run_<timestamp>).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    paths_path    = Path(args.paths)
    solution_path = Path(args.solution)
    out_dir       = Path(args.out_dir) if args.out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load inputs ---------------------------------------------------------
    dataset = load_paths(paths_path)
    selected_ids, selected_names, phase1_meta = _load_solution(solution_path, dataset)

    print(
        f"[phase-II] Flows={dataset.n_flows} | "
        f"Switches total={dataset.n_switches} | "
        f"Selected={len(selected_ids)} | "
        f"Strategy={args.strategy}"
    )

    # --- assign --------------------------------------------------------------
    assignments_idx: Dict[int, int] = assign_flows(
        dataset, selected_ids, strategy=args.strategy
    )

    # --- human-readable names ------------------------------------------------
    flow_assignments: Dict[str, str] = {
        dataset.fid_to_name[f_idx]: dataset.sid_to_name[sid]
        for f_idx, sid in assignments_idx.items()
    }

    # --- statistics ----------------------------------------------------------
    stats = assignment_stats(dataset, assignments_idx)

    # --- validation ----------------------------------------------------------
    eval_result = evaluate_assignment(dataset, assignments_idx)

    # --- write outputs -------------------------------------------------------
    _write_solution(
        out_dir, args.strategy, selected_ids, selected_names,
        flow_assignments, stats, eval_result, phase1_meta,
    )
    _write_assignments(out_dir, flow_assignments)
    _write_summary(out_dir, dataset, args.strategy, selected_names, flow_assignments, stats, eval_result)

    print(
        f"[phase-II] Assigned {len(flow_assignments)} flows | "
        f"Imbalance(bytes)={stats['imbalance_bytes']}x | "
        f"Coverage OK={eval_result['coverage_ok']} | "
        f"out={out_dir}"
    )


# ---------------------------------------------------------------------------
# Load Phase I solution
# ---------------------------------------------------------------------------

def _load_solution(solution_path: Path, dataset: FlowDataset):
    """Read solution.json, resolve switch names → integer IDs in this dataset."""
    with solution_path.open("r", encoding="utf-8") as fh:
        sol = json.load(fh)

    selected_names = sol.get("selected_switch_names", [])
    selected_ids_raw = sol.get("selected_switch_ids", [])

    # Build name → sid mapping from this dataset.
    name_to_sid = {name: idx for idx, name in enumerate(dataset.sid_to_name)}

    # Prefer names (portable across re-numbered datasets); fall back to ids.
    if selected_names:
        selected_ids = [name_to_sid[n] for n in selected_names if n in name_to_sid]
        missing = [n for n in selected_names if n not in name_to_sid]
        if missing:
            print(f"[phase-II] WARNING: {len(missing)} selected switch names not found in paths file: {missing[:5]}")
    else:
        selected_ids = list(selected_ids_raw)

    # Carry forward Phase I metadata for provenance in output.
    phase1_meta = {
        "phase1_status":   sol.get("status"),
        "phase1_objective": sol.get("objective"),
        "phase1_solution_file": str(solution_path),
    }

    return selected_ids, selected_names, phase1_meta


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_solution(
    out_dir: Path,
    strategy: str,
    selected_ids,
    selected_names,
    flow_assignments: Dict[str, str],
    stats: Dict,
    eval_result: Dict,
    phase1_meta: Dict,
) -> None:
    solution: Dict[str, Any] = {
        **phase1_meta,
        "assignment_strategy":   strategy,
        "selected_switch_ids":   selected_ids,
        "selected_switch_names": selected_names,
        "flow_assignments":      flow_assignments,
        "switch_load":  {k: v["flow_count"] for k, v in stats["per_switch"].items() if v["flow_count"] > 0},
        "switch_bytes": {k: v["total_bytes"] for k, v in stats["per_switch"].items() if v["total_bytes"] > 0},
        "balance": {
            "imbalance_count": stats["imbalance_count"],
            "imbalance_bytes": stats["imbalance_bytes"],
            "std_count":       stats["std_count"],
            "std_bytes":       stats["std_bytes"],
        },
        "eval": eval_result,
    }
    path = out_dir / "solution.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(solution, fh, indent=2, sort_keys=True)


def _write_assignments(out_dir: Path, flow_assignments: Dict[str, str]) -> None:
    """Write a clean flow_id → switch_name mapping, one entry per line."""
    path = out_dir / "assignments.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(flow_assignments, fh, indent=2, sort_keys=True)


def _write_summary(
    out_dir: Path,
    dataset: FlowDataset,
    strategy: str,
    selected_names,
    flow_assignments: Dict[str, str],
    stats: Dict,
    eval_result: Dict,
) -> None:
    path = out_dir / "summary.txt"
    b = stats["balance"] if "balance" in stats else stats
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"Phase II — Flow Assignment\n")
        fh.write(f"{'='*50}\n")
        fh.write(f"Strategy          : {strategy}\n")
        fh.write(f"Total flows       : {dataset.n_flows}\n")
        fh.write(f"Assigned flows    : {len(flow_assignments)}\n")
        fh.write(f"Selected switches : {len(selected_names)}\n")
        fh.write(f"Coverage OK       : {eval_result['coverage_ok']}\n")
        fh.write(f"Capacity OK       : {eval_result['capacity_ok']}\n")
        if dataset.has_flow_sizes():
            total_gb = float(dataset.flow_sizes.sum()) / 1e9
            fh.write(f"Total traffic     : {total_gb:.3f} GB\n")
        fh.write(f"\nLoad Balance\n")
        fh.write(f"{'-'*50}\n")
        fh.write(f"Imbalance (flows) : {stats['imbalance_count']}x  (1.0 = perfect)\n")
        fh.write(f"Imbalance (bytes) : {stats['imbalance_bytes']}x  (1.0 = perfect)\n")
        fh.write(f"Std-dev (flows)   : {stats['std_count']}\n")
        fh.write(f"Std-dev (bytes)   : {stats['std_bytes']}\n")
        fh.write(f"\nPer-Switch Breakdown\n")
        fh.write(f"{'-'*50}\n")
        fh.write(f"  {'Switch':<22} {'Flows':>8} {'%Flows':>8} {'Bytes':>16} {'%Bytes':>8}\n")
        fh.write(f"  {'-'*64}\n")
        for sw, d in sorted(
            stats["per_switch"].items(), key=lambda kv: -kv[1]["total_bytes"]
        ):
            if d["flow_count"] == 0:
                continue
            fh.write(
                f"  {sw:<22} {d['flow_count']:>8} {d['pct_flows']:>7.2f}%"
                f" {d['total_bytes']:>16,.0f} {d['pct_bytes']:>7.2f}%\n"
            )
        if eval_result.get("coverage_errors"):
            fh.write(f"\nCoverage errors (first 10): {eval_result['coverage_errors'][:10]}\n")
        if eval_result.get("capacity_errors"):
            fh.write(f"\nCapacity errors (first 10): {eval_result['capacity_errors'][:10]}\n")


def _default_out_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("out") / f"run_{ts}"


if __name__ == "__main__":
    main()