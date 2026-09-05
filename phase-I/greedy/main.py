"""
Phase I — Greedy Switch Selection
===================================
Selects a near-minimum set of switches covering all flows using the standard
greedy set-cover algorithm (O(log n) approximation).

Output: solution.json with selected_switch_ids + selected_switch_names.
Assignment of flows to switches is handled by Phase II.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

from utils.data import FlowDataset, load_paths
from utils.eval import evaluate_cover


def greedy_set_cover(dataset: FlowDataset) -> Tuple[List[int], Set[int]]:
    """Standard greedy set cover: repeatedly pick the switch that covers the
    most currently-uncovered flows until all flows are covered."""
    switch_to_flows: List[Set[int]] = [set() for _ in range(dataset.n_switches)]
    for f_idx, switches in enumerate(dataset.P):
        for sid in switches:
            switch_to_flows[sid].add(f_idx)

    uncovered: Set[int] = set(range(dataset.n_flows))
    selected: List[int] = []

    while uncovered:
        best_sid  = None
        best_gain = 0
        for sid, flows in enumerate(switch_to_flows):
            gain = len(uncovered.intersection(flows))
            if gain > best_gain:
                best_gain = gain
                best_sid  = sid
        if best_sid is None or best_gain == 0:
            break
        selected.append(best_sid)
        uncovered.difference_update(switch_to_flows[best_sid])

    return selected, uncovered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase I greedy: select near-minimum switches to cover all flows."
    )
    parser.add_argument("--input",   required=True, help="Path to paths.jsonl[.gz].")
    parser.add_argument("--out-dir", help="Output directory (default: out/run_<timestamp>).")
    return parser.parse_args()


def main() -> None:
    args    = parse_args()
    dataset = load_paths(Path(args.input))
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_greedy(dataset, out_dir)


def _run_greedy(dataset: FlowDataset, out_dir: Path) -> None:
    selected_ids, _ = greedy_set_cover(dataset)
    cover_ok, uncovered_list = evaluate_cover(dataset, selected_ids)
    selected_names = [dataset.sid_to_name[sid] for sid in selected_ids]

    solution: Dict = {
        "status":                "Greedy",
        "objective":             len(selected_ids),
        "selected_switch_ids":   selected_ids,
        "selected_switch_names": selected_names,
        "coverage_ok":           cover_ok,
    }
    if not cover_ok:
        solution["uncovered_count"] = len(uncovered_list)

    solution_path = out_dir / "solution.json"
    with solution_path.open("w", encoding="utf-8") as fh:
        json.dump(solution, fh, indent=2, sort_keys=True)

    summary_path = out_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Phase I — Greedy Switch Selection\n")
        fh.write(f"{'='*40}\n")
        fh.write(f"Flows           : {dataset.n_flows}\n")
        fh.write(f"Switches total  : {dataset.n_switches}\n")
        fh.write(f"Selected        : {len(selected_ids)}\n")
        fh.write(f"Coverage OK     : {cover_ok}\n")
        fh.write(f"\nSelected switches: {', '.join(selected_names)}\n")
        fh.write(f"\nNext step → run Phase II to assign flows:\n")
        fh.write(f"  cd ../phase-II\n")
        fh.write(f"  ./run.sh ../dataset/out/<topology>/paths.jsonl.gz {out_dir}/solution.json\n")

    print(
        f"[phase-I greedy] Selected {len(selected_ids)} switches | "
        f"Coverage OK={cover_ok} | out={out_dir}"
    )
    print(f"  → Run Phase II:  cd ../phase-II && ./run.sh <paths.jsonl.gz> {out_dir}/solution.json")


def _default_out_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("out") / f"run_{ts}"


if __name__ == "__main__":
    main()