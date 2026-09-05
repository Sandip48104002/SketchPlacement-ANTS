"""
Phase III — CMS Simulation
===========================
Runs two kinds of simulation and compares them:

  OUR APPROACH (selective placement):
      One CMS per selected switch (Phase I output).
      Each flow updated in its ONE assigned switch only (Phase II output).
      Query:  single switch — assignments[flow_id].

  BASELINE (full deployment — all switches):
      One CMS per EVERY switch in the topology.
      Each flow updated in ALL switches on its path.
      Query:  min across all switches on the flow's path.

Metrics reported for each run:
    Estimation error  — mean/median/p95/p99/max relative error
    F1 score          — heavy-hitter detection at top-1% threshold
                        (precision, recall, F1)
    Throughput        — total CMS operations (updates + queries)
                        and ops-per-flow ratio vs baseline

Usage
-----
    cd phase-III

    # Our approach only (one strategy)
    python simulate.py \\
        --paths       ../dataset/out/abilene/paths.jsonl.gz \\
        --assignments ../phase-II/out/abilene_greedy/balance_size/assignments.json

    # All three strategies + baseline comparison
    python simulate.py \\
        --paths       ../dataset/out/abilene/paths.jsonl.gz \\
        --assignments ../phase-II/out/abilene_greedy/balance_size/assignments.json \\
                      ../phase-II/out/abilene_greedy/balance_count/assignments.json \\
                      ../phase-II/out/abilene_greedy/first_match/assignments.json \\
        --baseline \\
        --out-dir out/abilene_greedy
"""

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cms import CountMinSketch, demand_to_packets, WIDTH, DEPTH


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_paths_raw(paths_file: Path) -> List[Dict]:
    """Load paths.jsonl.gz → list of raw record dicts."""
    records = []
    opener = gzip.open if str(paths_file).endswith(".gz") else open
    with opener(paths_file, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_assignments(assignments_file: Path) -> Dict[str, str]:
    """Load assignments.json → { flow_id_str: switch_name }."""
    with assignments_file.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_demand_lookup(records: List[Dict]) -> Dict[str, int]:
    """flow_id_str → demand_bytes (int)."""
    lookup: Dict[str, int] = {}
    for r in records:
        fid  = str(r.get("id", r.get("flow_id", "")))
        size = r.get("demand_bytes") or r.get("demand") or 0
        lookup[fid] = int(size)
    return lookup


def _build_path_lookup(records: List[Dict]) -> Dict[str, List[str]]:
    """flow_id_str → list of switch names on its path."""
    lookup: Dict[str, List[str]] = {}
    for r in records:
        fid  = str(r.get("id", r.get("flow_id", "")))
        path = r.get("path") or r.get("nodes") or r.get("switches") or []
        lookup[fid] = [str(n) for n in path]
    return lookup


# ---------------------------------------------------------------------------
# F1 score — heavy hitter detection at top-1% threshold
# ---------------------------------------------------------------------------

def compute_f1(
    true_counts: np.ndarray,
    est_counts:  np.ndarray,
    percentile:  float = 99.0,
) -> Dict:
    """Compute precision, recall, and F1 for heavy-hitter detection.

    A flow is a heavy hitter if its TRUE packet count >= the top-1% threshold
    (99th percentile of all true counts).

    CMS never underestimates, so:
        FN = 0 always  →  recall = 1.0 always
        FP = non-heavy flows whose CMS estimate crosses the threshold due to
             collision inflation  →  drives precision down

    Parameters
    ----------
    true_counts : array of true packet counts, one per flow
    est_counts  : array of CMS estimates,      one per flow (same order)
    percentile  : threshold percentile (default 99.0 = top 1%)

    Returns
    -------
    Dict with threshold, n_true_hh, TP, FP, FN, TN,
              precision, recall, f1
    """
    threshold = float(np.percentile(true_counts, percentile))

    # Ground truth labels.
    true_hh = true_counts >= threshold   # True heavy hitters
    est_hh  = est_counts  >= threshold   # CMS-predicted heavy hitters

    TP = int(np.sum( true_hh &  est_hh))   # correctly flagged
    FP = int(np.sum(~true_hh &  est_hh))   # falsely flagged  (collision inflation)
    FN = int(np.sum( true_hh & ~est_hh))   # missed           (should be 0 for CMS)
    TN = int(np.sum(~true_hh & ~est_hh))   # correctly ignored

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "threshold_packets": int(threshold),
        "n_true_heavy":      int(np.sum(true_hh)),
        "n_est_heavy":       int(np.sum(est_hh)),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": round(precision, 6),
        "recall":    round(recall,    6),
        "f1":        round(f1,        6),
    }


# ---------------------------------------------------------------------------
# Throughput — total CMS operations (updates + queries)
# ---------------------------------------------------------------------------

def compute_throughput(
    n_flows:          int,
    total_update_ops: int,
    total_query_ops:  int,
) -> Dict:
    """Count total CMS hash+write operations for updates and queries.

    Each CMS update(flow, count) costs DEPTH operations
    (one hash + one array write per row).
    Each CMS query(flow) costs DEPTH operations
    (one hash + one array read per row).

    For the baseline, a flow with path length k costs k*DEPTH update ops
    and k*DEPTH query ops (one per switch on path).
    For our approach, each flow always costs exactly DEPTH update ops
    and DEPTH query ops regardless of path length.

    Parameters
    ----------
    n_flows          : number of flows simulated
    total_update_ops : sum of (path_length * DEPTH) over all flows  [baseline]
                       or  n_flows * DEPTH                          [selective]
    total_query_ops  : same structure as update ops
    """
    total_ops    = total_update_ops + total_query_ops
    ops_per_flow = total_ops / n_flows if n_flows > 0 else 0.0

    return {
        "total_update_ops": total_update_ops,
        "total_query_ops":  total_query_ops,
        "total_ops":        total_ops,
        "ops_per_flow":     round(ops_per_flow, 2),
    }


# ---------------------------------------------------------------------------
# Error aggregation helper (shared by both simulation functions)
# ---------------------------------------------------------------------------

def _aggregate_errors(
    per_switch_results: Dict[str, Dict],
    cms_map:            Dict[str, CountMinSketch],
    switches_in_use,
    true_arr:           np.ndarray,
    est_arr:            np.ndarray,
) -> Dict:
    """Build per-switch summary and overall stats including F1."""
    per_switch_summary = {}
    all_rel: List[float] = []
    all_abs: List[float] = []

    for sw, r in per_switch_results.items():
        rel = np.array(r["rel_errors"], dtype=float)
        ab  = np.array(r["abs_errors"], dtype=float)
        all_rel.extend(r["rel_errors"])
        all_abs.extend(r["abs_errors"])

        per_switch_summary[sw] = {
            "flow_count":      r["flow_count"],
            "memory_kb":       round(cms_map[sw].memory_bytes() / 1024, 1),
            "mean_rel_error":  round(float(rel.mean()), 6) if len(rel) else 0.0,
            "max_rel_error":   round(float(rel.max()),  6) if len(rel) else 0.0,
            "mean_abs_error":  round(float(ab.mean()),  2) if len(ab)  else 0.0,
            "max_abs_error":   int(ab.max())               if len(ab)  else 0,
            "overflow_count":  r["overflows"],
            "overflow_pct":    round(100.0 * r["overflows"] / r["flow_count"], 2)
                               if r["flow_count"] else 0.0,
            "total_true_pkts": r["total_true"],
            "total_est_pkts":  r["total_est"],
        }

    rel_arr = np.array(all_rel, dtype=float)
    ab_arr  = np.array(all_abs, dtype=float)

    # F1 computed over all flows globally (not per-switch).
    f1_stats = compute_f1(true_arr, est_arr, percentile=99.0)

    overall = {
        "flows_simulated":    sum(r["flow_count"] for r in per_switch_results.values()),
        # --- error ---
        "mean_rel_error":     round(float(rel_arr.mean()),            6) if len(rel_arr) else 0.0,
        "median_rel_error":   round(float(np.median(rel_arr)),        6) if len(rel_arr) else 0.0,
        "p95_rel_error":      round(float(np.percentile(rel_arr, 95)),6) if len(rel_arr) else 0.0,
        "p99_rel_error":      round(float(np.percentile(rel_arr, 99)),6) if len(rel_arr) else 0.0,
        "max_rel_error":      round(float(rel_arr.max()),             6) if len(rel_arr) else 0.0,
        "mean_abs_error":     round(float(ab_arr.mean()),             2) if len(ab_arr)  else 0.0,
        "max_abs_error":      int(ab_arr.max())                          if len(ab_arr)  else 0,
        # --- f1 ---
        "f1": f1_stats,
        # --- memory ---
        "cms_width":          WIDTH,
        "cms_depth":          DEPTH,
        "cms_memory_kb_each": round(CountMinSketch().memory_bytes() / 1024, 1),
        "total_switches":     len(switches_in_use),
        "total_memory_kb":    round(
            len(switches_in_use) * CountMinSketch().memory_bytes() / 1024, 1
        ),
    }

    return {"per_switch": per_switch_summary, "overall": overall}


# ---------------------------------------------------------------------------
# Simulation A — OUR APPROACH (selective placement + assignment)
# ---------------------------------------------------------------------------

def run_simulation(
    records:     List[Dict],
    assignments: Dict[str, str],
    label:       str = "",
) -> Dict:
    """
    OUR APPROACH
    ------------
    Update : each flow → its ONE assigned switch's CMS only.
    Query  : each flow → its ONE assigned switch's CMS only.

    Throughput ops per flow = DEPTH (update) + DEPTH (query) = 2 * DEPTH
    This is constant regardless of path length.
    """
    demand_lookup   = _build_demand_lookup(records)
    switches_in_use = set(assignments.values())
    cms_map: Dict[str, CountMinSketch] = {sw: CountMinSketch() for sw in switches_in_use}

    # --- UPDATE --------------------------------------------------------------
    # Each assigned flow costs exactly DEPTH ops (one hash+write per row).
    flows_skipped   = 0
    total_update_ops = 0

    for fid_str, switch_name in assignments.items():
        demand_bytes = demand_lookup.get(fid_str)
        if demand_bytes is None:
            flows_skipped += 1
            continue
        cms_map[switch_name].update(int(fid_str), demand_to_packets(demand_bytes))
        total_update_ops += DEPTH   # one hash + one write per row

    if flows_skipped:
        print(f"  [warn] {flows_skipped} flows in assignments not found in paths file")

    # --- QUERY ---------------------------------------------------------------
    # Each assigned flow costs exactly DEPTH ops (one hash+read per row).
    per_switch_results: Dict[str, Dict] = {sw: {
        "flow_count": 0, "total_true": 0, "total_est": 0,
        "abs_errors": [], "rel_errors": [], "overflows": 0,
    } for sw in switches_in_use}

    true_list: List[int] = []
    est_list:  List[int] = []
    total_query_ops = 0

    for fid_str, switch_name in assignments.items():
        demand_bytes = demand_lookup.get(fid_str)
        if demand_bytes is None:
            continue

        true_count = demand_to_packets(demand_bytes)
        est_count  = cms_map[switch_name].query(int(fid_str))  # ONE sketch only
        total_query_ops += DEPTH   # one hash + one read per row

        true_list.append(true_count)
        est_list.append(est_count)

        r = per_switch_results[switch_name]
        r["flow_count"]  += 1
        r["total_true"]  += true_count
        r["total_est"]   += est_count
        r["abs_errors"].append(abs(est_count - true_count))
        r["rel_errors"].append(
            abs(est_count - true_count) / true_count if true_count > 0 else 0.0
        )
        if est_count > true_count:
            r["overflows"] += 1

    true_arr = np.array(true_list, dtype=np.int64)
    est_arr  = np.array(est_list,  dtype=np.int64)

    result = _aggregate_errors(per_switch_results, cms_map, switches_in_use, true_arr, est_arr)

    # Throughput — ops count only, no timing.
    n_flows = len(true_list)
    result["overall"]["throughput"] = compute_throughput(
        n_flows, total_update_ops, total_query_ops
    )
    result["label"]    = label
    result["approach"] = "selective"
    return result


# ---------------------------------------------------------------------------
# Simulation B — BASELINE (all switches, all flows, path-min query)
# ---------------------------------------------------------------------------

def run_baseline_simulation(records: List[Dict], label: str = "baseline") -> Dict:
    """
    BASELINE — full deployment
    --------------------------
    Every switch in the topology gets a CMS.
    Every flow updated in ALL switches on its path.
    Query: MIN estimate across all path switches.

    Throughput ops per flow = path_length * DEPTH (update)
                            + path_length * DEPTH (query)
    This grows with topology diameter.
    """
    demand_lookup = _build_demand_lookup(records)
    path_lookup   = _build_path_lookup(records)

    all_switches: set = set()
    for path in path_lookup.values():
        all_switches.update(path)

    cms_map: Dict[str, CountMinSketch] = {sw: CountMinSketch() for sw in all_switches}

    # --- UPDATE  (every flow → ALL switches on its path) ---------------------
    # Cost per flow = path_length * DEPTH ops.
    total_update_ops = 0
    for r in records:
        fid_str      = str(r.get("id", r.get("flow_id", "")))
        demand_bytes = demand_lookup.get(fid_str, 0)
        pkt_count    = demand_to_packets(demand_bytes)
        path         = path_lookup.get(fid_str, [])
        for switch_name in path:
            cms_map[switch_name].update(int(fid_str), pkt_count)
            total_update_ops += DEPTH   # one hash + one write per row per switch

    # --- QUERY  (every flow → MIN across all path switches) ------------------
    # Cost per flow = path_length * DEPTH ops (query every switch then take min).
    per_switch_results: Dict[str, Dict] = {sw: {
        "flow_count": 0, "total_true": 0, "total_est": 0,
        "abs_errors": [], "rel_errors": [], "overflows": 0,
    } for sw in all_switches}

    true_list: List[int] = []
    est_list:  List[int] = []
    total_query_ops = 0

    for r in records:
        fid_str      = str(r.get("id", r.get("flow_id", "")))
        demand_bytes = demand_lookup.get(fid_str, 0)
        true_count   = demand_to_packets(demand_bytes)
        path         = path_lookup.get(fid_str, [])

        if not path:
            continue

        # Query every switch on the path — each costs DEPTH ops.
        estimates = {}
        for sw in path:
            estimates[sw]    = cms_map[sw].query(int(fid_str))
            total_query_ops += DEPTH

        best_switch = min(estimates, key=estimates.get)
        est_count   = estimates[best_switch]

        true_list.append(true_count)
        est_list.append(est_count)

        r2 = per_switch_results[best_switch]
        r2["flow_count"]  += 1
        r2["total_true"]  += true_count
        r2["total_est"]   += est_count
        r2["abs_errors"].append(abs(est_count - true_count))
        r2["rel_errors"].append(
            abs(est_count - true_count) / true_count if true_count > 0 else 0.0
        )
        if est_count > true_count:
            r2["overflows"] += 1

    true_arr = np.array(true_list, dtype=np.int64)
    est_arr  = np.array(est_list,  dtype=np.int64)

    result = _aggregate_errors(per_switch_results, cms_map, all_switches, true_arr, est_arr)

    n_flows = len(true_list)
    result["overall"]["throughput"] = compute_throughput(
        n_flows, total_update_ops, total_query_ops
    )
    result["label"]    = label
    result["approach"] = "baseline"
    return result


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_results(out_dir: Path, result: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "cms_results.json").open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)

    o        = result["overall"]
    f1       = o["f1"]
    tput     = o["throughput"]
    label    = result["label"]
    approach = result.get("approach", "")
    txt_path = out_dir / "cms_summary.txt"

    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Phase III — CMS Simulation\n")
        fh.write(f"{'='*60}\n")
        fh.write(f"Label             : {label}\n")
        fh.write(f"Approach          : {approach}\n")
        fh.write(f"CMS width         : {o['cms_width']:,} columns (2^16)\n")
        fh.write(f"CMS depth         : {o['cms_depth']} rows\n")
        fh.write(f"Memory per switch : {o['cms_memory_kb_each']} KB\n")
        fh.write(f"Switches with CMS : {o['total_switches']}\n")
        fh.write(f"Total CMS memory  : {o['total_memory_kb']} KB\n")
        fh.write(f"Flows simulated   : {o['flows_simulated']:,}\n")
        if approach == "baseline":
            fh.write(f"Query method      : min across all path switches\n")
        else:
            fh.write(f"Query method      : single assigned switch\n")

        fh.write(f"\nEstimation Error (relative = |est-true| / true)\n")
        fh.write(f"{'-'*60}\n")
        fh.write(f"  Mean   : {o['mean_rel_error']*100:>10.4f}%\n")
        fh.write(f"  Median : {o['median_rel_error']*100:>10.4f}%\n")
        fh.write(f"  p95    : {o['p95_rel_error']*100:>10.4f}%\n")
        fh.write(f"  p99    : {o['p99_rel_error']*100:>10.4f}%\n")
        fh.write(f"  Max    : {o['max_rel_error']*100:>10.4f}%\n")
        fh.write(f"  Mean abs error   : {o['mean_abs_error']:.2f} packets\n")
        fh.write(f"  Max  abs error   : {o['max_abs_error']:,} packets\n")

        fh.write(f"\nHeavy-Hitter Detection (top 1% threshold)\n")
        fh.write(f"{'-'*60}\n")
        fh.write(f"  Threshold        : >= {f1['threshold_packets']:,} packets\n")
        fh.write(f"  True heavy hitters  : {f1['n_true_heavy']:,}\n")
        fh.write(f"  Estimated heavy     : {f1['n_est_heavy']:,}\n")
        fh.write(f"  TP={f1['TP']:,}  FP={f1['FP']:,}  FN={f1['FN']:,}  TN={f1['TN']:,}\n")
        fh.write(f"  Precision        : {f1['precision']:.6f}\n")
        fh.write(f"  Recall           : {f1['recall']:.6f}\n")
        fh.write(f"  F1 Score         : {f1['f1']:.6f}\n")

        fh.write(f"\nThroughput (CMS operations count)\n")
        fh.write(f"{'-'*60}\n")
        fh.write(f"  Total update ops : {tput['total_update_ops']:>15,}\n")
        fh.write(f"  Total query ops  : {tput['total_query_ops']:>15,}\n")
        fh.write(f"  Total ops        : {tput['total_ops']:>15,}\n")
        fh.write(f"  Ops per flow     : {tput['ops_per_flow']:>15.2f}\n")

        fh.write(f"\nPer-Switch Breakdown\n")
        fh.write(f"{'-'*60}\n")
        fh.write(
            f"  {'Switch':<22} {'Flows':>7} {'MeanErr%':>9} "
            f"{'MaxErr%':>9} {'Overflow%':>10} {'Mem(KB)':>8}\n"
        )
        fh.write(f"  {'-'*69}\n")
        for sw, s in sorted(
            result["per_switch"].items(), key=lambda kv: -kv[1]["flow_count"]
        ):
            if s["flow_count"] == 0:
                continue
            fh.write(
                f"  {sw:<22} {s['flow_count']:>7,} "
                f"{s['mean_rel_error']*100:>8.4f}% "
                f"{s['max_rel_error']*100:>8.4f}% "
                f"{s['overflow_pct']:>9.2f}% "
                f"{s['memory_kb']:>7.1f}\n"
            )
    print(f"  → {txt_path}")


def write_comparison(
    out_dir:  Path,
    results:  List[Dict],
    baseline: Optional[Dict],
) -> None:
    """Full comparison table: error + F1 + throughput + memory, all in one view."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cms_comparison.txt"

    with path.open("w", encoding="utf-8") as fh:
        fh.write("Phase III — Full Comparison: Our Approach vs Baseline\n")
        fh.write(f"{'='*110}\n\n")

        # ---- Error + memory table ------------------------------------------
        fh.write("[ Estimation Error & Memory ]\n")
        fh.write(
            f"  {'Label':<22} {'Approach':<12} {'Switches':>9} {'Mem(KB)':>8} "
            f"{'Mean%':>9} {'Median%':>9} {'p95%':>8} {'p99%':>8} {'Max%':>10}\n"
        )
        fh.write(f"  {'-'*97}\n")
        for r in results:
            o = r["overall"]
            fh.write(
                f"  {r['label']:<22} {r.get('approach',''):<12} "
                f"{o['total_switches']:>9} {o['total_memory_kb']:>8.1f} "
                f"{o['mean_rel_error']*100:>8.4f}% "
                f"{o['median_rel_error']*100:>8.4f}% "
                f"{o['p95_rel_error']*100:>7.4f}% "
                f"{o['p99_rel_error']*100:>7.4f}% "
                f"{o['max_rel_error']*100:>9.4f}%\n"
            )
        if baseline:
            o = baseline["overall"]
            fh.write(f"  {'-'*97}\n")
            fh.write(
                f"  {baseline['label']:<22} {baseline.get('approach',''):<12} "
                f"{o['total_switches']:>9} {o['total_memory_kb']:>8.1f} "
                f"{o['mean_rel_error']*100:>8.4f}% "
                f"{o['median_rel_error']*100:>8.4f}% "
                f"{o['p95_rel_error']*100:>7.4f}% "
                f"{o['p99_rel_error']*100:>7.4f}% "
                f"{o['max_rel_error']*100:>9.4f}%\n"
            )

        # ---- F1 table -------------------------------------------------------
        fh.write(f"\n[ Heavy-Hitter Detection — top 1% threshold ]\n")
        fh.write(
            f"  {'Label':<22} {'Approach':<12} {'Threshold':>12} "
            f"{'TrueHH':>8} {'EstHH':>8} {'TP':>7} {'FP':>7} {'FN':>5} "
            f"{'Prec':>8} {'Recall':>8} {'F1':>8}\n"
        )
        fh.write(f"  {'-'*107}\n")
        for r in [*results] + ([baseline] if baseline else []):
            if r is None:
                continue
            o  = r["overall"]
            f1 = o["f1"]
            sep = "  " if r.get("approach") != "baseline" else f"  {'-'*105}\n  "
            if r.get("approach") == "baseline":
                fh.write(f"  {'-'*105}\n")
            fh.write(
                f"  {r['label']:<22} {r.get('approach',''):<12} "
                f"{f1['threshold_packets']:>12,} "
                f"{f1['n_true_heavy']:>8,} {f1['n_est_heavy']:>8,} "
                f"{f1['TP']:>7,} {f1['FP']:>7,} {f1['FN']:>5,} "
                f"{f1['precision']:>8.4f} {f1['recall']:>8.4f} {f1['f1']:>8.4f}\n"
            )

        # ---- Throughput table -----------------------------------------------
        fh.write(f"\n[ Throughput — CMS Operations Count ]\n")
        fh.write(
            f"  {'Label':<22} {'Approach':<12} "
            f"{'UpdateOps':>14} {'QueryOps':>12} {'TotalOps':>12} {'Ops/Flow':>10}\n"
        )
        fh.write(f"  {'-'*84}\n")
        for r in results:
            o    = r["overall"]
            tput = o["throughput"]
            fh.write(
                f"  {r['label']:<22} {r.get('approach',''):<12} "
                f"{tput['total_update_ops']:>14,} "
                f"{tput['total_query_ops']:>12,} "
                f"{tput['total_ops']:>12,} "
                f"{tput['ops_per_flow']:>10.2f}\n"
            )
        if baseline:
            o    = baseline["overall"]
            tput = o["throughput"]
            fh.write(f"  {'-'*84}\n")
            fh.write(
                f"  {baseline['label']:<22} {baseline.get('approach',''):<12} "
                f"{tput['total_update_ops']:>14,} "
                f"{tput['total_query_ops']:>12,} "
                f"{tput['total_ops']:>12,} "
                f"{tput['ops_per_flow']:>10.2f}\n"
            )

        # ---- Savings summary ------------------------------------------------
        if baseline:
            b_mem  = baseline["overall"]["total_memory_kb"]
            b_ops  = baseline["overall"]["throughput"]["total_ops"]
            b_f1   = baseline["overall"]["f1"]["f1"]
            fh.write(f"\n[ Savings vs Baseline ]\n")
            fh.write(f"  {'Label':<22} {'MemSaving%':>12} {'OpsSaving%':>12} {'F1 Delta':>10}\n")
            fh.write(f"  {'-'*58}\n")
            for r in results:
                our_mem = r["overall"]["total_memory_kb"]
                our_ops = r["overall"]["throughput"]["total_ops"]
                our_f1  = r["overall"]["f1"]["f1"]
                mem_s   = 100.0 * (b_mem - our_mem) / b_mem if b_mem > 0 else 0.0
                ops_s   = 100.0 * (b_ops - our_ops) / b_ops if b_ops > 0 else 0.0
                f1_d    = our_f1 - b_f1
                fh.write(
                    f"  {r['label']:<22} {mem_s:>11.1f}% {ops_s:>11.1f}% "
                    f"{f1_d:>+10.4f}\n"
                )

    print(f"\n[comparison] → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase III: CMS simulation with error, F1, and throughput comparison."
    )
    parser.add_argument("--paths",       required=True,
                        help="Path to paths.jsonl[.gz].")
    parser.add_argument("--assignments", required=True, nargs="+",
                        help="One or more assignments.json files from Phase II.")
    parser.add_argument("--labels",      nargs="+",
                        help="Labels for each assignments file.")
    parser.add_argument("--baseline",    action="store_true",
                        help="Also run the full-deployment baseline.")
    parser.add_argument("--out-dir",     default="out/simulation",
                        help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args       = parse_args()
    paths_file = Path(args.paths)
    out_dir    = Path(args.out_dir)
    asgn_files = [Path(p) for p in args.assignments]

    if args.labels:
        if len(args.labels) != len(asgn_files):
            raise ValueError("--labels count must match --assignments count")
        labels = args.labels
    else:
        labels = [p.parent.name for p in asgn_files]

    print(f"[phase-III] Loading paths from {paths_file} ...")
    records = load_paths_raw(paths_file)
    print(f"[phase-III] Loaded {len(records):,} flow records")

    our_results: List[Dict] = []
    for asgn_file, label in zip(asgn_files, labels):
        print(f"\n[phase-III] Simulating (selective): {label}")
        assignments = load_assignments(asgn_file)
        print(f"  Flows in assignments : {len(assignments):,}")

        result = run_simulation(records, assignments, label=label)
        o      = result["overall"]
        tput   = o["throughput"]
        f1     = o["f1"]
        print(
            f"  Switches : {o['total_switches']} | "
            f"Memory : {o['total_memory_kb']} KB | "
            f"Mean err : {o['mean_rel_error']*100:.4f}% | "
            f"F1 : {f1['f1']:.4f} | "
            f"Ops/flow : {tput['ops_per_flow']:.1f}"
        )
        write_results(out_dir / label, result)
        our_results.append(result)

    baseline_result: Optional[Dict] = None
    if args.baseline:
        print(f"\n[phase-III] Simulating (baseline — all switches, path-min) ...")
        baseline_result = run_baseline_simulation(records, label="baseline")
        o    = baseline_result["overall"]
        tput = o["throughput"]
        f1   = o["f1"]
        print(
            f"  Switches : {o['total_switches']} | "
            f"Memory : {o['total_memory_kb']} KB | "
            f"Mean err : {o['mean_rel_error']*100:.4f}% | "
            f"F1 : {f1['f1']:.4f} | "
            f"Ops/flow : {tput['ops_per_flow']:.1f}"
        )
        write_results(out_dir / "baseline", baseline_result)

    if len(our_results) > 1 or baseline_result:
        write_comparison(out_dir, our_results, baseline_result)

    print(f"\n[phase-III] Done. Results in {out_dir}")


if __name__ == "__main__":
    main()