# SketchPlacement

A four-stage pipeline for **sketch-based network flow monitoring**. Given a real
network topology and simulated traffic flows, it determines:

1. **Which switches** to instrument with a sketch (Phase I — ILP)
2. **Which switch monitors which flow**, with traffic load balancing (Phase II)
3. **How accurately** each approach monitors flows using Count-Min Sketches,
   with comparison against the full-deployment baseline (Phase III)

---

## How It Works

```
┌───────────┐  paths.jsonl.gz  ┌───────────┐  selected_switches.json  ┌───────────┐  assignments.json  ┌───────────┐
│  Dataset  │ ────────────────▶│  Phase I  │ ───────────────────────▶│  Phase II │ ──────────────────▶│ Phase III │
│Generation │                  │  (Select  │                          │  (Assign  │                    │   (CMS    │
│           │                  │ Switches) │                          │  Flows +  │                    │Simulation)│
└───────────┘                  └───────────┘                          │ Balancing)│                    └───────────┘
  topology +                   ILP — min switches                     └───────────┘                    error / F1
  flow model                   C1: every flow covered                 balance_size                     throughput
                               C2: assign → selected                  balance_count                    vs baseline
```

### Phase I — ILP (updated)

The ILP now solves a **joint selection + assignment** problem in a single model:

**Variables**

- `x[s] ∈ {0,1}` — switch `s` is selected (instrumented with a sketch)
- `y[f,s] ∈ {0,1}` — flow `f` is assigned to switch `s` (only defined for `s` on the path of `f`)

**Objective**

```
minimise   Σ_s  x[s]
```

**Constraints**

| ID | Constraint | Purpose |
|---|---|---|
| C1 | `∀ f: Σ_{s ∈ P(f)} y[f,s] = 1` | Every flow assigned to exactly one switch on its path |
| C2 | `∀ f, s ∈ P(f): y[f,s] ≤ x[s]` | A flow can only go to a selected switch |

The model uses **equality** in C1 (not `≥ 1` as in a pure set-cover), which
prevents duplicate monitoring and makes the assignment unambiguous for Phase II.
Load balancing across selected switches is handled entirely by Phase II.

> **Solver:** The ILP uses **Gurobi** via `gurobipy`. A valid Gurobi licence
> is required. Free academic licences are available at
> [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/).

### Phase II — Load Balancing (unchanged)

Given the selected switches from Phase I, assigns every flow to exactly one
switch using one of three strategies (`balance_size`, `balance_count`,
`first_match`).

### Phase III — CMS Simulation (unchanged)

Simulates a Count-Min Sketch on every selected switch and measures estimation
error, heavy-hitter F1, and throughput. Compares against the full-deployment
baseline.

---

## Directory Structure

```
SketchPlacement/
├── dataset/                    Dataset generation
│   ├── configs/
│   │   ├── run_abilene.yaml    Abilene (12 nodes)
│   │   ├── run_cogent.yaml     Cogent  (197 nodes)
│   │   └── run_geant.yaml      GEANT   (40 nodes)
│   ├── src/
│   │   ├── load_topology.py    Load GraphML, annotate link capacities
│   │   ├── generate_flows.py   Simulate flows (gravity / size_dist model)
│   │   ├── compute_paths.py    Shortest-path routing + capacity admission
│   │   └── run.py              Pipeline orchestrator
│   └── run_dataset.sh
│
├── phase-I/                    Switch selection ILP
│   ├── main.py                 Entry point
│   ├── ilp/ilp.py              build_core_model, solve_model, extract helpers
│   ├── greedy/main.py          Greedy set-cover entry point (no solver needed)
│   └── utils/
│       ├── data.py             FlowDataset dataclass, load_paths
│       └── eval.py             evaluate_cover, evaluate_assignment
│
├── phase-II/                   Flow assignment + load balancing
│   ├── main.py
│   └── run.sh
│
├── phase-III/                  CMS simulation + evaluation
│   ├── cms.py                  CountMinSketch class (w=2^16, d=3, int32)
│   ├── simulate.py             Simulation engine — selective + baseline
│   └── run.sh
│
├── requirements.txt
└── README.md
```

---

## Prerequisites

### Python packages

```bash
pip install -r requirements.txt
```

| Package | Version | Purpose |
|---|---|---|
| `networkx` | ≥ 3.2 | Topology loading and shortest-path routing |
| `numpy` | ≥ 1.24 | Array operations, statistics |
| `PyYAML` | ≥ 6.0 | Config file parsing |
| `gurobipy` | ≥ 11.0 | ILP solver for Phase I (requires a Gurobi licence) |
| `mmh3` | ≥ 4.0 | MurmurHash3 for CMS hash functions (Phase III) |

> **Gurobi licence required.** Phase I uses Gurobi as its ILP solver.
> Free academic licences are available at
> [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/).
> Run `grbgetkey YOUR-KEY` after registration to activate. The Greedy solver
> in Phase I does **not** require Gurobi.

### Topology data

GraphML files from [Topology Zoo](http://www.topology-zoo.org/) placed under
`dataset/data/topozoo/`:

```
dataset/data/topozoo/
├── Abilene.graphml
├── Cogentco.graphml
└── Geant2012.graphml
```

---

## Full Pipeline

### Step 1 — Generate Dataset

```bash
cd dataset
./run_dataset.sh configs/run_abilene.yaml   # Abilene  (12 nodes)
./run_dataset.sh configs/run_cogent.yaml    # Cogent   (197 nodes)
./run_dataset.sh configs/run_geant.yaml     # GEANT    (40 nodes)
```

On **Windows (PowerShell)**:

```powershell
cd dataset
python src/run.py --config configs/run_abilene.yaml
python src/run.py --config configs/run_cogent.yaml
python src/run.py --config configs/run_geant.yaml
```

**Output** written to `dataset/out/<topology>/`:

```
dataset/out/abilene/
├── topology.json       Node and edge list with link capacities
├── flows.csv           src, dst, demand for every generated flow
└── paths.jsonl.gz      One JSON record per flow: path, hops, cost, demand_bytes
```

---

### Step 2 — Phase I: Select Switches (ILP)

Run from inside `phase-I/`. The ILP jointly selects the minimum number of
switches and produces an initial flow assignment (C1 + C2 only — no load
balancing here).

```bash
cd phase-I
python main.py --input ../dataset/out/abilene/paths.jsonl.gz --out-dir ../out/abilene
```

On **Windows (PowerShell)**:

```powershell
cd phase-I
python main.py --input ../dataset/out/abilene/paths.jsonl.gz --out-dir ../out/abilene
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--input` | (required) | Path to `paths.jsonl[.gz]` |
| `--out-dir` | `out/run_<timestamp>` | Output directory |
| `--time-limit` | `600` | Solver time limit in seconds |
| `--skip-solve` | off | Write LP file only, do not solve |
| `--mode` | `solve` | `solve`, `preprocess`, or `eval` |

**Output** written to `--out-dir`:

```
out/abilene/
├── model.lp                 LP formulation (for inspection)
├── selected_switches.json   List of selected switch names  → Phase II input
├── assignments.json         Initial {flow_id: switch_name} map
├── solution.json            Full solver details (status, objective, eval)
└── summary.txt              Human-readable report
```

#### Greedy alternative (no solver, instant)

```bash
cd phase-I
python -m greedy.main --input ../dataset/out/abilene/paths.jsonl.gz --out-dir ../out/abilene_greedy
```

---

### Step 3 — Phase II: Assign Flows + Load Balance

Reads `selected_switches.json` from Phase I and assigns every flow to exactly
one selected switch using a load-balancing strategy.

```bash
cd phase-II
python main.py \
  --paths    ../dataset/out/abilene/paths.jsonl.gz \
  --switches ../out/abilene/selected_switches.json \
  --strategy balance_size \
  --out-dir  out/abilene/balance_size
```

On **Windows (PowerShell)**:

```powershell
cd phase-II
python main.py --paths ../dataset/out/abilene/paths.jsonl.gz --switches ../out/abilene/selected_switches.json --strategy balance_size --out-dir out/abilene/balance_size
```

**Assignment strategies:**

| Strategy | How it assigns | Best for |
|---|---|---|
| `balance_size` | Least total bytes so far; processes largest flows first (LPT). | Equalising **traffic volume** — default |
| `balance_count` | Fewest flows so far. | Equalising **flow count** per switch |
| `first_match` | First selected switch on the path. No balancing. | Baseline comparison only |

**Run all three strategies:**

```bash
for strategy in balance_size balance_count first_match; do
  python main.py \
    --paths    ../dataset/out/abilene/paths.jsonl.gz \
    --switches ../out/abilene/selected_switches.json \
    --strategy $strategy \
    --out-dir  out/abilene/$strategy
done
```

**Output** written to `--out-dir`:

```
phase-II/out/abilene/balance_size/
├── assignments.json    Clean { "flow_id": "switch_name", ... }  → Phase III input
├── solution.json       Provenance + balance stats + eval
└── summary.txt         Per-switch breakdown with imbalance metrics
```

---

### Step 4 — Phase III: CMS Simulation

Simulates a Count-Min Sketch on every selected switch and evaluates accuracy.

```bash
cd phase-III
python simulate.py \
  --paths       ../dataset/out/abilene/paths.jsonl.gz \
  --assignments ../phase-II/out/abilene/balance_size/assignments.json \
                ../phase-II/out/abilene/balance_count/assignments.json \
                ../phase-II/out/abilene/first_match/assignments.json \
  --baseline \
  --out-dir     out/abilene
```

On **Windows (PowerShell)**:

```powershell
cd phase-III
python simulate.py --paths ../dataset/out/abilene/paths.jsonl.gz --assignments ../phase-II/out/abilene/balance_size/assignments.json --baseline --out-dir out/abilene
```

**CMS parameters (fixed):**

| Parameter | Value | Notes |
|---|---|---|
| Width `w` | 2^16 = 65,536 columns | Per row |
| Depth `d` | 3 rows | Independent hash functions |
| Counter type | `int32` | Per cell |
| Hash function | MurmurHash3 (mmh3) | Seed 0/1/2 per row |
| Update value | `ceil(demand_bytes / 1500)` | Packet count estimate |
| Memory per switch | 768 KB | 3 × 65,536 × 4 bytes |

**Metrics reported:**

- **Estimation error** — mean/median/p95/p99/max relative error `|est−true|/true`
- **Heavy-hitter F1** — precision/recall/F1 at the top-1% packet-count threshold
- **Throughput** — total CMS hash + array operations; selective vs baseline

**Output:**

```
phase-III/out/abilene/
├── balance_size/   { cms_results.json, cms_summary.txt }
├── balance_count/  { cms_results.json, cms_summary.txt }
├── first_match/    { cms_results.json, cms_summary.txt }
├── baseline/       { cms_results.json, cms_summary.txt }
└── cms_comparison.txt    Side-by-side table of all runs
```

---

## Running the Full Pipeline — All Topologies

**Linux / macOS:**

```bash
# Step 1 — Dataset
cd dataset
for cfg in configs/run_abilene.yaml configs/run_cogent.yaml configs/run_geant.yaml; do
  ./run_dataset.sh $cfg
done

# Step 2 — Phase I (ILP on all topologies)
cd ../phase-I
for topo in abilene cogent geant; do
  python main.py --input ../dataset/out/$topo/paths.jsonl.gz --out-dir ../out/$topo
done

# Step 3 — Phase II (all strategies on all topologies)
cd ../phase-II
for topo in abilene cogent geant; do
  for strategy in balance_size balance_count first_match; do
    python main.py \
      --paths    ../dataset/out/$topo/paths.jsonl.gz \
      --switches ../out/$topo/selected_switches.json \
      --strategy $strategy \
      --out-dir  out/$topo/$strategy
  done
done

# Step 4 — Phase III (all strategies + baseline on all topologies)
cd ../phase-III
for topo in abilene cogent geant; do
  python simulate.py \
    --paths       ../dataset/out/$topo/paths.jsonl.gz \
    --assignments ../phase-II/out/$topo/balance_size/assignments.json \
                  ../phase-II/out/$topo/balance_count/assignments.json \
                  ../phase-II/out/$topo/first_match/assignments.json \
    --baseline \
    --out-dir     out/$topo
done
```

**Windows (PowerShell) — run each topology manually:**

```powershell
# Abilene — full pipeline
cd phase-I
python main.py --input ../dataset/out/abilene/paths.jsonl.gz --out-dir ../out/abilene
cd ../phase-II
python main.py --paths ../dataset/out/abilene/paths.jsonl.gz --switches ../out/abilene/selected_switches.json --strategy balance_size --out-dir out/abilene/balance_size
python main.py --paths ../dataset/out/abilene/paths.jsonl.gz --switches ../out/abilene/selected_switches.json --strategy balance_count --out-dir out/abilene/balance_count
python main.py --paths ../dataset/out/abilene/paths.jsonl.gz --switches ../out/abilene/selected_switches.json --strategy first_match --out-dir out/abilene/first_match
cd ../phase-III
python simulate.py --paths ../dataset/out/abilene/paths.jsonl.gz --assignments ../phase-II/out/abilene/balance_size/assignments.json ../phase-II/out/abilene/balance_count/assignments.json ../phase-II/out/abilene/first_match/assignments.json --baseline --out-dir out/abilene
```

---

## Output Summary

After running the full pipeline:

```
dataset/out/
├── abilene/   { topology.json, flows.csv, paths.jsonl.gz }
├── cogent/    { topology.json, flows.csv, paths.jsonl.gz }
└── geant/     { topology.json, flows.csv, paths.jsonl.gz }

out/                              ← Phase I output
├── abilene/   { model.lp, selected_switches.json, assignments.json, solution.json, summary.txt }
├── cogent/    { ... }
└── geant/     { ... }

phase-II/out/
├── abilene/
│   ├── balance_size/   { assignments.json, solution.json, summary.txt }
│   ├── balance_count/  { assignments.json, solution.json, summary.txt }
│   └── first_match/    { assignments.json, solution.json, summary.txt }
├── cogent/   ...
└── geant/    ...

phase-III/out/
├── abilene/
│   ├── balance_size/   { cms_results.json, cms_summary.txt }
│   ├── balance_count/  { cms_results.json, cms_summary.txt }
│   ├── first_match/    { cms_results.json, cms_summary.txt }
│   ├── baseline/       { cms_results.json, cms_summary.txt }
│   └── cms_comparison.txt
├── cogent/   ...
└── geant/    ...
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: gurobipy` | `pip install gurobipy` |
| `ModuleNotFoundError: mmh3` | `pip install mmh3` |
| `FileNotFoundError: paths.jsonl.gz` | Run Step 1 (dataset generation) first |
| `selected_switches.json not found` | Run Phase I before Phase II |
| Phase I takes too long | Add `--time-limit 1200`; or use the greedy solver |
| Gurobi `License has expired` | Renew at [portal.gurobi.com](https://portal.gurobi.com) and run `grbgetkey YOUR-KEY` |