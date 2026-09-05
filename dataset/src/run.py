import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

from src.compute_paths import compute_shortest_paths, write_paths_jsonl_gz
from src.generate_flows import generate_flows, write_flows_csv
from src.load_topology import export_topology, load_topology
from src.utils import ensure_out_dir, load_config, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic flow/path dataset from a topology.")
    parser.add_argument("-c", "--config", default="configs/run.yaml", help="Path to YAML run configuration.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    config_path = Path(args.config)
    config = load_config(config_path)

    seed = int(config.get("seed", 0))
    set_seed(seed)

    topology_path = Path(config["topology"])
    out_dir = Path(config.get("out_dir", "out") )
    ensure_out_dir(out_dir)

    topology_name = config.get("topology_name") or topology_path.stem
    directed = bool(config.get("directed", False))
    weight_attr = config.get("edge_weight_attr")

    logging.info("Loading topology %s (directed=%s)", topology_path, directed)
    graph = load_topology(topology_path, directed=directed)
    export_topology(graph, out_dir / "topology.json", topology_name)
    logging.info("Topology nodes=%d edges=%d", graph.number_of_nodes(), graph.number_of_edges())

    flow_count = int(config.get("flow_count", 1000))
    flow_model = config.get("flow_model", "gravity")
    demand_scale = float(config.get("demand_scale", 10.0))
    demand_sigma = float(config.get("demand_sigma", 0.8))
    allow_self = bool(config.get("allow_self_flows", False))

    logging.info("Generating %d flows with model=%s", flow_count, flow_model)
    flows = generate_flows(
        graph,
        count=flow_count,
        model=flow_model,
        demand_scale=demand_scale,
        demand_sigma=demand_sigma,
        allow_self_flows=allow_self,
    )
    write_flows_csv(flows, out_dir / "flows.csv")

    # Optional per-flow size in Kbps; flows whose path bottleneck is smaller
    # are rejected (no capacity on path → no flow assigned).
    flow_size_kb: Optional[float] = config.get("flow_size_kb")  # None = no filter
    if flow_size_kb is not None:
        logging.info(
            "Capacity admission filter enabled: flow_size=%.1f Kbps (%.8f Gbps)",
            float(flow_size_kb),
            float(flow_size_kb) * 8 / 1_000_000,
        )

    logging.info("Computing shortest paths for %d flows", len(flows))
    paths = compute_shortest_paths(
        graph, flows, weight_attr=weight_attr, flow_size_kb=flow_size_kb
    )
    write_paths_jsonl_gz(paths, out_dir / "paths.jsonl.gz")

    logging.info("Completed run. Outputs written to %s", out_dir)


if __name__ == "__main__":
    main()
