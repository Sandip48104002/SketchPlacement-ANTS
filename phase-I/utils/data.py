import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class FlowDataset:
    P: List[List[int]]
    sid_to_name: List[str]
    fid_to_name: List[str]
    capacities: Optional[np.ndarray]
    # Per-flow size in bytes. None when the JSONL has no size field at all.
    # Individual entries are 0.0 for flows that had no size in the record.
    flow_sizes: Optional[np.ndarray] = field(default=None)

    @property
    def n_flows(self) -> int:
        return len(self.P)

    @property
    def n_switches(self) -> int:
        return len(self.sid_to_name)

    def has_flow_sizes(self) -> bool:
        """True when at least one flow carries a non-zero size."""
        return self.flow_sizes is not None and bool(np.any(self.flow_sizes > 0))


def _open_any(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_paths(path: Path, capacity_path: Optional[Path] = None) -> FlowDataset:
    sid_to_name: List[str] = []
    name_to_sid: Dict[str, int] = {}
    fid_to_name: List[str] = []
    P: List[List[int]] = []
    raw_sizes: List[float] = []
    any_size_found = False

    with _open_any(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            flow_id_val = record.get("flow_id", record.get("id"))
            if flow_id_val is None:
                raise KeyError("Flow record missing 'flow_id' or 'id'")
            flow_id = str(flow_id_val)

            path_nodes: Sequence[str] = (
                record.get("path") or record.get("nodes") or record.get("switches")
            )
            if path_nodes is None:
                raise KeyError(f"Flow {flow_id} missing path/nodes")

            fid_to_name.append(flow_id)

            switch_ids: List[int] = []
            for node in path_nodes:
                node_name = str(node)
                sid = name_to_sid.get(node_name)
                if sid is None:
                    sid = len(sid_to_name)
                    name_to_sid[node_name] = sid
                    sid_to_name.append(node_name)
                switch_ids.append(sid)
            if not switch_ids:
                raise ValueError(f"Flow {flow_id} has empty path")
            P.append(switch_ids)

            # Prefer demand_bytes (size_dist model), fall back to demand (gravity model).
            size_val = record.get("demand_bytes") or record.get("demand")
            if size_val is not None:
                raw_sizes.append(float(size_val))
                any_size_found = True
            else:
                raw_sizes.append(0.0)

    flow_sizes = np.array(raw_sizes, dtype=float) if any_size_found else None
    capacities = (
        _load_capacities(capacity_path, name_to_sid, len(sid_to_name))
        if capacity_path
        else None
    )
    return FlowDataset(
        P=P,
        sid_to_name=sid_to_name,
        fid_to_name=fid_to_name,
        capacities=capacities,
        flow_sizes=flow_sizes,
    )


def _load_capacities(
    path: Path, name_to_sid: Dict[str, int], n_switches: int
) -> np.ndarray:
    with _open_any(path) as fh:
        cap_map = json.load(fh)
    caps = np.zeros(n_switches, dtype=float)
    for sw_name, cap_val in cap_map.items():
        sid = name_to_sid.get(sw_name)
        if sid is None:
            raise ValueError(f"Capacity entry for unknown switch {sw_name}")
        caps[sid] = float(cap_val)
    return caps


def coverage_sets(dataset: FlowDataset) -> List[Tuple[int, List[int]]]:
    """Return list of (flow_idx, switch_ids) for convenience."""
    return [(f_idx, switches) for f_idx, switches in enumerate(dataset.P)]