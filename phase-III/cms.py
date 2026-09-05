"""
Count-Min Sketch (CMS)
======================
Fixed parameters:
    width  w = 2^16 = 65,536  columns
    depth  d = 3              rows
    dtype     int32
    flow_id   numeric integer (the 'id' field from paths.jsonl.gz)

Each row uses a different MurmurHash3 seed so the three hash functions
are independent.  The update value is the packet count for the flow
(ceil(demand_bytes / MTU)), added in a single vectorised operation —
no per-packet loop.

Public API
----------
    cms = CountMinSketch()
    cms.update(flow_id: int, packet_count: int)
    cms.query(flow_id: int) -> int
    cms.reset()
"""

import math
from typing import Dict

import mmh3          # pip install mmh3
import numpy as np


# ---------------------------------------------------------------------------
# Fixed CMS parameters
# ---------------------------------------------------------------------------

WIDTH  : int = 1 << 16          # 2^16 = 65,536 columns
DEPTH  : int = 3                # 3 rows / hash functions
MTU    : int = 1_500            # bytes — standard Ethernet MTU
SEEDS  : tuple = (0, 1, 2)      # one seed per row  (must be len == DEPTH)


# ---------------------------------------------------------------------------
# CountMinSketch
# ---------------------------------------------------------------------------

class CountMinSketch:
    """One CMS instance — intended to live on a single selected switch."""

    def __init__(self) -> None:
        # shape: (DEPTH, WIDTH), all zeros, int32
        self.table: np.ndarray = np.zeros((DEPTH, WIDTH), dtype=np.int32)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def update(self, flow_id: int, packet_count: int) -> None:
        """Add packet_count to all d cells for flow_id.

        Equivalent to calling a single-packet update packet_count times,
        but done in one numpy assignment — safe even for 1 GB flows.

        int32 overflow: np.int32 wraps silently on overflow.  With w=65536
        and only ~500 k flows the per-cell collision load stays small enough
        that overflow is not a practical concern.
        """
        if packet_count <= 0:
            return
        inc = np.int32(min(packet_count, np.iinfo(np.int32).max))
        for row, seed in enumerate(SEEDS):
            col = mmh3.hash(str(flow_id), seed, signed=False) % WIDTH
            # np.int32 arithmetic wraps on overflow (C-style) — acceptable here.
            self.table[row, col] = np.int32(
                int(self.table[row, col]) + int(inc)
            )

    def query(self, flow_id: int) -> int:
        """Return the minimum counter across all rows for flow_id.

        This is the CMS point-query estimate.  It always >= true count
        (overestimation only, never underestimation).
        """
        return int(min(
            self.table[row, mmh3.hash(str(flow_id), seed, signed=False) % WIDTH]
            for row, seed in enumerate(SEEDS)
        ))

    def reset(self) -> None:
        """Zero all counters (reuse the same sketch for a new window)."""
        self.table[:] = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def memory_bytes(self) -> int:
        return self.table.nbytes          # DEPTH * WIDTH * 4

    def __repr__(self) -> str:
        return (
            f"CountMinSketch(width={WIDTH}, depth={DEPTH}, "
            f"memory={self.memory_bytes() / 1024:.1f} KB)"
        )


# ---------------------------------------------------------------------------
# Packet-count helper
# ---------------------------------------------------------------------------

def demand_to_packets(demand_bytes: float, mtu: int = MTU) -> int:
    """Convert a flow's byte demand to a packet count using ceiling division.

    A 1 GB flow → ceil(1_000_000_000 / 1500) = 666_667 packets.
    A 180 B flow → ceil(180 / 1500) = 1 packet.
    """
    if demand_bytes <= 0:
        return 1          # every flow has at least one packet
    return math.ceil(demand_bytes / mtu)