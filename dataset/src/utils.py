import json
import logging
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def setup_logging() -> None:
    """Configure a simple logger for the CLI tools."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def sanitize_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert edge/node attributes into JSON-serializable values."""
    sanitized: Dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized
