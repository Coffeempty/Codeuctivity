"""YAML config loading utilities."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed config as a dictionary.
    """
    with open(Path(path)) as f:
        return yaml.safe_load(f)
