import yaml
from typing import Dict, Any


def load_config(path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
