import json
import tempfile
from pathlib import Path
import numpy as np
import torch
import pytest

from iv_vae.src.dataio.parse_json import process_single_json
from iv_vae.src.dataio.make_grid import get_grid_bins


@pytest.fixture
def synthetic_json_file():
    data = [
        {"k": 0.1, "T_days": 30, "iv": 0.2},
        {"log_moneyness": -0.2, "ttm": 60, "implied_volatility": 0.25},
        {
            "strike": 105,
            "spot": 100,
            "time_to_maturity": 0.25,
            "iv": 0.3,
        },  # ttm in years
        {"k": 0.9, "T_days": 500, "iv": 0.4},  # Out of bounds
    ]
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(data, f)
        return f.name


def test_process_single_json(synthetic_json_file):
    k_bins, t_bins = get_grid_bins(-0.5, 0.5, 7, 365, 64)
    result = process_single_json(synthetic_json_file, k_bins, t_bins)

    assert result is not None
    surface, mask = result["surface"], result["mask"]

    assert surface.shape == (64, 64)
    assert mask.shape == (64, 64)
    assert mask.dtype == bool

    # Expect 3 valid points to be gridded
    assert np.sum(mask) == 3

    # Clean up the temp file
    Path(synthetic_json_file).unlink()
