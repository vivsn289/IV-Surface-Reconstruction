import argparse
import json
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, Any, List

from iv_vae.src.dataio.make_grid import get_grid_bins, grid_and_average_iv


def robust_schema_adapter(record: Dict[str, Any]) -> Dict[str, float]:
    """
    Adapts a raw data record to a unified schema.
    Handles varying field names for key financial metrics.
    """
    # Log-moneyness 'k'
    if "k" in record:
        k = record["k"]
    elif "log_moneyness" in record:
        k = record["log_moneyness"]
    elif "strike" in record and "spot" in record and record["spot"] > 0:
        k = np.log(record["strike"] / record["spot"])
    else:
        return None

    # Time-to-maturity 'T_days'
    if "T_days" in record:
        T_days = record["T_days"]
    elif "ttm" in record:
        T_days = record["ttm"]
    elif "time_to_maturity" in record:  # Assuming in years
        T_days = record["time_to_maturity"] * 365.0
    else:
        return None

    # Implied volatility 'iv'
    if "iv" in record:
        iv = record["iv"]
    elif "implied_volatility" in record:
        iv = record["implied_volatility"]
    else:
        return None

    if any(val is None or not np.isfinite(val) for val in [k, T_days, iv]):
        return None

    return {"k": k, "T_days": T_days, "iv": iv}


def process_single_json(
    file_path: str, k_bins: np.ndarray, t_bins: np.ndarray
) -> Dict[str, np.ndarray]:
    """Processes a single JSON file into a gridded IV surface and mask."""
    with open(file_path, "r") as f:
        data = json.load(f)

    adapted_records = [robust_schema_adapter(rec) for rec in data]
    adapted_records = [rec for rec in adapted_records if rec is not None]

    if not adapted_records:
        return None

    df = pd.DataFrame(adapted_records)
    surface, mask = grid_and_average_iv(df, k_bins, t_bins)
    return {"surface": surface, "mask": mask}


def main():
    parser = argparse.ArgumentParser(
        description="Parse and cache IV surfaces from JSON files."
    )
    parser.add_argument(
        "--input_glob",
        type=str,
        required=True,
        help="Glob pattern for input JSON files.",
    )
    parser.add_argument("--kmin", type=float, required=True, help="Min log-moneyness.")
    parser.add_argument("--kmax", type=float, required=True, help="Max log-moneyness.")
    parser.add_argument(
        "--tmin_days", type=int, required=True, help="Min time-to-maturity in days."
    )
    parser.add_argument(
        "--tmax_days", type=int, required=True, help="Max time-to-maturity in days."
    )
    parser.add_argument(
        "--grid", type=int, required=True, help="Grid size (e.g., 64 for 64x64)."
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Output path for the cached .pt file."
    )
    args = parser.parse_args()

    k_bins, t_bins = get_grid_bins(
        args.kmin, args.kmax, args.tmin_days, args.tmax_days, args.grid
    )

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise ValueError(f"No files found for glob pattern: {args.input_glob}")

    all_surfaces = []
    all_masks = []

    for file in tqdm(files, desc="Processing JSON files"):
        result = process_single_json(file, k_bins, t_bins)
        if result:
            all_surfaces.append(result["surface"])
            all_masks.append(result["mask"])

    surfaces_tensor = torch.from_numpy(
        np.array(all_surfaces, dtype=np.float32)
    ).unsqueeze(1)
    masks_tensor = torch.from_numpy(np.array(all_masks, dtype=bool)).unsqueeze(1)

    # Compute normalization stats on the training set only (will be done in dataset creation)
    # For now, we save the raw data and compute stats later.

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "surfaces": surfaces_tensor,
            "masks": masks_tensor,
            "files": [Path(f).name for f in files],
        },
        output_path,
    )

    print(f"Saved cached data to {output_path}")
    print(f"Dataset shape: {surfaces_tensor.shape}")


if __name__ == "__main__":
    main()
