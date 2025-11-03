import numpy as np
import pandas as pd
from typing import Tuple


def get_grid_bins(
    k_min: float, k_max: float, t_min_days: int, t_max_days: int, grid_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates the bin edges for the log-moneyness and time-to-maturity grid."""
    k_bins = np.linspace(k_min, k_max, grid_size + 1)
    t_bins = np.linspace(t_min_days, t_max_days, grid_size + 1)
    return k_bins, t_bins


def grid_and_average_iv(
    df: pd.DataFrame, k_bins: np.ndarray, t_bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bins the raw IV data onto a grid and averages values within each cell.

    Args:
        df (pd.DataFrame): DataFrame with columns 'k', 'T_days', 'iv'.
        k_bins (np.ndarray): Bin edges for log-moneyness.
        t_bins (np.ndarray): Bin edges for time-to-maturity.

    Returns:
        A tuple containing:
        - surface (np.ndarray): The (64, 64) gridded IV surface.
        - mask_obs (np.ndarray): A (64, 64) boolean mask where True indicates an observed value.
    """
    grid_size = len(k_bins) - 1
    df["k_bin"] = pd.cut(df["k"], bins=k_bins, labels=False, include_lowest=True)
    df["t_bin"] = pd.cut(df["T_days"], bins=t_bins, labels=False, include_lowest=True)

    # Drop data outside the grid
    df.dropna(subset=["k_bin", "t_bin"], inplace=True)
    df["k_bin"] = df["k_bin"].astype(int)
    df["t_bin"] = df["t_bin"].astype(int)

    # Group by grid cell and average IV
    grouped = df.groupby(["t_bin", "k_bin"])["iv"].mean()

    surface = np.zeros((grid_size, grid_size), dtype=np.float32)
    mask_obs = np.zeros((grid_size, grid_size), dtype=bool)

    for (t_idx, k_idx), avg_iv in grouped.items():
        surface[t_idx, k_idx] = avg_iv
        mask_obs[t_idx, k_idx] = True

    return surface, mask_obs
