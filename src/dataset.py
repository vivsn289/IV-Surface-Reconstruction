# dataset_summary.py
# Summarizes a saved IV-surface dataset bundle (.pt) produced by your pipeline.

import torch
import numpy as np

# === Update this path if needed ===
DATASET_PT = "/home/harshil/IV-Surface-Reconstruction/artifacts/spy_iv_2019_2024.pt"


def build_time_splits(dates, ratios=(0.7, 0.15, 0.15)):
    """Return (train_idx, val_idx, test_idx) by sorting dates and splitting 70/15/15."""
    order = sorted([(d, i) for i, d in enumerate(dates)], key=lambda t: t[0])
    idxs = [i for _, i in order]
    N = len(idxs)
    n_train = int(N * ratios[0])
    n_val = int(N * ratios[1])
    train = idxs[:n_train]
    val = idxs[n_train : n_train + n_val]
    test = idxs[n_train + n_val :]
    return train, val, test


def summarize(dataset_path: str):
    bundle = torch.load(dataset_path, map_location="cpu")

    # Core tensors
    x = bundle["x"]  # [N,1,H,W] (possibly normalized)
    m = bundle["mask"].bool()  # [N,1,H,W]
    dates = bundle["dates"]
    meta = bundle.get("meta", {})
    norm_info = meta.get("normalization", {})
    H = x.shape[2]
    W = x.shape[3]
    N = x.shape[0]

    # Splits by date
    tr_idx, va_idx, te_idx = build_time_splits(dates)

    # Observation fractions
    # Per day: (# observed pixels) / (H*W)
    m_float = m.float()
    obs_counts = m_float.sum(dim=(1, 2, 3))  # [N]
    obs_frac = (obs_counts / float(H * W)).numpy()  # [N]
    mean_obs = float(obs_frac.mean())
    min_obs = float(obs_frac.min())
    max_obs = float(obs_frac.max())

    # IV range (observed)
    iv_min = None
    iv_max = None
    norm_name = str(norm_info.get("norm", "unknown"))

    if norm_name == "global_minmax":
        # Original units are available via vmin/vmax
        iv_min = float(norm_info.get("vmin", np.nan))
        iv_max = float(norm_info.get("vmax", np.nan))
    elif norm_name == "none":
        # Values in x are already in original units; restrict to observed cells
        observed_vals = x[m].numpy()
        if observed_vals.size > 0:
            iv_min = float(np.min(observed_vals))
            iv_max = float(np.max(observed_vals))
    elif norm_name == "per_cell_zscore":
        # Without per-cell unnormalization here, report normalized range on observed cells
        observed_vals = x[m].numpy()
        if observed_vals.size > 0:
            iv_min = float(np.min(observed_vals))
            iv_max = float(np.max(observed_vals))
        norm_name += " (normalized units)"
    else:
        # Unknown normalization; report range in stored units on observed cells
        observed_vals = x[m].numpy()
        if observed_vals.size > 0:
            iv_min = float(np.min(observed_vals))
            iv_max = float(np.max(observed_vals))

    # ---- Print in requested format ----
    def f3(z):
        return f"{z:.3f}"

    def f4(z):
        return f"{z:.4f}"

    print(f"Days total — {N}")
    print(f"Train / Val / Test — {len(tr_idx)} / {len(va_idx)} / {len(te_idx)}")
    print(f"Mean obs. frac. — {f3(mean_obs)}")
    print(f"Min/Max obs. frac. — {f3(min_obs)} / {f3(max_obs)}")

    if (
        iv_min is not None
        and iv_max is not None
        and np.isfinite(iv_min)
        and np.isfinite(iv_max)
    ):
        print(
            f"IV range (obs) [{f4(iv_min)}, {f4(iv_max)}]"
            + ("" if norm_name == "global_minmax" else f"  ({norm_name})")
        )
    else:
        print("IV range (obs) [N/A, N/A]")


if __name__ == "__main__":
    summarize(DATASET_PT)
