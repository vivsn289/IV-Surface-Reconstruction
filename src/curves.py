# project_charts.py
# Generates:
#  - mask_density_over_time.png
#  - training_curves_val.png        (val loss / val KL if available)
#  - masked_mae_val.png             (val MAE if available; else val MSE/RMSE proxy)
#
# Searches for metrics in artifacts/: training_metrics.jsonl -> training_metrics.csv -> *.log

import os
import re
import json
import glob
import math
import csv
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt

# -------- Paths (edit if needed) --------
ART_DIR = "/home/harshil/IV-Surface-Reconstruction/artifacts"
DATASET_PT = os.path.join(ART_DIR, "spy_iv_2019_2024.pt")

MASK_DENSITY_PNG = os.path.join(ART_DIR, "mask_density_over_time.png")
TRAIN_CURVES_PNG = os.path.join(ART_DIR, "training_curves_val.png")
MAE_CURVE_PNG = os.path.join(ART_DIR, "masked_mae_val.png")
MASK_DENSITY_CSV = os.path.join(ART_DIR, "mask_density_over_time.csv")


# -------- Utils --------
def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _f3(x):
    return f"{x:.3f}"


def _f4(x):
    return f"{x:.4f}"


# -------- Part 1: Mask Density Over Time --------
def mask_density_over_time(bundle_path: str) -> Tuple[List[str], np.ndarray]:
    b = torch.load(bundle_path, map_location="cpu")
    m = b["mask"].bool().numpy()  # [N,1,H,W]
    dates = list(b["dates"])  # list[str]
    N, _, H, W = m.shape

    # sort by date (string ISO format assumed)
    order = sorted([(d, i) for i, d in enumerate(dates)], key=lambda t: t[0])
    idx = [i for _, i in order]
    dates_sorted = [dates[i] for i in idx]
    m_sorted = m[idx]  # [N,1,H,W]

    # obs fraction per day
    obs_counts = m_sorted.sum(axis=(1, 2, 3)).astype(np.float64)  # [N]
    obs_frac = obs_counts / float(H * W)  # [N]
    return dates_sorted, obs_frac


def plot_mask_density(
    dates: List[str], obs_frac: np.ndarray, png_path: str, csv_path: str
):
    _ensure_dir(png_path)
    # Save CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "obs_fraction"])
        for d, v in zip(dates, obs_frac):
            w.writerow([d, f"{v:.6f}"])

    # Plot
    x = np.arange(len(dates))
    plt.figure(figsize=(11, 3.4))
    plt.plot(x, obs_frac)
    plt.title("Mask Density Over Time (Observed Fraction per Day)")
    plt.xlabel("Time (days, sorted)")
    plt.ylabel("Observed fraction")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    plt.close()
    print(f"[OK] Mask density: {png_path}")
    print(f"[OK] Mask density CSV: {csv_path}")
    print(
        f"    Mean={_f3(obs_frac.mean())} | Min={_f3(obs_frac.min())} | Max={_f3(obs_frac.max())}"
    )


# -------- Part 2: Metrics parsing (Val loss/KL/MAE/MSE) --------
def find_metrics_files(art_dir: str) -> Dict[str, Optional[str]]:
    # Priority order
    candidates = {
        "jsonl": os.path.join(art_dir, "training_metrics.jsonl"),
        "csv": os.path.join(art_dir, "training_metrics.csv"),
        "log": None,
    }
    if not os.path.exists(candidates["jsonl"]):
        candidates["jsonl"] = None
    if not os.path.exists(candidates["csv"]):
        candidates["csv"] = None

    # find a log file (train_*.log or *.log)
    if candidates["jsonl"] is None and candidates["csv"] is None:
        logs = sorted(glob.glob(os.path.join(art_dir, "*.log")))
        candidates["log"] = logs[0] if logs else None
    return candidates


def parse_jsonl_metrics(path: str) -> Dict[str, List[float]]:
    out = {"epoch": [], "val_loss": [], "val_kl": [], "val_mae": [], "val_mse": []}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            e = j.get("epoch")
            if e is None:
                continue
            out["epoch"].append(int(e))
            # populate if present
            for k_json, k_std in [
                ("val_loss", "val_loss"),
                ("val_kl", "val_kl"),
                ("val_mae", "val_mae"),
                ("val_mse", "val_mse"),
                ("val_rec", "val_mse"),  # alias used in earlier script
            ]:
                v = j.get(k_json)
                if v is not None:
                    out[k_std].append(float(v))
                else:
                    # keep same length (None) to align later
                    if k_std in ("val_loss", "val_kl", "val_mae", "val_mse"):
                        out[k_std].append(np.nan)
    return out


def parse_csv_metrics(path: str) -> Dict[str, List[float]]:
    out = {"epoch": [], "val_loss": [], "val_kl": [], "val_mae": [], "val_mse": []}
    with open(path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if "epoch" not in row:
                continue
            out["epoch"].append(int(row["epoch"]))

            def grab(*keys):
                for k in keys:
                    if k in row and row[k] not in ("", None):
                        try:
                            return float(row[k])
                        except:
                            pass
                return np.nan

            out["val_loss"].append(grab("val_loss"))
            out["val_kl"].append(grab("val_kl"))
            out["val_mae"].append(grab("val_mae"))
            out["val_mse"].append(grab("val_mse", "val_rec"))
    return out


def parse_text_log(path: str) -> Dict[str, List[float]]:
    # Parses lines like:
    # "Epoch 03/25 | lr=... | train_rec=0.12345 | val_rec=0.10101 | mode=..."
    out = {"epoch": [], "val_loss": [], "val_kl": [], "val_mae": [], "val_mse": []}
    pat = re.compile(
        r"Epoch\s+(\d+)[^\|]*\|\s*lr=.*?\|\s*train_rec=([0-9.]+)\s*\|\s*val_rec=([0-9.]+)"
    )
    # Note: earlier training printed only masked-MSE ("val_rec") and not val KL/loss.
    with open(path, "r") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            epoch = int(m.group(1))
            val_mse = float(m.group(3))
            out["epoch"].append(epoch)
            out["val_mse"].append(val_mse)
            out["val_loss"].append(np.nan)  # not available
            out["val_kl"].append(np.nan)  # not available
            out["val_mae"].append(np.nan)  # not available
    return out


def load_metrics_any(art_dir: str) -> Dict[str, List[float]]:
    found = find_metrics_files(art_dir)
    if found["jsonl"]:
        print(f"[INFO] Using metrics from {found['jsonl']}")
        return parse_jsonl_metrics(found["jsonl"])
    if found["csv"]:
        print(f"[INFO] Using metrics from {found['csv']}")
        return parse_csv_metrics(found["csv"])
    if found["log"]:
        print(f"[INFO] Parsing training log {found['log']}")
        return parse_text_log(found["log"])
    print("[WARN] No metrics file found (jsonl/csv/log). Curves may be partial.")
    return {"epoch": [], "val_loss": [], "val_kl": [], "val_mae": [], "val_mse": []}


# -------- Part 3: Plots --------
def plot_training_curves_val(metrics: Dict[str, List[float]], png_path: str):
    ep = np.array(metrics["epoch"], dtype=float)
    val_loss = np.array(metrics["val_loss"], dtype=float)
    val_kl = np.array(metrics["val_kl"], dtype=float)

    if ep.size == 0:
        print("[WARN] No epochs in metrics. Skipping training_curves_val.png")
        return

    have_loss = np.isfinite(val_loss).any()
    have_kl = np.isfinite(val_kl).any()

    if not have_loss and not have_kl:
        print("[WARN] No val loss/KL in metrics. Skipping training_curves_val.png")
        return

    plt.figure(figsize=(8.5, 4))
    if have_loss:
        plt.plot(ep, val_loss, label="Val Loss")
    if have_kl:
        plt.plot(ep, val_kl, label="Val KL")
    plt.title("Training Curves (Validation)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _ensure_dir(png_path)
    plt.savefig(png_path, dpi=120)
    plt.close()
    print(f"[OK] Training curves (val): {png_path}")


def plot_masked_mae_val(metrics: Dict[str, List[float]], png_path: str):
    ep = np.array(metrics["epoch"], dtype=float)
    val_mae = np.array(metrics["val_mae"], dtype=float)
    val_mse = np.array(metrics["val_mse"], dtype=float)

    if ep.size == 0:
        print("[WARN] No epochs in metrics. Skipping masked_mae_val.png")
        return

    have_mae = np.isfinite(val_mae).any()
    have_mse = np.isfinite(val_mse).any()

    plt.figure(figsize=(8.5, 4))
    title = "Masked-MAE vs Epoch [val]"
    if have_mae:
        plt.plot(ep, val_mae, label="Val MAE")
    elif have_mse:
        # If only MSE is available, plot MSE and RMSE as proxies with clear labels.
        plt.plot(ep, val_mse, label="Val MSE")
        rmse = np.sqrt(np.clip(val_mse, 0, None))
        plt.plot(ep, rmse, label="Val RMSE (√MSE)")
        title += "  (no MAE logged; showing MSE/RMSE)"
    else:
        print("[WARN] No val MAE/MSE in metrics. Skipping masked_mae_val.png")
        plt.close()
        return

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _ensure_dir(png_path)
    plt.savefig(png_path, dpi=120)
    plt.close()
    print(f"[OK] Masked error curve (val): {png_path}")


# -------- Main --------
if __name__ == "__main__":
    # 1) Mask density over time (from dataset)
    print("[INFO] Loading dataset for mask density...")
    ds_dates, ds_obs_frac = mask_density_over_time(DATASET_PT)
    plot_mask_density(ds_dates, ds_obs_frac, MASK_DENSITY_PNG, MASK_DENSITY_CSV)

    # 2) Training curves (from metrics if available)
    print("[INFO] Loading metrics for curves...")
    metrics = load_metrics_any(ART_DIR)
    plot_training_curves_val(metrics, TRAIN_CURVES_PNG)
    plot_masked_mae_val(metrics, MAE_CURVE_PNG)
