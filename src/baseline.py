# baselines_task_B.py
# Baseline methods for IV surface reconstruction & inpainting:
#  - Bicubic (scattered) interpolation
#  - PCA/SVD low-rank reconstruction (with partial-observation solve)
#  - Soft-Impute (nuclear-norm matrix completion)
#
# Outputs:
#  - CSV metrics in SAVE_DIR/baselines_metrics.csv
#  - Example panels (PNG) per baseline in SAVE_DIR/
#  - Summary plots (PNG) in SAVE_DIR/
#
# Assumes dataset bundle saved at DATASET_PT with keys:
#  x [N,1,H,W]   (min-max in [0,1] over observed pixels),
#  mask [N,1,H,W], dates (strings), T_grid [H], k_grid [W].

import os
import numpy as np
import torch
import pandas as pd
from typing import List
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from numpy.linalg import svd

# Optional SSIM (fallback to None if skimage not installed)
try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    ssim = None

# -------------------- CONFIG --------------------
DATASET_PT = "/home/harshil/IV-Surface-Reconstruction/artifacts/spy_iv_2019_2024.pt"
SAVE_DIR = "/home/harshil/IV-Surface-Reconstruction/artifacts"
os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 1337
np.random.seed(SEED)

# Inpainting evaluation keep ratio (observed kept; rest becomes hidden target)
INPAINT_KEEP = 0.30

# PCA ranks to evaluate
PCA_RANKS = [8, 16, 32, 64]

# Soft-Impute params
SOFT_IMPUTE_LAMBDA = 0.5
SOFT_IMPUTE_MAX_ITERS = 200
SOFT_IMPUTE_TOL = 1e-4

# Number of example panels per baseline
NUM_PANELS = 3


# -------------------- UTILS --------------------
def masked_mse(x, xhat, mask, eps=1e-8):
    num = ((x - xhat) ** 2 * mask).sum()
    den = mask.sum() + eps
    return float(num / den)


def masked_mae(x, xhat, mask, eps=1e-8):
    num = (np.abs(x - xhat) * mask).sum()
    den = mask.sum() + eps
    return float(num / den)


def try_ssim(x, xhat):
    if ssim is None:
        return np.nan
    try:
        return float(ssim(x, xhat, data_range=1.0))
    except Exception:
        return np.nan


def count_no_arb_violations(xhat, T_vec):
    """
    xhat: [H,W] normalized sigma in [0,1]
    T_vec: [H] maturities (years)
    Returns: (convexity_violations, calendar_violations)
    """
    H, W = xhat.shape
    # Convexity along k (second finite difference >= 0 for vol)
    if W >= 3:
        d2 = xhat[:, 2:] - 2 * xhat[:, 1:-1] + xhat[:, :-2]
        convex_viol = (d2 < 0).sum()
    else:
        convex_viol = 0

    # Calendar monotonicity for total variance w = sigma^2 * T
    w = (xhat**2) * T_vec[:, None]
    if H >= 2:
        dT = w[1:, :] - w[:-1, :]
        cal_viol = (dT < 0).sum()
    else:
        cal_viol = 0
    return int(convex_viol), int(cal_viol)


def save_panel(path, target, recon, obs_mask, title):
    err = np.zeros_like(target)
    eval_mask = obs_mask.astype(bool)
    err[eval_mask] = target[eval_mask] - recon[eval_mask]

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
    im0 = axs[0].imshow(target, origin="lower", aspect="auto")
    axs[0].set_title("Target σ")
    im1 = axs[1].imshow(recon, origin="lower", aspect="auto")
    axs[1].set_title("Recon σ̂")
    im2 = axs[2].imshow(err, origin="lower", aspect="auto")
    axs[2].set_title("Error (eval cells)")
    for ax in axs:
        ax.set_xlabel("k index")
        ax.set_ylabel("T index")
    fig.suptitle(title)
    fig.colorbar(im0, ax=axs[0], fraction=0.046)
    fig.colorbar(im1, ax=axs[1], fraction=0.046)
    fig.colorbar(im2, ax=axs[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_splits(dates: List[str], ratios=(0.7, 0.15, 0.15)):
    order = sorted([(d, i) for i, d in enumerate(dates)], key=lambda t: t[0])
    idxs = [i for _, i in order]
    N = len(idxs)
    n_train = int(N * ratios[0])
    n_val = int(N * ratios[1])
    train = idxs[:n_train]
    val = idxs[n_train : n_train + n_val]
    test = idxs[n_train + n_val :]
    return train, val, test


# -------------------- BASELINE: Bicubic (scattered) --------------------
def bicubic_inpaint(surface, mask, k_grid, T_grid):
    """
    Scattered interpolation over observed cells using griddata:
      - Try 'cubic', fallback to 'linear', then 'nearest' for any remaining nans.
    surface: [H,W]
    mask:    [H,W] bool (observed)
    Returns reconstructed [H,W]
    """
    H, W = surface.shape
    KK, TT = np.meshgrid(k_grid, T_grid)  # [H,W] each
    pts = np.stack([KK[mask], TT[mask]], axis=1)  # [n,2]
    vals = surface[mask]

    grid_pts = np.stack([KK.ravel(), TT.ravel()], axis=1)

    if pts.shape[0] < 4:
        return np.zeros_like(surface)

    recon = griddata(pts, vals, grid_pts, method="cubic")
    if np.isnan(recon).any():
        recon2 = griddata(pts, vals, grid_pts, method="linear")
        sel = np.isnan(recon)
        recon[sel] = recon2[sel]
    if np.isnan(recon).any():
        recon2 = griddata(pts, vals, grid_pts, method="nearest")
        sel = np.isnan(recon)
        recon[sel] = recon2[sel]
    return recon.reshape(H, W)


# -------------------- BASELINE: PCA --------------------
class PCABaseline:
    """
    PCA/SVD baseline trained on train surfaces with pixel-wise mean imputation.
    Provides:
      - reconstruct_full(x) : denoise recon with all observed mask available
      - inpaint_partial(x, keep_mask) : solve latent coeffs from partial observed pixels
    """

    def __init__(self, train_X: np.ndarray, train_M: np.ndarray, ranks: List[int]):
        """
        train_X: [N,1,H,W]
        train_M: [N,1,H,W] (bool or {0,1})
        """
        N, _, H, W = train_X.shape
        P = H * W

        X0 = train_X[:, 0].astype(np.float64)  # [N,H,W]
        M0 = train_M[:, 0].astype(bool)  # [N,H,W]

        # Pixel-wise mean over observed entries ONLY
        obs_sum = (X0 * M0).sum(axis=0)  # [H,W]
        obs_cnt = M0.sum(axis=0).clip(1)  # [H,W]
        self.mu_img = obs_sum / obs_cnt  # [H,W]
        self.H, self.W, self.P = H, W, P

        # Impute missing entries with mu and center
        X_imputed = np.where(~M0, self.mu_img[None, ...], X0)  # [N,H,W]
        Xc = (X_imputed - self.mu_img[None, ...]).reshape(N, P)  # [N,P]

        U, S, Vt = svd(Xc, full_matrices=False)
        self.U, self.S, self.Vt = U, S, Vt
        self.ranks = ranks

    def reconstruct_full(
        self, x_img: np.ndarray, m_img: np.ndarray, r: int
    ) -> np.ndarray:
        x_flat = x_img.reshape(1, -1)  # [1,P]
        mu = self.mu_img.reshape(1, -1)  # [1,P]
        V_r = self.Vt[:r, :]  # [r,P]
        a = (x_flat - mu) @ V_r.T  # [1,r]
        xhat = mu + a @ V_r  # [1,P]
        return xhat.reshape(self.H, self.W)

    def inpaint_partial(
        self, x_img: np.ndarray, keep_mask: np.ndarray, r: int
    ) -> np.ndarray:
        mu = self.mu_img
        V_r = self.Vt[:r, :]  # [r,P]
        obs_idx = np.flatnonzero(keep_mask.reshape(-1))
        if obs_idx.size < r:
            return self.reconstruct_full(x_img, keep_mask, r)

        y = (x_img - mu).reshape(-1)[obs_idx][:, None]  # [n_obs,1]
        W = V_r[:, obs_idx].T  # [n_obs,r]
        A = W.T @ W + 1e-6 * np.eye(r)
        b = W.T @ y
        a = np.linalg.solve(A, b).reshape(1, -1)  # [1,r]
        xhat = mu.reshape(1, -1) + a @ V_r  # [1,P]
        return xhat.reshape(self.H, self.W)


# -------------------- BASELINE: Soft-Impute --------------------
def soft_impute_single(
    X_missing: np.ndarray,
    M_obs: np.ndarray,
    lam=SOFT_IMPUTE_LAMBDA,
    max_iters=SOFT_IMPUTE_MAX_ITERS,
    tol=SOFT_IMPUTE_TOL,
):
    """
    Soft-Impute (Mazumder et al.) matrix completion for a single 2D surface.
    X_missing: [H,W] values on observed cells, 0 elsewhere
    M_obs:     [H,W] boolean mask of observed
    Returns completed [H,W].
    """
    X = X_missing.astype(np.float64, copy=True)
    Y = X.copy()  # current estimate
    prev = None

    for it in range(max_iters):
        # Reconstruct: keep ground truth on observed, current estimate on missing
        X_obs = M_obs * X + (~M_obs) * Y
        U, s, Vt = svd(X_obs, full_matrices=False)
        s_shrunk = np.maximum(s - lam, 0.0)
        Y_new = (U * s_shrunk) @ Vt

        if prev is not None:
            denom = np.linalg.norm(prev) + 1e-12
            diff = np.linalg.norm(Y_new - prev) / denom
            if diff < tol:
                Y = Y_new
                break
        prev = Y_new
        Y = Y_new

    return Y


# -------------------- EVAL + PLOTS --------------------
def main():
    bundle = torch.load(DATASET_PT, map_location="cpu")
    X = bundle["x"].numpy()  # [N,1,H,W] in [0,1]
    M = bundle["mask"].numpy().astype(bool)
    dates = bundle["dates"]
    T_grid = bundle["T_grid"].numpy()
    k_grid = bundle["k_grid"].numpy()
    N, _, H, W = X.shape

    train_idx, val_idx, test_idx = build_splits(dates)
    print(
        f"Dataset: N={N}, train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )

    # Prepare PCA baseline (fit on train only)
    pca = PCABaseline(train_X=X[train_idx], train_M=M[train_idx], ranks=PCA_RANKS)

    rows = []  # for CSV

    # Choose a few evenly spaced test examples for panels
    panel_ids = [
        test_idx[i]
        for i in np.linspace(
            0, max(0, len(test_idx) - 1), num=min(NUM_PANELS, len(test_idx)), dtype=int
        )
    ]

    def eval_and_log(name, recon_fn, mode):
        """
        recon_fn(i, mode) should return reconstructed [H,W] for sample i.
        mode: "recon" or "inpaint"
        """
        mse_all, mae_all, ssim_all, cviol_all, tviol_all = [], [], [], [], []
        for i in test_idx:
            target = X[i, 0]
            obs = M[i, 0]
            # For metrics:
            if mode == "recon":
                eval_mask = obs.astype(np.float32)
            else:
                # hide a fraction of observed cells
                rng = np.random.RandomState(SEED + i)
                keep = (rng.rand(H, W) < INPAINT_KEEP) & obs
                # we evaluate on the hidden observed cells
                eval_mask = (obs & (~keep)).astype(np.float32)

            recon = recon_fn(i, mode)

            mse_all.append(masked_mse(target, recon, eval_mask))
            mae_all.append(masked_mae(target, recon, eval_mask))
            ssim_all.append(try_ssim(target, recon))
            cv, tv = count_no_arb_violations(recon, T_grid)
            cviol_all.append(cv)
            tviol_all.append(tv)

        row = {
            "baseline": name,
            "mode": mode,
            "inpaint_keep": INPAINT_KEEP if mode == "inpaint" else 1.0,
            "mse_mean": float(np.mean(mse_all)),
            "mae_mean": float(np.mean(mae_all)),
            "ssim_mean": float(np.nanmean(ssim_all)),
            "convex_viol_mean": float(np.mean(cviol_all)),
            "calendar_viol_mean": float(np.mean(tviol_all)),
            "n_test": len(test_idx),
        }
        rows.append(row)
        print(
            f"{name} [{mode}] -> MSE {row['mse_mean']:.5f} | MAE {row['mae_mean']:.5f} | SSIM {row['ssim_mean']:.4f}"
        )

        # Save panels
        for j, idx in enumerate(panel_ids):
            target = X[idx, 0]
            obs = M[idx, 0]
            if mode == "recon":
                keep = obs
                eval_mask = obs.astype(np.float32)
            else:
                rng = np.random.RandomState(SEED + idx)
                keep = (rng.rand(H, W) < INPAINT_KEEP) & obs
                eval_mask = (obs & (~keep)).astype(np.float32)
            recon = recon_fn(idx, mode, keep_override=keep)
            pth = os.path.join(SAVE_DIR, f"baseline_{name}_{mode}_{j}.png")
            save_panel(pth, target, recon, eval_mask, f"{name} [{mode}] sample#{j}")

    # ---- Bicubic baseline ----
    def bicubic_recon_fn(i, mode, keep_override=None):
        img = X[i, 0]
        obs = M[i, 0]
        if mode == "recon":
            use_mask = obs
        else:
            if keep_override is None:
                rng = np.random.RandomState(SEED + i)
                use_mask = (rng.rand(H, W) < INPAINT_KEEP) & obs
            else:
                use_mask = keep_override
        return bicubic_inpaint(
            img,
            use_mask,
            k_grid=bundle["k_grid"].numpy(),
            T_grid=bundle["T_grid"].numpy(),
        )

    eval_and_log("bicubic", bicubic_recon_fn, "recon")
    eval_and_log("bicubic", bicubic_recon_fn, "inpaint")

    # ---- PCA baseline ----
    def pca_recon_fn_for_rank(r):
        def fn(i, mode, keep_override=None):
            img = X[i, 0]
            obs = M[i, 0]
            if mode == "recon":
                return pca.reconstruct_full(img, obs, r=r)
            else:
                if keep_override is None:
                    rng = np.random.RandomState(SEED + i)
                    keep = (rng.rand(H, W) < INPAINT_KEEP) & obs
                else:
                    keep = keep_override
                return pca.inpaint_partial(img, keep, r=r)

        return fn

    for r in PCA_RANKS:
        eval_and_log(f"pca_r{r}", pca_recon_fn_for_rank(r), "recon")
        eval_and_log(f"pca_r{r}", pca_recon_fn_for_rank(r), "inpaint")

    # ---- Soft-Impute baseline ----
    def soft_impute_recon_fn(i, mode, keep_override=None):
        img = X[i, 0]
        obs = M[i, 0]
        if mode == "recon":
            use_mask = obs
        else:
            if keep_override is None:
                rng = np.random.RandomState(SEED + i)
                use_mask = (rng.rand(H, W) < INPAINT_KEEP) & obs
            else:
                use_mask = keep_override
        Xobs = np.zeros_like(img)
        Xobs[use_mask] = img[use_mask]
        recon = soft_impute_single(
            Xobs,
            use_mask,
            lam=SOFT_IMPUTE_LAMBDA,
            max_iters=SOFT_IMPUTE_MAX_ITERS,
            tol=SOFT_IMPUTE_TOL,
        )
        recon = np.clip(recon, 0.0, 1.0)
        return recon

    eval_and_log("soft_impute", soft_impute_recon_fn, "recon")
    eval_and_log("soft_impute", soft_impute_recon_fn, "inpaint")

    # Save CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(SAVE_DIR, "baselines_metrics.csv")
    df.to_csv(csv_path, index=False)
    print("Saved baseline metrics to:", csv_path)

    # -------------------- Diagnostics / Plots --------------------
    # Bar charts per metric across baselines (mode in label)
    labels = df.apply(lambda r: f"{r['baseline']} ({r['mode']})", axis=1)
    for metric in ["mse_mean", "mae_mean", "ssim_mean"]:
        fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(df) + 2), 4))
        ax.bar(range(len(df)), df[metric].values)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(metric.replace("_", " ").upper())
        ax.set_ylabel(metric)
        fig.tight_layout()
        fig.savefig(os.path.join(SAVE_DIR, f"{metric}_bars.png"), dpi=130)
        plt.close(fig)

    # PCA rank curves (recon vs inpaint)
    pca_df = df[df["baseline"].str.startswith("pca_r")].copy()
    if not pca_df.empty:
        pca_df["rank"] = pca_df["baseline"].str.extract(r"pca_r(\d+)").astype(int)
        for metric in ["mse_mean", "mae_mean", "ssim_mean"]:
            fig, ax = plt.subplots(figsize=(6.8, 4.2))
            for mode in ["recon", "inpaint"]:
                sub = pca_df[pca_df["mode"] == mode].sort_values("rank")
                if not sub.empty:
                    ax.plot(sub["rank"], sub[metric], marker="o", label=mode)
            ax.set_xlabel("PCA Rank")
            ax.set_ylabel(metric)
            ax.set_title(f"PCA {metric} vs rank")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(SAVE_DIR, f"pca_{metric}_vs_rank.png"), dpi=130)
            plt.close(fig)

    # Violation bars
    for metric in ["convex_viol_mean", "calendar_viol_mean"]:
        fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(df) + 2), 4))
        ax.bar(range(len(df)), df[metric].values)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(metric.replace("_", " ").upper())
        ax.set_ylabel(metric)
        fig.tight_layout()
        fig.savefig(os.path.join(SAVE_DIR, f"{metric}_bars.png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
