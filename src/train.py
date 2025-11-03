# train_vae.py  (fixed KL + full reporting)
# Logs [Val] curves (loss/KL/MAE), prints Tables 1&2 (CSV), qualitative montage,
# evaluates Bicubic (recon & inpainting), and uses correct KL reduction.
# Optional: KL warm-up during the early epochs.

import os
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# ---- optional baseline dependency (with graceful fallback) ----
try:
    from scipy.interpolate import griddata

    SCIPY_OK = True
except Exception:
    SCIPY_OK = False
    griddata = None

# ------------------ config ------------------
DATASET_PT = "/home/harshil/IV-Surface-Reconstruction/artifacts/spy_iv_2019_2024.pt"
SAVE_DIR = "/home/harshil/IV-Surface-Reconstruction/artifacts"

LATENT_DIM = 48
EPOCHS = 40
WARMUP_RECON_EPOCHS = 5  # recon→inpaint curriculum switch
KL_WARMUP_EPOCHS = 5  # anneal β from 0→BETA over these epochs
BATCH_SIZE = 64
LR = 1e-3
BETA = 2.0
LAMBDA_TV = 5e-5
LAMBDA_ARB = 5e-3
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)


# ------------------ utils ------------------
def to_device(*t):
    return [x.to(DEVICE) for x in t]


def masked_mse(x, xhat, mask, eps=1e-8):
    num = (mask * (x - xhat) ** 2).sum()
    den = mask.sum().clamp_min(eps)
    return num / den


def masked_mae(x, xhat, mask, eps=1e-8):
    num = (mask * (x - xhat).abs()).sum()
    den = mask.sum().clamp_min(eps)
    return num / den


def tv_loss(xhat, eps=1e-6):
    dx = xhat[:, :, 1:, :] - xhat[:, :, :-1, :]
    dy = xhat[:, :, :, 1:] - xhat[:, :, :, :-1]
    dx_c = dx[:, :, :, :-1]
    dy_c = dy[:, :, :-1, :]
    tv = torch.sqrt(dx_c.pow(2) + dy_c.pow(2) + eps)
    return tv.mean()


def convexity_penalty_k(xhat):
    d2 = xhat[..., 2:] - 2 * xhat[..., 1:-1] + xhat[..., :-2]
    return torch.relu(-d2).mean()


def calendar_penalty_T(xhat, T_vec):
    w = (xhat**2) * T_vec.view(1, 1, -1, 1)
    dT = w[..., 1:, :] - w[..., :-1, :]
    return torch.relu(-dT).mean()


def kl_standard(mu, logvar):
    """
    KL(q||p) with q = N(mu, diag(exp(logvar))), p = N(0, I)
    Reduction: sum over latent dims, mean over batch. Returns scalar.
    """
    # kl_per_dim: [B, D]
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl_per_dim.sum(dim=1).mean()
    return kl


def save_panel(img_path, x, xhat, mask, title):
    x = x.squeeze(0).numpy()
    xhat = xhat.squeeze(0).numpy()
    m = mask.squeeze(0).numpy().astype(bool)
    err = np.zeros_like(x)
    err[m] = x[m] - xhat[m]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    im0 = axs[0].imshow(x, origin="lower", aspect="auto")
    axs[0].set_title("Target σ")
    im1 = axs[1].imshow(xhat, origin="lower", aspect="auto")
    axs[1].set_title("Recon σ̂")
    im2 = axs[2].imshow(err, origin="lower", aspect="auto")
    axs[2].set_title("Error (obs)")
    for ax in axs:
        ax.set_xlabel("k")
        ax.set_ylabel("T idx")
    fig.suptitle(title)
    fig.colorbar(im0, ax=axs[0], fraction=0.046)
    fig.colorbar(im1, ax=axs[1], fraction=0.046)
    fig.colorbar(im2, ax=axs[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(img_path, dpi=120)
    plt.close(fig)


def save_presentation_plot(montage_path, val_triplet, test_triplet):
    """
    Generates a 2x3 plot for presentation, focusing on clarity.
    Row 1: Reconstruction (validation set)
    Row 2: Inpainting (test set)
    """
    (xv, mv, xhatv) = val_triplet
    (xt, mkeep, xhat_t, loss_mask_t) = test_triplet

    # Move to CPU, squeeze, and convert to numpy
    xv = xv.squeeze().cpu().numpy()
    mv = mv.squeeze().cpu().numpy().astype(bool)
    xhatv = xhatv.squeeze().cpu().numpy()
    xt = xt.squeeze().cpu().numpy()
    mkeep = mkeep.squeeze().cpu().numpy().astype(bool)
    xhat_t = xhat_t.squeeze().cpu().numpy()

    # Get coordinates for scatter plot
    val_coords = np.argwhere(mv)
    test_coords = np.argwhere(mkeep)

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fig.suptitle("Qualitative VAE Performance", fontsize=16)

    # --- Row 1: Reconstruction ---
    vmin1, vmax1 = min(xv.min(), xhatv.min()), max(xv.max(), xhatv.max())

    # 1.1: Ground Truth
    im11 = axs[0, 0].imshow(
        xv, origin="lower", aspect="auto", vmin=vmin1, vmax=vmax1, cmap="viridis"
    )
    axs[0, 0].set_title("1.1: Ground Truth (Validation)")
    fig.colorbar(im11, ax=axs[0, 0])

    # 1.2: VAE Reconstruction
    im12 = axs[0, 1].imshow(
        xhatv, origin="lower", aspect="auto", vmin=vmin1, vmax=vmax1, cmap="viridis"
    )
    axs[0, 1].scatter(val_coords[:, 1], val_coords[:, 0], s=1, c="red", marker=".")
    axs[0, 1].set_title("1.2: VAE Reconstruction (Observed points in red)")
    fig.colorbar(im12, ax=axs[0, 1])

    # 1.3: Absolute Error
    err_v = np.abs(xv - xhatv)
    im13 = axs[0, 2].imshow(err_v, origin="lower", aspect="auto", cmap="magma")
    axs[0, 2].set_title("1.3: Absolute Error")
    fig.colorbar(im13, ax=axs[0, 2])

    # --- Row 2: Inpainting ---
    vmin2, vmax2 = min(xt.min(), xhat_t.min()), max(xt.max(), xhat_t.max())

    # 2.1: Ground Truth
    im21 = axs[1, 0].imshow(
        xt, origin="lower", aspect="auto", vmin=vmin2, vmax=vmax2, cmap="viridis"
    )
    axs[1, 0].set_title("2.1: Ground Truth (Test)")
    fig.colorbar(im21, ax=axs[1, 0])

    # 2.2: VAE Inpainting
    im22 = axs[1, 1].imshow(
        xhat_t, origin="lower", aspect="auto", vmin=vmin2, vmax=vmax2, cmap="viridis"
    )
    axs[1, 1].scatter(test_coords[:, 1], test_coords[:, 0], s=1, c="red", marker=".")
    axs[1, 1].set_title("2.2: VAE Inpainting (Input points in red)")
    fig.colorbar(im22, ax=axs[1, 1])

    # 2.3: Absolute Error
    err_t = np.abs(xt - xhat_t)
    im23 = axs[1, 2].imshow(err_t, origin="lower", aspect="auto", cmap="magma")
    axs[1, 2].set_title("2.3: Absolute Error")
    fig.colorbar(im23, ax=axs[1, 2])

    for ax in axs.flat:
        ax.set_xlabel("Moneyness (k)")
        ax.set_ylabel("Time to Expiry (T)")

    fig.savefig(montage_path, dpi=150)
    plt.close(fig)


# ------------------ data ------------------
class IVBundle(Dataset):
    def __init__(self, bundle, indices: List[int]):
        self.x = bundle["x"].float()
        self.mask = bundle["mask"].bool()
        self.dates = bundle["dates"]
        self.idx = indices

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        n = self.idx[i]
        return self.x[n], self.mask[n], n


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


# ------------------ model ------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),  # 64->32
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),  # 32->16
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),  # 16->8
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),  # 8->4
        )
        self.mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.net(x)
        h = h.view(x.size(0), -1)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 128, 4, 4)
        return self.deconv(h)


class BetaVAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.enc = Encoder(latent_dim)
        self.dec = Decoder(latent_dim)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_in):
        mu, logvar = self.enc(x_in)
        z = self.reparam(mu, logvar)
        xhat = self.dec(z)
        return xhat, mu, logvar


# ------------------ baseline: bicubic ------------------
def bicubic_grid_fit(
    x_np: np.ndarray, m_np: np.ndarray, T_grid: np.ndarray, k_grid: np.ndarray
) -> np.ndarray:
    """
    Fit a surface from observed cells only using scipy.griddata with method='cubic' (fallback to 'linear'/'nearest').
    x_np, m_np: [H,W]
    returns xhat_full: [H,W]
    """
    H, W = x_np.shape
    Ti, Kj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    obs = m_np.astype(bool)
    if obs.sum() < 5 or not SCIPY_OK:
        xhat = np.copy(x_np)
        xhat[~obs] = np.nan
        xhat = pd_fillna_2d(xhat)
        return xhat

    points = np.column_stack([T_grid[Ti[obs]], k_grid[Kj[obs]]])
    values = x_np[obs]
    gridT, gridK = np.meshgrid(T_grid, k_grid, indexing="ij")

    for method in ("cubic", "linear", "nearest"):
        try:
            xhat = griddata(points, values, (gridT, gridK), method=method)
            if np.isnan(xhat).all():
                continue
            if np.isnan(xhat).any():
                xhat_nn = griddata(points, values, (gridT, gridK), method="nearest")
                xhat = np.where(np.isnan(xhat), xhat_nn, xhat)
            return xhat
        except Exception:
            continue

    xhat = np.copy(x_np)
    xhat[~obs] = np.nan
    xhat = pd_fillna_2d(xhat)
    return xhat


def pd_fillna_2d(a: np.ndarray) -> np.ndarray:
    """Simple 2D nearest-fill using repeated 1D forward/backward fills (no pandas dependency)."""
    x = a.copy()
    # forward/backward along rows
    for _ in range(2):
        mask = np.isnan(x)
        idx = np.where(~mask, np.arange(x.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = x[np.arange(x.shape[0])[:, None], idx]
        x = np.where(mask, out, x)
        x = np.flip(x, axis=1)
    x = np.flip(x, axis=1)
    # forward/backward along cols
    x = x.T
    for _ in range(2):
        mask = np.isnan(x)
        idx = np.where(~mask, np.arange(x.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = x[np.arange(x.shape[0])[:, None], idx]
        x = np.where(mask, out, x)
        x = np.flip(x, axis=1)
    x = np.flip(x, axis=1).T
    return x


# ------------------ evaluation helpers ------------------
def eval_model_recon(model, loader, T_vec):
    model.eval()
    tot_mae = 0.0
    tot_mse = 0.0
    tot_kl = 0.0
    tot_loss = 0.0
    cnt = 0
    with torch.no_grad():
        for xb, mb, _ in loader:
            xb, mb = to_device(xb, mb)
            x_in2 = torch.cat([xb * mb.float(), mb.float()], dim=1)
            xhat, mu, logvar = model(x_in2)
            rec = masked_mse(xb, xhat, mb.float())
            mae = masked_mae(xb, xhat, mb.float())
            kl = kl_standard(mu, logvar)  # FIXED KL
            tv = tv_loss(xhat)
            arb = convexity_penalty_k(xhat) + calendar_penalty_T(xhat, T_vec)
            loss = rec + BETA * kl + LAMBDA_TV * tv + LAMBDA_ARB * arb
            B = xb.size(0)
            tot_mae += mae.item() * B
            tot_mse += rec.item() * B
            tot_kl += kl.item() * B
            tot_loss += loss.item() * B
            cnt += B
    return dict(
        mae=tot_mae / cnt, mse=tot_mse / cnt, kl=tot_kl / cnt, loss=tot_loss / cnt
    )


def eval_model_inpaint(model, loader, keep_frac=0.3):
    model.eval()
    tot_mae = 0.0
    tot_mse = 0.0
    cnt = 0
    with torch.no_grad():
        for xb, mb, _ in loader:
            xb, mb = to_device(xb, mb)
            rand = torch.rand_like(mb.float())
            keep = (mb & (rand < keep_frac)).float()
            loss_mask = (mb & (~keep.bool())).float()  # held-out observed
            x_in2 = torch.cat([xb * keep, keep], dim=1)
            xhat, _, _ = model(x_in2)
            mae = masked_mae(xb, xhat, loss_mask)
            mse = masked_mse(xb, xhat, loss_mask)
            B = xb.size(0)
            tot_mae += mae.item() * B
            tot_mse += mse.item() * B
            cnt += B
    return dict(mae=tot_mae / cnt, mse=tot_mse / cnt)


def eval_bicubic_recon(bundle, indices):
    if not SCIPY_OK:
        return dict(mae=np.nan, mse=np.nan)
    x = bundle["x"].numpy()
    m = bundle["mask"].numpy()
    T = bundle["T_grid"].numpy()
    K = bundle["k_grid"].numpy()
    tot_mae = 0.0
    tot_mse = 0.0
    cnt = 0
    for n in indices:
        xn = x[n, 0]
        mn = m[n, 0]
        obs = mn.astype(bool)
        if obs.sum() == 0:
            continue
        xhat = bicubic_grid_fit(xn, obs, T, K)
        mae = np.abs(xn[obs] - xhat[obs]).mean()
        mse = ((xn[obs] - xhat[obs]) ** 2).mean()
        tot_mae += mae
        tot_mse += mse
        cnt += 1
    return dict(mae=tot_mae / max(1, cnt), mse=tot_mse / max(1, cnt))


def eval_bicubic_inpaint(bundle, indices, keep_frac=0.3):
    if not SCIPY_OK:
        return dict(mae=np.nan, mse=np.nan)
    rng = np.random.default_rng(SEED)
    x = bundle["x"].numpy()
    m = bundle["mask"].numpy()
    T = bundle["T_grid"].numpy()
    K = bundle["k_grid"].numpy()
    tot_mae = 0.0
    tot_mse = 0.0
    cnt = 0
    for n in indices:
        xn = x[n, 0]
        mn = m[n, 0].astype(bool)
        if mn.sum() == 0:
            continue
        keep = np.zeros_like(mn, dtype=bool)
        keep[mn] = rng.random(mn.sum()) < keep_frac
        heldout = mn & (~keep)
        if heldout.sum() == 0:
            continue
        xfit = np.full_like(xn, np.nan)
        xfit[keep] = xn[keep]
        xhat = bicubic_grid_fit(xfit, keep.astype(bool), T, K)
        mae = np.abs(xn[heldout] - xhat[heldout]).mean()
        mse = ((xn[heldout] - xhat[heldout]) ** 2).mean()
        tot_mae += mae
        tot_mse += mse
        cnt += 1
    return dict(mae=tot_mae / max(1, cnt), mse=tot_mse / max(1, cnt))


def plot_val_curves(hist, save_dir):
    epochs = np.arange(1, len(hist["val_loss"]) + 1)
    # Loss & KL (twin axes)
    fig, ax1 = plt.subplots(figsize=(7.2, 4))
    l1 = ax1.plot(epochs, hist["val_loss"], label="Val Loss")[0]
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val Loss")
    ax2 = ax1.twinx()
    l2 = ax2.plot(epochs, hist["val_kl"], label="Val KL")[0]
    ax2.set_ylabel("Val KL")
    ax1.legend([l1, l2], ["Val Loss", "Val KL"], loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "val_loss_kl.png"), dpi=130)
    plt.close(fig)
    # Masked-MAE vs epoch
    fig = plt.figure(figsize=(7.2, 4))
    plt.plot(epochs, hist["val_mae"])
    plt.xlabel("Epoch")
    plt.ylabel("Val Masked-MAE")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_masked_mae.png"), dpi=130)
    plt.close(fig)


def print_table(title, rows: List[Dict], cols: List[str]):
    print("\n" + title)
    widths = [max(len(col), *(len(str(r[col])) for r in rows)) for col in cols]
    line = " | ".join(col.ljust(w) for col, w in zip(cols, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))


def save_csv(path, rows: List[Dict], cols: List[str]):
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c) for c in cols})


def kl_weight(epoch: int) -> float:
    """Linear KL warm-up 0→BETA over KL_WARMUP_EPOCHS, then fixed at BETA."""
    if KL_WARMUP_EPOCHS <= 0:
        return BETA
    return BETA * min(1.0, epoch / float(KL_WARMUP_EPOCHS))


# ------------------ training ------------------
class Trainer:
    def __init__(self):
        self.hist = {"val_loss": [], "val_kl": [], "val_mae": []}

    def train(self):
        bundle = torch.load(DATASET_PT, map_location="cpu")
        dates = bundle["dates"]
        T_grid = bundle["T_grid"].float().to(DEVICE)

        train_idx, val_idx, test_idx = build_splits(dates)
        ds_train = IVBundle(bundle, train_idx)
        ds_val = IVBundle(bundle, val_idx)

        train_loader = DataLoader(
            ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            ds_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
        )

        model = BetaVAE(LATENT_DIM).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        best_val_mse = float("inf")
        ckpt_path = os.path.join(SAVE_DIR, "vae_beta.pt")

        for epoch in range(1, EPOCHS + 1):
            model.train()
            inpainting_phase = epoch > WARMUP_RECON_EPOCHS
            epoch_rec = 0.0
            seen = 0

            for xb, mb, _ in train_loader:
                xb, mb = to_device(xb, mb)
                B = xb.size(0)
                if inpainting_phase:
                    keep_prob = np.random.uniform(0.2, 0.8)
                    rand = torch.rand_like(mb.float())
                    input_mask = (mb & (rand < keep_prob)).float()
                    loss_mask = (mb & (~(input_mask.bool()))).float()
                    if loss_mask.sum() < 10:
                        loss_mask = mb.float()
                else:
                    input_mask = mb.float()
                    loss_mask = mb.float()

                x_in2 = torch.cat([xb * input_mask, input_mask], dim=1)
                xhat, mu, logvar = model(x_in2)

                rec = masked_mse(xb, xhat, loss_mask)
                kl = kl_standard(mu, logvar)  # FIXED KL
                tv = tv_loss(xhat)
                arb = convexity_penalty_k(xhat) + calendar_penalty_T(xhat, T_grid)

                loss = rec + kl_weight(epoch) * kl + LAMBDA_TV * tv + LAMBDA_ARB * arb

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                epoch_rec += rec.item() * B
                seen += B

            sched.step()

            # ---- validation (compute loss/KL/MAE with full obs) ----
            val_stats = eval_model_recon(model, val_loader, T_grid)
            self.hist["val_loss"].append(val_stats["loss"])
            self.hist["val_kl"].append(val_stats["kl"])
            self.hist["val_mae"].append(val_stats["mae"])

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | lr={sched.get_last_lr()[0]:.2e} | "
                f"val_loss={val_stats['loss']:.8f} | val_KL={val_stats['kl']:.8f} | "
                f"val_MAE={val_stats['mae']:.8f} | mode={'INPAINT' if inpainting_phase else 'RECON'} "
                f"| kl_w={kl_weight(epoch):.3f}"
            )

            if val_stats["mse"] < best_val_mse:
                best_val_mse = val_stats["mse"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": dict(
                            LATENT_DIM=LATENT_DIM,
                            BETA=BETA,
                            LAMBDA_TV=LAMBDA_TV,
                            LAMBDA_ARB=LAMBDA_ARB,
                        ),
                    },
                    ckpt_path,
                )

            # Save a panel occasionally
            if epoch % 5 == 0:
                try:
                    xb, mb, _ = next(iter(val_loader))
                    xb, mb = xb.to(DEVICE), mb.to(DEVICE)
                    x_in2 = torch.cat([xb * mb.float(), mb.float()], dim=1)
                    with torch.no_grad():
                        xhat, _, _ = model(x_in2)
                    pth = os.path.join(SAVE_DIR, f"vae_panel_e{epoch}.png")
                    save_panel(
                        pth, xb[0].cpu(), xhat[0].cpu(), mb[0].cpu(), f"Epoch {epoch}"
                    )
                except Exception:
                    pass

        print(f"Saved best model to: {ckpt_path}")
        # Curves
        plot_val_curves(self.hist, SAVE_DIR)

        # ---- Tables & qualitative panels ----
        self.report_tables_and_panels(ckpt_path, bundle, val_idx, test_idx)
        return ckpt_path, bundle

    def report_tables_and_panels(self, ckpt_path, bundle, val_idx, test_idx):
        # Load best model
        model = BetaVAE(LATENT_DIM).to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state["model"])
        model.eval()

        # Dataloaders
        ds_val = IVBundle(bundle, val_idx)
        ds_test = IVBundle(bundle, test_idx)
        val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

        T_vec = bundle["T_grid"].float().to(DEVICE)

        # Model metrics
        val_recon = eval_model_recon(model, val_loader, T_vec)
        test_recon = eval_model_recon(model, test_loader, T_vec)
        val_inpaint = eval_model_inpaint(model, val_loader, keep_frac=0.3)

        # Bicubic baselines
        bic_val_recon = eval_bicubic_recon(bundle, val_idx)
        bic_val_inpaint = eval_bicubic_inpaint(bundle, val_idx, keep_frac=0.3)

        # ---- Table 1: Reconstruction (observed cells only) ----
        table1_cols = ["Model", "Split", "Masked-MAE", "MSE", "Notes"]
        table1_rows = [
            {
                "Model": "Bicubic (baseline)",
                "Split": "Val",
                "Masked-MAE": f"{bic_val_recon['mae']:.6f}",
                "MSE": f"{bic_val_recon['mse']:.6f}",
                "Notes": "interpolation" if SCIPY_OK else "fallback fill",
            },
            {
                "Model": "β-VAE (β=2)",
                "Split": "Val",
                "Masked-MAE": f"{val_recon['mae']:.6f}",
                "MSE": f"{val_recon['mse']:.6f}",
                "Notes": "warmup/best",
            },
            {
                "Model": "β-VAE (β=2)",
                "Split": "Test",
                "Masked-MAE": f"{test_recon['mae']:.6f}",
                "MSE": f"{test_recon['mse']:.6f}",
                "Notes": "best ckpt",
            },
        ]
        print_table(
            "Table 1. Reconstruction (observed cells only)", table1_rows, table1_cols
        )
        save_csv(
            os.path.join(SAVE_DIR, "table1_reconstruction.csv"),
            table1_rows,
            table1_cols,
        )

        # ---- Table 2: Inpainting (held-out observed cells, keep=0.3) ----
        table2_cols = ["Model", "Split", "Masked-MAE", "MSE", "Notes"]
        table2_rows = [
            {
                "Model": "Bicubic (baseline)",
                "Split": "Val",
                "Masked-MAE": f"{bic_val_inpaint['mae']:.6f}",
                "MSE": f"{bic_val_inpaint['mse']:.6f}",
                "Notes": "on held-out" if SCIPY_OK else "fallback fill",
            },
            {
                "Model": "β-VAE (β=2)",
                "Split": "Val",
                "Masked-MAE": f"{val_inpaint['mae']:.6f}",
                "MSE": f"{val_inpaint['mse']:.6f}",
                "Notes": "on held-out (keep=0.3)",
            },
        ]
        print_table(
            "Table 2. Inpainting (held-out observed cells)", table2_rows, table2_cols
        )
        save_csv(
            os.path.join(SAVE_DIR, "table2_inpainting.csv"), table2_rows, table2_cols
        )

        # ---- Qualitative montage ----
        try:
            # one val sample (recon)
            v_n = val_idx[len(val_idx) // 2]
            xv = bundle["x"][v_n : v_n + 1].to(DEVICE)
            mv = bundle["mask"][v_n : v_n + 1].to(DEVICE)
            x_in2 = torch.cat([xv * mv.float(), mv.float()], dim=1)
            with torch.no_grad():
                xhatv, _, _ = model(x_in2)

            # one test sample (inpaint with keep=0.3)
            t_n = test_idx[len(test_idx) // 2]
            xt = bundle["x"][t_n : t_n + 1].to(DEVICE)
            mt = bundle["mask"][t_n : t_n + 1].to(DEVICE)
            rand = torch.rand_like(mt.float())
            keep = (mt & (rand < 0.3)).float()
            loss_mask_t = (mt & (~keep.bool())).float()
            x_in2_t = torch.cat([xt * keep, keep], dim=1)
            with torch.no_grad():
                xhat_t, _, _ = model(x_in2_t)

            montage_path = os.path.join(SAVE_DIR, "qualitative_montage.png")
            save_presentation_plot(
                montage_path,
                (xv[0], mv[0], xhatv[0]),
                (xt[0], keep[0], xhat_t[0], loss_mask_t[0]),
            )
            print("Saved presentation plot:", montage_path)
        except Exception as e:
            print("Montage failed:", repr(e))


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()
