# evaluate_cvae.py
# Evaluates CVAE checkpoints on your IV-surface dataset:
# - Recon MAE (observed cells)
# - Inpaint MAE (on hidden observed cells), conditioning on the *visible* subset
# - Saves a few fixed-scale panels for visual sanity checks

import os
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ------------------ CONFIG ------------------
DATASET_PT = "/home/harshil/IV-Surface-Reconstruction/artifacts/spy_iv_2019_2024.pt"
CKPTS = [
    "/home/harshil/IV-Surface-Reconstruction/artifacts/cvae.pt",  # best
    "/home/harshil/IV-Surface-Reconstruction/artifacts/cvae_last.pt",  # last (optional)
]
SAVE_DIR = "/home/harshil/IV-Surface-Reconstruction/artifacts/eval_cvae"

BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPAINT_KEEP = 0.30  # visible fraction during inpainting eval
INPAINT_REPEATS = 3  # Monte-Carlo repeats since masks are random
PANEL_SAMPLES = 3  # how many panels per ckpt/mode

os.makedirs(SAVE_DIR, exist_ok=True)
torch.set_float32_matmul_precision("high")


# ------------------ UTILS ------------------
def masked_mae(x, xhat, mask, eps=1e-8):
    num = (mask * (x - xhat).abs()).sum()
    den = mask.sum().clamp_min(eps)
    return num / den


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


def save_panel(img_path, x, xhat, mask, title, vmin=0.0, vmax=1.0):
    x = x.squeeze(0).cpu().numpy()
    xhat = xhat.squeeze(0).cpu().numpy()
    m = mask.squeeze(0).cpu().numpy().astype(bool)
    err = np.zeros_like(x)
    err[m] = x[m] - xhat[m]

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    im0 = axs[0].imshow(x, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    axs[0].set_title("Target σ")
    im1 = axs[1].imshow(xhat, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    axs[1].set_title("Recon σ̂")
    im2 = axs[2].imshow(err, origin="lower", aspect="auto")
    axs[2].set_title("Error (eval cells)")
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


# ------------------ DATA ------------------
class IVBundle(Dataset):
    def __init__(self, bundle, indices: List[int]):
        self.x = bundle["x"].float()
        self.m = bundle["mask"].bool()
        self.idx = indices

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        n = self.idx[i]
        return self.x[n], self.m[n], n


# ------------------ CONDITION FEATURES ------------------
def compute_conditioning(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """
    x: [B,1,H,W], m: [B,1,H,W] (bool/float)
    returns c: [B,3] = [mean_obs, slope_T, skew_k], roughly standardized via tanh
    """
    mf = m.float()
    eps = 1e-8
    B, _, H, W = x.shape

    # mean observed
    mean_obs = (x * mf).sum((1, 2, 3)) / mf.sum((1, 2, 3)).clamp_min(eps)

    # slope along T (last third - first third)
    t_third = max(1, H // 3)
    top = slice(0, t_third)
    bot = slice(H - t_third, H)
    top_mean = (x[:, :, top, :] * mf[:, :, top, :]).sum((1, 2, 3)) / mf[
        :, :, top, :
    ].sum((1, 2, 3)).clamp_min(eps)
    bot_mean = (x[:, :, bot, :] * mf[:, :, bot, :]).sum((1, 2, 3)) / mf[
        :, :, bot, :
    ].sum((1, 2, 3)).clamp_min(eps)
    slope_T = bot_mean - top_mean

    # skew along k (right - left)
    mid = W // 2
    left_mean = (x[:, :, :, :mid] * mf[:, :, :, :mid]).sum((1, 2, 3)) / mf[
        :, :, :, :mid
    ].sum((1, 2, 3)).clamp_min(eps)
    right_mean = (x[:, :, :, mid:] * mf[:, :, :, mid:]).sum((1, 2, 3)) / mf[
        :, :, :, mid:
    ].sum((1, 2, 3)).clamp_min(eps)
    skew_k = right_mean - left_mean

    c = torch.stack([mean_obs, slope_T, skew_k], dim=1)  # [B,3]
    c = torch.tanh(
        2.0 * (c - c.mean(dim=0, keepdim=True)) / (c.std(dim=0, keepdim=True) + 1e-6)
    )
    return c


# ------------------ MODEL (same arch as train_cvae.py) ------------------
class EncCond(nn.Module):
    def __init__(self, latent_dim, cond_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),  # 64->32
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),  # 32->16
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),  # 16->8
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),  # 8->4
        )
        # CHANGED: use cond_dim (not hard-coded 3)
        self.fc_h = nn.Linear(128 * 4 * 4 + cond_dim, 256)
        # CHANGED: output dims follow latent_dim
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x2, c):
        h = self.conv(x2).view(x2.size(0), -1)
        h = torch.cat([h, c], dim=1)
        h = F.relu(self.fc_h(h))
        return self.mu(h), self.logvar(h)


class DecCond(nn.Module):
    def __init__(self, latent_dim, cond_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim + cond_dim, 128 * 4 * 4)
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

    def forward(self, z, c):
        zc = torch.cat([z, c], dim=1)
        h = self.fc(zc).view(z.size(0), 128, 4, 4)
        return self.deconv(h)


class CVAE(nn.Module):
    def __init__(self, latent_dim=32, cond_dim=3):
        super().__init__()
        self.enc = EncCond(latent_dim, cond_dim)
        self.dec = DecCond(latent_dim, cond_dim)

    @staticmethod
    def reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x2, c):
        mu, logvar = self.enc(x2, c)
        z = self.reparam(mu, logvar)
        xhat = self.dec(z, c)
        return xhat, mu, logvar


# ------------------ EVALUATION ------------------
def load_model(ckpt_path: str, device=DEVICE):
    # CHANGED: read dims from checkpoint cfg or infer from weights
    state = torch.load(ckpt_path, map_location=device)
    cfg = state.get("cfg", {})
    # fallback inference if cfg missing
    sd = state["model"]
    inferred_latent = sd["enc.mu.weight"].shape[0] if "enc.mu.weight" in sd else 32
    inferred_in_features = (
        sd["dec.fc.weight"].shape[1] if "dec.fc.weight" in sd else (inferred_latent + 3)
    )
    inferred_cond = inferred_in_features - inferred_latent

    latent_dim = int(cfg.get("LATENT_DIM", inferred_latent))
    cond_dim = int(cfg.get("COND_DIM", inferred_cond))

    model = CVAE(latent_dim=latent_dim, cond_dim=cond_dim).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, latent_dim, cond_dim


def evaluate(bundle, ckpt_path: str):
    print(f"\n=== {os.path.basename(ckpt_path)} ===")
    # CHANGED: load returns model and dims
    model, latent_dim, cond_dim = load_model(ckpt_path)

    dates = bundle["dates"]
    _, _, test_idx = build_splits(dates)
    ds_test = IVBundle(bundle, test_idx)
    loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

    # --- Recon MAE ---
    mae_sum, pix_sum = 0.0, 0.0
    with torch.no_grad():
        for xb, mb, _ in loader:
            xb = xb.to(DEVICE)
            mb = mb.to(DEVICE)
            c = compute_conditioning(xb, mb)
            x_in2 = torch.cat([xb * mb.float(), mb.float()], dim=1)
            xhat, _, _ = model(x_in2, c)
            eval_mask = mb.float()
            mae_sum += (eval_mask * (xb - xhat).abs()).sum().item()
            pix_sum += eval_mask.sum().item()
    recon_mae = mae_sum / max(1, pix_sum)
    print(f"Recon MAE (obs cells): {recon_mae:.5f}")

    # --- Inpaint MAE (Monte-Carlo over random visible sets) ---
    inpaint_maes = []
    with torch.no_grad():
        for rep in range(INPAINT_REPEATS):
            mae_sum, pix_sum = 0.0, 0.0
            for xb, mb, _ in loader:
                xb = xb.to(DEVICE)
                mb = mb.to(DEVICE)
                keep = (mb & (torch.rand_like(mb.float()) < INPAINT_KEEP)).float()
                # condition on what you can see:
                c = compute_conditioning(xb * keep, keep.bool())
                x_in2 = torch.cat([xb * keep, keep], dim=1)
                xhat, _, _ = model(x_in2, c)
                eval_mask = (mb & (~keep.bool())).float()
                mae_sum += (eval_mask * (xb - xhat).abs()).sum().item()
                pix_sum += eval_mask.sum().item()
            inpaint_maes.append(mae_sum / max(1, pix_sum))
    inpaint_mae = float(np.mean(inpaint_maes))
    print(
        f"Inpaint MAE (hidden obs cells, keep={INPAINT_KEEP:.2f}, reps={INPAINT_REPEATS}): {inpaint_mae:.5f}"
    )

    # --- Panels ---
    test_idxs = test_idx
    picks = np.linspace(
        0, len(test_idxs) - 1, num=min(PANEL_SAMPLES, len(test_idxs)), dtype=int
    )
    for j, idx in enumerate([test_idxs[i] for i in picks]):
        xb = bundle["x"][idx : idx + 1].to(DEVICE)
        mb = bundle["mask"][idx : idx + 1].to(DEVICE)

        # Recon panel (full mask input)
        with torch.no_grad():
            c = compute_conditioning(xb, mb)
            x_in2 = torch.cat([xb * mb.float(), mb.float()], dim=1)
            xhat, _, _ = model(x_in2, c)
        out_p = os.path.join(SAVE_DIR, f"{os.path.basename(ckpt_path)}_recon_{j}.png")
        save_panel(out_p, xb[0], xhat[0], mb[0].float(), f"Recon #{j}")

        # Inpaint panel (condition on visible subset)
        keep = (mb & (torch.rand_like(mb.float()) < INPAINT_KEEP)).float()
        with torch.no_grad():
            c = compute_conditioning(xb * keep, keep.bool())
            x_in2 = torch.cat([xb * keep, keep], dim=1)
            xhat, _, _ = model(x_in2, c)
        eval_mask = (mb & (~keep.bool())).float()
        out_p = os.path.join(SAVE_DIR, f"{os.path.basename(ckpt_path)}_inpaint_{j}.png")
        save_panel(out_p, xb[0], xhat[0], eval_mask[0], f"Inpaint #{j}")


def main():
    print("[EVAL] Loading dataset bundle...")
    bundle = torch.load(DATASET_PT, map_location="cpu")
    # ensure float/bool dtypes
    bundle["x"] = bundle["x"].float()
    bundle["mask"] = bundle["mask"].bool()

    for ck in CKPTS:
        if os.path.exists(ck):
            evaluate(bundle, ck)
        else:
            print(f"[WARN] Checkpoint not found: {ck}")


if __name__ == "__main__":
    main()
