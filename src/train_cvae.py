# train_cvae.py
# Conditional VAE for IV-surface reconstruction & inpainting
# Conditioning vector c is computed per-surface:
#   [ mean_sigma_obs, term_structure_slope, skew ]
# - Mask-aware training (recon warmup -> inpainting)
# - Loss: masked MSE + β*KL + λ_TV*TV + λ_arb*(convexity_k + calendar_monotonicity_T)

import os
from dataclasses import dataclass
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# ------------------ config ------------------
DATASET_PT = "/home/harshil/IV-Surface-Reconstruction/artifacts/spy_iv_2019_2024.pt"
SAVE_DIR = "/home/harshil/IV-Surface-Reconstruction/artifacts"

LATENT_DIM = 32
COND_DIM = 3  # mean, slope(T), skew(k)
EPOCHS = 40
WARMUP_RECON_EPOCHS = 10  # first N epochs: pure reconstruction
BATCH_SIZE = 64
LR = 1e-3
BETA = 1.00  # β in β-VAE
LAMBDA_TV = 1e-5
LAMBDA_ARB = 1e-2
SEED = 2025
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


def tv_loss(xhat, eps=1e-6):
    # forward differences
    dx = xhat[..., 1:, :] - xhat[..., :-1, :]
    dy = xhat[..., :, 1:] - xhat[..., :, :-1]
    # crop to common region (H-1, W-1) before combining to isotropic TV
    dx_c = dx[..., :, :-1]
    dy_c = dy[..., :-1, :]
    return (dx_c.pow(2) + dy_c.pow(2) + eps).sqrt().mean()


def convexity_penalty_k(xhat):
    d2 = xhat[..., 2:] - 2 * xhat[..., 1:-1] + xhat[..., :-2]
    return torch.relu(-d2).mean()


def calendar_penalty_T(xhat, T_vec):
    w = (xhat**2) * T_vec.view(1, 1, -1, 1)
    dT = w[..., 1:, :] - w[..., :-1, :]
    return torch.relu(-dT).mean()


def save_panel(img_path, x, xhat, mask, title):
    x = x.squeeze(0).cpu().numpy()
    xhat = xhat.squeeze(0).cpu().numpy()
    m = mask.squeeze(0).cpu().numpy().astype(bool)
    err = np.zeros_like(x)
    err[m] = x[m] - xhat[m]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    im0 = axs[0].imshow(x, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
    axs[0].set_title("Target σ")
    im1 = axs[1].imshow(xhat, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
    axs[1].set_title("Recon σ̂")
    im2 = axs[2].imshow(err, origin="lower", aspect="auto")
    axs[2].set_title("Error (obs cells)")
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


# ------------------ conditioning features ------------------
def compute_conditioning(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """
    x: [B,1,H,W] normalized σ in [0,1]
    m: [B,1,H,W] bool mask
    Returns c: [B, COND_DIM] = [mean_obs, slope_T, skew_k]
    """
    B, _, H, W = x.shape
    mf = m.float()
    eps = 1e-8

    # mean observed sigma
    sum_obs = (x * mf).sum(dim=(1, 2, 3))
    cnt_obs = mf.sum(dim=(1, 2, 3)).clamp_min(eps)
    mean_obs = sum_obs / cnt_obs

    # slope along T: avg σ in last 1/3 rows - first 1/3 rows
    t_third = max(1, H // 3)
    top = slice(0, t_third)
    bot = slice(H - t_third, H)

    top_sum = (x[:, :, top, :] * mf[:, :, top, :]).sum(dim=(1, 2, 3))
    top_cnt = (mf[:, :, top, :]).sum(dim=(1, 2, 3)).clamp_min(eps)
    bot_sum = (x[:, :, bot, :] * mf[:, :, bot, :]).sum(dim=(1, 2, 3))
    bot_cnt = (mf[:, :, bot, :]).sum(dim=(1, 2, 3)).clamp_min(eps)
    slope_T = (bot_sum / bot_cnt) - (top_sum / top_cnt)

    # skew along k: right half - left half
    mid = W // 2
    left_sum = (x[:, :, :, :mid] * mf[:, :, :, :mid]).sum(dim=(1, 2, 3))
    left_cnt = (mf[:, :, :, :mid]).sum(dim=(1, 2, 3)).clamp_min(eps)
    right_sum = (x[:, :, :, mid:] * mf[:, :, :, mid:]).sum(dim=(1, 2, 3))
    right_cnt = (mf[:, :, :, mid:]).sum(dim=(1, 2, 3)).clamp_min(eps)
    skew_k = (right_sum / right_cnt) - (left_sum / left_cnt)

    c = torch.stack([mean_obs, slope_T, skew_k], dim=1)  # [B,3]
    # standardize roughly to ~[-1,1] range to help training
    c = torch.tanh(
        2.0 * (c - c.mean(dim=0, keepdim=True)) / (c.std(dim=0, keepdim=True) + 1e-6)
    )
    return c


# ------------------ data ------------------
class IVBundle(Dataset):
    def __init__(self, bundle, indices: List[int]):
        self.x = bundle["x"].float()  # [N,1,H,W]
        self.mask = bundle["mask"].bool()  # [N,1,H,W]
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
class EncCond(nn.Module):
    def __init__(self, latent_dim, cond_dim):
        super().__init__()
        # input: 2 channels (x⊙mask_in, mask_in)
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
        self.fc_h = nn.Linear(128 * 4 * 4 + cond_dim, 256)
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
            nn.ReLU(inplace=True),  # 4->8
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),  # 8->16
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),  # 16->32
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),  # 32->64
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),  # normalized inputs in [0,1]
        )

    def forward(self, z, c):
        zc = torch.cat([z, c], dim=1)
        h = self.fc(zc).view(z.size(0), 128, 4, 4)
        return self.deconv(h)


class CVAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, cond_dim=COND_DIM):
        super().__init__()
        self.enc = EncCond(latent_dim, cond_dim)
        self.dec = DecCond(latent_dim, cond_dim)

    @staticmethod
    def reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x2, c):
        mu, logvar = self.enc(x2, c)
        z = self.reparam(mu, logvar)
        xhat = self.dec(z, c)
        return xhat, mu, logvar


# ------------------ training ------------------
def train():
    bundle = torch.load(DATASET_PT, map_location="cpu")
    x = bundle["x"].float()  # [N,1,H,W]
    m = bundle["mask"].bool()  # [N,1,H,W]
    dates = bundle["dates"]
    T_grid = bundle["T_grid"].float().to(DEVICE)  # [H]

    train_idx, val_idx, test_idx = build_splits(dates)
    ds_train = IVBundle(bundle, train_idx)
    ds_val = IVBundle(bundle, val_idx)

    train_loader = DataLoader(
        ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        ds_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    model = CVAE(LATENT_DIM, COND_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val = float("inf")
    ckpt_path = os.path.join(SAVE_DIR, "cvae.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_rec = 0.0

        inpainting_phase = epoch > WARMUP_RECON_EPOCHS

        for xb, mb, _ in train_loader:
            xb, mb = to_device(xb, mb)  # [B,1,H,W]

            # conditioning vector c from current target + mask
            c = compute_conditioning(xb, mb)  # [B,3]

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

            x_in2 = torch.cat([xb * input_mask, input_mask], dim=1)  # [B,2,H,W]
            xhat, mu, logvar = model(x_in2, c)

            rec = masked_mse(xb, xhat, loss_mask)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            tv = tv_loss(xhat)
            arb = convexity_penalty_k(xhat) + calendar_penalty_T(xhat, T_grid)

            loss = rec + BETA * kl + LAMBDA_TV * tv + LAMBDA_ARB * arb

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            epoch_rec += rec.item() * xb.size(0)

        sched.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_rec, val_n = 0.0, 0
            for xb, mb, _ in val_loader:
                xb, mb = to_device(xb, mb)
                c = compute_conditioning(xb, mb)
                x_in2 = torch.cat([xb * mb.float(), mb.float()], dim=1)
                xhat, mu, logvar = model(x_in2, c)
                rec = masked_mse(xb, xhat, mb.float())
                val_rec += rec.item() * xb.size(0)
                val_n += xb.size(0)
            val_rec /= max(1, val_n)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | lr={sched.get_last_lr()[0]:.2e} | "
            f"train_rec={epoch_rec/len(ds_train):.5f} | val_rec={val_rec:.5f} | mode={'INPAINT' if inpainting_phase else 'RECON'}"
        )

        if val_rec < best_val:
            best_val = val_rec
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": dict(
                        LATENT_DIM=LATENT_DIM,
                        COND_DIM=COND_DIM,
                        BETA=BETA,
                        LAMBDA_TV=LAMBDA_TV,
                        LAMBDA_ARB=LAMBDA_ARB,
                    ),
                },
                ckpt_path,
            )

        # sample panel snapshots
        if epoch % 5 == 0:
            try:
                xb, mb, _ = next(iter(val_loader))
                xb, mb = xb.to(DEVICE), mb.to(DEVICE)
                c = compute_conditioning(xb, mb)
                x_in2 = torch.cat([xb * mb.float(), mb.float()], dim=1)
                with torch.no_grad():
                    xhat, _, _ = model(x_in2, c)
                pth = os.path.join(SAVE_DIR, f"cvae_panel_e{epoch}.png")
                save_panel(pth, xb[0], xhat[0], mb[0].float(), f"CVAE Epoch {epoch}")
            except Exception:
                pass

    print(f"Saved best CVAE to: {ckpt_path}")
    return ckpt_path, bundle


# ------------------ quick demos ------------------
def demo_reconstruct(ckpt_path, bundle, n_samples=3, mode="recon"):
    model = CVAE(LATENT_DIM, COND_DIM).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state["model"])
    model.eval()

    x = bundle["x"].float()
    m = bundle["mask"].bool()

    def split_idx(dates):
        return build_splits(dates)

    _, _, test_idx = split_idx(bundle["dates"])
    picks = np.linspace(
        0, len(test_idx) - 1, num=min(n_samples, len(test_idx)), dtype=int
    )

    for j, idx in enumerate([test_idx[i] for i in picks]):
        xb = x[idx : idx + 1].to(DEVICE)
        mb = m[idx : idx + 1].to(DEVICE)
        c = compute_conditioning(xb, mb)

        if mode == "inpaint":
            rand = torch.rand_like(mb.float())
            keep = (mb & (rand < 0.3)).float()
            x_in2 = torch.cat([xb * keep, keep], dim=1)
            loss_mask = (mb & (~keep.bool())).float()
            mode_name = "inpaint"
        else:
            x_in2 = torch.cat([xb * mb.float(), mb.float()], dim=1)
            loss_mask = mb.float()
            mode_name = "recon"

        with torch.no_grad():
            xhat, _, _ = model(x_in2, c)

        out_path = os.path.join(SAVE_DIR, f"cvae_demo_{mode_name}_{j}.png")
        save_panel(out_path, xb[0], xhat[0], loss_mask[0], f"CVAE {mode_name} #{j}")
        print("Wrote:", out_path)


if __name__ == "__main__":
    ckpt, bundle = train()
    demo_reconstruct(ckpt, bundle, n_samples=3, mode="recon")
    demo_reconstruct(ckpt, bundle, n_samples=3, mode="inpaint")
