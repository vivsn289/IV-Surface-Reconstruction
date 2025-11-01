# train_vae.py
# Mask-aware β-VAE for IV-surface reconstruction & inpainting
# - Loads bundle saved by iv_surface_dataset.py
# - 2-channel input: [x_observed, mask_input]; decoder predicts 1 channel (sigma)
# - Loss: masked MSE + β*KL + λ_TV*TV + λ_arb*(convexity_k + calendar_monotonicity_T)
# - Mask curriculum: warmup epochs = full recon; then random input sparsity for inpainting

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
EPOCHS = 25
WARMUP_RECON_EPOCHS = 5  # first N epochs: pure reconstruction
BATCH_SIZE = 64
LR = 1e-3
BETA = 2.0  # β-VAE
LAMBDA_TV = 1e-4
LAMBDA_ARB = 1e-2
SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)


# ------------------ utils ------------------
def to_device(*t):
    return [x.to(DEVICE) for x in t]


def masked_mse(x, xhat, mask, eps=1e-8):
    # x, xhat, mask: [B,1,H,W]
    num = (mask * (x - xhat) ** 2).sum()
    den = mask.sum().clamp_min(eps)
    return num / den


def tv_loss(xhat, eps=1e-6):
    # xhat: [B,1,H,W]
    dx = xhat[:, :, 1:, :] - xhat[:, :, :-1, :]  # [B,1,H-1,W]
    dy = xhat[:, :, :, 1:] - xhat[:, :, :, :-1]  # [B,1,H,W-1]
    # crop to common interior so shapes match: [B,1,H-1,W-1]
    dx_c = dx[:, :, :, :-1]
    dy_c = dy[:, :, :-1, :]
    tv = torch.sqrt(dx_c.pow(2) + dy_c.pow(2) + eps)
    return tv.mean()


def convexity_penalty_k(xhat):
    # second finite difference along k (W axis)
    d2 = xhat[..., 2:] - 2 * xhat[..., 1:-1] + xhat[..., :-2]
    return torch.relu(-d2).mean()


def calendar_penalty_T(xhat, T_vec):
    # monotonicity of total variance along T (H axis)
    # xhat: [B,1,H,W], T_vec: [H]
    w = (xhat**2) * T_vec.view(1, 1, -1, 1)  # [B,1,H,W]
    dT = w[..., 1:, :] - w[..., :-1, :]
    return torch.relu(-dT).mean()


def save_panel(img_path, x, xhat, mask, title):
    # x, xhat, mask: [1,H,W] tensors on CPU
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


# ------------------ data ------------------
class IVBundle(Dataset):
    def __init__(self, bundle, indices: List[int]):
        self.x = bundle["x"].float()  # [N,1,H,W], already normalized
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
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # input channels=2 (x_input, mask_input)
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
            nn.ReLU(inplace=True),  # 4->8
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),  # 8->16
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),  # 16->32
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),  # 32->64
            nn.Conv2d(16, 1, 3, padding=1),  # output σ̂
            nn.Sigmoid(),  # since inputs were min-max normalized to [0,1]
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

    def forward(self, x_in):  # x_in: [B,2,H,W]
        mu, logvar = self.enc(x_in)
        z = self.reparam(mu, logvar)
        xhat = self.dec(z)
        return xhat, mu, logvar


# ------------------ training ------------------
def train():
    bundle = torch.load(DATASET_PT, map_location="cpu")
    x = bundle["x"]  # [N,1,H,W]
    m = bundle["mask"]  # [N,1,H,W]
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

    model = BetaVAE(LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val = float("inf")
    ckpt_path = os.path.join(SAVE_DIR, "vae_beta.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss, epoch_rec = 0.0, 0.0

        # choose training mode
        inpainting_phase = epoch > WARMUP_RECON_EPOCHS

        for xb, mb, _ in train_loader:
            xb, mb = to_device(xb, mb)  # [B,1,H,W]
            B, _, H, W = xb.shape

            # Build input mask for encoder
            if inpainting_phase:
                # random per-pixel keep prob between 0.2..0.8 over observed cells
                keep_prob = np.random.uniform(0.2, 0.8)
                rand = torch.rand_like(mb.float())
                input_mask = (mb & (rand < keep_prob)).float()
                loss_mask = (
                    mb & (~(input_mask.bool()))
                ).float()  # evaluate on hidden observed cells
                # if extremely sparse, fallback to some observed
                if loss_mask.sum() < 10:
                    loss_mask = mb.float()
            else:
                input_mask = mb.float()
                loss_mask = mb.float()

            x_in = xb * input_mask
            x_in2 = torch.cat([x_in, input_mask], dim=1)  # [B,2,H,W]

            xhat, mu, logvar = model(x_in2)

            # losses
            rec = masked_mse(xb, xhat, loss_mask)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            tv = tv_loss(xhat)
            arb = convexity_penalty_k(xhat) + calendar_penalty_T(xhat, T_grid)

            loss = rec + BETA * kl + LAMBDA_TV * tv + LAMBDA_ARB * arb

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            epoch_loss += loss.item() * B
            epoch_rec += rec.item() * B

        sched.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_rec = 0.0
            val_cnt = 0
            for xb, mb, _ in val_loader:
                xb, mb = to_device(xb, mb)
                # use full obs as input in val for stability
                x_in2 = torch.cat([xb * mb.float(), mb.float()], dim=1)
                xhat, mu, logvar = model(x_in2)
                rec = masked_mse(xb, xhat, mb.float())
                val_rec += rec.item() * xb.size(0)
                val_cnt += xb.size(0)
            val_rec /= max(1, val_cnt)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | lr={sched.get_last_lr()[0]:.2e} | "
            f"train_rec={epoch_rec/len(ds_train):.5f} | val_rec={val_rec:.5f} | mode={'INPAINT' if inpainting_phase else 'RECON'}"
        )

        # save best on val_rec
        if val_rec < best_val:
            best_val = val_rec
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

        # dump a small panel every few epochs
        if epoch % 5 == 0:
            try:
                xb, mb, _ = next(iter(val_loader))
                xb, mb = xb.to(DEVICE), mb.to(DEVICE)
                x_in2 = torch.cat([xb * mb.float(), mb.float()], dim=1)
                with torch.no_grad():
                    xhat, _, _ = model(x_in2)
                # save first sample panel
                pth = os.path.join(SAVE_DIR, f"vae_panel_e{epoch}.png")
                save_panel(
                    pth, xb[0].cpu(), xhat[0].cpu(), mb[0].cpu(), f"Epoch {epoch}"
                )
            except Exception as _:
                pass

    print(f"Saved best model to: {ckpt_path}")
    return ckpt_path, bundle


# ------------------ quick test/inference ------------------
def demo_reconstruct(ckpt_path, bundle, n_samples=3, mode="recon"):
    model = BetaVAE(LATENT_DIM).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state["model"])
    model.eval()

    x = bundle["x"].float()
    m = bundle["mask"].bool()

    # pick evenly spaced samples from test split
    _, _, test_idx = build_splits(bundle["dates"])
    picks = np.linspace(
        0, len(test_idx) - 1, num=min(n_samples, len(test_idx)), dtype=int
    )

    for j, idx in enumerate([test_idx[i] for i in picks]):
        xb = x[idx : idx + 1].to(DEVICE)  # [1,1,H,W]
        mb = m[idx : idx + 1].to(DEVICE)

        if mode == "inpaint":
            # hide most observed; keep 30%
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
            xhat, _, _ = model(x_in2)

        out_path = os.path.join(SAVE_DIR, f"demo_{mode_name}_{j}.png")
        save_panel(
            out_path,
            xb[0].cpu(),
            xhat[0].cpu(),
            loss_mask[0].cpu(),
            f"Demo {mode_name} #{j}",
        )
        print("Wrote:", out_path)


if __name__ == "__main__":
    ckpt, bundle = train()
    demo_reconstruct(ckpt, bundle, n_samples=3, mode="recon")
    demo_reconstruct(ckpt, bundle, n_samples=3, mode="inpaint")
