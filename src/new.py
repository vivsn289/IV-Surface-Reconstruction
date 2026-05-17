import os
import json
import copy
import csv
from typing import List, Dict, Tuple

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Try SciPy for bicubic
try:
    from scipy.interpolate import griddata
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    griddata = None

# ============================================================================
# FIXED: correct directory → repo root (NOT cwd)
# ============================================================================
curr_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# FIXED: CONFIG placed BEFORE ANY USE
# ============================================================================
CONFIG = {
    "dataset_path": os.path.join(curr_dir, "artifacts/spy_iv_2019_2024.pt"),
    "save_dir": os.path.join(curr_dir, "artifacts/latest"),

    "latent_dim": 128,
    "hidden_dims": [64, 128, 256, 512],
    "epochs": 150,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "seed": 42,

    "free_bits": 5.0,
    "lambda_tv": 1e-5,
    "lambda_arb": 0.02,
    "beta_max": 1.0,

    "snapshot_every": 25,
    "num_visual_samples": 4,
}

# ============================================================================
# SAFE DIRECTORY CREATION
# ============================================================================
os.makedirs(CONFIG["save_dir"], exist_ok=True)
SAMPLES_DIR = os.path.join(CONFIG["save_dir"], "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

# ============================================================================
# SEEDS
# ============================================================================
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def to_device(*tensors):
    return [t.to(DEVICE) for t in tensors]


def masked_mse(x, xhat, mask, eps=1e-8):
    return ((x - xhat) ** 2 * mask).sum() / mask.sum().clamp_min(eps)


def masked_mae(x, xhat, mask, eps=1e-8):
    return ((x - xhat).abs() * mask).sum() / mask.sum().clamp_min(eps)


def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def tv_loss(xhat):
    dh = torch.abs(xhat[:, :, 1:, :] - xhat[:, :, :-1, :])
    dw = torch.abs(xhat[:, :, :, 1:] - xhat[:, :, :, :-1])
    return dh.mean() + dw.mean()


def arbitrage_penalty(xhat, T_vec):
    k_diff2 = xhat[..., 2:] - 2 * xhat[..., 1:-1] + xhat[..., :-2]
    pen_conv = torch.relu(-k_diff2).mean()

    if T_vec is None:
        return pen_conv

    t_grid = T_vec.view(1, 1, -1, 1).to(xhat.device)
    total_var = (xhat**2) * t_grid
    dt = total_var[..., 1:, :] - total_var[..., :-1, :]
    pen_time = torch.relu(-dt).mean()
    return pen_conv + pen_time


# ============================================================================
# DATASET
# ============================================================================
class IVSurfaceDataset(Dataset):
    def __init__(self, bundle, indices):
        self.x = bundle["x"][indices].float()
        self.mask = bundle["mask"][indices].float()
        self.dates = [bundle["dates"][i] for i in indices]

        T_grid = bundle["T_grid"].float()
        k_grid = bundle["k_grid"].float()

        T_norm = (T_grid - T_grid.min()) / (T_grid.max() - T_grid.min() + 1e-6) * 2 - 1
        k_norm = (k_grid - k_grid.min()) / (k_grid.max() - k_grid.min() + 1e-6) * 2 - 1

        T_mesh, k_mesh = torch.meshgrid(T_norm, k_norm, indexing="ij")

        self.T_channel = T_mesh.unsqueeze(0).unsqueeze(0).repeat(len(indices), 1, 1, 1)
        self.k_channel = k_mesh.unsqueeze(0).unsqueeze(0).repeat(len(indices), 1, 1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.mask[idx],
            self.T_channel[idx],
            self.k_channel[idx],
            idx,
        )


# ============================================================================
# LOADER
# ============================================================================
def get_dataloaders(bundle_path, batch_size):
    bundle = torch.load(bundle_path, map_location="cpu")
    n = len(bundle["dates"])

    indices = np.arange(n)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    loaders = {
        "train": DataLoader(IVSurfaceDataset(bundle, indices[:n_train]),
                            batch_size=batch_size, shuffle=True),
        "val": DataLoader(IVSurfaceDataset(bundle, indices[n_train:n_train+n_val]),
                          batch_size=batch_size, shuffle=False),
        "test": DataLoader(IVSurfaceDataset(bundle, indices[n_train+n_val:]),
                           batch_size=batch_size, shuffle=False),
    }

    return loaders, {"T": bundle["T_grid"], "k": bundle["k_grid"]}, bundle


# ============================================================================
# MODEL BUILDING BLOCKS
# ============================================================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.net(x)


class CoordResNetEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, latent_dim):
        super().__init__()
        layers = [nn.Conv2d(in_channels, hidden_dims[0], 3, stride=2, padding=1)]

        for i in range(len(hidden_dims) - 1):
            layers.append(
                nn.Sequential(
                    ResBlock(hidden_dims[i]),
                    nn.Conv2d(hidden_dims[i], hidden_dims[i+1], 3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.SiLU()
                )
            )

        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flat_dim = hidden_dims[-1] * 4 * 4

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_var = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        h = self.pool(h).view(h.size(0), -1)
        return self.fc_mu(h), self.fc_var(h)


class CoordResNetDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims):
        super().__init__()
        hd = list(reversed(hidden_dims))

        self.initial_h = 4
        self.initial_ch = hd[0]
        self.fc_dec = nn.Linear(latent_dim, hd[0] * 4 * 4)

        layers = []
        for i in range(len(hd) - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hd[i], hd[i+1], 4, stride=2, padding=1),
                    nn.BatchNorm2d(hd[i+1]),
                    nn.SiLU(),
                    ResBlock(hd[i+1]),
                )
            )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(hd[-1], hd[-1], 4, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(hd[-1], 1, 3, padding=1),
            )
        )

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        h = self.fc_dec(z)
        h = h.view(z.size(0), self.initial_ch, self.initial_h, self.initial_h)
        return self.decoder(h)


class ResnetCoordVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = CoordResNetEncoder(4, config["hidden_dims"], config["latent_dim"])
        self.decoder = CoordResNetDecoder(config["latent_dim"], config["hidden_dims"])

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x, mask, t_grid, k_grid):
        x_in = torch.cat([x * mask, mask, t_grid, k_grid], dim=1)
        mu, logvar = self.encoder(x_in)
        z = self.reparameterize(mu, logvar)
        xhat = self.decoder(z)

        if xhat.shape[-2:] != x.shape[-2:]:
            xhat = F.interpolate(xhat, size=x.shape[-2:], mode="bilinear",
                                 align_corners=False)
        return xhat, mu, logvar


# ============================================================================
# BICUBIC FIT
# ============================================================================
def fit_bicubic(x_batch, mask_batch, T_grid, k_grid):
    if not SCIPY_OK:
        return x_batch

    preds = []
    T_mesh, K_mesh = np.meshgrid(T_grid.numpy(), k_grid.numpy(), indexing="ij")

    for i in range(x_batch.shape[0]):
        x_img = x_batch[i, 0].cpu().numpy()
        m_img = mask_batch[i, 0].cpu().numpy().astype(bool)

        if m_img.sum() < 4:
            preds.append(x_img)
            continue

        pts = np.column_stack((T_mesh[m_img], K_mesh[m_img]))
        vals = x_img[m_img]

        try:
            grid_z = griddata(pts, vals, (T_mesh, K_mesh), method="cubic")
            if np.isnan(grid_z).any():
                grid_z[np.isnan(grid_z)] = griddata(
                    pts, vals,
                    (T_mesh[np.isnan(grid_z)], K_mesh[np.isnan(grid_z)]),
                    method="nearest"
                )
            preds.append(grid_z)
        except Exception:
            preds.append(x_img)

    return np.array(preds)[:, None, :, :]


# ============================================================================
# TRAINING
# ============================================================================
def get_beta(epoch, total_epochs, max_beta):
    return max_beta * min(1.0, epoch / 20.0)


def train_model():
    loaders, grids, bundle = get_dataloaders(CONFIG["dataset_path"],
                                             CONFIG["batch_size"])
    T_grid = grids["T"].to(DEVICE)

    model = ResnetCoordVAE(CONFIG).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    history = {
        "train_loss": [], "train_mae": [], "train_kl": [], "train_tv": [],
        "train_arb": [], "val_mae": [], "val_kl": [], "lr": [], "beta": [],
    }

    best_loss = float("inf")
    best_state = None

    print("=" * 80)
    print("STARTING RESNET-COORD-CONV TRAINING")
    print("=" * 80)

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        beta = get_beta(epoch, CONFIG["epochs"], CONFIG["beta_max"])

        t_loss = t_mae = t_kl = t_tv = t_arb = 0

        pbar = tqdm(loaders["train"], desc=f"Ep {epoch}", leave=False)
        for x, mask, t_ch, k_ch, _ in pbar:
            x, mask, t_ch, k_ch = to_device(x, mask, t_ch, k_ch)
            xhat, mu, logvar = model(x, mask, t_ch, k_ch)

            rec = masked_mae(x, xhat, mask)
            kl_raw = kl_divergence(mu, logvar)
            kl_loss = beta * torch.max(
                kl_raw, torch.tensor(CONFIG["free_bits"], device=DEVICE)
            )
            tv = tv_loss(xhat)
            arb = arbitrage_penalty(xhat, T_grid)

            loss = rec + kl_loss + CONFIG["lambda_tv"] * tv + CONFIG["lambda_arb"] * arb

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_loss += loss.item()
            t_mae += rec.item()
            t_kl += kl_raw.item()
            t_tv += tv.item()
            t_arb += arb.item()

        num_t = len(loaders["train"])
        avg_loss = t_loss / num_t
        avg_mae = t_mae / num_t
        avg_kl = t_kl / num_t
        avg_tv = t_tv / num_t
        avg_arb = t_arb / num_t

        # Validation
        model.eval()
        v_mae = v_kl = 0
        with torch.no_grad():
            for x, mask, t_ch, k_ch, _ in loaders["val"]:
                x, mask, t_ch, k_ch = to_device(x, mask, t_ch, k_ch)
                xhat, mu, logvar = model(x, mask, t_ch, k_ch)
                v_mae += masked_mae(x, xhat, mask).item()
                v_kl += kl_divergence(mu, logvar).item()

        num_v = len(loaders["val"])
        v_mae /= num_v
        v_kl /= num_v

        scheduler.step(v_mae)
        lr_now = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_loss)
        history["train_mae"].append(avg_mae)
        history["train_kl"].append(avg_kl)
        history["train_tv"].append(avg_tv)
        history["train_arb"].append(avg_arb)
        history["val_mae"].append(v_mae)
        history["val_kl"].append(v_kl)
        history["lr"].append(lr_now)
        history["beta"].append(beta)

        if v_mae < best_loss:
            best_loss = v_mae
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state,
                       os.path.join(CONFIG["save_dir"], "best_coord_vae.pt"))

        # snapshots
        if epoch == 1 or epoch % CONFIG["snapshot_every"] == 0:
            save_reconstruction_examples(
                model, loaders["val"], epoch,
                save_dir=SAMPLES_DIR,
                max_samples=CONFIG["num_visual_samples"]
            )

        print(f"{epoch:04d} | "
              f"TrLoss={avg_loss:.4f} | TrMAE={avg_mae:.4f} | "
              f"ValMAE={v_mae:.4f} | ValKL={v_kl:.4f} | "
              f"Beta={beta:.3f} | LR={lr_now:.2e} | Best={best_loss:.4f}")

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # final snapshots
    save_reconstruction_examples(
        model, loaders["val"], CONFIG["epochs"],
        save_dir=SAMPLES_DIR, tag="final",
        max_samples=CONFIG["num_visual_samples"]
    )

    return model, history, loaders, grids, bundle


# ============================================================================
# REPORT / TABLES
# ============================================================================
def generate_report(model, history, loaders, grids, bundle):
    print("\nGenerating Report...")
    model.eval()

    save_history(history, CONFIG["save_dir"])

    # --- Reconstruction ---
    def eval_recon(loader):
        mae_vae, mse_vae = [], []
        mae_b, mse_b = [], []

        with torch.no_grad():
            for x, mask, t_ch, k_ch, _ in tqdm(loader, leave=False):
                xd, md, td, kd = to_device(x, mask, t_ch, k_ch)
                xhat, _, _ = model(xd, md, td, kd)

                mae_vae.append(masked_mae(xd, xhat, md).item())
                mse_vae.append(masked_mse(xd, xhat, md).item())

                if SCIPY_OK:
                    xbic = fit_bicubic(x, mask, grids["T"].cpu(), grids["k"].cpu())
                    xbic = torch.tensor(xbic).float().to(DEVICE)
                    mae_b.append(masked_mae(xd, xbic, md).item())
                    mse_b.append(masked_mse(xd, xbic, md).item())
                else:
                    mae_b.append(0.0)
                    mse_b.append(0.0)

        return {
            "VAE": (np.mean(mae_vae), np.mean(mse_vae)),
            "Bicubic": (np.mean(mae_b), np.mean(mse_b)),
        }

    val_res = eval_recon(loaders["val"])
    test_res = eval_recon(loaders["test"])

    print("\nTABLE I — Reconstruction")
    print(val_res)
    print(test_res)

    # Save CSV
    rows = [
        {"split": "val", "model": "VAE", "mae": val_res["VAE"][0], "mse": val_res["VAE"][1]},
        {"split": "val", "model": "Bicubic", "mae": val_res["Bicubic"][0], "mse": val_res["Bicubic"][1]},
        {"split": "test", "model": "VAE", "mae": test_res["VAE"][0], "mse": test_res["VAE"][1]},
        {"split": "test", "model": "Bicubic", "mae": test_res["Bicubic"][0], "mse": test_res["Bicubic"][1]},
    ]
    save_table_as_csv(
        rows, ["split", "model", "mae", "mse"],
        os.path.join(CONFIG["save_dir"], "table1_recon.csv")
    )

    # --- Inpainting ---
    def eval_inpaint(loader, keep_frac=0.3):
        mae_vae, mse_vae = [], []
        mae_b, mse_b = [], []

        with torch.no_grad():
            for x, mask, t_ch, k_ch, _ in tqdm(loader, leave=False):
                xd, md, td, kd = to_device(x, mask, t_ch, k_ch)

                rand = torch.rand_like(md) < keep_frac
                mask_in = md * rand.float()
                mask_loss = md * (~rand).float()

                if mask_loss.sum() == 0:
                    continue

                xhat, _, _ = model(xd, mask_in, td, kd)
                mae_vae.append(masked_mae(xd, xhat, mask_loss).item())
                mse_vae.append(masked_mse(xd, xhat, mask_loss).item())

                if SCIPY_OK:
                    xbic = fit_bicubic(x, mask_in.cpu(),
                                       grids["T"].cpu(), grids["k"].cpu())
                    xbic = torch.tensor(xbic).float().to(DEVICE)
                    mae_b.append(masked_mae(xd, xbic, mask_loss).item())
                    mse_b.append(masked_mse(xd, xbic, mask_loss).item())

        return {
            "VAE": (np.mean(mae_vae), np.mean(mse_vae)),
            "Bicubic": (np.mean(mae_b), np.mean(mse_b)),
        }

    inp_res = eval_inpaint(loaders["test"])
    print("\nTABLE II — Inpainting")
    print(inp_res)

    save_table_as_csv(
        [
            {"split": "test", "model": "VAE", "mae": inp_res["VAE"][0], "mse": inp_res["VAE"][1]},
            {"split": "test", "model": "Bicubic", "mae": inp_res["Bicubic"][0], "mse": inp_res["Bicubic"][1]},
        ],
        ["split", "model", "mae", "mse"],
        os.path.join(CONFIG["save_dir"], "table2_inpaint.csv")
    )

    # --- Training Curves
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(history["train_loss"]); axs[0, 0].plot(history["val_mae"])
    axs[0, 0].set_title("Loss & Val MAE")

    axs[0, 1].plot(history["train_mae"]); axs[0, 1].plot(history["val_mae"])
    axs[0, 1].set_title("MAE")

    axs[1, 0].plot(history["train_kl"]); axs[1, 0].plot(history["val_kl"])
    axs[1, 0].set_title("KL")

    axs[1, 1].plot(history["lr"])
    axs[1, 1].plot(history["beta"])
    axs[1, 1].set_title("LR & Beta")

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["save_dir"], "training_curves.png"), dpi=150)
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    model, history, loaders, grids, bundle = train_model()
    generate_report(model, history, loaders, grids, bundle)
    print("\n[SUCCESS] Script completed.")
