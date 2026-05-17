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
    # Average KL per-sample
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def tv_loss(xhat):
    dh = torch.abs(xhat[:, :, 1:, :] - xhat[:, :, :-1, :])
    dw = torch.abs(xhat[:, :, :, 1:] - xhat[:, :, :, :-1])
    return dh.mean() + dw.mean()


def arbitrage_penalty(xhat, T_vec):
    # Convexity in k (last dim)
    k_diff2 = xhat[..., 2:] - 2 * xhat[..., 1:-1] + xhat[..., :-2]
    pen_conv = torch.relu(-k_diff2).mean()

    # Monotonicity in T (2nd to last dim)
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
        self.x = bundle["x"][indices].float()  # [N, 1, H, W]
        self.mask = bundle["mask"][indices].float()  # [N, 1, H, W]
        self.dates = [bundle["dates"][i] for i in indices]

        # Pre-compute Coordinate Grids normalized to [-1, 1]
        T_grid = bundle["T_grid"].float()
        k_grid = bundle["k_grid"].float()

        # Normalize grids
        T_norm = (T_grid - T_grid.min()) / (T_grid.max() - T_grid.min() + 1e-6) * 2 - 1
        k_norm = (k_grid - k_grid.min()) / (k_grid.max() - k_grid.min() + 1e-6) * 2 - 1

        # Create meshgrid [H, W]
        T_mesh, k_mesh = torch.meshgrid(T_norm, k_norm, indexing="ij")

        # Expand to [N, 1, H, W]
        self.T_channel = T_mesh.unsqueeze(0).unsqueeze(0).repeat(len(indices), 1, 1, 1)
        self.k_channel = k_mesh.unsqueeze(0).unsqueeze(0).repeat(len(indices), 1, 1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Return: x, mask, T_channel, k_channel, idx
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
        "train": DataLoader(
            IVSurfaceDataset(bundle, indices[:n_train]),
            batch_size=batch_size,
            shuffle=True,
        ),
        "val": DataLoader(
            IVSurfaceDataset(bundle, indices[n_train : n_train + n_val]),
            batch_size=batch_size,
            shuffle=False,
        ),
        "test": DataLoader(
            IVSurfaceDataset(bundle, indices[n_train + n_val :]),
            batch_size=batch_size,
            shuffle=False,
        ),
    }
    grids = {"T": bundle["T_grid"], "k": bundle["k_grid"]}
    return loaders, grids, bundle


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
        modules = []
        # Input channels = 4 (Val, Mask, T_grid, k_grid)
        modules.append(nn.Conv2d(in_channels, hidden_dims[0], 3, stride=2, padding=1))

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    ResBlock(hidden_dims[i]),
                    nn.Conv2d(
                        hidden_dims[i], hidden_dims[i + 1], 3, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.SiLU(),
                )
            )

        self.encoder = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flat_dim = hidden_dims[-1] * 4 * 4
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_var = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        h = self.pool(h)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_var(h)


class CoordResNetDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims):
        super().__init__()
        self.hidden_dims = list(reversed(hidden_dims))
        self.initial_h = 4
        self.initial_ch = self.hidden_dims[0]
        self.fc_dec = nn.Linear(
            latent_dim, self.initial_ch * self.initial_h * self.initial_h
        )

        modules = []
        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.hidden_dims[i],
                        self.hidden_dims[i + 1],
                        4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.SiLU(),
                    ResBlock(self.hidden_dims[i + 1]),
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.hidden_dims[-1], self.hidden_dims[-1], 4, stride=2, padding=1
                ),
                nn.SiLU(),
                nn.Conv2d(self.hidden_dims[-1], 1, 3, padding=1),
            )
        )
        self.decoder = nn.Sequential(*modules)

    def forward(self, z):
        h = self.fc_dec(z)
        h = h.view(z.size(0), self.initial_ch, self.initial_h, self.initial_h)
        return self.decoder(h)


class ResnetCoordVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Input channels is now 4: Value, Mask, T_grid, k_grid
        self.encoder = CoordResNetEncoder(
            4, config["hidden_dims"], config["latent_dim"]
        )
        self.decoder = CoordResNetDecoder(config["latent_dim"], config["hidden_dims"])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, mask, t_grid, k_grid):
        # Concatenate all 4 channels
        x_in = torch.cat([x * mask, mask, t_grid, k_grid], dim=1)

        mu, logvar = self.encoder(x_in)
        z = self.reparameterize(mu, logvar)
        xhat = self.decoder(z)

        if xhat.shape[-2:] != x.shape[-2:]:
            xhat = F.interpolate(
                xhat, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        return xhat, mu, logvar


# ============================================================================
# BICUBIC FIT
# ============================================================================
def fit_bicubic(x_batch, mask_batch, t_grid, k_grid):
    if not SCIPY_OK:
        return x_batch
    preds = []
    T_mesh, K_mesh = np.meshgrid(t_grid.numpy(), k_grid.numpy(), indexing="ij")
    for i in range(x_batch.shape[0]):
        x_img = x_batch[i, 0].cpu().numpy()
        m_img = mask_batch[i, 0].cpu().numpy().astype(bool)
        if m_img.sum() < 4:
            preds.append(x_img)
            continue
        points = np.column_stack((T_mesh[m_img], K_mesh[m_img]))
        values = x_img[m_img]
        try:
            grid_z = griddata(points, values, (T_mesh, K_mesh), method="cubic")
            if np.isnan(grid_z).any():
                grid_z[np.isnan(grid_z)] = griddata(
                    points,
                    values,
                    (T_mesh[np.isnan(grid_z)], K_mesh[np.isnan(grid_z)]),
                    method="nearest",
                )
            preds.append(grid_z)
        except Exception:
            preds.append(x_img)
    return np.array(preds)[:, None, :, :]


def get_beta(epoch, total_epochs, max_beta):
    # Gentler schedule — hit max_beta by epoch ~20
    return max_beta * min(1.0, epoch / 20.0)


# ----------------- LOGGING / VISUAL HELPERS -----------------


def save_history(history: Dict[str, List[float]], save_dir: str):
    """Save training history to JSON + CSV for reproducible plots."""
    json_path = os.path.join(save_dir, "training_history.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)

    csv_path = os.path.join(save_dir, "training_history.csv")
    fieldnames = [
        "epoch",
        "train_loss",
        "train_mae",
        "train_kl",
        "train_tv",
        "train_arb",
        "val_mae",
        "val_kl",
        "lr",
        "beta",
    ]
    num_epochs = len(history["train_loss"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(num_epochs):
            writer.writerow(
                {
                    "epoch": i + 1,
                    "train_loss": history["train_loss"][i],
                    "train_mae": history["train_mae"][i],
                    "train_kl": history["train_kl"][i],
                    "train_tv": history["train_tv"][i],
                    "train_arb": history["train_arb"][i],
                    "val_mae": history["val_mae"][i],
                    "val_kl": history["val_kl"][i],
                    "lr": history["lr"][i],
                    "beta": history["beta"][i],
                }
            )


def save_table_as_csv(rows, fieldnames, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def save_reconstruction_examples(
    model: nn.Module,
    loader: DataLoader,
    epoch: int,
    save_dir: str,
    tag: str = "val",
    max_samples: int = 4,
):
    """Save a grid of original vs masked vs reconstructed surfaces for visual proof."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        batch = next(iter(loader))
        x, mask, t_ch, k_ch, idx = batch
        x_d, m_d, t_d, k_d = to_device(x, mask, t_ch, k_ch)
        xhat, _, _ = model(x_d, m_d, t_d, k_d)

    x_np = x_d.cpu().numpy()
    m_np = m_d.cpu().numpy()
    xhat_np = xhat.cpu().numpy()

    n_show = min(max_samples, x_np.shape[0])
    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
    if n_show == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n_show):
        obs = x_np[i, 0] * m_np[i, 0]
        gt = x_np[i, 0]
        rec = xhat_np[i, 0]

        axes[i, 0].imshow(obs)
        axes[i, 0].set_title("Observed (masked)")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(rec)
        axes[i, 2].set_title("VAE Recon")
        axes[i, 2].axis("off")

    plt.tight_layout()
    fname = os.path.join(save_dir, f"recon_{tag}_ep{epoch:03d}.png")
    plt.savefig(fname, dpi=150)
    plt.close()


# ----------------- TRAINING -----------------


def train_model():
    loaders, grids, bundle = get_dataloaders(
        CONFIG["dataset_path"], CONFIG["batch_size"]
    )
    T_grid = grids["T"].to(DEVICE)

    model = ResnetCoordVAE(CONFIG).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    history = {
        "train_loss": [],
        "train_mae": [],
        "train_kl": [],
        "train_tv": [],
        "train_arb": [],
        "val_mae": [],
        "val_kl": [],
        "lr": [],
        "beta": [],
    }
    best_loss = float("inf")
    best_model_state = None

    print(f"{'='*80}\nSTARTING RESNET-COORD-CONV TRAINING\n{'='*80}")
    print(
        f"{'Ep':<4} | {'Tr Loss':<10} | {'Tr MAE':<10} | {'Val MAE':<10} | "
        f"{'Val KL':<10} | {'Beta':<8} | {'LR':<10} | {'Best':<10}"
    )
    print("-" * 90)

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        beta = get_beta(epoch, CONFIG["epochs"], CONFIG["beta_max"])

        train_loss_acc = 0.0
        train_mae_acc = 0.0
        train_kl_acc = 0.0
        train_tv_acc = 0.0
        train_arb_acc = 0.0

        pbar = tqdm(loaders["train"], desc=f"Ep {epoch}", leave=False)
        for x, mask, t_ch, k_ch, _ in pbar:
            x, mask, t_ch, k_ch = to_device(x, mask, t_ch, k_ch)

            xhat, mu, logvar = model(x, mask, t_ch, k_ch)

            # L1 Loss (MAE) for sharper results
            recon_loss = masked_mae(x, xhat, mask)
            kl_raw = kl_divergence(mu, logvar)
            kl_loss = beta * torch.max(
                kl_raw, torch.tensor(CONFIG["free_bits"], device=DEVICE)
            )
            tv = tv_loss(xhat)
            arb = arbitrage_penalty(xhat, T_grid)

            loss = (
                recon_loss
                + kl_loss
                + CONFIG["lambda_tv"] * tv
                + CONFIG["lambda_arb"] * arb
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_acc += loss.item()
            train_mae_acc += recon_loss.item()
            train_kl_acc += kl_raw.item()
            train_tv_acc += tv.item()
            train_arb_acc += arb.item()

            pbar.set_postfix(
                {
                    "L1": f"{recon_loss.item():.4f}",
                    "KL": f"{kl_raw.item():.2f}",
                }
            )

        # Validation
        model.eval()
        val_mae_acc = 0.0
        val_kl_acc = 0.0
        with torch.no_grad():
            for x, mask, t_ch, k_ch, _ in loaders["val"]:
                x, mask, t_ch, k_ch = to_device(x, mask, t_ch, k_ch)
                xhat, mu, logvar = model(x, mask, t_ch, k_ch)
                val_mae_acc += masked_mae(x, xhat, mask).item()
                val_kl_acc += kl_divergence(mu, logvar).item()

        num_train_batches = len(loaders["train"])
        num_val_batches = len(loaders["val"])

        avg_train = train_loss_acc / num_train_batches
        avg_train_mae = train_mae_acc / num_train_batches
        avg_train_kl = train_kl_acc / num_train_batches
        avg_train_tv = train_tv_acc / num_train_batches
        avg_train_arb = train_arb_acc / num_train_batches

        avg_val_mae = val_mae_acc / num_val_batches
        avg_val_kl = val_kl_acc / num_val_batches

        scheduler.step(avg_val_mae)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train)
        history["train_mae"].append(avg_train_mae)
        history["train_kl"].append(avg_train_kl)
        history["train_tv"].append(avg_train_tv)
        history["train_arb"].append(avg_train_arb)
        history["val_mae"].append(avg_val_mae)
        history["val_kl"].append(avg_val_kl)
        history["lr"].append(current_lr)
        history["beta"].append(beta)

        if avg_val_mae < best_loss:
            best_loss = avg_val_mae
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(
                best_model_state, os.path.join(CONFIG["save_dir"], "best_coord_vae.pt")
            )

        print(
            f"{epoch:<4} | {avg_train:<10.5f} | {avg_train_mae:<10.5f} | "
            f"{avg_val_mae:<10.5f} | {avg_val_kl:<10.5f} | {beta:<8.4f} | "
            f"{current_lr:<10.2e} | {best_loss:<10.5f}"
        )

        # Save recon snapshots during training as proof
        if (epoch == 1) or (epoch % CONFIG["snapshot_every"] == 0):
            save_reconstruction_examples(
                model,
                loaders["val"],
                epoch,
                save_dir=SAMPLES_DIR,
                tag="val",
                max_samples=CONFIG["num_visual_samples"],
            )

    # Load best state (by val MAE)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final recon grid after training
    save_reconstruction_examples(
        model,
        loaders["val"],
        epoch=CONFIG["epochs"],
        save_dir=SAMPLES_DIR,
        tag="val_final",
        max_samples=CONFIG["num_visual_samples"],
    )

    return model, history, loaders, grids, bundle


# ----------------- REPORT / EVAL -----------------


def generate_report(model, history, loaders, grids, bundle):
    print("\nGenerating Report...")
    model.eval()

    # Save training history JSON + CSV
    save_history(history, CONFIG["save_dir"])

    # --- TABLE I: Reconstruction ---
    def eval_recon(loader):
        mae_vae, mse_vae = [], []
        mae_base, mse_base = [], []
        with torch.no_grad():
            for x, mask, t_ch, k_ch, _ in tqdm(loader, desc="Eval Recon", leave=False):
                x_d, m_d, t_d, k_d = to_device(x, mask, t_ch, k_ch)
                xhat, _, _ = model(x_d, m_d, t_d, k_d)
                mae_vae.append(masked_mae(x_d, xhat, m_d).item())
                mse_vae.append(masked_mse(x_d, xhat, m_d).item())
                if SCIPY_OK:
                    xhat_bic = fit_bicubic(x, mask, grids["T"].cpu(), grids["k"].cpu())
                    xhat_bic = torch.tensor(xhat_bic).float().to(DEVICE)
                    mae_base.append(masked_mae(x_d, xhat_bic, m_d).item())
                    mse_base.append(masked_mse(x_d, xhat_bic, m_d).item())
                else:
                    mae_base.append(0.0)
                    mse_base.append(0.0)
        return {
            "VAE": (np.mean(mae_vae), np.mean(mse_vae)),
            "Bicubic": (np.mean(mae_base), np.mean(mse_base)),
        }

    res_val = eval_recon(loaders["val"])
    res_test = eval_recon(loaders["test"])

    print("\nTABLE I: Reconstruction performance.")
    print(f"{'Model':<20} | {'Split':<10} | {'Masked-MAE':<12} | {'MSE':<12}")
    print("-" * 60)
    print(
        f"{'Bicubic':<20} | {'Val':<10} | {res_val['Bicubic'][0]:.6f}     | {res_val['Bicubic'][1]:.6f}"
    )
    print(
        f"{'Coord-VAE':<20} | {'Val':<10} | {res_val['VAE'][0]:.6f}     | {res_val['VAE'][1]:.6f}"
    )
    print(
        f"{'Bicubic':<20} | {'Test':<10} | {res_test['Bicubic'][0]:.6f}     | {res_test['Bicubic'][1]:.6f}"
    )
    print(
        f"{'Coord-VAE':<20} | {'Test':<10} | {res_test['VAE'][0]:.6f}     | {res_test['VAE'][1]:.6f}"
    )

    # Save TABLE I as CSV
    rows_t1 = [
        {
            "split": "val",
            "model": "Bicubic",
            "masked_mae": res_val["Bicubic"][0],
            "mse": res_val["Bicubic"][1],
        },
        {
            "split": "val",
            "model": "Coord-VAE",
            "masked_mae": res_val["VAE"][0],
            "mse": res_val["VAE"][1],
        },
        {
            "split": "test",
            "model": "Bicubic",
            "masked_mae": res_test["Bicubic"][0],
            "mse": res_test["Bicubic"][1],
        },
        {
            "split": "test",
            "model": "Coord-VAE",
            "masked_mae": res_test["VAE"][0],
            "mse": res_test["VAE"][1],
        },
    ]
    save_table_as_csv(
        rows_t1,
        fieldnames=["split", "model", "masked_mae", "mse"],
        path=os.path.join(CONFIG["save_dir"], "table1_recon.csv"),
    )

    # --- TABLE II: Inpainting ---
    def eval_inpaint(loader, keep_frac=0.3):
        mae_vae, mse_vae = [], []
        mae_base, mse_base = [], []
        with torch.no_grad():
            for x, mask, t_ch, k_ch, _ in tqdm(
                loader, desc="Eval Inpaint", leave=False
            ):
                x_d, m_d, t_d, k_d = to_device(x, mask, t_ch, k_ch)
                rand = torch.rand_like(m_d) < keep_frac
                mask_in = m_d * rand.float()
                mask_loss = m_d * (~rand).float()
                if mask_loss.sum() == 0:
                    continue

                xhat, _, _ = model(x_d, mask_in, t_d, k_d)
                mae_vae.append(masked_mae(x_d, xhat, mask_loss).item())
                mse_vae.append(masked_mse(x_d, xhat, mask_loss).item())
                if SCIPY_OK:
                    xhat_bic = fit_bicubic(
                        x.cpu(), mask_in.cpu(), grids["T"].cpu(), grids["k"].cpu()
                    )
                    xhat_bic = torch.tensor(xhat_bic).float().to(DEVICE)
                    mae_base.append(masked_mae(x_d, xhat_bic, mask_loss).item())
                    mse_base.append(masked_mse(x_d, xhat_bic, mask_loss).item())
        return {
            "VAE": (np.mean(mae_vae), np.mean(mse_vae)),
            "Bicubic": (np.mean(mae_base), np.mean(mse_base)),
        }

    inp_test = eval_inpaint(loaders["test"])

    print("\nTABLE II: Inpainting performance.")
    print(f"{'Model':<20} | {'Split':<10} | {'Masked-MAE':<12} | {'MSE':<12}")
    print("-" * 60)
    print(
        f"{'Bicubic':<20} | {'Test':<10} | {inp_test['Bicubic'][0]:.6f}     | {inp_test['Bicubic'][1]:.6f}"
    )
    print(
        f"{'Coord-VAE':<20} | {'Test':<10} | {inp_test['VAE'][0]:.6f}     | {inp_test['VAE'][1]:.6f}"
    )

    rows_t2 = [
        {
            "split": "test",
            "model": "Bicubic",
            "masked_mae": inp_test["Bicubic"][0],
            "mse": inp_test["Bicubic"][1],
        },
        {
            "split": "test",
            "model": "Coord-VAE",
            "masked_mae": inp_test["VAE"][0],
            "mse": inp_test["VAE"][1],
        },
    ]
    save_table_as_csv(
        rows_t2,
        fieldnames=["split", "model", "masked_mae", "mse"],
        path=os.path.join(CONFIG["save_dir"], "table2_inpaint.csv"),
    )

    # --- Training Curves Plots ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Loss + Val MAE
    axs[0, 0].plot(history["train_loss"], label="Train Total Loss")
    axs[0, 0].plot(history["val_mae"], label="Val MAE (masked)")
    axs[0, 0].set_title("Total Loss & Val MAE")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].legend()

    # MAE: train vs val
    axs[0, 1].plot(history["train_mae"], label="Train MAE")
    axs[0, 1].plot(history["val_mae"], label="Val MAE")
    axs[0, 1].set_title("Reconstruction MAE")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].legend()

    # KL
    axs[1, 0].plot(history["train_kl"], label="Train KL")
    axs[1, 0].plot(history["val_kl"], label="Val KL")
    axs[1, 0].set_title("KL Divergence")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].legend()

    # LR & Beta
    ax_lr = axs[1, 1]
    ax_lr.plot(history["lr"], label="LR")
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_title("Learning Rate & Beta Schedule")

    ax_beta = ax_lr.twinx()
    ax_beta.plot(history["beta"], label="Beta", linestyle="--")
    # Combine legends
    lines, labels = ax_lr.get_legend_handles_labels()
    lines2, labels2 = ax_beta.get_legend_handles_labels()
    ax_lr.legend(lines + lines2, labels + labels2, loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["save_dir"], "training_curves.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    trained_model, hist, dls, grids, data_bundle = train_model()
    generate_report(trained_model, hist, dls, grids, data_bundle)
    print("\n[SUCCESS] Script completed.")
