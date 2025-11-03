import argparse
import json
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np

from iv_vae.src.dataio.dataset import IVSurfaceDataset
from iv_vae.src.models.beta_vae import BetaVAE
from iv_vae.src.models.losses import (
    masked_mse_loss,
    kl_divergence_loss,
    total_variation_loss,
)
from iv_vae.src.utils.seed import seed_everything
from iv_vae.src.utils.logging import TrainingLogger
from iv_vae.src.utils.metrics import masked_mae


def train_epoch(model, dataloader, optimizer, beta, tv_lambda, device):
    model.train()
    total_loss, total_recon, total_kl, total_tv = 0, 0, 0, 0
    for batch in dataloader:
        surfaces = batch["surface"].to(device)
        masks = batch["mask_obs"].to(device)

        optimizer.zero_grad()
        recon_x, mu, logvar = model(surfaces)

        recon_loss = masked_mse_loss(recon_x, surfaces, masks)
        kl_loss = kl_divergence_loss(mu, logvar)
        tv_loss = total_variation_loss(recon_x)

        loss = recon_loss + beta * kl_loss + tv_lambda * tv_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_tv += tv_loss.item()

    return (
        total_loss / len(dataloader),
        total_recon / len(dataloader),
        total_kl / len(dataloader),
        total_tv / len(dataloader),
    )


def validate_epoch(model, dataloader, beta, tv_lambda, device):
    model.eval()
    total_loss, total_masked_mae = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            surfaces = batch["surface"].to(device)
            masks = batch["mask_obs"].to(device)

            recon_x, mu, logvar = model(surfaces)

            recon_loss = masked_mse_loss(recon_x, surfaces, masks)
            kl_loss = kl_divergence_loss(mu, logvar)
            tv_loss = total_variation_loss(recon_x)

            loss = recon_loss + beta * kl_loss + tv_lambda * tv_loss
            total_loss += loss.item()

            # Use the inpainting MAE on the validation set for early stopping
            total_masked_mae += masked_mae(recon_x, surfaces, ~masks).item()

    return total_loss / len(dataloader), total_masked_mae / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train a Beta-VAE model.")
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tv", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = IVSurfaceDataset(args.cache, device=device)

    with open(Path(args.splits_dir) / "train.json", "r") as f:
        train_indices = json.load(f)
    with open(Path(args.splits_dir) / "val.json", "r") as f:
        val_indices = json.load(f)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = BetaVAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    logger = TrainingLogger(log_dir="artifacts/logs", run_name=f"beta{args.beta}")

    best_val_mae = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss, recon_loss, kl_loss, tv_loss = train_epoch(
            model, train_loader, optimizer, args.beta, args.tv, device
        )
        val_loss, val_mae = validate_epoch(
            model, val_loader, args.beta, args.tv, device
        )
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Masked MAE: {val_mae:.4f}"
        )

        logger.log_epoch(
            {
                "train_loss": train_loss,
                "recon_loss": recon_loss,
                "kl_loss": kl_loss,
                "tv_loss": tv_loss,
                "val_loss": val_loss,
                "val_masked_mae": val_mae,
                "lr": scheduler.get_last_lr()[0],
            }
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), save_dir / "best.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    logger.save()
    print(f"Training complete. Best model saved to {save_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
