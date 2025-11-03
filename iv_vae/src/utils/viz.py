import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset
import umap

from iv_vae.src.dataio.dataset import IVSurfaceDataset
from iv_vae.src.models.beta_vae import BetaVAE


def plot_surface(ax, data, title, vmin, vmax):
    im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Log-Moneyness (k)")
    ax.set_ylabel("Time-to-Maturity (T)")
    return im


def plot_reconstruction_comparison(dataset, model, indices, out_dir, num_samples=5):
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    for i, idx in enumerate(indices[:num_samples]):
        sample = dataset[idx]
        surface_norm = sample["surface"].unsqueeze(0)
        surface_raw = sample["surface_raw"].squeeze().numpy()

        with torch.no_grad():
            recon_norm, _, _ = model(surface_norm)
        recon_raw = dataset.denormalize(recon_norm).squeeze().cpu().numpy()

        error = np.abs(recon_raw - surface_raw)
        vmin, vmax = surface_raw.min(), surface_raw.max()

        plot_surface(axes[i, 0], surface_raw, f"Original (Day {idx})", vmin, vmax)
        plot_surface(axes[i, 1], recon_raw, "VAE Reconstruction", vmin, vmax)
        im = plot_surface(axes[i, 2], error, "Absolute Error", 0, error.max())

        fig.colorbar(im, ax=axes[i, 2])

    plt.tight_layout()
    plt.savefig(out_dir / "reconstruction_comparison.png")
    plt.close()


def plot_inpainting_triptych(dataset, model, indices, mask_ratios, out_dir):
    for ratio in mask_ratios:
        fig, axes = plt.subplots(len(indices), 3, figsize=(15, 5 * len(indices)))
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            surface_norm = sample["surface"].unsqueeze(0)
            surface_raw = sample["surface_raw"].squeeze().numpy()
            mask_obs = sample["mask_obs"].squeeze()

            # Create synthetic mask
            unobs_indices = torch.argwhere(~mask_obs)
            num_to_hide = int(len(unobs_indices) * ratio)
            hide_indices = unobs_indices[
                torch.randperm(len(unobs_indices))[:num_to_hide]
            ]
            synth_mask = mask_obs.clone()
            synth_mask[hide_indices[:, 0], hide_indices[:, 1]] = False

            masked_surface_norm = surface_norm * synth_mask.float()

            with torch.no_grad():
                recon_norm, _, _ = model(masked_surface_norm)
            recon_raw = dataset.denormalize(recon_norm).squeeze().cpu().numpy()

            error = np.abs(recon_raw - surface_raw)
            vmin, vmax = surface_raw.min(), surface_raw.max()

            plot_surface(
                axes[i, 0],
                surface_raw * synth_mask.numpy(),
                f"Observed (Ratio {ratio})",
                vmin,
                vmax,
            )
            plot_surface(axes[i, 1], recon_raw, "VAE Inpainting", vmin, vmax)
            im = plot_surface(axes[i, 2], error, "Absolute Error", 0, error.max())
            fig.colorbar(im, ax=axes[i, 2])

        plt.tight_layout()
        plt.savefig(out_dir / f"inpainting_triptych_ratio_{ratio}.png")
        plt.close()


def plot_latent_space(dataset, model, indices, out_dir):
    latents = []
    with torch.no_grad():
        for idx in indices:
            surface_norm = dataset[idx]["surface"].unsqueeze(0)
            mu, _ = model.encode(surface_norm)
            latents.append(mu.squeeze().cpu().numpy())
    latents = np.array(latents)

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(latents)

    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=indices, cmap="viridis", s=10)
    plt.colorbar(label="Day Index")
    plt.title("Latent Space Embedding (UMAP)")
    plt.savefig(out_dir / "latent_space_umap.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--mask_ratios", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cpu")  # Visualization on CPU
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = IVSurfaceDataset(args.cache, device=device)

    with open(Path(args.splits_dir) / "test.json", "r") as f:
        test_indices = json.load(f)

    model = BetaVAE(latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    sample_indices = np.random.choice(
        test_indices, args.num_samples, replace=False
    ).tolist()

    print("Generating reconstruction comparison plot...")
    plot_reconstruction_comparison(
        dataset, model, sample_indices, out_dir, args.num_samples
    )

    print("Generating inpainting triptychs...")
    plot_inpainting_triptych(dataset, model, sample_indices, args.mask_ratios, out_dir)

    print("Generating latent space plot...")
    plot_latent_space(dataset, model, test_indices, out_dir)

    print(f"Visualizations saved to {out_dir}")


if __name__ == "__main__":
    main()
