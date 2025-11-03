import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from iv_vae.src.dataio.dataset import IVSurfaceDataset
from iv_vae.src.models.beta_vae import BetaVAE
from iv_vae.src.utils.metrics import calculate_metrics


def evaluate_vae(model, dataset, indices, mask_ratios, device):
    model.eval()
    test_loader = DataLoader(Subset(dataset, indices), batch_size=64, shuffle=False)

    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating VAE"):
            surfaces = batch["surface"].to(device)
            masks_obs = batch["mask_obs"].to(device)
            surfaces_raw = batch["surface_raw"].to(device)

            recon_norm, _, _ = model(surfaces)
            recon_raw = dataset.denormalize(recon_norm)

            # Natural inpainting
            for i in range(surfaces.size(0)):
                metrics = calculate_metrics(
                    recon_raw[i].squeeze().cpu().numpy(),
                    surfaces_raw[i].squeeze().cpu().numpy(),
                    masks_obs[i].squeeze().cpu().numpy(),
                )
                results.append({"model": "BetaVAE", "mask_ratio": "natural", **metrics})

            # Synthetic inpainting
            for ratio in mask_ratios:
                # Create synthetic masks
                synth_masks = masks_obs.clone()
                for i in range(surfaces.size(0)):
                    unobs_indices = torch.argwhere(~synth_masks[i])
                    num_to_hide = int(len(unobs_indices) * ratio)
                    hide_indices = unobs_indices[
                        torch.randperm(len(unobs_indices))[:num_to_hide]
                    ]
                    synth_masks[i][
                        hide_indices[:, 0], hide_indices[:, 1], hide_indices[:, 2]
                    ] = False

                # Re-run model on masked input
                masked_surfaces = surfaces * synth_masks.float()
                recon_norm_synth, _, _ = model(masked_surfaces)
                recon_raw_synth = dataset.denormalize(recon_norm_synth)

                for i in range(surfaces.size(0)):
                    metrics = calculate_metrics(
                        recon_raw_synth[i].squeeze().cpu().numpy(),
                        surfaces_raw[i].squeeze().cpu().numpy(),
                        synth_masks[i].squeeze().cpu().numpy(),
                    )
                    results.append({"model": "BetaVAE", "mask_ratio": ratio, **metrics})

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Beta-VAE model.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--mask_ratios", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = IVSurfaceDataset(args.cache, device=device)

    with open(Path(args.splits_dir) / "test.json", "r") as f:
        test_indices = json.load(f)

    model = BetaVAE(latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    vae_results = evaluate_vae(model, dataset, test_indices, args.mask_ratios, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vae_results.to_csv(out_dir / "vae_summary.csv", index=False)

    print(f"VAE evaluation results saved to {out_dir / 'vae_summary.csv'}")


if __name__ == "__main__":
    main()
