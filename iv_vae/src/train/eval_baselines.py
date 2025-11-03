import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Subset

from iv_vae.src.dataio.dataset import IVSurfaceDataset
from iv_vae.src.models.bicubic_baseline import bicubic_interpolation
from iv_vae.src.models.pca_baseline import (
    iterative_pca_imputation,
    apply_pca_reconstruction,
)
from iv_vae.src.utils.metrics import calculate_metrics


def evaluate_bicubic(dataset, indices, mask_ratios):
    results = []
    for idx in tqdm(indices, desc="Evaluating Bicubic"):
        sample = dataset[idx]
        surface_raw = sample["surface_raw"].squeeze().numpy()
        mask_obs = sample["mask_obs"].squeeze().numpy()

        # Natural inpainting
        recon = bicubic_interpolation(surface_raw * mask_obs, mask_obs)
        metrics = calculate_metrics(recon, surface_raw, mask_obs)
        results.append({"model": "Bicubic", "mask_ratio": "natural", **metrics})

        # Synthetic inpainting
        for ratio in mask_ratios:
            # Create synthetic mask
            unobs_indices = np.argwhere(~mask_obs)
            np.random.shuffle(unobs_indices)
            num_to_hide = int(len(unobs_indices) * ratio)
            synth_mask = np.copy(mask_obs)
            hide_indices = tuple(unobs_indices[:num_to_hide].T)
            synth_mask[hide_indices] = False

            recon = bicubic_interpolation(surface_raw * synth_mask, synth_mask)
            metrics = calculate_metrics(recon, surface_raw, synth_mask)
            results.append({"model": "Bicubic", "mask_ratio": ratio, **metrics})

    return pd.DataFrame(results)


def evaluate_pca(dataset, train_indices, test_indices, components_list, mask_ratios):
    train_surfaces = dataset.surfaces[train_indices].squeeze().numpy()
    train_masks = dataset.masks[train_indices].squeeze().numpy()

    test_surfaces = dataset.surfaces[test_indices].squeeze().numpy()
    test_masks = dataset.masks[test_indices].squeeze().numpy()

    results = []
    for k in components_list:
        print(f"Training PCA with {k} components...")
        pca = iterative_pca_imputation(train_surfaces, train_masks, n_components=k)

        # Natural inpainting
        recon = apply_pca_reconstruction(test_surfaces, test_masks, pca)
        for i in tqdm(range(len(test_surfaces)), desc=f"Evaluating PCA (k={k})"):
            metrics = calculate_metrics(recon[i], test_surfaces[i], test_masks[i])
            results.append({"model": f"PCA-{k}", "mask_ratio": "natural", **metrics})

        # Synthetic inpainting
        for ratio in mask_ratios:
            # Create synthetic masks for the whole test set
            synth_masks = []
            for mask in test_masks:
                unobs_indices = np.argwhere(~mask)
                np.random.shuffle(unobs_indices)
                num_to_hide = int(len(unobs_indices) * ratio)
                synth_mask = np.copy(mask)
                hide_indices = tuple(unobs_indices[:num_to_hide].T)
                synth_mask[hide_indices] = False
                synth_masks.append(synth_mask)
            synth_masks = np.array(synth_masks)

            recon = apply_pca_reconstruction(
                test_surfaces * synth_masks, synth_masks, pca
            )
            for i in range(len(test_surfaces)):
                metrics = calculate_metrics(recon[i], test_surfaces[i], synth_masks[i])
                results.append({"model": f"PCA-{k}", "mask_ratio": ratio, **metrics})

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models.")
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--mask_ratios", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    parser.add_argument("--pca_components", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    dataset = IVSurfaceDataset(args.cache)

    with open(Path(args.splits_dir) / "train.json", "r") as f:
        train_indices = json.load(f)
    with open(Path(args.splits_dir) / "test.json", "r") as f:
        test_indices = json.load(f)

    bicubic_results = evaluate_bicubic(dataset, test_indices, args.mask_ratios)
    pca_results = evaluate_pca(
        dataset, train_indices, test_indices, args.pca_components, args.mask_ratios
    )

    all_results = pd.concat([bicubic_results, pca_results])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(out_dir / "baselines_summary.csv", index=False)

    print(f"Baseline results saved to {out_dir / 'baselines_summary.csv'}")


if __name__ == "__main__":
    main()
