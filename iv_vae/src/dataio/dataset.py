import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import Tuple, Dict, List


class IVSurfaceDataset(Dataset):
    def __init__(self, cache_path: str, device: str = "cpu"):
        data = torch.load(cache_path, map_location=device)
        self.surfaces = data["surfaces"]
        self.masks = data["masks"]

        self.v_min, self.v_max = self.compute_norm_stats()
        self.normalized_surfaces = self.normalize(self.surfaces)

    def compute_norm_stats(self) -> Tuple[float, float]:
        # Important: Use only observed values for normalization stats
        observed_values = self.surfaces[self.masks]
        return observed_values.min().item(), observed_values.max().item()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.v_min) / (self.v_max - self.v_min)

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * (self.v_max - self.v_min) + self.v_min

    def __len__(self):
        return len(self.surfaces)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "surface": self.normalized_surfaces[idx],
            "mask_obs": self.masks[idx],
            "surface_raw": self.surfaces[idx],
        }


def create_splits(
    n_samples: int, train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0)
    indices = np.arange(n_samples)

    # Reproducible shuffle
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    train_indices = indices[:train_end].tolist()
    val_indices = indices[train_end:val_end].tolist()
    test_indices = indices[val_end:].tolist()

    return train_indices, val_indices, test_indices


def main():
    parser = argparse.ArgumentParser(description="Create and save dataset splits.")
    parser.add_argument(
        "--cache", type=str, required=True, help="Path to the cached .pt dataset."
    )
    parser.add_argument(
        "--make_splits", action="store_true", help="If set, create and save splits."
    )
    parser.add_argument("--train", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument(
        "--val", type=float, default=0.15, help="Validation split ratio."
    )
    parser.add_argument("--test", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits.")
    parser.add_argument("--out_dir", type=str, help="Directory to save split files.")
    args = parser.parse_args()

    if args.make_splits:
        if not args.out_dir:
            raise ValueError(
                "Output directory --out_dir must be provided to save splits."
            )

        dataset = IVSurfaceDataset(args.cache)
        train_idx, val_idx, test_idx = create_splits(
            len(dataset), args.train, args.val, args.test, args.seed
        )

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "train.json", "w") as f:
            json.dump(train_idx, f)
        with open(out_dir / "val.json", "w") as f:
            json.dump(val_idx, f)
        with open(out_dir / "test.json", "w") as f:
            json.dump(test_idx, f)

        print(f"Saved splits to {out_dir}")
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")


if __name__ == "__main__":
    main()
