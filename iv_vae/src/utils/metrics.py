import torch
from torch import Tensor
from skimage.metrics import structural_similarity as ssim
import numpy as np


def masked_mae(input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Computes the Mean Absolute Error on the masked region."""
    error = torch.abs(input - target)
    masked_error = error * mask.float()
    return masked_error.sum() / mask.sum().clamp(min=1.0)


def calculate_metrics(pred: np.ndarray, true: np.ndarray, mask_obs: np.ndarray):
    """
    Calculates reconstruction and inpainting metrics.

    Args:
        pred (np.ndarray): Predicted surface.
        true (np.ndarray): Ground truth surface.
        mask_obs (np.ndarray): Mask of observed pixels.

    Returns:
        dict: A dictionary of metrics.
    """
    # Ensure numpy arrays
    if isinstance(pred, Tensor):
        pred = pred.cpu().numpy()
    if isinstance(true, Tensor):
        true = true.cpu().numpy()
    if isinstance(mask_obs, Tensor):
        mask_obs = mask_obs.cpu().numpy()

    # Reconstruction metrics (on the full grid)
    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))
    ssim_val = ssim(pred, true, data_range=true.max() - true.min())

    # Inpainting metrics (on the unobserved grid)
    mask_unobs = ~mask_obs
    masked_mae_val = np.sum(np.abs(pred - true) * mask_unobs) / np.sum(mask_unobs)

    return {
        "recon_mse": mse,
        "recon_mae": mae,
        "recon_ssim": ssim_val,
        "inpaint_mae": masked_mae_val,
    }
