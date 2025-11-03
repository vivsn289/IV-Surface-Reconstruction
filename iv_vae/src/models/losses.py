import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


def masked_mse_loss(input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Computes the Mean Squared Error on the masked region."""
    error = (input - target).pow(2)
    masked_error = error * mask.float()
    return masked_error.sum() / mask.sum().clamp(min=1.0)


def kl_divergence_loss(mu: Tensor, logvar: Tensor) -> Tensor:
    """Computes the KL divergence between the posterior q(z|x) and a standard Gaussian prior p(z)."""
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def total_variation_loss(img: Tensor) -> Tensor:
    """Computes the isotropic Total Variation loss."""
    # Assumes img is a batch of images, shape (B, C, H, W)
    pixel_diff_h = img[:, :, 1:, :] - img[:, :, :-1, :]
    pixel_diff_w = img[:, :, :, 1:] - img[:, :, :, :-1]

    tv_h = torch.sum(pixel_diff_h.pow(2))
    tv_w = torch.sum(pixel_diff_w.pow(2))

    return tv_h + tv_w


# Optional: Proxy violation counters
def calendar_monotonicity_proxy(surface: Tensor, epsilon: float = 0.01) -> Tensor:
    """Counts violations where IV(T+dT) + eps < IV(T)."""
    violations = F.relu(surface[:, :, :-1, :] - surface[:, :, 1:, :] - epsilon)
    return (violations > 0).sum()


def convexity_in_k_proxy(surface: Tensor, epsilon: float = 0.01) -> Tensor:
    """Counts violations of convexity in the log-moneyness dimension."""
    # Discrete second derivative: f(x-1) - 2f(x) + f(x+1)
    second_diff = (
        surface[:, :, :, 2:] - 2 * surface[:, :, :, 1:-1] + surface[:, :, :, :-2]
    )
    violations = F.relu(-second_diff - epsilon)
    return (violations > 0).sum()
