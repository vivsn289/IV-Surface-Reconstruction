import torch
import pytest
from iv_vae.src.models.losses import masked_mse_loss, total_variation_loss


def test_masked_mse_loss():
    input = torch.ones(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask[:, :, :2, :] = True  # Mask the top half

    loss = masked_mse_loss(input, target, mask)
    assert torch.isclose(loss, torch.tensor(1.0))

    # Test with no mask
    full_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    loss_full = masked_mse_loss(input, target, full_mask)
    assert torch.isclose(loss_full, torch.tensor(1.0))


def test_total_variation_loss():
    img = torch.zeros(1, 1, 4, 4)
    img[:, :, 1, 1] = 1.0

    tv_loss = total_variation_loss(img)
    # Expecting non-zero loss due to the single hot pixel
    assert tv_loss > 0

    # Uniform image should have zero TV loss
    uniform_img = torch.ones(1, 1, 4, 4) * 0.5
    tv_loss_uniform = total_variation_loss(uniform_img)
    assert torch.isclose(tv_loss_uniform, torch.tensor(0.0))
