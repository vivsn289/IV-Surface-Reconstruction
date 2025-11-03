import torch
import pytest
from iv_vae.src.models.beta_vae import BetaVAE
from iv_vae.src.models.losses import masked_mse_loss, kl_divergence_loss


def test_train_step():
    model = BetaVAE(latent_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create a synthetic batch
    surface = torch.rand(4, 1, 64, 64)
    mask = torch.ones(4, 1, 64, 64, dtype=torch.bool)

    # Initial forward pass
    recon_x, mu, logvar = model(surface)
    initial_loss = masked_mse_loss(recon_x, surface, mask) + kl_divergence_loss(
        mu, logvar
    )

    # Perform a training step
    optimizer.zero_grad()
    initial_loss.backward()
    optimizer.step()

    # Forward pass after one step
    recon_x_after, mu_after, logvar_after = model(surface)
    loss_after = masked_mse_loss(recon_x_after, surface, mask) + kl_divergence_loss(
        mu_after, logvar_after
    )

    # Loss should decrease after one optimization step
    assert loss_after < initial_loss

    # Check shapes
    assert recon_x.shape == surface.shape
    assert mu.shape == (4, 8)
    assert logvar.shape == (4, 8)
