# β-VAE for Implied Volatility Surface Reconstruction

This repository contains a production-ready implementation of a mask-aware β-Variational Autoencoder (β-VAE) to reconstruct and inpaint implied volatility (IV) surfaces. The project treats IV surfaces as 64x64 images and compares the β-VAE's performance against Bicubic and PCA baselines.

## Project Structure

```
iv_vae/
  ├── README.md
  ├── pyproject.toml
  ├── Makefile
  ├── configs/
  │   └── default.yaml
  ├── data/
  ├── src/
  │   ├── dataio/
  │   ├── models/
  │   ├── train/
  │   └── utils/
  ├── scripts/
  │   └── run_all.sh
  └── artifacts/
      ├── cache/
      ├── splits/
      ├── ckpts/
      ├── logs/
      ├── figs/
      └── tables/
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd iv_vae
    ```

2.  **Create a Python environment and install dependencies:**
    This project uses Poetry for dependency management.
    ```bash
    pip install poetry
    poetry install
    ```
    Alternatively, you can create a virtual environment and install from `requirements.txt` if provided.

## Dataset

The dataset consists of daily SPY options quotes in JSON format, located in the `data/` directory. The data is parsed, adapted to a unified schema, and gridded to 64x64 surfaces representing log-moneyness vs. time-to-maturity.

## Running the Pipeline

You can run the entire pipeline using either the `Makefile` or the `run_all.sh` script.

**Using Make:**
```bash
make all
```

**Using the shell script:**
```bash
bash scripts/run_all.sh
```

This will execute the following steps sequentially:
1.  **Build Cache:** Parses raw JSONs and creates a cached `.pt` file.
2.  **Create Splits:** Generates train/validation/test splits.
3.  **Train β-VAE:** Trains the β-VAE model with β=2.
4.  **Evaluate Baselines:** Runs Bicubic and PCA baselines.
5.  **Evaluate β-VAE:** Evaluates the trained VAE model.
6.  **Generate Figures:** Creates visualizations of the results.
7.  **Summarize:** Combines all evaluation results into a single CSV.

## Outputs

All outputs are saved in the `artifacts/` directory:
-   `cache/`: Cached PyTorch dataset.
-   `splits/`: JSON files with train/val/test indices.
-   `ckpts/`: Trained model checkpoints.
-   `logs/`: Training logs in CSV format.
-   `figs/`: Generated plots, including:
    -   Reconstruction comparisons.
    -   Inpainting triptychs for different mask ratios.
    -   UMAP visualization of the latent space.
-   `tables/`: Evaluation results in CSV format, with a final `summary.csv`.

## Sample Figures

**Reconstruction Comparison:**
*(This would be an embedded image in a real README)*

**Inpainting Triptych (Mask Ratio 0.5):**
*(This would be an embedded image in a real README)*

**Latent Space:**
*(This would be an embedded image in a real README)*
