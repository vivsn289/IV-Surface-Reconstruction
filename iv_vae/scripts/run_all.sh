#!/bin/bash
set -e

# Variables
DATA_GLOB="data/spy_options_data_*.json"
CACHE="artifacts/cache/spy_iv_cache.pt"
SPLITS_DIR="artifacts/splits"
CKPT_DIR="artifacts/ckpts/beta2"
TABLES_DIR="artifacts/tables"
FIGS_DIR="artifacts/figs"
BETA=2

# 1. Build cache
echo "--- Building cache ---"
python -m iv_vae.src.dataio.parse_json \
  --input_glob "$DATA_GLOB" \
  --kmin -0.5 --kmax 0.5 --tmin_days 7 --tmax_days 365 \
  --grid 64 --out "$CACHE"

# 2. Create splits
echo "--- Creating splits ---"
python -m iv_vae.src.dataio.dataset \
  --cache "$CACHE" \
  --make_splits --train 0.70 --val 0.15 --test 0.15 \
  --out_dir "$SPLITS_DIR"

# 3. Train VAE
echo "--- Training VAE (beta=$BETA) ---"
python -m iv_vae.src.train.train_vae \
  --cache "$CACHE" \
  --splits_dir "$SPLITS_DIR" \
  --beta $BETA --latent_dim 32 --epochs 200 \
  --batch_size 64 --lr 1e-3 --tv 1e-6 \
  --save_dir "$CKPT_DIR"

# 4. Evaluate VAE
echo "--- Evaluating VAE ---"
python -m iv_vae.src.train.eval_vae \
  --ckpt "$CKPT_DIR/best.pt" \
  --cache "$CACHE" \
  --splits_dir "$SPLITS_DIR" \
  --mask_ratios 0.2 0.5 0.8 \
  --out_dir "$TABLES_DIR"

# 5. Run baselines
echo "--- Evaluating baselines ---"
python -m iv_vae.src.train.eval_baselines \
  --cache "$CACHE" \
  --splits_dir "$SPLITS_DIR" \
  --mask_ratios 0.2 0.5 0.8 \
  --pca_components 8 16 32 \
  --out_dir "$TABLES_DIR"

# 6. Generate figures
echo "--- Generating figures ---"
python -m iv_vae.src.utils.viz \
  --ckpt "$CKPT_DIR/best.pt" \
  --cache "$CACHE" \
  --splits_dir "$SPLITS_DIR" \
  --mask_ratios 0.2 0.5 0.8 \
  --out_dir "$FIGS_DIR"

# 7. Combine results
echo "--- Combining results ---"
python -c "import pandas as pd; pd.concat([pd.read_csv('$TABLES_DIR/vae_summary.csv'), pd.read_csv('$TABLES_DIR/baselines_summary.csv')]).to_csv('$TABLES_DIR/summary.csv', index=False)"

echo "--- All steps completed successfully ---"
