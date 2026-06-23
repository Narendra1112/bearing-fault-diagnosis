#!/bin/bash
set -e

MODEL_PATH="/app/models/best_cnn.pth"
TRAIN_PATH="/app/data/processed/train.npz"

if [ ! -f "$MODEL_PATH" ]; then
    echo "[startup] Downloading best_cnn.pth from HF Model repo..."
    python3 - <<'EOF'
import os
from huggingface_hub import hf_hub_download

dest_dir = "/app/models"
os.makedirs(dest_dir, exist_ok=True)

path = hf_hub_download(
    repo_id="Narendra1112/bearing-fault-cnn",
    repo_type="model",
    filename="best_cnn.pth",
    local_dir=dest_dir,
)
print(f"  -> saved to {path} ({os.path.getsize(path):,} bytes)", flush=True)
EOF
    echo "[startup] Model download complete."
else
    echo "[startup] Model already present at $MODEL_PATH"
fi

if [ ! -f "$TRAIN_PATH" ]; then
    echo "[startup] Downloading train.npz from HF Model repo..."
    python3 - <<'EOF'
import os
from huggingface_hub import hf_hub_download

dest_dir = "/app/data/processed"
os.makedirs(dest_dir, exist_ok=True)

path = hf_hub_download(
    repo_id="Narendra1112/bearing-fault-cnn",
    repo_type="model",
    filename="train.npz",
    local_dir=dest_dir,
)
print(f"  -> saved to {path} ({os.path.getsize(path):,} bytes)", flush=True)
EOF
    echo "[startup] Drift data download complete."
else
    echo "[startup] Train data already present at $TRAIN_PATH"
fi

exec uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 7860 \
    --workers 1 \
    --log-level info
