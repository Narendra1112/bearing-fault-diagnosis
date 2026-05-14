# ── Bearing Fault Diagnosis — Hugging Face Spaces ────────────────────────────
# HF Spaces requires the app to listen on port 7860.
# Build context: project root
#   docker build -t bearing-hf .
# -----------------------------------------------------------------------------

FROM python:3.11-slim

# System deps for scipy / PyWavelets / librosa native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (CPU-only torch to keep image under HF 10 GB limit) ──────────
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# ── Copy only what the API needs at runtime ───────────────────────────────────
COPY src/            ./src/
COPY models/         ./models/
COPY data/processed/ ./data/processed/

# ── Runtime config ────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=/tmp/mlruns

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "src.api:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
