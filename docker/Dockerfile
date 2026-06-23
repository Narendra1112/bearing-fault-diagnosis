# ── Bearing Fault Diagnosis — FastAPI inference service ──────────────────────
# Build context must be the PROJECT ROOT (one level above docker/):
#   docker build -f docker/Dockerfile -t bearing-api .
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# System deps for scipy / PyWavelets native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python deps first (layer-cached unless requirements change) ──────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project source ───────────────────────────────────────────────────────
# Only what the API needs at runtime — no notebooks, no raw .mat files
COPY src/           ./src/
COPY models/        ./models/
# data/processed/ is git-ignored; create empty dir so the app starts cleanly.
RUN mkdir -p ./data/processed

# ── Runtime config ────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=/mlruns

EXPOSE 8000

# Single worker: in-process drift CUSUM state and Prometheus metrics are not
# shared across workers. Scale horizontally behind a load balancer instead,
# with a shared Prometheus pushgateway / multiprocess mode if needed.
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
