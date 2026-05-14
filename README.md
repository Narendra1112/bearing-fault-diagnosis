# Bearing Fault Diagnosis

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-ff4b4b?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

End-to-end bearing fault diagnosis system using the CWRU (Case Western Reserve University) benchmark dataset. Classifies rolling-element bearing vibration signals into 10 fault categories using hand-crafted features + classical ML and a 1-D CNN, served through a production FastAPI layer with live drift monitoring.

---

## Problem Statement

Undetected bearing faults cause 40–50% of induction motor failures. Vibration-based condition monitoring can detect faults early, but manual inspection is impractical at scale. This project automates fault classification from raw accelerometer signals with >100% test accuracy on held-out CWRU data.

---

## Dataset

**CWRU Bearing Dataset** — 12 kHz Drive End accelerometer recordings at 1797 RPM under 4 operating conditions:

| Fault Type | Severities (inches) | Label IDs |
|------------|---------------------|-----------|
| Normal     | —                   | 0         |
| Ball       | 0.007, 0.014, 0.021 | 1, 2, 3   |
| Inner Race | 0.007, 0.014, 0.021 | 4, 5, 6   |
| Outer Race | 0.007, 0.014, 0.021 | 7, 8, 9   |

Signals are segmented into **1024-sample windows** (50% overlap) and per-window z-score normalised, yielding **5,440 windows** split 70/15/15 into train/val/test sets.

---

## Pipeline

```
data/raw/*.mat
    └─► preprocess.py   — segment + normalise → data/processed/*.npz
            ├─► features.py     — 11 hand-crafted features per window
            │       └─► train_ml.py   — Random Forest + SVM
            └─► train_cnn.py    — 1-D CNN (PyTorch)
                        └─► api.py          — FastAPI inference server
                                └─► dashboard.py  — Streamlit UI
```

### Feature Engineering (11 features)

`RMS`, `Kurtosis`, `Crest Factor`, `Skewness`, `Peak-to-Peak`, `FFT top-5 magnitudes`, `Envelope Spectrum RMS`

### CNN Architecture

| Layer        | Output Shape  |
|--------------|---------------|
| Input        | (1, 1024)     |
| ConvBlock ×4 | (256, 64)     |
| GlobalAvgPool| (256, 1)      |
| FC 256→128   | (128,)        |
| FC 128→10    | (10,)         |

Each ConvBlock: `Conv1d → BatchNorm → ReLU → Dropout → MaxPool`  
Total parameters: **168,490**

---

## Results

| Model         | Test Accuracy |
|---------------|---------------|
| Random Forest | 87.38%        |
| SVM (RBF)     | 85.78%        |
| **1-D CNN**   | **100.00%**   |

CNN training stopped at epoch 17 (best validation accuracy reached at epoch 7).

---

## Installation

```bash
# Clone the repo
git clone https://github.com/Narendra1112/bearing-fault-diagnosis.git
cd bearing-fault-diagnosis

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Download data and train

```bash
# Download CWRU .mat files into data/raw/
python src/download_data.py

# Preprocess → segment + normalise
python src/preprocess.py

# Extract hand-crafted features
python src/features.py

# Train classical ML models
python src/train_ml.py

# Train 1-D CNN
python src/train_cnn.py
```

---

## Running the Dashboard

```bash
streamlit run src/dashboard.py
```

Opens at **http://localhost:8501** with four sections:

- **Raw Signal Viewer** — pick a fault class, plot the vibration signal
- **FFT Spectrum** — frequency-domain view of the selected signal
- **Model Comparison** — bar chart: RF vs SVM vs CNN accuracy
- **Live Predictor** — load a random test window, run CNN inference, compare predicted vs actual

---

## Running the API

```bash
uvicorn src.api:app --port 8000
```

| Endpoint       | Method | Description                              |
|----------------|--------|------------------------------------------|
| `/health`      | GET    | Service status, uptime, model info       |
| `/classes`     | GET    | All 10 fault class names                 |
| `/predict`     | POST   | CNN inference on a 1024-sample window    |
| `/metrics`     | GET    | Rolling summary of last 100 predictions  |
| `/drift`       | GET    | Data-drift status vs training reference  |

Interactive docs: **http://localhost:8000/docs**

### Sample request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"signal": [0.123, -0.456, ...]}'   # 1024 floats
```

### Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

Starts the API on `:8000` and MLflow UI on `:5000`.

---

## Monitoring

- **PredictionMonitor** (`src/monitor.py`) — thread-safe rolling window of the last 100 predictions, backed by MLflow
- **DriftDetector** (`src/drift_detector.py`) — flags incoming signals whose kurtosis, peak-to-peak, or crest factor deviate more than 2 std from the training distribution

---

## Real-World Applications

| Industry          | Use Case                                               |
|-------------------|--------------------------------------------------------|
| Manufacturing     | Predictive maintenance on conveyor and spindle motors  |
| Wind Energy       | Gearbox and generator bearing health monitoring        |
| Railways          | Axle bearing fault detection from on-board sensors     |
| HVAC              | Early fault detection in compressor bearings           |
| Aerospace         | Engine bearing diagnostics during ground tests         |

---

## Project Structure

```
bearing-fault-diagnosis/
├── data/
│   ├── raw/          # CWRU .mat files (git-ignored)
│   └── processed/    # Segmented windows as .npz (git-ignored)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/           # Saved weights (git-ignored)
├── notebooks/
│   └── 01_eda.ipynb
├── outputs/
│   └── figures/      # Confusion matrices, training curves
├── src/
│   ├── api.py
│   ├── dashboard.py
│   ├── download_data.py
│   ├── drift_detector.py
│   ├── features.py
│   ├── load_data.py
│   ├── monitor.py
│   ├── preprocess.py
│   ├── test_api.py
│   ├── train_cnn.py
│   └── train_ml.py
└── requirements.txt
```

---

## License

MIT
