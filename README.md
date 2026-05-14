---
title: Bearing Fault Diagnosis
emoji: 🔧
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

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
    └─► preprocess.py   — segment + normalise -> data/processed/*.npz
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
| ConvBlock x4 | (256, 64)     |
| GlobalAvgPool| (256, 1)      |
| FC 256->128  | (128,)        |
| FC 128->10   | (10,)         |

Each ConvBlock: `Conv1d -> BatchNorm -> ReLU -> Dropout -> MaxPool`
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

## API Endpoints

| Endpoint       | Method | Description                              |
|----------------|--------|------------------------------------------|
| `/health`      | GET    | Service status, uptime, model info       |
| `/classes`     | GET    | All 10 fault class names                 |
| `/predict`     | POST   | CNN inference on a 1024-sample window    |
| `/metrics`     | GET    | Rolling summary of last 100 predictions  |
| `/drift`       | GET    | Data-drift status vs training reference  |

Interactive docs: **/docs**

### Sample request

```bash
curl -X POST <space-url>/predict \
  -H "Content-Type: application/json" \
  -d '{"signal": [0.123, -0.456, ...]}'   # 1024 floats
```

---

## Installation (local)

```bash
git clone https://github.com/Narendra1112/bearing-fault-diagnosis.git
cd bearing-fault-diagnosis

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

### Download data and train

```bash
python src/download_data.py    # download CWRU .mat files
python src/preprocess.py       # segment + normalise
python src/features.py         # extract features
python src/train_ml.py         # Random Forest + SVM
python src/train_cnn.py        # 1-D CNN
```

### Run locally

```bash
# API
uvicorn src.api:app --port 8000

# Dashboard
streamlit run src/dashboard.py
```

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
├── Dockerfile        # Hugging Face Spaces entry point
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
├── render.yaml       # Render deploy config (reference)
└── requirements.txt
```

---

## License

MIT
