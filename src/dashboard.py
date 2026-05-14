"""
dashboard.py — Streamlit bearing fault diagnosis dashboard

Sections
--------
  1. Raw Signal Viewer  — pick a fault class, plot raw vibration window
  2. FFT Spectrum        — frequency spectrum of the selected window
  3. Model Comparison    — bar chart: RF vs SVM vs CNN accuracy
  4. Live Predictor      — random test window through CNN, predicted vs actual

Run:
    streamlit run src/dashboard.py
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bearing Fault Diagnosis",
    page_icon="⚙",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FS = 12_000   # Drive End sampling rate (Hz)

CLASS_NAMES = [
    "normal",
    "ball_0.007", "ball_0.014", "ball_0.021",
    "ir_0.007",   "ir_0.014",   "ir_0.021",
    "or_0.007",   "or_0.014",   "or_0.021",
]

MODEL_RESULTS = {
    "Random Forest": 87.38,
    "SVM (RBF)":     85.78,
    "1D-CNN":       100.00,
}


# ---------------------------------------------------------------------------
# CNN model definition (must match train_cnn.py exactly)
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x):
        return self.block(x)


class BearingCNN(nn.Module):
    def __init__(self, n_classes=10, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   32,  kernel=7, dropout=0.1),
            ConvBlock(32,  64,  kernel=5, dropout=0.1),
            ConvBlock(64,  128, kernel=3, dropout=0.2),
            ConvBlock(128, 256, kernel=3, dropout=0.2),
        )
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


# ---------------------------------------------------------------------------
# Cached data & model loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading test data...")
def load_test_data():
    path = ROOT / "data" / "processed" / "test.npz"
    d = np.load(path)
    return d["X"], d["y"]


@st.cache_resource(show_spinner="Loading CNN model...")
def load_cnn_model():
    ckpt = ROOT / "models" / "best_cnn.pth"
    model = BearingCNN(n_classes=10)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Plot helpers  (return Figure objects — st.pyplot consumes them)
# ---------------------------------------------------------------------------
def plot_signal(window: np.ndarray, title: str) -> plt.Figure:
    t = np.arange(len(window)) / FS * 1000   # ms
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, window, linewidth=0.7, color="#1f77b4")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (normalised)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_fft(window: np.ndarray, title: str) -> plt.Figure:
    n    = len(window)
    win  = np.hanning(n)
    amps = np.abs(np.fft.rfft(window.astype(np.float64) * win)) * (2.0 / n)
    freq = np.fft.rfftfreq(n, d=1.0 / FS)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(freq, amps, linewidth=0.7, color="#ff7f0e")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim(0, FS / 2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_model_comparison() -> plt.Figure:
    names  = list(MODEL_RESULTS.keys())
    accs   = list(MODEL_RESULTS.values())
    colors = ["#aec7e8", "#aec7e8", "#1f77b4"]   # highlight CNN

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, accs, color=colors, edgecolor="white", width=0.5)

    # Annotate each bar
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{acc:.2f}%",
            ha="center", va="bottom", fontweight="bold", fontsize=11,
        )

    ax.set_ylim(80, 103)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Model Comparison — CWRU Bearing Fault (10 classes)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_probabilities(proba: np.ndarray, predicted: int, actual: int) -> plt.Figure:
    colors = []
    for i in range(len(CLASS_NAMES)):
        if i == actual and i == predicted:
            colors.append("#2ca02c")   # green — correct
        elif i == predicted:
            colors.append("#d62728")   # red — wrong prediction
        elif i == actual:
            colors.append("#ff7f0e")   # orange — missed actual
        else:
            colors.append("#aec7e8")   # grey

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.barh(CLASS_NAMES, proba * 100, color=colors, edgecolor="white")
    ax.set_xlabel("Confidence (%)")
    ax.set_title("CNN Prediction Confidence")
    ax.set_xlim(0, 105)
    ax.grid(True, axis="x", alpha=0.3)

    for i, (p, name) in enumerate(zip(proba, CLASS_NAMES)):
        if p > 0.005:
            ax.text(p * 100 + 0.5, i, f"{p*100:.1f}%", va="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#2ca02c", label="Correct prediction"),
        Patch(color="#d62728", label="Wrong prediction"),
        Patch(color="#ff7f0e", label="Missed actual class"),
        Patch(color="#aec7e8", label="Other"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main dashboard layout
# ---------------------------------------------------------------------------
st.title("Bearing Fault Diagnosis Dashboard")
st.caption("CWRU Dataset  |  12 kHz Drive End  |  10-class fault classification")
st.markdown("---")

# Load resources once
X_test, y_test = load_test_data()
cnn_model      = load_cnn_model()

# ============================================================
# Section 1 — Raw Signal Viewer
# ============================================================
st.header("1  Raw Signal Viewer")

col_ctrl, col_info = st.columns([1, 2])
with col_ctrl:
    selected_class = st.selectbox(
        "Select fault class",
        options=list(range(len(CLASS_NAMES))),
        format_func=lambda i: f"{i} — {CLASS_NAMES[i]}",
    )
    # Pick a random window of the chosen class
    class_indices = np.where(y_test == selected_class)[0]
    if len(class_indices) == 0:
        st.warning(f"No test windows for class {selected_class}.")
        st.stop()

    if st.button("Shuffle window", key="shuffle_signal"):
        st.session_state["window_idx"] = int(
            np.random.choice(class_indices)
        )
    if "window_idx" not in st.session_state or \
            y_test[st.session_state["window_idx"]] != selected_class:
        st.session_state["window_idx"] = int(class_indices[0])

    window_idx = st.session_state["window_idx"]

with col_info:
    st.metric("Selected class", CLASS_NAMES[selected_class])
    st.metric("Window index (in test set)", window_idx)
    st.metric("Windows available for this class", len(class_indices))

selected_window = X_test[window_idx]
st.pyplot(plot_signal(selected_window, f"Raw window — {CLASS_NAMES[selected_class]}"))

# ============================================================
# Section 2 — FFT Spectrum
# ============================================================
st.markdown("---")
st.header("2  FFT Spectrum")
st.pyplot(plot_fft(selected_window, f"Amplitude Spectrum — {CLASS_NAMES[selected_class]}"))

col_a, col_b, col_c = st.columns(3)
fft_amps = np.abs(np.fft.rfft(selected_window.astype(np.float64))) * (2.0 / len(selected_window))
fft_freq = np.fft.rfftfreq(len(selected_window), d=1.0 / FS)
peak_idx  = np.argmax(fft_amps)
col_a.metric("Peak frequency",  f"{fft_freq[peak_idx]:.1f} Hz")
col_b.metric("Peak amplitude",  f"{fft_amps[peak_idx]:.4f}")
col_c.metric("Signal RMS",      f"{np.sqrt(np.mean(selected_window**2)):.4f}")

# ============================================================
# Section 3 — Model Comparison
# ============================================================
st.markdown("---")
st.header("3  Model Comparison")

col_chart, col_table = st.columns([2, 1])
with col_chart:
    st.pyplot(plot_model_comparison())
with col_table:
    st.markdown("#### Accuracy on 816 test windows")
    st.markdown("")
    for model_name, acc in MODEL_RESULTS.items():
        delta = f"+{acc - min(MODEL_RESULTS.values()):.2f}%" \
                if acc != min(MODEL_RESULTS.values()) else "baseline"
        st.metric(model_name, f"{acc:.2f}%", delta=delta)

# ============================================================
# Section 4 — Live Predictor
# ============================================================
st.markdown("---")
st.header("4  Live Predictor")
st.markdown(
    "Pick any test window, run it through the saved CNN, "
    "and compare the prediction against the ground-truth label."
)

pred_col, result_col = st.columns([1, 2])

with pred_col:
    pred_class_filter = st.selectbox(
        "Filter by true class (or 'All')",
        options=["All"] + CLASS_NAMES,
        key="pred_filter",
    )
    if pred_class_filter == "All":
        candidate_indices = np.arange(len(y_test))
    else:
        filt_id = CLASS_NAMES.index(pred_class_filter)
        candidate_indices = np.where(y_test == filt_id)[0]

    if st.button("Run on random window", type="primary", key="run_pred"):
        st.session_state["pred_idx"] = int(np.random.choice(candidate_indices))

    if "pred_idx" not in st.session_state:
        st.session_state["pred_idx"] = int(candidate_indices[0])

    pred_idx    = st.session_state["pred_idx"]
    pred_window = X_test[pred_idx]
    true_label  = int(y_test[pred_idx])

    # CNN inference
    x_tensor = torch.from_numpy(
        pred_window[np.newaxis, np.newaxis, :].astype(np.float32)
    )
    with torch.no_grad():
        logits = cnn_model(x_tensor)
        proba  = torch.softmax(logits, dim=1).numpy()[0]

    pred_label = int(np.argmax(proba))
    correct    = pred_label == true_label

    st.metric("Test window index", pred_idx)
    st.metric("True class",        CLASS_NAMES[true_label])
    st.metric(
        "Predicted class",
        CLASS_NAMES[pred_label],
        delta="Correct" if correct else "Wrong",
        delta_color="normal" if correct else "inverse",
    )
    st.metric("Confidence", f"{proba[pred_label]*100:.1f}%")

with result_col:
    st.pyplot(plot_probabilities(proba, pred_label, true_label))
    st.pyplot(plot_signal(pred_window, f"Input window (test idx={pred_idx})"))

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "CWRU Bearing Fault Diagnosis  |  "
    "PyTorch 1D-CNN  |  scikit-learn RF & SVM  |  "
    "Built with Streamlit"
)
