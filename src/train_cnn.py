"""
train_cnn.py — 1-D CNN bearing fault classifier (PyTorch)

Architecture
------------
  Input  : (batch, 1, 1024)  — raw z-score normalised signal window
  Block 1: Conv1d(1,   32, k=7) -> BN -> ReLU -> MaxPool(2)   => (32, 512)
  Block 2: Conv1d(32,  64, k=5) -> BN -> ReLU -> MaxPool(2)   => (64, 256)
  Block 3: Conv1d(64, 128, k=3) -> BN -> ReLU -> MaxPool(2)   => (128, 128)
  Block 4: Conv1d(128,256, k=3) -> BN -> ReLU -> MaxPool(2)   => (256,  64)
  GlobalAveragePooling1D                                       => (256,)
  FC(256 -> 128) -> ReLU -> Dropout(0.4)
  FC(128 ->  10)                                               => logits

Training
--------
  Loss      : weighted CrossEntropyLoss  (handles class imbalance)
  Optimiser : Adam  lr=1e-3, weight_decay=1e-4
  Scheduler : CosineAnnealingLR
  Epochs    : up to 50, early stopping patience=10 on val accuracy
  Checkpoint: models/best_cnn.pth  (best val-acc weights)

Run:
    python src/train_cnn.py
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

ROOT        = Path(__file__).resolve().parent.parent
MODELS_DIR  = ROOT / "models"
FIGURES_DIR = ROOT / "outputs" / "figures"
sys.path.insert(0, str(ROOT / "src"))

from preprocess import load_processed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "normal",
    "ball_0.007", "ball_0.014", "ball_0.021",
    "ir_0.007",   "ir_0.014",   "ir_0.021",
    "or_0.007",   "or_0.014",   "or_0.021",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    """Wraps (N, 1024) signal windows + integer labels for DataLoader."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Add channel dim: (N, 1024) -> (N, 1, 1024)
        self.X = torch.from_numpy(X[:, np.newaxis, :].astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d(2)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float = 0.0):
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
    """
    4-block 1-D CNN for 10-class bearing fault diagnosis.
    Accepts input shape (batch, 1, 1024).
    """

    def __init__(self, n_classes: int = 10, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   32,  kernel=7, dropout=0.1),   # -> (32,  512)
            ConvBlock(32,  64,  kernel=5, dropout=0.1),   # -> (64,  256)
            ConvBlock(64,  128, kernel=3, dropout=0.2),   # -> (128, 128)
            ConvBlock(128, 256, kernel=3, dropout=0.2),   # -> (256,  64)
        )
        # Collapse time dimension regardless of input length
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Class weights for imbalanced dataset
# ---------------------------------------------------------------------------

def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """
    Inverse-frequency weighting: w_c = N / (n_classes * count_c).
    Upweights rare fault classes relative to the abundant normal class.
    """
    counts = Counter(y.tolist())
    total  = len(y)
    weights = np.array([
        total / (n_classes * counts.get(c, 1))
        for c in range(n_classes)
    ], dtype=np.float32)
    return torch.from_numpy(weights).to(DEVICE)


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------

def _run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
) -> tuple[float, float]:
    """Forward (+ backward if optimizer given). Returns (avg_loss, accuracy)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = correct = total = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            correct    += (logits.argmax(dim=1) == y_batch).sum().item()
            total      += len(y_batch)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Training curve plot
# ---------------------------------------------------------------------------

def _plot_history(history: dict, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ep = range(1, len(history["train_loss"]) + 1)

    ax1.plot(ep, history["train_loss"], label="Train")
    ax1.plot(ep, history["val_loss"],   label="Val")
    ax1.set_title("Loss per epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ep, [a * 100 for a in history["train_acc"]], label="Train")
    ax2.plot(ep, [a * 100 for a in history["val_acc"]],   label="Val")
    ax2.set_title("Accuracy per epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("BearingCNN — Training History", fontsize=13)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cnn_training_history.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Training curves saved -> {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(
    cm:          np.ndarray,
    class_names: list,
    out_dir:     Path,
) -> None:
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("BearingCNN — Confusion Matrix (normalised)", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cm_CNN.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved -> {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_cnn(
    epochs:       int   = 50,
    batch_size:   int   = 128,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    patience:     int   = 10,
    n_classes:    int   = 10,
) -> BearingCNN:

    # ---- Data ---------------------------------------------------------------
    print("Loading processed splits...")
    X_train, y_train = load_processed("train")
    X_val,   y_val   = load_processed("val")
    X_test,  y_test  = load_processed("test")

    print(f"  Train : {X_train.shape}  Val : {X_val.shape}  Test : {X_test.shape}")
    print(f"  Device: {DEVICE}\n")

    train_loader = DataLoader(
        WindowDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        WindowDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        WindowDataset(X_test, y_test),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # ---- Model / loss / optimiser ------------------------------------------
    model     = BearingCNN(n_classes=n_classes).to(DEVICE)
    weights   = compute_class_weights(y_train, n_classes)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # ---- Print model summary ------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"BearingCNN  |  trainable parameters: {total_params:,}")
    print("=" * 72)
    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Train Acc':>10}  "
          f"{'Val Loss':>9}  {'Val Acc':>9}  {'LR':>9}")
    print("=" * 72)

    # ---- Training loop ------------------------------------------------------
    ckpt_path      = MODELS_DIR / "best_cnn.pth"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc   = -1.0
    patience_count = 0
    history        = {"train_loss": [], "train_acc": [],
                      "val_loss":   [], "val_acc":   []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = _run_epoch(model, val_loader,   criterion, None)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        improved = "*" if va_acc > best_val_acc else " "
        print(
            f"{epoch:>6}  {tr_loss:>11.4f}  {tr_acc*100:>9.2f}%  "
            f"{va_loss:>9.4f}  {va_acc*100:>8.2f}%  {current_lr:>9.2e}  {improved}"
        )

        if va_acc > best_val_acc:
            best_val_acc   = va_acc
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no val improvement for {patience} epochs).")
                break

    print("=" * 72)
    print(f"\n  Best val accuracy : {best_val_acc*100:.2f}%")
    print(f"  Checkpoint saved  -> {ckpt_path.relative_to(ROOT)}")

    # ---- Plots --------------------------------------------------------------
    _plot_history(history, FIGURES_DIR)

    # ---- Test evaluation ----------------------------------------------------
    print("\n" + "=" * 72)
    print("  Test-set evaluation (best checkpoint)")
    print("=" * 72)

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch.to(DEVICE))
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_true.extend(y_batch.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_true)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n  Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")

    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
    )
    print("\n  Classification Report:\n")
    for line in report.splitlines():
        print("    " + line)

    # Confusion matrix — text
    cm = confusion_matrix(y_true, y_pred)
    w  = max(len(n) for n in CLASS_NAMES) + 2
    print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
    header = " " * w + "  ".join(f"{n:>{w}}" for n in CLASS_NAMES)
    print("  " + header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>{w}}" for v in row)
        print(f"  {CLASS_NAMES[i]:>{w}}  {row_str}")

    # Confusion matrix — PNG
    _plot_confusion_matrix(cm, CLASS_NAMES, FIGURES_DIR)

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_cnn(
        epochs=50,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-4,
        patience=10,
        n_classes=10,
    )
