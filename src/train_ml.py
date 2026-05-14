"""
train_ml.py — Classical ML classifier training and evaluation

Models  : Random Forest, SVM (RBF kernel)
Input   : hand-crafted feature matrices from data/processed/*_features.npz
Output  : accuracy + confusion matrix printed; best model → models/best_ml_model.pkl

Run:
    python src/train_ml.py
"""

import sys
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT        = Path(__file__).resolve().parent.parent
MODELS_DIR  = ROOT / "models"
FIGURES_DIR = ROOT / "outputs" / "figures"
sys.path.insert(0, str(ROOT / "src"))

from features import load_features, FEATURE_NAMES

# Class names in class-id order (matches preprocess.py LABEL_ENCODING)
CLASS_NAMES = [
    "normal",
    "ball_0.007", "ball_0.014", "ball_0.021",
    "ir_0.007",   "ir_0.014",   "ir_0.021",
    "or_0.007",   "or_0.014",   "or_0.021",
]


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_models() -> dict:
    """Return a dict of {name: estimator} to train and compare."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "SVM": Pipeline([
            # SVM is not scale-invariant; StandardScaler is mandatory here
            ("scaler", StandardScaler()),
            ("clf",    SVC(
                C=10.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=42,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _print_confusion_matrix(cm: np.ndarray, class_names: list, model_name: str) -> None:
    """Pretty-print a confusion matrix to stdout."""
    w = max(len(n) for n in class_names) + 2
    header = " " * w + "  ".join(f"{n:>{w}}" for n in class_names)
    print(f"\n  Confusion matrix — {model_name}")
    print("  " + header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>{w}}" for v in row)
        print(f"  {class_names[i]:>{w}}  {row_str}")


def _save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: list,
    model_name: str,
    out_dir: Path,
) -> None:
    """Save a normalised confusion-matrix heatmap as a PNG."""
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"{model_name} — Confusion Matrix (normalised)", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cm_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix plot saved -> {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_and_evaluate() -> None:
    # ---- Load features -----------------------------------------------------
    print("Loading feature matrices...")
    X_train, y_train = load_features("train")
    X_val,   y_val   = load_features("val")
    X_test,  y_test  = load_features("test")

    # Merge train + val for final fit (val was only needed for hyper-param tuning)
    X_fit = np.concatenate([X_train, X_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)

    print(f"  Train+Val : {X_fit.shape}   Test : {X_test.shape}")
    print(f"  Features  : {FEATURE_NAMES}\n")

    models      = build_models()
    results     = {}
    best_acc    = -1.0
    best_name   = None
    best_model  = None

    # ---- Train & evaluate each model ---------------------------------------
    for name, model in models.items():
        print("=" * 60)
        print(f"  Model : {name}")
        print("=" * 60)

        print(f"  Training on {len(X_fit):,} windows...")
        model.fit(X_fit, y_fit)

        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        cm     = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=CLASS_NAMES,
            zero_division=0,
        )

        print(f"\n  Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"\n  Classification Report:\n")
        # Indent the report for readability
        for line in report.splitlines():
            print("    " + line)

        _print_confusion_matrix(cm, CLASS_NAMES, name)
        _save_confusion_matrix_plot(cm, CLASS_NAMES, name, FIGURES_DIR)

        # Save every model individually
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{name}.pkl"
        joblib.dump(model, model_path)
        print(f"  Model saved -> {model_path.relative_to(ROOT)}")

        results[name] = {"accuracy": acc, "model": model}

        if acc > best_acc:
            best_acc   = acc
            best_name  = name
            best_model = model

    # ---- Save best model ---------------------------------------------------
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, res in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
        marker = "  <-- BEST" if name == best_name else ""
        print(f"  {name:<20s}  accuracy = {res['accuracy']:.4f}{marker}")

    best_path = MODELS_DIR / "best_ml_model.pkl"
    joblib.dump(best_model, best_path)
    print(f"\n  Best model : {best_name}  (acc={best_acc:.4f})")
    print(f"  Saved      -> {best_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_and_evaluate()
