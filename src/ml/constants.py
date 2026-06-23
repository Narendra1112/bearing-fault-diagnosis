"""
ml/constants.py — Single source of truth for ML constants.

CLASS_NAMES was previously duplicated across 12+ files. Any module needing
the fault class list MUST import it from here to prevent silent label drift.
"""

from __future__ import annotations

# 10-class CWRU fault taxonomy. Order is fixed and matches the trained model's
# output layer — DO NOT reorder without retraining.
CLASS_NAMES: list[str] = [
    "normal",
    "ball_0.007", "ball_0.014", "ball_0.021",
    "ir_0.007",   "ir_0.014",   "ir_0.021",
    "or_0.007",   "or_0.014",   "or_0.021",
]

N_CLASSES: int = len(CLASS_NAMES)   # 10
