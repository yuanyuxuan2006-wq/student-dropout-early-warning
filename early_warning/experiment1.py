"""Experiment 1: How do the learning rate and L2 strength affect gradient descent?

(a) learning rate -> convergence speed vs oscillation/divergence
(b) L2 strength   -> under/over-fitting, measured on the VALIDATION set only
"""
import numpy as np
import pandas as pd

from .metrics import evaluate, tune_threshold
from .model import LogisticRegressionGD, log_loss

LEARNING_RATES = [0.001, 0.01, 0.1, 1.0, 5.0, 20.0]
L2_VALUES = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def gd_status(model, window=100):
    """Describe how a run ended: diverged, oscillating (loss went UP recently),
    converged, or still decreasing when the iteration budget ran out."""
    recent = np.diff(model.history["train"][-window:])
    if model.diverged:
        return "diverged"
    if np.any(recent > 1e-12):
        return "oscillating"
    return "converged" if model.converged else "still decreasing"


def run(data):
    X_train, y_train, X_val, y_val = data.X_train, data.y_train, data.X_val, data.y_val

    lr_runs = {}
    rows = []
    for lr in LEARNING_RATES:
        m = LogisticRegressionGD(lr=lr, l2=0.01).fit(X_train, y_train, X_val, y_val)
        lr_runs[lr] = m
        rows.append({"learning_rate": lr, "iterations": m.n_iters_run, "status": gd_status(m),
                     "final_train_loss": m.history["train"][-1],
                     "final_val_loss": m.history["val"][-1]})
    lr_table = pd.DataFrame(rows)
    print("(a) Learning-rate sweep (l2 = 0.01)")
    print(lr_table.round(4).to_string(index=False), "\n")

    # Best learning rate = lowest validation loss among runs that converged.
    stable = lr_table[lr_table["status"] == "converged"]
    best_lr = float(stable.loc[stable["final_val_loss"].idxmin(), "learning_rate"])

    l2_runs = {}
    rows = []
    for l2 in L2_VALUES:
        m = LogisticRegressionGD(lr=best_lr, l2=l2).fit(X_train, y_train, X_val, y_val)
        l2_runs[l2] = m
        val_probs = m.predict_proba(X_val)
        thr, _ = tune_threshold(y_val, val_probs)
        val_metrics = evaluate(y_val, (val_probs >= thr).astype(int), val_probs)
        rows.append({"l2": l2, "train_logloss": log_loss(y_train, m.predict_proba(X_train)),
                     "val_logloss": log_loss(y_val, val_probs), "threshold": thr,
                     "val_recall": val_metrics["recall"], "val_f1": val_metrics["f1"],
                     "val_precision@100": val_metrics["precision@100"],
                     "weight_norm": float(np.linalg.norm(m.w))})
    l2_table = pd.DataFrame(rows)
    print(f"(b) L2 sweep (learning rate = {best_lr})")
    print(l2_table.round(4).to_string(index=False), "\n")

    best_l2 = float(l2_table.loc[l2_table["val_logloss"].idxmin(), "l2"])
    print(f"Selected on validation: learning rate = {best_lr}, l2 = {best_l2}")

    return {"lr_runs": lr_runs, "lr_table": lr_table, "l2_runs": l2_runs, "l2_table": l2_table,
            "best_lr": best_lr, "best_l2": best_l2}
