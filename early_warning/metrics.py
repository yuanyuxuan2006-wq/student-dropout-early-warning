"""Dropout-class metrics, precision@k for the advisor's capacity, and threshold tuning."""
import numpy as np
import pandas as pd


def safe_div(a, b):
    return a / b if b else 0.0


def evaluate(y_true, y_pred, scores=None, k=100):
    """Dropout-class metrics. precision@k uses the ranking scores (higher = riskier)."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision, recall = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
    result = {"accuracy": (tp + tn) / len(y_true), "precision": precision, "recall": recall,
              "f1": safe_div(2 * precision * recall, precision + recall),
              "TP": tp, "FP": fp, "FN": fn, "TN": tn}
    # Constant scores (majority baseline) give no ranking, so precision@k is undefined.
    has_ranking = scores is not None and np.ptp(scores) > 0
    result[f"precision@{k}"] = precision_at_k(y_true, scores, k) if has_ranking else np.nan
    return result


def precision_at_k(y_true, scores, k=100):
    """Of the k highest-risk students, the fraction who actually dropped out."""
    top_k = np.argsort(-scores, kind="stable")[:k]
    return float(np.mean(y_true[top_k]))


def tune_threshold(y_true, probs, min_recall=0.70, grid=np.arange(0.05, 0.951, 0.01)):
    """Pick the threshold with the best F1 among those reaching min_recall (validation only).

    If no threshold reaches min_recall, fall back to the one with the highest recall.
    """
    rows = [{"threshold": t, **evaluate(y_true, (probs >= t).astype(int))} for t in grid]
    table = pd.DataFrame(rows)
    ok = table[table["recall"] >= min_recall]
    best = ok.loc[ok["f1"].idxmax()] if len(ok) else table.loc[table["recall"].idxmax()]
    return float(best["threshold"]), table
