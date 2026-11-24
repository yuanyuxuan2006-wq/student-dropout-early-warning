"""Experiment 4 (deep-learning extension): does a neural network beat logistic regression?

REPORT.md (section 4, "Linearity") lists as a limitation that logistic regression cannot model
interactions between features. An MLP can. This experiment tests whether that matters here,
using exactly the same data, splits, preprocessing, threshold rule and metrics as before:

(a) choose the MLP architecture on the VALIDATION set (same selection rule as Experiment 1:
    lowest validation log loss)
(b) one-time test comparison: from-scratch LogReg vs PyTorch LogReg (control) vs MLP
(c) robustness: repeat the whole pipeline on the same 5 random splits as Experiment 3
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import FIGURES_DIR
from .data import make_splits
from .experiment2 import make_subgroups
from .experiment3 import ROBUST_SEEDS
from .metrics import evaluate, tune_threshold
from .mlp import MLP, TorchLogReg, predict_proba, train_model
from .model import LogisticRegressionGD, log_loss
from .pipeline import fit_pipeline

# Candidate architectures. Everything else (optimiser, learning rate, batch size,
# early stopping) is held fixed so the comparison is about the architecture.
CONFIGS = [
    {"hidden": (16,), "dropout": 0.0},
    {"hidden": (16,), "dropout": 0.3},
    {"hidden": (64, 32), "dropout": 0.0},
    {"hidden": (64, 32), "dropout": 0.3},
    {"hidden": (128, 64), "dropout": 0.3},
    {"hidden": (128, 64), "dropout": 0.5},
]
TRAIN_KW = {"lr": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "max_epochs": 300, "patience": 20}


def describe(cfg):
    return f"MLP {'-'.join(map(str, cfg['hidden']))}, dropout {cfg['dropout']}"


def make_mlp(n_inputs, cfg):
    return lambda: MLP(n_inputs, hidden=cfg["hidden"], dropout=cfg["dropout"])


def score(y_val, val_probs, y_test, test_probs):
    """Tune the threshold on validation (same rule as the original model), then evaluate on test."""
    thr, _ = tune_threshold(y_val, val_probs)
    m = evaluate(y_test, (test_probs >= thr).astype(int), test_probs)
    m["threshold"] = thr
    m["auc"] = roc_auc_score(y_test, test_probs)
    return m


def run(data, exp1, device="cpu"):
    X_train, X_val, X_test = data.X_train, data.X_val, data.X_test
    y_train, y_val, y_test = data.y_train, data.y_val, data.y_test
    n_inputs = X_train.shape[1]

    # ------------------------------------------------------------------------------------------
    # (a) Architecture selection on validation only
    # ------------------------------------------------------------------------------------------
    rows, runs = [], {}
    for cfg in CONFIGS:
        model, hist = train_model(make_mlp(n_inputs, cfg), X_train, y_train, X_val, y_val,
                                  device=device, **TRAIN_KW)
        val_probs = predict_proba(model, X_val, device)
        thr, _ = tune_threshold(y_val, val_probs)
        val_m = evaluate(y_val, (val_probs >= thr).astype(int), val_probs)
        runs[describe(cfg)] = (cfg, model, hist)
        rows.append({"architecture": describe(cfg),
                     "params": sum(p.numel() for p in model.parameters()),
                     "best_epoch": hist["best_epoch"],
                     "train_logloss": log_loss(y_train, predict_proba(model, X_train, device)),
                     "val_logloss": log_loss(y_val, val_probs),
                     "val_f1": val_m["f1"], "val_precision@100": val_m["precision@100"]})
    arch_table = pd.DataFrame(rows)

    # Reference row: the original from-scratch model on the same validation data
    lr_model = LogisticRegressionGD(lr=exp1["best_lr"], l2=exp1["best_l2"]).fit(X_train, y_train)
    lr_val = lr_model.predict_proba(X_val)
    thr, _ = tune_threshold(y_val, lr_val)
    lr_val_m = evaluate(y_val, (lr_val >= thr).astype(int), lr_val)
    reference = pd.DataFrame([{"architecture": "LogReg GD (reference)", "params": n_inputs + 1,
                               "best_epoch": np.nan,
                               "train_logloss": log_loss(y_train, lr_model.predict_proba(X_train)),
                               "val_logloss": log_loss(y_val, lr_val), "val_f1": lr_val_m["f1"],
                               "val_precision@100": lr_val_m["precision@100"]}])
    print("(a) Architecture selection (validation set only)")
    print(pd.concat([arch_table, reference], ignore_index=True).round(4).to_string(index=False), "\n")

    best_name = arch_table.loc[arch_table["val_logloss"].idxmin(), "architecture"]
    best_cfg, best_mlp, best_hist = runs[best_name]
    print(f"Selected on validation log loss: {best_name}\n")

    # ------------------------------------------------------------------------------------------
    # (b) One-time test comparison
    # ------------------------------------------------------------------------------------------
    torch_lr, _ = train_model(lambda: TorchLogReg(n_inputs), X_train, y_train, X_val, y_val,
                              device=device, **TRAIN_KW)
    probs = {
        "LogReg GD (from scratch)": (lr_val, lr_model.predict_proba(X_test)),
        "LogReg (PyTorch control)": (predict_proba(torch_lr, X_val, device),
                                     predict_proba(torch_lr, X_test, device)),
        best_name: (predict_proba(best_mlp, X_val, device), predict_proba(best_mlp, X_test, device)),
    }

    subgroups = make_subgroups(data.X_test_df)
    hidden = (data.X_test_df["sem1_pass_rate"].to_numpy() >= 0.5) & (y_test == 1)
    rows = []
    for name, (pv, pt) in probs.items():
        m = score(y_val, pv, y_test, pt)
        pred = (pt >= m["threshold"]).astype(int)
        rows.append({"model": name, "threshold": m["threshold"], "recall": m["recall"],
                     "precision": m["precision"], "f1": m["f1"], "auc": m["auc"],
                     "precision@100": m["precision@100"],
                     "hidden_dropouts_caught": pred[hidden].mean(),
                     "recall_scholarship": evaluate(y_test[subgroups["Scholarship"]],
                                                    pred[subgroups["Scholarship"]])["recall"],
                     "recall_age<=20": evaluate(y_test[subgroups["Age <= 20"]],
                                                pred[subgroups["Age <= 20"]])["recall"]})
    test_table = pd.DataFrame(rows).set_index("model")
    print("(b) TEST SET comparison (thresholds tuned on validation, test used once)")
    print(test_table.round(3).T.to_string(), "\n")

    p_gd, p_torch = probs["LogReg GD (from scratch)"][1], probs["LogReg (PyTorch control)"][1]
    print(f"Control check: from-scratch vs PyTorch logistic regression -> probability correlation "
          f"{np.corrcoef(p_gd, p_torch)[0, 1]:.4f}, max |difference| {np.max(np.abs(p_gd - p_torch)):.3f}\n")

    # ------------------------------------------------------------------------------------------
    # (c) Robustness: same 5 seeds as Experiment 3, whole pipeline repeated
    # ------------------------------------------------------------------------------------------
    rows = []
    for s in ROBUST_SEEDS:
        tr_df, va_df, te_df, y_tr, y_va, y_te = make_splits(data.features_df, data.y_all, seed=s)
        pp, gd, _ = fit_pipeline(tr_df, va_df, y_tr, y_va, exp1["best_lr"], exp1["best_l2"])
        Xtr, Xva, Xte = pp.transform(tr_df), pp.transform(va_df), pp.transform(te_df)
        lr_m = score(y_va, gd.predict_proba(Xva), y_te, gd.predict_proba(Xte))
        mlp, _ = train_model(make_mlp(Xtr.shape[1], best_cfg), Xtr, y_tr, Xva, y_va,
                             seed=s, device=device, **TRAIN_KW)
        mlp_m = score(y_va, predict_proba(mlp, Xva, device), y_te, predict_proba(mlp, Xte, device))
        rows.append({"seed": s, "logreg_f1": lr_m["f1"], "mlp_f1": mlp_m["f1"],
                     "logreg_auc": lr_m["auc"], "mlp_auc": mlp_m["auc"],
                     "logreg_p@100": lr_m["precision@100"], "mlp_p@100": mlp_m["precision@100"]})
    robust = pd.DataFrame(rows)
    print("(c) Robustness across the same 5 random splits as Experiment 3")
    print(robust.round(3).to_string(index=False))
    for metric in ["f1", "auc", "p@100"]:
        lr_col, mlp_col = robust[f"logreg_{metric}"], robust[f"mlp_{metric}"]
        wins = int((mlp_col > lr_col).sum())
        print(f"  {metric:6s} LogReg {lr_col.mean():.3f} +/- {lr_col.std():.3f} | "
              f"MLP {mlp_col.mean():.3f} +/- {mlp_col.std():.3f} | "
              f"mean difference {mlp_col.mean() - lr_col.mean():+.3f} | MLP better on {wins}/5 splits")
    print()

    return {"arch_table": arch_table, "best_name": best_name, "best_hist": best_hist,
            "runs": runs, "test_table": test_table, "robust": robust}


def figure3(exp4, show=False):
    """(a) Learning curves with and without dropout, (b) F1 per random split."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, style in [("MLP 64-32, dropout 0.0", "tab:red"), ("MLP 64-32, dropout 0.3", "tab:blue")]:
        _, _, hist = exp4["runs"][name]
        epochs = np.arange(1, len(hist["train"]) + 1)
        ax1.plot(epochs, hist["train"], color=style, ls="--", lw=1.2, label=f"{name}: train")
        ax1.plot(epochs, hist["val"], color=style, lw=2, label=f"{name}: validation")
        ax1.axvline(hist["best_epoch"], color=style, lw=0.8, alpha=0.6)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Log loss")
    ax1.set_title("(a) Learning curves: vertical lines = best epoch kept by early stopping")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    robust = exp4["robust"]
    x = np.arange(len(robust))
    ax2.plot(x, robust["logreg_f1"], "o-", lw=2, label="Logistic regression (from scratch)")
    ax2.plot(x, robust["mlp_f1"], "s-", lw=2, label=f"{exp4['best_name']}")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"seed {s}" for s in robust["seed"]])
    ax2.set_ylabel("Test dropout F1")
    ax2.set_title("(b) Same 5 random splits as Experiment 3")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("Figure 3. Deep-learning extension: MLP vs logistic regression")
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / "figure3_mlp_vs_logreg.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path
