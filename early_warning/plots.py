"""Figures 1 and 2 of the report. Each function saves a PNG and returns its path."""
import matplotlib.pyplot as plt
import numpy as np

from .config import FIGURES_DIR
from .experiment1 import L2_VALUES, LEARNING_RATES, gd_status


def _save(fig, name, show):
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def figure1(exp1, show=False):
    """How the learning rate and L2 strength shape gradient descent (Experiment 1)."""
    lr_runs, l2_table, best_l2 = exp1["lr_runs"], exp1["l2_table"], exp1["best_l2"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(LEARNING_RATES)))
    for color, lr in zip(colors, LEARNING_RATES):
        m = lr_runs[lr]
        iterations = np.arange(1, len(m.history["train"]) + 1)  # start at 1 so the log axis shows iteration 1
        ax1.plot(iterations, m.history["train"], color=color, lw=1.5,
                 label=f"lr = {lr:g} ({gd_status(m)}, {m.n_iters_run} it)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Gradient descent iteration (log scale)")
    ax1.set_ylabel("Training objective: log loss + L2 (log scale)")
    ax1.set_title("(a) Learning rate: too small is slow, too large oscillates")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    x_pos = np.arange(len(L2_VALUES))
    ax2.plot(x_pos, l2_table["train_logloss"], "o-", label="Train log loss")
    ax2.plot(x_pos, l2_table["val_logloss"], "s-", label="Validation log loss")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{v:g}" for v in L2_VALUES])
    ax2.set_xlabel("L2 regularisation strength")
    ax2.set_ylabel("Log loss")
    ax2.axvline(L2_VALUES.index(best_l2), color="grey", ls="--", label=f"Selected l2 = {best_l2:g}")
    ax2_r = ax2.twinx()
    ax2_r.plot(x_pos, l2_table["val_f1"], "d:", color="tab:red", label="Validation F1 (right axis)")
    ax2_r.set_ylabel("Validation dropout F1", color="tab:red")
    lines = ax2.get_legend_handles_labels()
    lines_r = ax2_r.get_legend_handles_labels()
    ax2.legend(lines[0] + lines_r[0], lines[1] + lines_r[1], fontsize=8, loc="upper left")
    ax2.set_title("(b) L2 strength: train/validation gap and F1")
    ax2.grid(alpha=0.3)

    fig.suptitle("Figure 1. Gradient descent behaviour (training and validation data only)")
    fig.tight_layout()
    return _save(fig, "figure1_gradient_descent.png", show)


def recall_at_n(y_true, scores):
    """Fraction of all dropouts found after contacting the top-N ranked students, N = 1..len."""
    order = np.argsort(-scores, kind="stable")
    return np.cumsum(y_true[order]) / y_true.sum()


def figure2(data, exp2, show=False):
    """(a) What the advisor gains from the model vs the rule when contacting the top-N
    students, and (b) whether the model misses some groups more than others (fairness)."""
    y_test = data.y_test
    fair_full, fair_ns, recall_full = exp2["fair_full"], exp2["fair_ns"], exp2["recall_full"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    n_contacted = np.arange(1, len(y_test) + 1)
    ax1.plot(n_contacted, recall_at_n(y_test, exp2["test_probs"]), lw=2, label="Model (with sensitive)")
    ax1.plot(n_contacted, recall_at_n(y_test, exp2["test_probs_ns"]), lw=2, ls="--", label="Model (no sensitive)")
    ax1.plot(n_contacted, recall_at_n(y_test, exp2["rule_scores"]), lw=2, label="Semester-1 rule ranking")
    ax1.plot(n_contacted, n_contacted / len(y_test), color="grey", ls=":", label="Random order")
    ax1.axvline(100, color="black", lw=1, alpha=0.6)
    ax1.text(105, 0.05, "advisor capacity\n(100 students)", fontsize=8)
    ax1.axhline(0.70, color="tab:red", lw=1, alpha=0.6)
    ax1.text(len(y_test) * 0.62, 0.66, "target recall 0.70", fontsize=8, color="tab:red")
    ax1.set_xlabel("Number of students contacted (highest risk first)")
    ax1.set_ylabel("Share of all test dropouts reached (recall)")
    ax1.set_title("(a) Dropouts reached vs advisor effort (test set)")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(alpha=0.3)

    groups = list(exp2["subgroups"])
    x_pos = np.arange(len(groups))
    width = 0.38
    ax2.bar(x_pos - width / 2, fair_full["recall"], width, label="With sensitive features")
    ax2.bar(x_pos + width / 2, fair_ns["recall"], width, label="Without sensitive features")
    ax2.axhline(recall_full, color="tab:blue", ls="--", lw=1)
    ax2.axhspan(recall_full - 0.10, recall_full, color="tab:blue", alpha=0.08,
                label="Within 10 pts of overall recall (with sensitive)")
    for i, n_drop in enumerate(fair_full["n_dropouts"]):
        ax2.text(i, 0.02, f"{n_drop}\ndropouts", ha="center", fontsize=7, color="white")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(groups, rotation=30, ha="right")
    ax2.set_ylim(0, 1.25)
    ax2.set_yticks(np.arange(0, 1.01, 0.2))
    ax2.set_ylabel("Dropout recall")
    ax2.set_title("(b) Subgroup recall: which students does the model miss?")
    ax2.legend(fontsize=8, loc="upper left", ncol=2)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Figure 2. Test-set value for the advisor and fairness across subgroups")
    fig.tight_layout()
    return _save(fig, "figure2_advisor_value_fairness.png", show)
