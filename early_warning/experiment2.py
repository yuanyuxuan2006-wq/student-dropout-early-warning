"""Experiment 2: Final test evaluation against baselines, implementation check, and a
fairness comparison of the model WITH vs WITHOUT sensitive features.

The test set is used here for the first (and only) time.
"""
import numpy as np
import pandas as pd

from .baselines import majority_baseline, semester1_rule
from .config import SEED, SENSITIVE_COLS
from .metrics import evaluate, tune_threshold
from .model import LogisticRegressionGD
from .pipeline import fit_pipeline, sklearn_equivalent


def make_subgroups(X_test_df):
    return {
        "Age <= 20": X_test_df["Age at enrollment"].to_numpy() <= 20,
        "Age > 20": X_test_df["Age at enrollment"].to_numpy() > 20,
        "Scholarship": X_test_df["Scholarship holder"].to_numpy() == 1,
        "No scholarship": X_test_df["Scholarship holder"].to_numpy() == 0,
        "Debtor": X_test_df["Debtor"].to_numpy() == 1,
        "Not debtor": X_test_df["Debtor"].to_numpy() == 0,
        "Female": X_test_df["Gender"].to_numpy() == 0,   # dataset coding: 1 = male, 0 = female
        "Male": X_test_df["Gender"].to_numpy() == 1,
    }


def subgroup_table(y_test, pred, subgroups, label):
    overall = evaluate(y_test, pred)["recall"]
    rows = []
    for group, mask in subgroups.items():
        m = evaluate(y_test[mask], pred[mask])
        rows.append({"model": label, "subgroup": group, "n": int(mask.sum()),
                     "n_dropouts": int(y_test[mask].sum()), "recall": m["recall"],
                     "precision": m["precision"], "gap_vs_overall": m["recall"] - overall})
    return pd.DataFrame(rows), overall


def run(data, exp1):
    X_train, X_val, X_test = data.X_train, data.X_val, data.X_test
    y_train, y_val, y_test = data.y_train, data.y_val, data.y_test
    X_test_df = data.X_test_df
    best_lr, best_l2 = exp1["best_lr"], exp1["best_l2"]

    # 2a) Final model: hyperparameters from Experiment 1, threshold tuned on validation.
    final_model = LogisticRegressionGD(lr=best_lr, l2=best_l2).fit(X_train, y_train, X_val, y_val)
    threshold, val_threshold_table = tune_threshold(y_val, final_model.predict_proba(X_val))
    test_probs = final_model.predict_proba(X_test)
    test_pred = (test_probs >= threshold).astype(int)

    maj_pred, maj_scores = majority_baseline(len(y_test))
    rule_pred, rule_scores = semester1_rule(X_test_df)

    results = pd.DataFrame({
        "Majority class": evaluate(y_test, maj_pred, maj_scores),
        "Semester-1 rule": evaluate(y_test, rule_pred, rule_scores),
        f"LogReg GD (t={threshold:.2f})": evaluate(y_test, test_pred, test_probs),
    }).T
    print("TEST SET RESULTS (Dropout = positive class)")
    print(results.round(3).to_string(), "\n")

    # 2b) Verify the from-scratch implementation against scikit-learn on identical inputs.
    sk_model = sklearn_equivalent(best_l2, len(y_train)).fit(X_train, y_train)
    sk_probs = sk_model.predict_proba(X_test)[:, 1]
    agree_05 = np.mean((sk_probs >= 0.5) == (test_probs >= 0.5))
    agree_t = np.mean((sk_probs >= threshold) == (test_probs >= threshold))
    print(f"Agreement with scikit-learn: {agree_05:.2%} at t=0.5, {agree_t:.2%} at t={threshold:.2f} | "
          f"max |prob difference| = {np.max(np.abs(sk_probs - test_probs)):.4f} | "
          f"weight correlation = {np.corrcoef(sk_model.coef_[0], final_model.w)[0, 1]:.4f}\n")

    # 2c) Fairness: subgroup recall for the full model vs a model without sensitive features.
    pp_ns, model_ns, threshold_ns = fit_pipeline(data.X_train_df, data.X_val_df, y_train, y_val,
                                                 best_lr, best_l2, exclude_cols=SENSITIVE_COLS)
    test_probs_ns = model_ns.predict_proba(pp_ns.transform(X_test_df))
    test_pred_ns = (test_probs_ns >= threshold_ns).astype(int)

    subgroups = make_subgroups(X_test_df)
    fair_full, recall_full = subgroup_table(y_test, test_pred, subgroups, "With sensitive")
    fair_ns, recall_ns = subgroup_table(y_test, test_pred_ns, subgroups, "Without sensitive")
    fairness = pd.concat([fair_full, fair_ns], ignore_index=True)
    print("Subgroup dropout recall on TEST (gap = subgroup recall - overall recall)")
    print(fairness.round(3).to_string(index=False), "\n")

    max_gap_full = fair_full["gap_vs_overall"].min()
    max_gap_ns = fair_ns["gap_vs_overall"].min()
    ns_metrics = evaluate(y_test, test_pred_ns, test_probs_ns)
    print(f"Overall recall: with sensitive = {recall_full:.3f}, without = {recall_ns:.3f} "
          f"(change {100 * (recall_ns - recall_full):+.1f} pts); F1 with = {results.iloc[2]['f1']:.3f}, "
          f"without = {ns_metrics['f1']:.3f}")
    print(f"Largest shortfall below overall recall: with sensitive = {100 * max_gap_full:.1f} pts, "
          f"without = {100 * max_gap_ns:.1f} pts\n")

    # 2d) Investigate the largest gap: is it real, and WHY does it happen?
    worst = fair_full.loc[fair_full["gap_vs_overall"].idxmin()]
    worst_mask = subgroups[worst["subgroup"]] & (y_test == 1)
    rng = np.random.default_rng(SEED)
    worst_preds = test_pred[worst_mask]
    boot = [rng.choice(worst_preds, size=len(worst_preds), replace=True).mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"Worst subgroup: '{worst['subgroup']}' recall {worst['recall']:.3f} from only "
          f"{int(worst['n_dropouts'])} dropouts; 95% bootstrap CI [{lo:.2f}, {hi:.2f}]")
    other_dropouts = ~subgroups[worst["subgroup"]] & (y_test == 1)
    signal_cols = ["sem1_pass_rate", "Tuition fees up to date", "Debtor"]
    why = pd.DataFrame({
        f"{worst['subgroup']} dropouts": X_test_df[worst_mask][signal_cols].mean(),
        "All other dropouts": X_test_df[other_dropouts][signal_cols].mean(),
    })
    print("Mean values of the model's strongest signals among actual dropouts:")
    print(why.round(3).to_string(), "\n")

    # 2e) H1 success criteria (evaluated honestly, pass or fail).
    model_row, rule_row = results.iloc[2], results.iloc[1]
    criteria = pd.DataFrame([
        ("1. Dropout recall >= 0.70", f"{model_row['recall']:.3f}", model_row["recall"] >= 0.70),
        ("2. F1 >= rule F1 + 0.05", f"{model_row['f1']:.3f} vs {rule_row['f1']:.3f}",
         model_row["f1"] >= rule_row["f1"] + 0.05),
        ("3. Precision@100 > rule", f"{model_row['precision@100']:.2f} vs {rule_row['precision@100']:.2f}",
         model_row["precision@100"] > rule_row["precision@100"]),
        ("4. Subgroup recall gap <= 10 pts", f"{-100 * max_gap_full:.1f} pts (no-sensitive: {-100 * max_gap_ns:.1f})",
         max_gap_full >= -0.10 or (max_gap_ns >= -0.10 and recall_full - recall_ns <= 0.03)),
        ("5. >= 98% agreement with sklearn", f"{agree_t:.2%}", agree_t >= 0.98),
    ], columns=["criterion", "value", "met"])
    print("H1 SUCCESS CRITERIA (criterion 6 is checked in Experiment 3)")
    print(criteria.to_string(index=False))

    return {"final_model": final_model, "threshold": threshold, "test_probs": test_probs,
            "test_pred": test_pred, "rule_pred": rule_pred, "rule_scores": rule_scores,
            "results": results, "test_probs_ns": test_probs_ns, "subgroups": subgroups,
            "fair_full": fair_full, "fair_ns": fair_ns, "recall_full": recall_full,
            "criteria": criteria}
