"""Experiment 3: Robustness across random splits and error analysis on the test set."""
import pandas as pd

from .baselines import semester1_rule
from .config import SENSITIVE_COLS
from .data import make_splits
from .metrics import evaluate
from .pipeline import fit_pipeline, top_contributors

ROBUST_SEEDS = [0, 1, 2, 3, 42]
SUMMARY_COLS = ["risk", "sem1_pass_rate", "Curricular units 1st sem (grade)",
                "Tuition fees up to date", "Debtor", "Scholarship holder", "Age at enrollment"]


def run(data, exp1, exp2):
    best_lr, best_l2 = exp1["best_lr"], exp1["best_l2"]
    X_test, X_test_df, y_test = data.X_test, data.X_test_df, data.y_test
    feature_names = data.feature_names
    final_model = exp2["final_model"]
    test_probs, test_pred, rule_pred = exp2["test_probs"], exp2["test_pred"], exp2["rule_pred"]

    # 3a) Repeat the WHOLE pipeline (split -> preprocess -> train -> tune -> test) on 5 seeds.
    rows = []
    for s in ROBUST_SEEDS:
        tr_df, va_df, te_df, y_tr, y_va, y_te = make_splits(data.features_df, data.y_all, seed=s)
        pp_s, m_s, thr_s = fit_pipeline(tr_df, va_df, y_tr, y_va, best_lr, best_l2)
        probs_s = m_s.predict_proba(pp_s.transform(te_df))
        model_m = evaluate(y_te, (probs_s >= thr_s).astype(int), probs_s)
        rule_m = evaluate(y_te, *semester1_rule(te_df))
        rows.append({"seed": s, "threshold": thr_s, "model_recall": model_m["recall"],
                     "model_f1": model_m["f1"], "model_p@100": model_m["precision@100"],
                     "rule_f1": rule_m["f1"], "rule_p@100": rule_m["precision@100"]})
    robust = pd.DataFrame(rows)
    print("(a) Robustness across 5 random splits")
    print(robust.round(3).to_string(index=False))
    f1_range = robust["model_f1"].max() - robust["model_f1"].min()
    print(f"Model F1: mean {robust['model_f1'].mean():.3f}, range {f1_range:.3f} "
          f"-> criterion 6 (range <= 0.08) {'MET' if f1_range <= 0.08 else 'NOT MET'}; "
          f"model beats rule F1 on {int((robust['model_f1'] > robust['rule_f1']).sum())}/5 seeds\n")

    # 3b) Targeted group: dropouts who passed at least half their semester-1 units.
    #     The rule cannot flag these students by definition - can the model?
    passed_half = X_test_df["sem1_pass_rate"].to_numpy() >= 0.5
    hidden_dropouts = passed_half & (y_test == 1)
    print(f"(b) 'Hidden' dropouts (passed >= half their units): {hidden_dropouts.sum()} of "
          f"{y_test.sum()} test dropouts")
    print(f"    caught by the rule: {rule_pred[hidden_dropouts].mean():.1%} | "
          f"caught by the model: {test_pred[hidden_dropouts].mean():.1%}")
    fp_rate_passed = test_pred[passed_half & (y_test == 0)].mean()
    print(f"    cost: model flags {fp_rate_passed:.1%} of non-dropouts who also passed >= half\n")

    # 3c) Error analysis on the test set.
    test_view = X_test_df.copy()
    test_view["risk"] = test_probs
    test_view["dropout"] = y_test
    test_view["flagged"] = test_pred

    false_neg = test_view[(test_view["dropout"] == 1) & (test_view["flagged"] == 0)]
    false_pos = test_view[(test_view["dropout"] == 0) & (test_view["flagged"] == 1)]
    true_pos = test_view[(test_view["dropout"] == 1) & (test_view["flagged"] == 1)]

    print(f"(c) Error profile: mean feature values for each outcome group "
          f"(FN = {len(false_neg)}, FP = {len(false_pos)}, TP = {len(true_pos)})")
    profile = pd.DataFrame({"Missed dropouts (FN)": false_neg[SUMMARY_COLS].mean(),
                            "Caught dropouts (TP)": true_pos[SUMMARY_COLS].mean(),
                            "False alarms (FP)": false_pos[SUMMARY_COLS].mean()})
    print(profile.round(3).to_string(), "\n")

    # The 20 missed dropouts the model was MOST confident about (most dangerous misses)
    fn_review = false_neg.sort_values("risk").head(20)
    print("20 most confidently missed dropouts (lowest predicted risk):")
    print(fn_review[SUMMARY_COLS].round(2).to_string(), "\n")
    fn_patterns = {
        "fees up to date AND passed >= 80% of units": ((fn_review["Tuition fees up to date"] == 1)
                                                       & (fn_review["sem1_pass_rate"] >= 0.8)).mean(),
        "scholarship holder": (fn_review["Scholarship holder"] == 1).mean(),
        "age <= 20": (fn_review["Age at enrollment"] <= 20).mean(),
    }
    print("Patterns among these 20:", {k: f"{v:.0%}" for k, v in fn_patterns.items()}, "\n")

    # False positives: were they flagged because of SENSITIVE features or academic ones?
    sensitive_prefixes = tuple(SENSITIVE_COLS)
    X_test_pos = {idx: i for i, idx in enumerate(X_test_df.index)}
    fp_review = false_pos.sort_values("risk", ascending=False).head(10)
    rows = []
    for idx, row in fp_review.iterrows():
        top3 = top_contributors(final_model, X_test[X_test_pos[idx]], feature_names, k=3)
        rows.append({"risk": round(row["risk"], 3),
                     "top_3_reasons": "; ".join(f"{n} ({c:+.2f})" for n, c in top3),
                     "top_reason_is_sensitive": bool(top3) and top3[0][0].startswith(sensitive_prefixes)})
    fp_table = pd.DataFrame(rows, index=fp_review.index)
    print("10 most confident false alarms and the features that drove each score:")
    print(fp_table.to_string(), "\n")
    print(f"Share of these false alarms whose #1 reason is a sensitive feature: "
          f"{fp_table['top_reason_is_sensitive'].mean():.0%}\n")

    # 3d) Weight inspection: which features does the model rely on?
    weights = pd.Series(final_model.w, index=feature_names).sort_values()
    print("Largest weights that INCREASE dropout risk:")
    print(weights.tail(8)[::-1].round(3).to_string())
    print("\nLargest weights that DECREASE dropout risk:")
    print(weights.head(8).round(3).to_string(), "\n")

    # 3e) The advisor's actual output: a ranked contact list with reasons.
    contact_list = test_view.sort_values("risk", ascending=False).head(10)[["risk", "dropout"]].copy()
    contact_list["top_3_reasons"] = [
        "; ".join(n for n, _ in top_contributors(final_model, X_test[X_test_pos[i]], feature_names))
        for i in contact_list.index]
    print("Example advisor contact list (top 10 of the test set):")
    print(contact_list.round(3).to_string())

    return {"robust": robust, "profile": profile, "fp_table": fp_table, "weights": weights,
            "contact_list": contact_list}
