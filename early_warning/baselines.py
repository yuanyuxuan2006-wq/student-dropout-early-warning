"""Non-learning baselines the model must beat."""
import numpy as np


def majority_baseline(n):
    """Predict 'Not Dropout' for everyone. Scores are constant, so it has no ranking."""
    return np.zeros(n, dtype=int), np.zeros(n)


def semester1_rule(df):
    """Flag students who passed fewer than half of their enrolled semester-1 units.

    Ranking score for precision@100: lower pass rate = riskier; ties broken by lower
    semester-1 grade (grade is on a 0-20 scale, so /1000 never overrides the pass rate).
    """
    pass_rate = df["sem1_pass_rate"].to_numpy()
    grade = df["Curricular units 1st sem (grade)"].to_numpy()
    flags = (pass_rate < 0.5).astype(int)
    scores = (1 - pass_rate) + (20 - grade) / 1000
    return flags, scores
