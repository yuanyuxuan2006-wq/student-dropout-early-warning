"""Sanity checks on the building blocks (edge cases). Run with:  python -m pytest"""
import numpy as np
import pandas as pd

from early_warning.metrics import evaluate, precision_at_k
from early_warning.model import LogisticRegressionGD, log_loss, sigmoid
from early_warning.pipeline import sklearn_equivalent
from early_warning.preprocessing import Preprocessor


def test_sigmoid_is_stable():
    assert np.isclose(sigmoid(np.array([0.0]))[0], 0.5)
    assert np.all(np.isfinite(sigmoid(np.array([-1000.0, 1000.0]))))   # no overflow


def test_log_loss_never_takes_log_of_zero():
    assert np.isfinite(log_loss(np.array([1, 0]), np.array([0.0, 1.0])))


def test_gradient_descent_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X @ np.array([1.5, -2.0, 0.5, 0.0]) + rng.normal(size=300) > 0).astype(int)
    l2 = 0.01
    mine = LogisticRegressionGD(lr=1.0, l2=l2, n_iters=5000, tol=1e-10).fit(X, y)
    sk = sklearn_equivalent(l2, len(y)).fit(X, y)
    assert np.allclose(mine.w, sk.coef_[0], atol=1e-3)
    assert np.isclose(mine.b, sk.intercept_[0], atol=1e-3)


def test_too_large_learning_rate_is_reported_not_crashed():
    X = np.array([[1.0], [2.0], [3.0], [4.0]]) * 100
    y = np.array([0, 0, 1, 1])
    m = LogisticRegressionGD(lr=1e6, l2=1.0).fit(X, y)
    assert m.diverged or not m.converged


def test_evaluate_and_precision_at_k():
    y = np.array([1, 0, 1, 0])
    m = evaluate(y, np.array([1, 1, 0, 0]), scores=np.array([0.9, 0.8, 0.2, 0.1]), k=2)
    assert m["TP"] == 1 and m["FP"] == 1 and m["FN"] == 1 and m["TN"] == 1
    assert m["precision@2"] == 0.5
    assert np.isnan(evaluate(y, np.zeros(4, dtype=int), scores=np.zeros(4))["precision@100"])
    assert precision_at_k(y, np.array([0.9, 0.1, 0.8, 0.2]), k=2) == 1.0


def test_preprocessor_merges_rare_levels_and_handles_unseen():
    train = pd.DataFrame({"cat": [1] * 40 + [2] * 5, "num": np.arange(45.0)})
    prep = Preprocessor(["cat"], min_count=30).fit(train)
    assert prep.feature_names_ == ["num", "cat=1", "cat=other"]
    out = prep.transform(pd.DataFrame({"cat": [99], "num": [0.0]}))   # unseen level -> "other"
    assert out.shape == (1, 3) and np.all(np.isfinite(out))
