"""Sanity checks for the PyTorch extension. Run with:  python -m pytest"""
import numpy as np
import torch

from early_warning.mlp import MLP, TorchLogReg, predict_proba, train_model
from early_warning.model import LogisticRegressionGD


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X @ np.array([1.5, -2.0, 0.5, 0.0]) + rng.normal(size=n) > 0).astype(int)
    return X[: n // 2], y[: n // 2], X[n // 2:], y[n // 2:]


def test_architecture_output_shape():
    x = torch.randn(8, 10)
    assert MLP(10, hidden=(16, 8)).eval()(x).shape == (8,)
    assert TorchLogReg(10)(x).shape == (8,)


def test_pytorch_logreg_agrees_with_from_scratch_model():
    X_tr, y_tr, X_va, y_va = _toy()
    gd = LogisticRegressionGD(lr=1.0, l2=0.0, n_iters=5000, tol=1e-10).fit(X_tr, y_tr)
    tl, _ = train_model(lambda: TorchLogReg(4), X_tr, y_tr, X_va, y_va, lr=1e-2,
                        weight_decay=0.0, max_epochs=300, patience=300)
    agree = np.mean((gd.predict_proba(X_va) >= 0.5) == (predict_proba(tl, X_va) >= 0.5))
    assert agree >= 0.95


def test_early_stopping_restores_best_epoch():
    X_tr, y_tr, X_va, y_va = _toy()
    _, hist = train_model(lambda: MLP(4, hidden=(64,), dropout=0.0), X_tr, y_tr, X_va, y_va,
                          max_epochs=60, patience=5)
    assert 1 <= hist["best_epoch"] <= len(hist["val"])
    assert hist["val"][hist["best_epoch"] - 1] == min(hist["val"])


def test_training_is_reproducible():
    X_tr, y_tr, X_va, y_va = _toy()
    runs = [train_model(lambda: MLP(4, hidden=(8,)), X_tr, y_tr, X_va, y_va, max_epochs=5,
                        seed=7)[0] for _ in range(2)]
    assert np.allclose(predict_proba(runs[0], X_va), predict_proba(runs[1], X_va))
