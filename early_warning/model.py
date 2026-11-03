"""Logistic regression implemented from scratch (sigmoid, log loss, gradient descent)."""
import numpy as np


def sigmoid(z):
    """Numerically stable sigmoid: never calls exp() on a large positive number."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def log_loss(y, p, eps=1e-12):
    """Mean binary cross-entropy. p is clipped so log(0) can never occur."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class LogisticRegressionGD:
    """Logistic regression trained from scratch with full-batch gradient descent.

    Objective:  J(w, b) = mean log loss  +  (l2 / 2) * ||w||^2     (bias is not regularised)
    Gradients:  dJ/dw = X^T (p - y) / n + l2 * w ,   dJ/db = mean(p - y)
    """

    def __init__(self, lr=0.1, l2=0.01, n_iters=3000, tol=1e-7, diverge_factor=10.0):
        self.lr = lr
        self.l2 = l2
        self.n_iters = n_iters
        self.tol = tol                        # stop when the loss improves by less than this
        self.diverge_factor = diverge_factor  # stop if loss exceeds this * starting loss

    def fit(self, X, y, X_val=None, y_val=None):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.history = {"train": [], "val": []}
        self.converged = False
        self.diverged = False

        for it in range(self.n_iters):
            p = sigmoid(X @ self.w + self.b)
            train_loss = log_loss(y, p) + 0.5 * self.l2 * float(self.w @ self.w)
            self.history["train"].append(train_loss)
            if X_val is not None:
                self.history["val"].append(log_loss(y_val, self.predict_proba(X_val)))

            # Divergence check: a too-large step makes the loss explode or become NaN.
            if not np.isfinite(train_loss) or train_loss > self.diverge_factor * self.history["train"][0]:
                self.diverged = True
                break
            # Convergence check: loss has stopped changing.
            if it > 0 and abs(self.history["train"][-2] - train_loss) < self.tol:
                self.converged = True
                break

            error = p - y
            grad_w = X.T @ error / n + self.l2 * self.w
            grad_b = float(error.mean())
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

        self.n_iters_run = it + 1
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
