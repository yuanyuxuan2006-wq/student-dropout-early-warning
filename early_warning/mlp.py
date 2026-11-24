"""PyTorch models for the deep-learning extension (Experiment 4).

The ARCHITECTURE (what the network looks like) is kept separate from the TRAINING code (how it
learns), so one `train_model` function can train both models below. The only thing that
changes between logistic regression and the MLP is the class passed in.
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from .config import SEED


# ----------------------------------------------------------------------------------------------
# Architectures
# ----------------------------------------------------------------------------------------------
class TorchLogReg(nn.Module):
    """The same model as LogisticRegressionGD (z = w.x + b), written in PyTorch.

    Used as a control: if PyTorch's training code is correct, this should behave almost exactly
    like the from-scratch model, so any difference the MLP shows comes from the architecture.
    """

    def __init__(self, n_inputs):
        super().__init__()
        self.linear = nn.Linear(n_inputs, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)          # logits; sigmoid is applied in the loss


class MLP(nn.Module):
    """Multi-layer perceptron: [Linear -> ReLU -> Dropout] x len(hidden) -> Linear.

    The hidden layers with ReLU let the model learn non-linear combinations of features
    (e.g. age x evening attendance), which logistic regression cannot represent.
    Dropout randomly zeroes hidden units during training to reduce overfitting.
    """

    def __init__(self, n_inputs, hidden=(64, 32), dropout=0.3):
        super().__init__()
        layers, prev = [], n_inputs
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)]
            prev = width
        layers.append(nn.Linear(prev, 1))          # one output: dropout logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ----------------------------------------------------------------------------------------------
# Training code (shared by every architecture)
# ----------------------------------------------------------------------------------------------
def _tensor(a, device):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)


def train_model(make_model, X_train, y_train, X_val, y_val, lr=1e-3, weight_decay=1e-4,
                max_epochs=300, batch_size=64, patience=20, seed=SEED, device="cpu"):
    """Train with mini-batch Adam and early stopping on validation log loss.

    make_model:   function returning a fresh nn.Module (called after seeding, so weight
                  initialisation is reproducible).
    weight_decay: L2 penalty applied by the optimiser (the PyTorch version of `l2`).
    patience:     stop after this many epochs without a new best validation loss; the
                  weights from the best epoch are restored (this is early stopping).

    Returns (model, history) where history has per-epoch "train" and "val" log loss and
    "best_epoch" (1-based).
    """
    torch.manual_seed(seed)
    model = make_model().to(device)
    loss_fn = nn.BCEWithLogitsLoss()               # numerically stable sigmoid + log loss
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xtr, ytr = _tensor(X_train, device), _tensor(y_train, device)
    Xva, yva = _tensor(X_val, device), _tensor(y_val, device)
    gen = torch.Generator().manual_seed(seed)      # reproducible shuffling
    n = len(Xtr)

    history = {"train": [], "val": [], "best_epoch": 0}
    best_val, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()                              # dropout ON
        order = torch.randperm(n, generator=gen).to(device)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()                        # backpropagation
            opt.step()                             # gradient step
            total += loss.item() * len(idx)
        history["train"].append(total / n)

        model.eval()                               # dropout OFF
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), yva).item()
        history["val"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val, bad_epochs = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)              # roll back to the best epoch
    model.eval()
    return model, history


def predict_proba(model, X, device="cpu"):
    """Dropout probabilities as a numpy array (same interface as LogisticRegressionGD)."""
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(_tensor(X, device))).cpu().numpy()
