"""Deep-learning extension: compare a PyTorch MLP with the from-scratch logistic regression.

Usage:  python run_mlp.py                 (CPU, figure saved to figures/)
        python run_mlp.py --device cuda   (use the GPU)
        python run_mlp.py --show          (also open the figure)

CPU is the default on purpose: with 2,654 training rows and a small network, copying each
mini-batch to the GPU can cost more time than the GPU saves.
"""
import argparse
import contextlib
import io
import sys
import time

import matplotlib
import numpy as np
import torch

from early_warning import experiment1, experiment4
from early_warning.config import SEED
from early_warning.pipeline import load_dataset


def section(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    parser.add_argument("--show", action="store_true", help="display the figure interactively")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA requested but torch.cuda.is_available() is False")

    np.random.seed(SEED)
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | device {args.device}")
    start = time.time()

    data = load_dataset(verbose=False)
    # Re-use Experiment 1 exactly as main.py does, so the logistic regression baseline gets the
    # same validation-selected learning rate and L2 (its printout is hidden here).
    with contextlib.redirect_stdout(io.StringIO()):
        exp1 = experiment1.run(data)
    print(f"Logistic regression baseline: lr = {exp1['best_lr']}, l2 = {exp1['best_l2']} "
          f"(from Experiment 1) | model inputs: {data.X_train.shape[1]}")

    section("EXPERIMENT 4: MLP vs logistic regression")
    exp4 = experiment4.run(data, exp1, device=args.device)

    section("FIGURE")
    print("Saved", experiment4.figure3(exp4, show=args.show))
    print(f"Total time: {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
