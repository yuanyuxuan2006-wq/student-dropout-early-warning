"""Run the whole project: data -> Experiments 1-3 -> Figures 1-2.

Usage:  python main.py            (figures saved to figures/)
        python main.py --show     (also open the figures in a window)
"""
import argparse
import sys

import matplotlib
import numpy as np
import pandas as pd
import sklearn

from early_warning import experiment1, experiment2, experiment3, plots
from early_warning.config import SEED
from early_warning.pipeline import load_dataset


def section(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    np.random.seed(SEED)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    # Dependencies (documented for reproducibility)
    print(f"Python {sys.version.split()[0]} | numpy {np.__version__} | pandas {pd.__version__} | "
          f"scikit-learn {sklearn.__version__} | matplotlib {matplotlib.__version__}")

    section("DATA")
    data = load_dataset()
    section("EXPERIMENT 1: learning rate and L2 strength")
    exp1 = experiment1.run(data)
    section("EXPERIMENT 2: test results, sklearn check, fairness")
    exp2 = experiment2.run(data, exp1)
    section("EXPERIMENT 3: robustness and error analysis")
    experiment3.run(data, exp1, exp2)

    section("FIGURES")
    print("Saved", plots.figure1(exp1, show=args.show))
    print("Saved", plots.figure2(data, exp2, show=args.show))


if __name__ == "__main__":
    main()
