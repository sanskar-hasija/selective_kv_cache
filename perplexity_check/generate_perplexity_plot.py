import argparse
import time
import itertools
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


def plot(
    output_dir: str = "experiments",
    title: Optional[str] = None,
    perplexity_limit: Optional[float] = None,
    skip_first: int = 100,
):
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize = (25,15))
    ax.set_xlabel("Input Sequence Length")

    for file in output_dir.glob("*.csv"):
        experiment = file.stem
        df = pd.read_csv(file)
        df = df.groupby(['input_length']).mean()
        X = df.index[skip_first:]
        Y = df["overall_ppl"][skip_first:]
        Y = np.log(Y)
        ax.plot(X, Y, "-", label=f"{experiment}")
    ax.set_ylabel("Perplexity (log), lower is better")
    if perplexity_limit:
        ax.set_ylim(top=min(ax.get_ylim()[1], perplexity_limit))
    ax.legend(loc=[1, 2, 7][0]) 

    ax.set_title(title.replace("\\n", "\n") if title else "Log perplexity as a function of input lengths")
    fig.tight_layout()

    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./experiments")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--log_perplexity_limit", type=float, default=5.0)
    # Perplexity starts a bit unstable, so we skip the start
    parser.add_argument("--skip_first", type=int, default=100)

    args = parser.parse_args()

    figure = plot(
        output_dir=args.output_dir,
        title=args.title,
        perplexity_limit=args.log_perplexity_limit,
        skip_first=args.skip_first,
    )

    save_path = f"perplexity_plot.png"
    plt.savefig(save_path, dpi=600)
    print(f"plot saved to {save_path}")


if __name__ == "__main__":
    main()