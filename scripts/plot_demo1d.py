""" Figures of the local 1D two-mode demo (scripts/demo1d_local.sh): final model samples against the target density, and the left-mode mass during training (the target weight of the left mode is 0.1).
    python scripts/plot_demo1d.py [--runs demo_as demo_as_barker demo_as_mh] [--seed 0]  ->  results/demo1d/demo1d_hist.png, demo1d_leftmass.png and a text summary """
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils.dist_utils import GMM1D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "results" / "demo1d"
LABEL = {"demo_as": "AS", "demo_as_barker": "AS + Barker", "demo_as_division": "AS + division", "demo_as_mh": "AS + MH"}


def evals(run):
    out = []
    for line in (run / "train.log").read_text().splitlines():
        if "left_mass" in line and "{" in line:
            try:
                out.append(ast.literal_eval(line[line.index("{"):]))
            except Exception:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["demo_as", "demo_as_barker", "demo_as_division"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    g = GMM1D()
    xs = torch.linspace(-6, 6, 600)
    dens = torch.exp(g.log_prob(xs[:, None]).reshape(-1)).numpy()
    runs = [(r, ROOT / f"{r}_s{a.seed}") for r in a.runs if (ROOT / f"{r}_s{a.seed}" / "samples_final.pt").exists()]
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    lines = []
    fig2, ax2 = plt.subplots(figsize=(5, 3.2))
    for ax, (r, d) in zip(axes, runs):
        x = torch.load(d / "samples_final.pt", map_location="cpu", weights_only=False)["model_samples"].reshape(-1).numpy()
        ax.hist(x, bins=80, range=(-6, 6), density=True, color="0.6", label="model")
        ax.plot(xs.numpy(), dens, "k", lw=1.5, label="target")
        left = float((x < 0).mean())
        ax.set_title(f"{LABEL[r]}: left mass {left:.2f} (target 0.10)", fontsize=9)
        ax.set_xlabel("x")
        lm = np.array(json.loads((ROOT / "leftmass_evals.json").read_text())[r])           # left_mass of the evaluations every 100 epochs (printed by the runs; train.log does not keep them)
        ax2.plot(np.arange(len(lm)) * 100, lm, marker="o", ms=3, label=LABEL[r])
        lines.append(f"{LABEL[r]:14} final-sample left mass {left:.3f}; evaluations (every 100 epochs): " + " ".join(f"{v:.2f}" for v in lm) + f"; mean of the last 5: {lm[-5:].mean():.3f} ± {lm[-5:].std(ddof=1) / np.sqrt(5):.3f}")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "demo1d_hist.png", dpi=200)
    ax2.axhline(0.1, color="k", ls="--", lw=1, label="target 0.1")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("left-mode mass")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(ROOT / "demo1d_leftmass.png", dpi=200)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
