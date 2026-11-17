""" Figures of the local 1D two-mode demo (scripts/demo1d_local.sh): final model samples against the target density (combined figure and one image per method), and the left-mode mass during training (target weight of the left mode: 0.1).
    python scripts/plot_demo1d.py                 # target with modes at -3 and +3 (runs demo_as, demo_as_barker, demo_as_division)
    python scripts/plot_demo1d.py --wide          # target with modes at -5 and +5 (runs demo_wide_as, demo_wide_as_barker, demo_wide_as_division)
    -> results/demo1d/demo1d[_wide]_hist.png, _leftmass.png, _AS.png, _barker.png, _division.png and a text summary; the left-mass series are read from results/demo1d/leftmass_evals.json (printed by the runs, not kept in train.log) """
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
LABEL = {"as": "AS", "as_barker": "AS + Barker", "as_division": "AS + division"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--narrow", action="store_true", help="modes at -1 and +1, plain AS only: one image of the final samples (same form as the others)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pre, tag, loc, R = ("demo_wide_", "demo1d_wide", (-5.0, 5.0), 8) if a.wide else ("demo_narrow_", "demo1d_narrow", (-1.0, 1.0), 4) if a.narrow else ("demo_", "demo1d", (-3.0, 3.0), 6)
    runs = [(k, ROOT / f"{pre}{k}_s{a.seed}") for k in LABEL if (ROOT / f"{pre}{k}_s{a.seed}" / "samples_final.pt").exists()]
    g = GMM1D(loc=loc)
    xs = torch.linspace(-R, R, 800)
    dens = torch.exp(g.log_prob(xs[:, None]).reshape(-1)).numpy()
    ev = json.loads((ROOT / "leftmass_evals.json").read_text())
    ymax = 0.75 if not (a.wide or a.narrow) else 0.8
    samples = {k: torch.load(d / "samples_final.pt", map_location="cpu", weights_only=False)["model_samples"].reshape(-1).numpy() for k, d in runs}

    def panel(ax, k, x):
        ax.hist(x, bins=100, range=(-R, R), density=True, color="0.6", label="model")
        ax.plot(xs.numpy(), dens, "k", lw=1.5, label="target")
        ax.set_ylim(0, ymax)
        ax.set_xlabel("x")
        ax.set_title(f"{LABEL[k]}: left mass {float((x < 0).mean()):.2f} (target 0.10)", fontsize=9)

    if a.narrow:                                                                                   # plain AS only: the image of the final samples (the evolution figures come from plot_demo1d_evolution.py)
        for k, d in runs:
            f1, a1 = plt.subplots(figsize=(4.2, 3.2))
            panel(a1, k, samples[k])
            a1.set_ylabel("density")
            a1.legend(fontsize=8)
            f1.tight_layout()
            f1.savefig(ROOT / f"{tag}_{'AS' if k == 'as' else k.replace('as_', '')}.png", dpi=200)
            print(f"{LABEL[k]:14} final-sample left mass {float((samples[k] < 0).mean()):.3f} (10 000 samples)")
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.2), sharey=True)
    for ax, (k, d) in zip(np.atleast_1d(axes), runs):
        panel(ax, k, samples[k])
    np.atleast_1d(axes)[0].set_ylabel("density")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / f"{tag}_hist.png", dpi=200)
    for k, d in runs:                                                                                 # one image per method, same axes
        f1, a1 = plt.subplots(figsize=(4.2, 3.2))
        panel(a1, k, samples[k])
        a1.set_ylabel("density")
        a1.legend(fontsize=8)
        f1.tight_layout()
        f1.savefig(ROOT / f"{tag}_{k.replace('as_', '') if k != 'as' else 'AS'}.png", dpi=200)
        plt.close(f1)
    fig2, ax2 = plt.subplots(figsize=(5, 3.2))
    lines = []
    for k, d in runs:
        lm = np.array(ev[f"{pre}{k}"])
        ax2.plot(np.arange(len(lm)) * 100, lm, marker="o", ms=3, label=LABEL[k])
        lines.append(f"{LABEL[k]:14} final-sample left mass {float((samples[k] < 0).mean()):.3f}; evaluations: " + " ".join(f"{v:.2f}" for v in lm) + f"; mean of the last 5: {lm[-5:].mean():.3f} +- {lm[-5:].std(ddof=1) / np.sqrt(5):.3f}")
    ax2.axhline(0.1, color="k", ls="--", lw=1, label="target 0.1")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("left-mode mass")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(ROOT / f"{tag}_leftmass.png", dpi=200)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
