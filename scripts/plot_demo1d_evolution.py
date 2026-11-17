""" Evolution of the plain-AS model distribution on the NARROW 1D target (modes at -1 and +1, weights 0.1 / 0.9), from the samples of every evaluation (train.py +dump_eval_samples=true, eval every 20 epochs),
    and the trend of the plain-AS left-mode mass with the distance between the modes (+-1, +-3, +-5; the +-3 and +-5 runs have one evaluation every 100 epochs, series in figures/demo1d_leftmass_evals.json).
    python scripts/plot_demo1d_evolution.py -> results/demo1d/demo1d_narrow_{evolution,leftmass}.png, demo1d_separation.png and a text summary """
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
run = ROOT / "demo_narrow_as_s0"
files = sorted((run / "eval_samples").glob("ep*.pt"))
epochs = np.array([int(f.stem[2:]) for f in files])
S = [torch.load(f, map_location="cpu").reshape(-1).numpy() for f in files]
lm = np.array([float((x < 0).mean()) for x in S])
g = GMM1D(loc=(-1.0, 1.0))
xs = torch.linspace(-4, 4, 800)
dens = torch.exp(g.log_prob(xs[:, None]).reshape(-1)).numpy()

sel = [0, 20, 100, 300, 600, 980]                                                                  # snapshots shown
fig, axes = plt.subplots(2, 3, figsize=(10, 5.4), sharex=True, sharey=True)
for ax, e in zip(axes.reshape(-1), sel):
    i = int(np.where(epochs == e)[0][0])
    ax.hist(S[i], bins=70, range=(-4, 4), density=True, color="0.6")
    ax.plot(xs.numpy(), dens, "k", lw=1.5)
    ax.set_title(f"epoch {e}: left mass {lm[i]:.2f} (target 0.10)", fontsize=9)
for ax in axes[1]:
    ax.set_xlabel("x")
for ax in axes[:, 0]:
    ax.set_ylabel("density")
fig.tight_layout()
fig.savefig(ROOT / "demo1d_narrow_evolution.png", dpi=200)

fig2, ax2 = plt.subplots(figsize=(5.4, 3.2))
ax2.plot(epochs, lm, marker="o", ms=3, lw=1, label="AS (modes at -1, +1)")
ax2.axhline(0.1, color="k", ls="--", lw=1, label="target 0.1")
ax2.set_xlabel("epoch")
ax2.set_ylabel("left-mode mass")
ax2.set_ylim(0, 0.3)
ax2.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig(ROOT / "demo1d_narrow_leftmass.png", dpi=200)

ev = json.loads((ROOT / "leftmass_evals.json").read_text())
last = {  # same epochs (500-900, every 100) for the three targets
    1: np.array([lm[np.where(epochs == e)[0][0]] for e in (500, 600, 700, 800, 900)]),
    3: np.array(ev["demo_as"][-5:]),
    5: np.array(ev["demo_wide_as"][-5:]),
}
fig3, ax3 = plt.subplots(figsize=(4.6, 3.2))
xs3 = sorted(last)
m = [last[k].mean() for k in xs3]
se = [last[k].std(ddof=1) / np.sqrt(len(last[k])) for k in xs3]
ax3.errorbar([2 * k for k in xs3], m, yerr=se, fmt="o-", capsize=3, label="plain AS (epochs 500-900)")
ax3.axhline(0.1, color="k", ls="--", lw=1, label="target 0.1")
ax3.set_xlabel("distance between the modes")
ax3.set_ylabel("left-mode mass")
ax3.legend(fontsize=8)
fig3.tight_layout()
fig3.savefig(ROOT / "demo1d_separation.png", dpi=200)

print(f"narrow target (modes at -1, +1), plain AS, {len(epochs)} evaluations every 20 epochs: left mass min {lm.min():.3f}, max {lm.max():.3f}; epoch 0: {lm[0]:.3f}; mean of all {lm.mean():.3f} +- {lm.std(ddof=1) / np.sqrt(len(lm)):.3f}; epochs 500-900 (every 100): {last[1].mean():.3f} +- {last[1].std(ddof=1) / np.sqrt(5):.3f}")
print("plain AS, left mass (mean +- SE over the evaluations at epochs 500-900): " + "; ".join(f"modes at +-{k}: {last[k].mean():.3f} +- {last[k].std(ddof=1) / np.sqrt(5):.3f}" for k in xs3))
