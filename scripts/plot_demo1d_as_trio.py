""" One figure with the three plain-AS runs of the 1D demo (modes at +-1, +-3, +-5; weights 0.1 / 0.9, scale 0.5): final samples (10 000) against the target density.
    python scripts/plot_demo1d_as_trio.py -> results/demo1d/demo1d_AS_trio.png """
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
CASES = [("demo_narrow_as_s0", 1.0, 4), ("demo_as_s0", 3.0, 6), ("demo_wide_as_s0", 5.0, 8)]      # run, mode position, half-width of the x axis
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.2), sharey=True)
for ax, (run, m, R) in zip(axes, CASES):
    x = torch.load(ROOT / run / "samples_final.pt", map_location="cpu", weights_only=False)["model_samples"].reshape(-1).numpy()
    xs = torch.linspace(-R, R, 800)
    dens = torch.exp(GMM1D(loc=(-m, m)).log_prob(xs[:, None]).reshape(-1)).numpy()
    ax.hist(x, bins=100, range=(-R, R), density=True, color="0.6", label="AS")
    ax.plot(xs.numpy(), dens, "k", lw=1.5, label="target")
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("x")
    ax.set_title(f"modes at ±{m:g}: left mass {float((x < 0).mean()):.2f} (target 0.10)", fontsize=9)
axes[0].set_ylabel("density")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "demo1d_AS_trio.png", dpi=200)
