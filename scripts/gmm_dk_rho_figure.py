""" Coverage and beta versus rho = sqrt(d nu1)/|mu| (predictions_dk.md, addendum 2). One colour per (d, K), one marker per Delta, open marker = sigma_max ablation run. """
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

rows = json.load(open(sys.argv[2]))
COL = {(8, 2): "#2a78d6", (8, 8): "#eb6834", (32, 2): "#1baf7a", (32, 8): "#6250d6"}
MK = {4.0: "o", 5.0: "s", 6.0: "^", 7.0: "D", 8.0: "v"}
INK, MUTED, BG = "#0b0b0b", "#52514e", "#fcfcfb"
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=BG)
for r in rows:
    c, m = COL[(r["d"], r["K"])], MK[r["delta"]]
    kw = dict(marker=m, ms=9, color=c, mec=c if not r["abl"] else c, mfc=BG if r["abl"] else c, mew=1.8, ls="none")
    ax[0].plot(r["rho"], r["coverage"] / r["K"], **kw)
    if np.isfinite(r["beta"]):
        ax[1].errorbar(r["rho"], r["beta"], yerr=r["beta_se"] if np.isfinite(r["beta_se"]) else 0, capsize=3, **kw)
ax[0].set(title="Coverage (fraction of modes with p̂ ≥ 0.25 p)", ylim=(-0.03, 1.08))
ax[1].set(title="Shrinkage β (0: right weights; −1: uniform)", ylim=(-0.8, 3.6))
ax[1].axhline(0, color=MUTED, lw=1, ls=":")
for a in ax:
    a.set_xscale("log"); a.set_xticks([1.5, 2, 3, 4, 6]); a.set_xticklabels(["1.5", "2", "3", "4", "6"]); a.set_xlabel("ρ = √(d ν₁) / ‖μ‖  (typical base radius / mode radius)", color=MUTED)
    a.set_facecolor(BG); a.grid(color="#e4e3de", lw=0.8); a.spines[["top", "right"]].set_visible(False); a.title.set_fontsize(10.5); a.title.set_color(INK); a.tick_params(colors=MUTED)
h1 = [Line2D([], [], marker="o", ls="none", color=c, ms=8, label=f"d = {d}, K = {k}") for (d, k), c in COL.items()]
h2 = [Line2D([], [], marker=m, ls="none", color=MUTED, ms=8, label=f"Δ = {int(D)}") for D, m in MK.items()]
h3 = [Line2D([], [], marker="o", ls="none", color=MUTED, mfc=BG, mew=1.8, ms=8, label="σ_max × 0.5 or × 2")]
l1 = ax[0].legend(handles=h1, frameon=False, fontsize=8.5, loc="lower left"); ax[0].add_artist(l1)
ax[1].legend(handles=h2 + h3, frameon=False, fontsize=8.5, loc="upper left", ncol=2)
fig.suptitle("AS on the (d, K, Δ) target, one seed, 24 runs: coverage and β versus ρ (H_ρ: a common threshold in ρ would give a step at one ρ)", fontsize=10.5, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(sys.argv[1], dpi=150, facecolor=BG); print("saved", sys.argv[1])
