""" Figure of the (d, K, Delta) sweep: coverage, beta and path-IS ESS versus Delta, one curve per (d, K). Reads the JSON of scripts/gmm_dk_analysis.py.
    Path-IS was only computed for the 5 pilot runs (the lot cap was reached before the IS of the sweep). """
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = [r for f in sys.argv[2:] for r in json.load(open(f))]
res = [r for r in res if "_sx" not in r["run"]]
ESS = {(8, 2, 8): 0.4524, (8, 8, 8): 0.3924, (32, 2, 8): 0.1007, (32, 8, 8): 0.0658, (32, 8, 4): 0.1656}   # pilot analysis (results/gmm_dk/analysis.log)
STYLE = {(8, 2): ("#2a78d6", "o", "-"), (8, 8): ("#eb6834", "s", "--"), (32, 2): ("#1baf7a", "^", "-"), (32, 8): ("#6250d6", "D", "--")}
INK, MUTED = "#0b0b0b", "#52514e"
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2), facecolor="#fcfcfb")
for key, (c, m, ls) in STYLE.items():
    rows = sorted([r for r in res if (r["d"], r["K"]) == key], key=lambda r: r["delta"])
    D = [r["delta"] for r in rows]
    lab = f"d = {key[0]}, K = {key[1]}"
    off = {(8, 2): -0.06, (8, 8): -0.02, (32, 2): 0.02, (32, 8): 0.06}[key]        # small horizontal offsets so overlapping points stay visible
    ax[0].plot(np.array(D) + off, [r["coverage"] / r["K"] for r in rows], color=c, marker=m, ls=ls, lw=2, ms=7, mec="#fcfcfb", label=lab)
    b = np.array([r["beta"] for r in rows], float); se = np.array([r["beta_se"] for r in rows], float)
    ok = np.isfinite(b)
    ax[1].errorbar(np.array(D)[ok] + off, b[ok], yerr=np.where(np.isfinite(se[ok]), se[ok], 0), color=c, marker=m, ls=ls, lw=2, ms=7, mec="#fcfcfb", capsize=3, label=lab)
    e = [(r["delta"], ESS[(r["d"], r["K"], r["delta"])]) for r in rows if (r["d"], r["K"], r["delta"]) in ESS]
    if e:
        ax[2].plot([x for x, _ in e], [y for _, y in e], color=c, marker=m, ls=ls, lw=2, ms=8, mec="#fcfcfb", label=lab)
ax[0].set(title="Coverage (fraction of modes with p̂ ≥ 0.25 p)", xlabel="Δ (distance between modes, in std)", ylim=(-0.03, 1.08))
ax[1].axhline(0, color=MUTED, lw=1, ls=":"); ax[1].axhline(-1, color=MUTED, lw=1, ls=":")
ax[1].set(title="Shrinkage β (0: right weights, −1: uniform)", xlabel="Δ (distance between modes, in std)", ylim=(-1.2, 3.4))
ax[1].text(4.05, -0.95, "β = −1", color=MUTED, fontsize=8); ax[1].text(4.05, 0.05, "β = 0", color=MUTED, fontsize=8)
ax[2].set(title="Path-IS ESS fraction (pilot runs only)", xlabel="Δ (distance between modes, in std)", ylim=(0, 0.55))
for a in ax:
    a.set_facecolor("#fcfcfb"); a.grid(color="#e4e3de", lw=0.8); a.spines[["top", "right"]].set_visible(False); a.set_xticks([4, 5, 6, 7, 8])
    a.title.set_fontsize(10.5); a.title.set_color(INK); a.tick_params(colors=MUTED); a.xaxis.label.set_color(MUTED)
ax[0].legend(frameon=False, fontsize=9, loc="lower left")
fig.suptitle("AS on the (d, K, Δ) target, one seed: coverage, shrinkage β and path-IS ESS versus Δ  (β undefined where < 2 modes are covered)", fontsize=10.5, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(sys.argv[1], dpi=150, facecolor=fig.get_facecolor())
print("saved", sys.argv[1])
