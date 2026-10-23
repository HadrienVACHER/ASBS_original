""" 2x2 at d = 36 (small 128x4 / big 256x6 controller) x (AS / Barker-threshold gamma = 2), seeds 0-2, 2000 epochs.
    Final time-averaged metrics (30 last evaluations) and LEARNING CURVES of the time-averaged |log R| and the worst mode
    against the epochs and against the cumulative number of energy calls.

      python scripts/analyse_capacity.py [--small-tag git:f0f31e6] [--window 10]

    Time-averaged = the per-mode ratios are averaged over a window of `window` evaluations (one every 50 epochs) before
    |log R|, the light-quarter ratio and the minimum are taken.
    Cost per epoch = 512 samples x (1 gradient for the adjoint + [Barker] 1 E(x1) + E(translates) + denoising gradients and
    energies). E and ∇E count as one call each (∇E is typically 2-3x more expensive than E).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils.dist_utils import GMM  # noqa: E402

CELLS = {
    "AS small": ("gmm36d_as_cmp", "small"),
    "Barker small": ("gmm36d_as_barker_g2_thr", "small"),
    "AS big": ("gmm36d_as_big_cap", "any"),
    "Barker big": ("gmm36d_as_barker_g2_thr_big", "any"),
}
D = 36
BATCH = 512


def find_runs(api, small_tag):
    runs = {c: {} for c in CELLS}
    for r in api.runs("CCDS_NTU/adjoint_sampler_cv", per_page=200, order="-created_at"):
        for cell, (base, kind) in CELLS.items():
            m = re.match(rf"^{re.escape(base)}_K{D}_s(\d)_spread", r.name)
            if not m or r.summary.get("_step", 0) < 50:
                continue
            if kind == "small" and small_tag not in r.tags:
                continue
            seed = int(m.group(1))
            if seed not in runs[cell]: # most recent first
                runs[cell][seed] = r
    return runs


def load(run):
    keys = [f"mode_ratio/{j}" for j in range(D)]
    h = run.history(keys=["mode_tv"] + keys, samples=4000, pandas=True, x_axis="_step").dropna(subset=["mode_tv"])
    steps, R = h["_step"].values.astype(float), h[keys].values
    ck = ["mh/energy_evals_per_sample", "denoise/grad_evals_per_sample", "denoise/E_evals_per_sample"]
    c = run.history(keys=ck, samples=4000, pandas=True, x_axis="_step")
    per_epoch = np.full(int(c["_step"].max()) + 1 if len(c) else int(steps.max()) + 1, 1.0)  # AS: one ∇E per sample
    if "mh/energy_evals_per_sample" in c:
        cc = c.set_index("_step").reindex(range(len(per_epoch))).ffill().bfill()
        per_epoch = 2.0 + cc["mh/energy_evals_per_sample"].values + cc.get("denoise/grad_evals_per_sample", 0).values \
            + cc.get("denoise/E_evals_per_sample", 0).values
    cum = np.cumsum(per_epoch) * BATCH
    return steps, R, cum


def metrics(Rm, w, light):
    return (float(np.mean(np.abs(np.log(np.maximum(Rm, 1e-6))))), float(Rm[light].mean()), float(Rm.min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small-tag", default="git:f0f31e6")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--out", default="results/capacity")
    args = ap.parse_args()
    import wandb
    api = wandb.Api(timeout=120)
    g = GMM(dim=D, K=D, seed=0, layout="spread", weight_law="loguniform", weight_ratio=20, radius=4.25)
    w = g.weights.numpy().astype(float); w = w / w.sum()
    light = np.argsort(w)[:D // 4]

    runs = find_runs(api, args.small_tag)
    data = {}
    for cell, rr in runs.items():
        print(f"{cell:13s}: seeds found {sorted(rr)} (steps {[rr[s].summary.get('_step') for s in sorted(rr)]})")
        data[cell] = {s: load(r) for s, r in rr.items()}

    print(f"\nFinal (last 30 evaluations, ratios averaged over the window), mean ± std over seeds")
    print(f"{'cell':13s}{'n':>3}{'|log R|':>16}{'light quarter':>18}{'min R':>14}")
    for cell, d in data.items():
        if not d:
            continue
        vals = np.array([metrics(R[-30:].mean(0), w, light) for (_, R, _) in d.values()])
        sd = vals.std(0) if len(vals) > 1 else np.zeros(3)
        print(f"{cell:13s}{len(d):3d}{vals[:, 0].mean():9.3f}±{sd[0]:.3f}{vals[:, 1].mean():12.3f}±{sd[1]:.3f}{vals[:, 2].mean():9.3f}±{sd[2]:.3f}")

    # learning curves
    curves = {}
    for cell, d in data.items():
        for s, (steps, R, cum) in d.items():
            W = args.window
            ep, ls, mn, cs = [], [], [], []
            for i in range(W, len(steps) + 1):
                m = metrics(R[i - W:i].mean(0), w, light)
                ep.append(steps[i - 1]); ls.append(m[0]); mn.append(m[2]); cs.append(cum[min(int(steps[i - 1]), len(cum) - 1)])
            curves.setdefault(cell, []).append((np.array(ep), np.array(ls), np.array(mn), np.array(cs)))
    print("\nlearning curves: |log R| (time-averaged over the last", args.window, "evaluations) at epochs 500 / 1000 / 1500 / 2000, and the worst mode")
    for cell, cs_ in curves.items():
        row = []
        for target in (500, 1000, 1500, 2000):
            v = [(c[1][np.argmin(abs(c[0] - target))], c[2][np.argmin(abs(c[0] - target))]) for c in cs_ if len(c[0])]
            row.append(f"{np.mean([a for a, _ in v]):.3f}/{np.mean([b for _, b in v]):.2f}")
        print(f"  {cell:13s} " + "  ".join(row) + f"   cumulative cost at epoch 2000: {np.mean([c[3][-1] for c in cs_]) / 1e6:6.1f} M calls")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(2, 2, figsize=(11, 7))
        colors = {"AS small": "tab:blue", "Barker small": "tab:orange", "AS big": "tab:cyan", "Barker big": "tab:red"}
        for cell, cs_ in curves.items():
            for k, (row, xi) in enumerate(((0, 0), (0, 3), (1, 0), (1, 3))):
                pass
            for (row, col, yi, xi) in ((0, 0, 1, 0), (0, 1, 1, 3), (1, 0, 2, 0), (1, 1, 2, 3)):
                for c in cs_:
                    ax[row, col].plot(c[xi], c[yi], color=colors[cell], alpha=0.35, lw=1)
                n = min(len(c[0]) for c in cs_)
                ax[row, col].plot(np.mean([c[xi][:n] for c in cs_], 0), np.mean([c[yi][:n] for c in cs_], 0), color=colors[cell], lw=2, label=cell)
        for a_, (xl, yl) in zip(ax.ravel(), (("epoch", "|log R|"), ("cumulative energy calls", "|log R|"), ("epoch", "min R"), ("cumulative energy calls", "min R"))):
            a_.set_xlabel(xl); a_.set_ylabel(yl)
        ax[0, 1].set_xscale("log"); ax[1, 1].set_xscale("log"); ax[0, 0].legend()
        fig.tight_layout(); fig.savefig(out / "capacity_curves.png", dpi=110)
        print(f"\nsaved {out}/capacity_curves.png")
    except Exception as ex:
        print("figure skipped:", ex)


if __name__ == "__main__":
    main()
