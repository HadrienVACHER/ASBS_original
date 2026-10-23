""" Final GMM figure and summary table (d = 24, 36): model ratio R̄_j against log w_j, one curve per cell
    (AS / Barker x small 128x4 / big 256x6), mean ± std over the 3 seeds (ratios averaged over the last 30 evaluations),
    and the table |log R̄|, light quarter, min R̄, energy_w2, energy calls per gradient update, learning curves.

      python scripts/plot_gmm_final.py [--out results/gmm_final]

    Cells are looked up in W&B by experiment name; a cell with no finished run is reported as missing and skipped.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils.dist_utils import GMM  # noqa: E402

BATCH, UPDATES_PER_EPOCH = 512, 100
# (label, name pattern with {D}, tag required or None, line style)
CELLS = [
    ("AS small", "gmm{D}d_as_cmp", "git:f0f31e6", "-"),
    ("Barker g=3 small", "gmm{D}d_as_barker_g3_thr", "git:f0f31e6", "-"),
    ("Barker g=2 small", "gmm{D}d_as_barker_g2_thr", "git:f0f31e6", ":"),
    ("AS big", "gmm{D}d_as_big_cap", None, "--"),
    ("Barker g=2 big", "gmm{D}d_as_barker_g2_thr_big", None, "--"),
    ("Barker g=3 big", "gmm{D}d_as_barker_g3_thr_big", None, "--"),
]
COLORS = {"AS small": "tab:blue", "AS big": "tab:cyan", "Barker g=3 small": "tab:red", "Barker g=2 small": "tab:orange",
          "Barker g=2 big": "tab:purple", "Barker g=3 big": "tab:brown"}


def find(runs, base, D, tag):
    out = {}
    for r in runs:
        m = re.match(rf"^{re.escape(base)}_K{D}_s(\d)_spread", r.name)
        if m and r.state == "finished" and (tag is None or tag in r.tags) and int(m.group(1)) not in out:
            out[int(m.group(1))] = r
    return out


def load(run, D):
    keys = ["energy_w2"] + [f"mode_ratio/{j}" for j in range(D)]
    h = run.history(keys=keys, samples=6000, pandas=True, x_axis="_step").dropna(subset=["energy_w2"])
    steps, R, ew2 = h["_step"].values.astype(float), h[keys[1:]].values, h["energy_w2"].values
    c = run.history(keys=["mh/energy_evals_per_sample", "denoise/grad_evals_per_sample", "denoise/E_evals_per_sample"],
                    samples=6000, pandas=True, x_axis="_step")
    if "mh/energy_evals_per_sample" in c:
        cc = c.set_index("_step").ffill().bfill()
        per_epoch = BATCH * (2.0 + cc["mh/energy_evals_per_sample"] + cc.get("denoise/grad_evals_per_sample", 0)
                             + cc.get("denoise/E_evals_per_sample", 0)).mean()
    else:
        per_epoch = BATCH * 1.0 # AS: one gradient per sample
    return steps, R, ew2, per_epoch / UPDATES_PER_EPOCH


def metrics(Rm, light):
    return float(np.mean(np.abs(np.log(np.maximum(Rm, 1e-6))))), float(Rm[light].mean()), float(Rm.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gmm_final")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb
    runs = list(wandb.Api(timeout=120).runs("CCDS_NTU/adjoint_sampler_cv", per_page=300, order="-created_at"))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    lines = []
    for ax, D in zip(axes, (24, 36)):
        g = GMM(dim=D, K=D, seed=0, layout="spread", weight_law="loguniform", weight_ratio=20, radius=4.25)
        w = g.weights.numpy().astype(float); w /= w.sum()
        order, light = np.argsort(w), np.argsort(w)[:D // 4]
        x = np.log(w[order])
        for label, pat, tag, ls in CELLS:
            rr = find(runs, pat.format(D=D), D, tag)
            if not rr:
                lines.append(f"d={D:2d} {label:18s} : MISSING")
                continue
            data = {s: load(r, D) for s, r in sorted(rr.items())}
            Rm = np.array([d[1][-30:].mean(0) for d in data.values()])           # (seeds, modes)
            mt = np.array([metrics(r, light) for r in Rm])
            ew = np.array([d[2][-30:].mean() for d in data.values()])
            cost = np.mean([d[3] for d in data.values()])
            # learning curves of |log R| (10-evaluation window) at epochs 500 ... 2000
            lc = []
            for ep in (500, 1000, 1500, 2000):
                v = []
                for steps, R, _, _ in data.values():
                    i = int(np.argmin(abs(steps - (ep - 1))))
                    v.append(np.mean(np.abs(np.log(np.maximum(R[max(i - 9, 0):i + 1].mean(0), 1e-6)))))
                lc.append(np.mean(v))
            mu, sd = Rm.mean(0)[order], Rm.std(0)[order]
            ax.plot(x, mu, ls, color=COLORS[label], lw=1.8, label=f"{label} (n={len(data)})")
            ax.fill_between(x, mu - sd, mu + sd, color=COLORS[label], alpha=0.12)
            lines.append(f"d={D:2d} {label:18s} n={len(data)} |logR| {mt[:, 0].mean():.3f}±{mt[:, 0].std():.3f}  light {mt[:, 1].mean():.3f}±{mt[:, 1].std():.3f}  "
                         f"minR {mt[:, 2].mean():.3f}±{mt[:, 2].std():.3f}  e_w2 {ew.mean():.2f}±{ew.std():.2f}  calls/update {cost:6.1f}  "
                         f"|logR|@500/1000/1500/2000 {lc[0]:.3f}/{lc[1]:.3f}/{lc[2]:.3f}/{lc[3]:.3f}")
        ax.axhline(1, color="k", lw=0.7)
        ax.set_xlabel("log w_j"); ax.set_ylabel("model ratio R̄_j (30 last evaluations)"); ax.set_title(f"d = K = {D}")
        ax.set_ylim(0, 1.5); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "gmm_ratio_vs_logw.png", dpi=130)
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text)
    print(text)
    print(f"\nsaved {out}/gmm_ratio_vs_logw.png and summary.txt")


if __name__ == "__main__":
    main()
