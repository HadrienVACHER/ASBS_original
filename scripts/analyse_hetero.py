""" Analyse of the heterogeneous-GMM runs (version 2, geometry chosen on AS alone): per cell |log R̄|, light quarter, min R̄,
    energy_w2, energy calls per gradient update, and the weight error against the CUMULATIVE energy calls.

      python scripts/analyse_hetero.py --geom "radius_min=4.5 radius_max=6.5 weight_ratio=20 scale_min=0.6 scale_max=1.4 min_sep=2.5"
                                       [--prefix hetero24d_v2] [--cells as barker_affine ...] [--out results/hetero_v2]

    Cells (experiment-name suffix): as | barker_transl | barker_affine | barker_aff_s8 | barker_aff_s4 | barker_affine_oracle.
    Ratios R̄_j are averaged over the last 30 evaluations; |log R̄| = mean_j |log R̄_j|; the learning curve uses a window of 10
    evaluations. Calls per epoch: 512 x (1 gradient for the adjoint + [Barker] E(x1) + E(translates) + denoising gradients/energies)
    + [affine, Laplace covariances] 2 D gradients per centre (one refit per epoch). E and ∇E count as one call each.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils.dist_utils import HeteroGMM  # noqa: E402

D, BATCH, UPDATES = 24, 512, 100
CELLS = [  # (suffix, label, colour, linestyle, laplace covariances every epoch)
    ("as", "AS", "tab:blue", "-", False),
    ("barker_transl", "Barker translation", "tab:orange", "-", False),
    ("barker_affine", "Barker affine (full)", "tab:red", "-", True),
    ("barker_aff_s8", "Barker affine s=8", "tab:green", "--", True),
    ("barker_aff_s4", "Barker affine s=4", "tab:purple", "--", True),
    ("barker_affine_oracle", "Barker affine, oracle", "k", ":", False),
]


def parse_geom(s):
    out = {}
    for kv in s.split():
        k, v = kv.split("=")
        try:
            out[k] = int(v) if k == "weight_ratio" and float(v).is_integer() else float(v)
        except ValueError:
            out[k] = v # layout=simplex
    return out


def load(run):
    keys = ["energy_w2"] + [f"mode_ratio/{j}" for j in range(D)]
    h = run.history(keys=keys, samples=6000, pandas=True, x_axis="_step").dropna(subset=["energy_w2"])
    steps, ew2, R = h["_step"].values.astype(float), h["energy_w2"].values, h[keys[1:]].values
    c = run.history(keys=["mh/energy_evals_per_sample", "denoise/grad_evals_per_sample", "denoise/E_evals_per_sample", "centres/n"],
                    samples=6000, pandas=True, x_axis="_step")
    n_ep = int(steps.max()) + 1
    per_epoch = np.full(n_ep, float(BATCH))
    n_cent = np.zeros(n_ep)
    if "mh/energy_evals_per_sample" in c:
        cc = c.set_index("_step").reindex(range(n_ep)).ffill().bfill()
        per_epoch = BATCH * (2.0 + cc["mh/energy_evals_per_sample"].values + cc["denoise/grad_evals_per_sample"].values
                             + cc["denoise/E_evals_per_sample"].values)
        n_cent = cc["centres/n"].values if "centres/n" in cc else np.full(n_ep, float(D))
    return steps, R, ew2, per_epoch, n_cent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", required=True, help='ex: "radius_min=4.5 radius_max=6.5 weight_ratio=20 scale_min=0.6 scale_max=1.4 min_sep=2.5"')
    ap.add_argument("--prefix", default="hetero24d_v2")
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--out", default="results/hetero_v2")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb

    g = HeteroGMM(dim=D, K=D, seed=0, **parse_geom(args.geom))
    w = g.weights.numpy().astype(float); w /= w.sum()
    light = np.argsort(w)[:D // 4]
    runs = list(wandb.Api(timeout=120).runs("CCDS_NTU/adjoint_sampler_cv", per_page=300, order="-created_at"))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    print(f"cible: poids {w.min():.4f}..{w.max():.4f}; quart léger = modes {sorted(light.tolist())}")
    print(f"{'cellule':22s}{'n':>3}{'|log R|':>16}{'quart léger':>16}{'min R':>14}{'energy_w2':>13}{'appels/màj':>11}")
    summary = {}
    for suffix, label, col, ls, laplace in CELLS:
        if args.cells and suffix not in args.cells:
            continue
        seeds = {}
        for r in runs:
            m = re.match(rf"^{re.escape(args.prefix)}_{suffix}_K{D}_s(\d)(?:_random|_simplex)?$", r.name)
            if m and r.state == "finished" and int(m.group(1)) not in seeds:
                seeds[int(m.group(1))] = r
        if not seeds:
            print(f"{label:22s}  (aucun run terminé)")
            continue
        data = [load(r) for _, r in sorted(seeds.items())]
        vals, ew, calls, curves = [], [], [], []
        for steps, R, ew2, per_epoch, n_cent in data:
            Rm = R[-30:].mean(0)
            vals.append((np.mean(np.abs(np.log(np.maximum(Rm, 1e-3)))), Rm[light].mean(), Rm.min()))
            ew.append(ew2[-30:].mean())
            extra = (2.0 * D * n_cent) if laplace else 0.0
            pe = per_epoch + extra
            calls.append(pe.mean() / UPDATES)
            cum = np.cumsum(pe)
            win = 10
            ep = [steps[i - 1] for i in range(win, len(steps) + 1)]
            ls_ = [np.mean(np.abs(np.log(np.maximum(R[i - win:i].mean(0), 1e-3)))) for i in range(win, len(steps) + 1)]
            curves.append((np.array(ep), np.array(ls_), cum[np.minimum(np.array(ep).astype(int), len(cum) - 1)]))
        v = np.array(vals)
        sd = v.std(0) if len(v) > 1 else np.zeros(3)
        print(f"{label:22s}{len(v):3d}{v[:, 0].mean():9.3f}±{sd[0]:.3f}{v[:, 1].mean():9.3f}±{sd[1]:.3f}{v[:, 2].mean():8.3f}±{sd[2]:.3f}"
              f"{np.mean(ew):8.2f}±{np.std(ew):.2f}{np.mean(calls):11.1f}")
        summary[label] = (v[:, 0].mean(), np.mean(calls))
        n = min(len(c[0]) for c in curves)
        e_mean, l_mean = curves[0][0][:n], np.mean([c[1][:n] for c in curves], 0)
        l_sd, c_mean = np.std([c[1][:n] for c in curves], 0), np.mean([c[2][:n] for c in curves], 0)
        for a_, xs in ((ax[0], e_mean), (ax[1], c_mean)):
            a_.plot(xs, l_mean, ls, color=col, lw=2, label=label)
            a_.fill_between(xs, l_mean - l_sd, l_mean + l_sd, color=col, alpha=0.12)
        ax[2].scatter([np.mean(calls)], [v[:, 0].mean()], color=col, s=60, label=label)
    ax[0].set_xlabel("epoch"); ax[1].set_xlabel("cumulative energy calls"); ax[1].set_xscale("log")
    ax[2].set_xlabel("energy calls per gradient update"); ax[2].set_xscale("log")
    for a_ in ax:
        a_.set_ylabel("|log R̄| (10-evaluation window)" if a_ is not ax[2] else "|log R̄| (last 30 evaluations)")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "hetero_weight_error.png", dpi=130)
    print(f"\nsaved {out}/hetero_weight_error.png")


if __name__ == "__main__":
    main()
