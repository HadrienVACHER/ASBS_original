""" Where does a trained controller leak mass? Committed mass ŵ_k(s) = E_{X_s ~ model}[π*(k | X_s)], π*(k | x) the posterior
    of mode k given X_s = x under the TRUE target (closed form on the GMM, unit-variance modes, Dirac start). Under the exact
    drift X_s is the bridge to p1 and ŵ(s) = w for every s (a martingale: flat curve). Where ŵ_k(s) drops localises the leak:
    early (s <~ 0.15) the correction is poorly learned, late a leak near the mode that no small-t control variate can fix.

      python scripts/committed_mass.py --dim 36 --K 36 --sigma-max 2.4509 --focus 22 \
          --runs as=results/committed_mass/as oracle=results/committed_mass/oracle thr=results/committed_mass/thr

    A run directory holds config.yaml (controller architecture) and controller_final.pt (dump_final=true).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import hydra  # noqa: E402
from test_mh_matcher import posterior  # noqa: E402
from adjoint_samplers.components.sde import ControlledSDE, VESDE, sdeint  # noqa: E402
from adjoint_samplers.utils.dist_utils import GMM  # noqa: E402


@torch.no_grad()
def committed_masses(target, ref_sde, controller, N, nfe, s_grid, device, batch=5000):
    sde = ControlledSDE(ref_sde, controller).to(device)
    timesteps = torch.linspace(0, 1, nfe + 1, device=device)
    idx = [int(round(s * nfe)) for s in s_grid]
    acc = torch.zeros(len(idx), target.K, device=device)
    n_done = 0
    for _ in range(0, N, batch):
        x0 = torch.zeros(batch, target.dim, device=device)
        xs = sdeint(sde, x0, timesteps, only_boundary=False)
        for a, i in enumerate(idx):
            t = timesteps[i].reshape(1)
            nut = ref_sde._pt_gauss_param(t, torch.zeros_like(t))[1].item()
            nu1 = ref_sde._pt_gauss_param(torch.ones(1, device=device), torch.zeros(1, device=device))[1].item()
            kappa = nut / nu1
            if i >= nfe: # t = 1: the posterior is the Dirac at x, the mode is the nearest
                acc[a] += torch.bincount(target.assign(xs[i]), minlength=target.K).float()
            else:
                acc[a] += posterior(target, kappa, nu1, xs[i])[0].sum(0)
        n_done += batch
    return acc / n_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--radius", type=float, default=4.25)
    ap.add_argument("--weight-ratio", type=float, default=20.0)
    ap.add_argument("--sigma-max", type=float, required=True)
    ap.add_argument("--sigma-min", type=float, default=0.001)
    ap.add_argument("--nfe", type=int, default=200)
    ap.add_argument("--N", type=int, default=20000)
    ap.add_argument("--runs", nargs="+", required=True, help="name=run_dir")
    ap.add_argument("--which", default="controller", choices=["controller", "ema"])
    ap.add_argument("--focus", type=int, nargs="*", default=[])
    ap.add_argument("--out", default="results/committed_mass/curves")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    K = args.K or args.dim
    target = GMM(dim=args.dim, K=K, seed=0, layout="spread", weight_law="loguniform", weight_ratio=args.weight_ratio, radius=args.radius)
    target.K = K
    target.to(device)
    w = target.weights
    order = torch.argsort(w)
    light = order[:K // 4]
    ref_sde = VESDE(sigma_min=args.sigma_min, sigma_max=args.sigma_max).to(device)
    s_grid = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

    results = {}
    for item in args.runs:
        name, path = item.split("=")
        cfg = OmegaConf.load(Path(path) / "config.yaml")
        ckpt = torch.load(Path(path) / "controller_final.pt", map_location="cpu")
        controller = hydra.utils.instantiate(cfg.controller).to(device)
        sd = ckpt[args.which] if ckpt.get(args.which) is not None else ckpt["controller"]
        controller.load_state_dict(sd)
        controller.eval()
        torch.manual_seed(0)
        m = committed_masses(target, ref_sde, controller, args.N, args.nfe, s_grid, device)
        results[name] = m.cpu()
        print(f"\n=== {name}: {ckpt.get('exp_name')} ({args.which}), N = {args.N}")
        print(f"{'s':>6}{'TV(ŵ(s), w)':>13}{'light quarter ratio':>21}{'min ratio':>11}" + "".join(f"{'mode ' + str(j) + ' ratio':>15}" for j in args.focus))
        for a, s in enumerate(s_grid):
            r = m[a] / w
            tv = 0.5 * (m[a] - w).abs().sum().item()
            print(f"{s:6.2f}{tv:13.4f}{r[light].mean().item():21.3f}{r.min().item():11.3f}" + "".join(f"{r[j].item():15.3f}" for j in args.focus))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"s": s_grid, "w": w.cpu().tolist(), "runs": {n: m.tolist() for n, m in results.items()}}, open(out / "committed_mass.json", "w"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        for n, m in results.items():
            r = (m / w.cpu())
            ax[0].plot(s_grid, 0.5 * (m - w.cpu()).abs().sum(1), marker="o", label=n)
            ax[1].plot(s_grid, r[:, light.cpu()].mean(1), marker="o", label=n)
            if args.focus:
                ax[2].plot(s_grid, r[:, args.focus[0]], marker="o", label=n)
        ax[0].set_ylabel("TV(ŵ(s), w)"); ax[1].set_ylabel("light-quarter ratio"); ax[2].set_ylabel(f"mode {args.focus[0] if args.focus else '-'} ratio")
        for a_ in ax:
            a_.set_xlabel("s"); a_.axhline(1 if a_ is not ax[0] else 0, color="k", lw=0.5); a_.legend()
        fig.tight_layout(); fig.savefig(out / "committed_mass.png", dpi=110)
        print(f"\nsaved {out}/committed_mass.json and .png")
    except Exception as ex:
        print("figure skipped:", ex)


if __name__ == "__main__":
    main()
