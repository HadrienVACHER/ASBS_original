""" Independent Metropolis-Hastings on the demo target, no Adjoint Sampling.

    Proposal r: the target with equal mode weights. Target p = e^{-E} / Z. Acceptance

        α(x, x') = min(1, e^{-[E(x') - E(x)]} r(x) / r(x'))

    python scripts/imh_demo.py [--chains 2000 --steps 200 --seed 0 --wandb]

    Saves imh_demo.png (histograms, mode weight and W2 along the chain) in --out.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adjoint_samplers.energies.dist_energy import DistEnergy
from adjoint_samplers.utils.dist_utils import GMM1D


def imh_step(x, E, logr, energy, r):
    y = r.sample([x.shape[0],]).reshape(x.shape)
    Ey = energy.eval(y).reshape(-1)
    logry = r.log_prob(y).reshape(-1)
    log_alpha = - (Ey - E) + logr - logry
    accept = torch.rand_like(E).log() < log_alpha
    return (
        torch.where(accept[:, None], y, x),
        torch.where(accept, Ey, E),
        torch.where(accept, logry, logr),
        accept.float().mean().item(),
    )


def metrics(x, p, ref):
    """ left-mode mass (target p.weights[0]), |left_mass - target| and 1D W2 to the target (quantile coupling) """
    x = x.reshape(-1)
    locs = p.distr.component_distribution.base_dist.loc.reshape(-1)
    left = ((x - locs[0]).abs() < (x - locs[1]).abs()).float().mean().item()
    q = ((torch.arange(len(x)) + 0.5) / len(x) * len(ref)).long()
    w2 = (x.sort().values - ref[q]).pow(2).mean().sqrt().item()
    return {"left_mass": left, "mode_tv": abs(left - p.weights[0].item()), "w2": w2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/imh_demo")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    p = GMM1D()
    r = p.uniform_weights()
    energy = DistEnergy(p)
    ref = p.sample([100000,]).reshape(-1).sort().values

    x = r.sample([args.chains,]).reshape(-1, 1) # chains start from the proposal
    E, logr = energy.eval(x).reshape(-1), r.log_prob(x).reshape(-1)

    if args.wandb:
        import wandb
        wandb.init(project="adjoint_sampler_cv", name=f"imh_demo_c{args.chains}_s{args.seed}", config=vars(args))

    hist = []
    m0 = metrics(x, p, ref)
    hist.append({"step": 0, "accept_rate": float("nan"), **m0})
    for step in range(1, args.steps + 1):
        x, E, logr, acc = imh_step(x, E, logr, energy, r)
        hist.append({"step": step, "accept_rate": acc, **metrics(x, p, ref)})
        if args.wandb:
            wandb.log({k: v for k, v in hist[-1].items() if k != "step"}, step=step)
        if step in (1, 5, 10, 25, 50, 100, args.steps):
            h = hist[-1]
            print(f"step {step:4d}  accept {acc:.3f}  left_mass {h['left_mass']:.3f}  "
                  f"mode_tv {h['mode_tv']:.3f}  w2 {h['w2']:.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = [h["step"] for h in hist]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    grid = torch.linspace(-6, 6, 400).unsqueeze(-1)
    ax[0].hist(x.reshape(-1), bins=60, density=True, alpha=0.6, label=f"IMH chains, step {args.steps}")
    ax[0].plot(grid, p.log_prob(grid).exp(), "k", label="target p")
    ax[0].plot(grid, r.log_prob(grid).exp(), "k--", label="proposal r")
    ax[0].legend()
    ax[0].set_title("samples")
    ax[1].plot(steps, [h["left_mass"] for h in hist])
    ax[1].axhline(p.weights[0].item(), color="k", ls="--")
    ax[1].set_title("left-mode mass (target 0.1)")
    ax[2].plot(steps[1:], [h["accept_rate"] for h in hist[1:]])
    ax[2].set_title("acceptance rate")
    for a in ax[1:]:
        a.set_xlabel("MH step")
    fig.tight_layout()
    fig.savefig(out / "imh_demo.png", dpi=120)
    print("saved", out / "imh_demo.png")
    if args.wandb:
        wandb.log({"imh_demo": wandb.Image(str(out / "imh_demo.png"))})
        wandb.finish()


if __name__ == "__main__":
    main()
