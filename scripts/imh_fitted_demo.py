""" Independent MH on the demo target with the proposal r fitted on samples of a "buffer".

    1. buffer: samples of the model (--buffer file.pt, an (N, 1) tensor) or, by default, a stand-in with the
       failure of Adjoint Sampling: right modes, wrong weights (--model-weight) and slightly wide modes
       (--model-std).
    2. r: a GMM with M components fitted by EM (k-means++ start), then equal weights 1/M (`--keep-weights`
       keeps the fitted ones) and covariances inflated by --inflate.
    3. independent MH with proposal r on the target p = e^{-E} / Z:
           α(x, x') = min(1, e^{-[E(x') - E(x)]} r(x) / r(x'))

    python scripts/imh_fitted_demo.py [--components 4 --buffer-size 10000 --chains 2000 --steps 200]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import distributions

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adjoint_samplers.components.matcher import _kmeans
from adjoint_samplers.energies.dist_energy import DistEnergy
from adjoint_samplers.utils.dist_utils import GMM1D
from imh_demo import imh_step, metrics


class FittedGMM:
    """ mixture of Gaussians with full covariances: sample and log_prob, as the proposal r needs """
    def __init__(self, weights, means, covs):
        self.weights, self.means, self.covs = weights, means, covs
        self.distr = distributions.MixtureSameFamily(
            distributions.Categorical(weights),
            distributions.MultivariateNormal(means, covariance_matrix=covs),
        )

    def sample(self, shape):
        return self.distr.sample(torch.Size(shape))

    def log_prob(self, x):
        return self.distr.log_prob(x)


def fit_gmm(x, M, iters=100, reg=1e-4, tol=1e-6, init_means=None):
    """ EM for a Gaussian mixture on x (N, D), full covariances, from k-means++ (Lloyd) centres
        (or from `init_means`)
    """
    N, D = x.shape
    means = _kmeans(x, M, iters=20) if init_means is None else init_means.clone()
    a = torch.cdist(x, means).argmin(-1)
    weights = torch.bincount(a, minlength=M).float().clamp(min=1) / N
    eye = torch.eye(D, device=x.device, dtype=x.dtype)
    covs = torch.stack([
        torch.cov(x[a == k].T).reshape(D, D) + reg * eye if (a == k).sum() > 1 else eye
        for k in range(M)
    ])
    prev = -float("inf")
    for _ in range(iters):
        comp = distributions.MultivariateNormal(means, covariance_matrix=covs)
        logp = comp.log_prob(x.unsqueeze(1)) + weights.log() # (N, M)
        ll = torch.logsumexp(logp, -1)
        resp = (logp - ll.unsqueeze(-1)).exp()
        Nk = resp.sum(0).clamp(min=1e-8)
        weights = Nk / N
        means = (resp.T @ x) / Nk[:, None]
        diff = x.unsqueeze(1) - means # (N, M, D)
        covs = torch.einsum("nk,nkd,nke->kde", resp, diff, diff) / Nk[:, None, None] + reg * eye
        if abs(ll.mean().item() - prev) < tol:
            break
        prev = ll.mean().item()
    return weights, means, covs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", default=None)
    ap.add_argument("--buffer-size", type=int, default=10000)
    ap.add_argument("--model-weight", type=float, default=0.45, help="stand-in buffer: weight of the left mode")
    ap.add_argument("--model-std", type=float, default=0.55, help="stand-in buffer: std of both modes")
    ap.add_argument("--components", type=int, default=4)
    ap.add_argument("--inflate", type=float, default=1.2, help="covariance factor")
    ap.add_argument("--keep-weights", action="store_true")
    ap.add_argument("--chains", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/imh_fitted_demo")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    p = GMM1D()
    energy = DistEnergy(p)
    ref = p.sample([100000,]).reshape(-1).sort().values

    if args.buffer:
        buf = torch.load(args.buffer).reshape(-1, 1)
    else:
        m = distributions.MixtureSameFamily(
            distributions.Categorical(torch.tensor([args.model_weight, 1 - args.model_weight])),
            distributions.Independent(distributions.Normal(
                torch.tensor([[-3.0], [3.0]]), torch.full((2, 1), args.model_std)), 1))
        buf = m.sample((args.buffer_size,))
    print(f"buffer: {len(buf)} samples, left share {(buf < 0).float().mean():.3f} (target {p.weights[0]:.1f})")

    w, mu, cov = fit_gmm(buf, args.components)
    print("fitted  weights", [round(v, 3) for v in w.tolist()], "means", [round(v, 2) for v in mu.reshape(-1).tolist()],
          "stds", [round(v, 2) for v in cov.reshape(-1).sqrt().tolist()])
    if not args.keep_weights:
        w = torch.full_like(w, 1 / len(w))
    r = FittedGMM(w, mu, cov * args.inflate)

    x = r.sample([args.chains,]).reshape(-1, 1)
    E, logr = energy.eval(x).reshape(-1), r.log_prob(x).reshape(-1)
    hist = [{"step": 0, "accept_rate": float("nan"), **metrics(x, p, ref)}]
    for step in range(1, args.steps + 1):
        x, E, logr, acc = imh_step(x, E, logr, energy, r)
        hist.append({"step": step, "accept_rate": acc, **metrics(x, p, ref)})
        if step in (1, 5, 10, 25, 50, 100, args.steps):
            h = hist[-1]
            print(f"step {step:4d}  accept {acc:.3f}  left_mass {h['left_mass']:.3f}  "
                  f"mode_tv {h['mode_tv']:.3f}  w2 {h['w2']:.3f}")
    tail = hist[len(hist) // 2:]
    print("second half of the chain: " + "  ".join(
        f"{k} {sum(h[k] for h in tail) / len(tail):.3f}" for k in ("accept_rate", "left_mass", "mode_tv", "w2")))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    grid = torch.linspace(-6, 6, 400).unsqueeze(-1)
    ax[0].hist(buf.reshape(-1), bins=60, density=True, alpha=0.3, label="buffer")
    ax[0].hist(x.reshape(-1), bins=60, density=True, alpha=0.5, label=f"IMH, step {args.steps}")
    ax[0].plot(grid, p.log_prob(grid).exp(), "k", label="target p")
    ax[0].plot(grid, r.log_prob(grid).exp(), "r--", label="fitted r")
    ax[0].legend()
    ax[1].plot([h["left_mass"] for h in hist])
    ax[1].axhline(p.weights[0].item(), color="k", ls="--")
    ax[1].set_title("left-mode mass (target 0.1)")
    ax[2].plot([h["accept_rate"] for h in hist])
    ax[2].set_title("acceptance rate")
    for a in ax[1:]:
        a.set_xlabel("MH step")
    fig.tight_layout()
    fig.savefig(out / "imh_fitted_demo.png", dpi=120)
    print("saved", out / "imh_fitted_demo.png")


if __name__ == "__main__":
    main()
