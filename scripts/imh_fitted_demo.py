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

from adjoint_samplers.components.mode_correction import fit_gmm, structure_cov  # noqa: F401 (re-exported)
from adjoint_samplers.energies.dist_energy import DistEnergy
from adjoint_samplers.utils.dist_utils import GMM1D
from imh_demo import imh_step, metrics


def load_samples(path, key="buffer_x1", device="cpu"):
    """ a tensor file, or the samples_final.pt of train.py (dump_final=true): the replay buffer `buffer_x1`
        (fitted on) and the fresh `model_samples` of the final model. Returns (x, model_samples or None).
    """
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict):
        return obj[key], obj.get("model_samples")
    return obj, None


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", default=None, help="samples_final.pt of train.py, or a tensor file")
    ap.add_argument("--buffer-key", default="buffer_x1", choices=["buffer_x1", "model_samples"])
    ap.add_argument("--buffer-size", type=int, default=10000)
    ap.add_argument("--model-weight", type=float, default=0.45, help="stand-in buffer: weight of the left mode")
    ap.add_argument("--model-std", type=float, default=0.55, help="stand-in buffer: std of both modes")
    ap.add_argument("--components", type=int, default=4)
    ap.add_argument("--inflate", type=float, default=1.2, help="covariance factor")
    ap.add_argument("--cov", default="full", choices=["full", "tied", "diag", "spherical"])
    ap.add_argument("--shrink", type=float, default=0.0)
    ap.add_argument("--keep-weights", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--chains", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/imh_fitted_demo")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    p = GMM1D()
    energy = DistEnergy(p)
    ref = p.sample([100000,]).reshape(-1).sort().values

    model_samples = None
    if args.buffer:
        buf, model_samples = load_samples(args.buffer, args.buffer_key)
        buf = buf.reshape(-1, 1)
    else:
        m = distributions.MixtureSameFamily(
            distributions.Categorical(torch.tensor([args.model_weight, 1 - args.model_weight])),
            distributions.Independent(distributions.Normal(
                torch.tensor([[-3.0], [3.0]]), torch.full((2, 1), args.model_std)), 1))
        buf = m.sample((args.buffer_size,))
    print(f"buffer: {len(buf)} samples, left share {(buf < 0).float().mean():.3f} (target {p.weights[0]:.1f})")

    w, mu, cov = fit_gmm(buf, args.components, cov_type=args.cov, shrink=args.shrink)
    print("fitted  weights", [round(v, 3) for v in w.tolist()], "means", [round(v, 2) for v in mu.reshape(-1).tolist()],
          "stds", [round(v, 2) for v in cov.reshape(-1).sqrt().tolist()])
    if not args.keep_weights:
        w = torch.full_like(w, 1 / len(w))
    r = FittedGMM(w, mu, cov * args.inflate)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project="adjoint_sampler_cv",
            name=f"imh_fitted_demo_{args.cov}_i{args.inflate}_M{args.components}",
            config=vars(args),
        )
        wandb.define_metric("step")
        wandb.define_metric("imh/*", step_metric="step")

    x = r.sample([args.chains,]).reshape(-1, 1)
    E, logr = energy.eval(x).reshape(-1), r.log_prob(x).reshape(-1)
    hist = [{"step": 0, "accept_rate": float("nan"), **metrics(x, p, ref)}]
    for step in range(1, args.steps + 1):
        x, E, logr, acc = imh_step(x, E, logr, energy, r)
        hist.append({"step": step, "accept_rate": acc, **metrics(x, p, ref)})
        if run:
            run.log({"step": step, **{f"imh/{k}": v for k, v in hist[-1].items() if k != "step"}})
        if step in (1, 5, 10, 25, 50, 100, args.steps):
            h = hist[-1]
            print(f"step {step:4d}  accept {acc:.3f}  left_mass {h['left_mass']:.3f}  "
                  f"mode_tv {h['mode_tv']:.3f}  w2 {h['w2']:.3f}")
    tail = hist[len(hist) // 2:]
    print("second half of the chain: " + "  ".join(
        f"{k} {sum(h[k] for h in tail) / len(tail):.3f}" for k in ("accept_rate", "left_mass", "mode_tv", "w2")))

    print("\n=== comparison (target left_mass 0.100; w2 noise floor about 0.2)")
    rows = {"AS buffer (fitted on)": metrics(buf, p, ref)}
    if model_samples is not None:
        rows["AS model samples"] = metrics(model_samples.reshape(-1, 1), p, ref)
    rows[f"IMH chain, last {len(tail)} steps"] = {k: sum(h[k] for h in tail) / len(tail) for k in ("left_mass", "mode_tv", "w2")}
    for name, m in rows.items():
        print(f"  {name:<34s} left_mass {m['left_mass']:.3f}  mode_tv {m['mode_tv']:.3f}  w2 {m['w2']:.3f}")
        if run:
            key = "imh_late" if name.startswith("IMH") else name.split(" (")[0].replace(" ", "_")
            for k, v in m.items():
                run.summary[f"{key}/{k}"] = v

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
    if run:
        run.log({"imh_fitted_demo": wandb.Image(str(out / "imh_fitted_demo.png"))})
        run.finish()


if __name__ == "__main__":
    main()
