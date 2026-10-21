""" Independent MH with a proposal r fitted on a "buffer", on the GMM target (no Adjoint Sampling).

    buffer: model samples (--buffer file.pt, (N, d)) or a stand-in for AS: right modes, weights flattened
            (π_k^power renormalised, --power < 1 under-weights the heavy modes) and modes slightly wide
            (--model-std).
    r:      GMM with M = --components fitted by EM (k-means++ start; --denoise starts k-means on
            x + ∇log p(x)), equal weights 1/M, covariances x --inflate.
    chain:  independent MH on p = e^{-E} / Z,  α = min(1, e^{-[E(x') - E(x)]} r(x) / r(x')).

    Reports the GMMEvaluator metrics (floors: exact samples of the same size) for the buffer, for the chain
    with the oracle proposal (the true modes with equal weights) and for the chain with the fitted one.

    python scripts/imh_fitted_gmm.py --dim 12 --K 12
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adjoint_samplers.components.evaluator import GMMEvaluator
from adjoint_samplers.components.matcher import _kmeans
from adjoint_samplers.energies.dist_energy import DistEnergy
from adjoint_samplers.utils.dist_utils import GMM
from imh_demo import imh_step
from imh_fitted_demo import FittedGMM, fit_gmm, load_samples

KEYS = ("mode_tv", "modes_covered", "eff_modes_ratio", "mode_spread", "energy_w2", "sample_w2")


def evaluate(evaluator, x):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        out = evaluator(x)
    return {k: out[k] for k in KEYS}


def mode_ratios(evaluator, x):
    """ share of the samples in mode k / target weight π_k, per mode (1: right proportion, 0: missed) """
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        m = evaluator.mode_metrics(x, per_mode=True)
    K = evaluator.dist.K
    return torch.tensor([m[f"mode_ratio/{k}"] for k in range(K)])


def fmt(m):
    return "  ".join(f"{k} {m[k]:6.3f}" for k in KEYS)


def run_chain(r, energy, evaluator, chains, steps, report, log=None, eval_every=10):
    """ log(step, dict): called with the acceptance every step and the evaluator metrics every
        `eval_every` steps (and at the reported steps)
    """
    x = r.sample([chains,]).reshape(chains, -1)
    E, logr = energy.eval(x).reshape(-1), r.log_prob(x).reshape(-1)
    if log:
        log(0, evaluate(evaluator, x))
    acc_hist = []
    for step in range(1, steps + 1):
        x, E, logr, acc = imh_step(x, E, logr, energy, r)
        acc_hist.append(acc)
        metrics = evaluate(evaluator, x) if step in report or step % eval_every == 0 else None
        if log:
            log(step, {"accept_rate": acc, **(metrics or {})})
        if step in report:
            print(f"    step {step:4d}  accept {acc:.3f}  {fmt(metrics)}")
    return x, sum(acc_hist[len(acc_hist) // 2:]) / max(1, len(acc_hist) - len(acc_hist) // 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=12)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--radius", type=float, default=4.25)
    ap.add_argument("--weight-ratio", type=float, default=20.0, help="max/min mode weight (loguniform law)")
    ap.add_argument("--out", default="results/imh_fitted_gmm")
    ap.add_argument("--target-seed", type=int, default=0)
    ap.add_argument("--buffer", default=None, help="samples_final.pt of train.py, or a tensor file")
    ap.add_argument("--buffer-key", default="buffer_x1", choices=["buffer_x1", "model_samples"])
    ap.add_argument("--buffer-size", type=int, default=10000)
    ap.add_argument("--power", type=float, default=0.3, help="stand-in buffer: weights π^power")
    ap.add_argument("--model-std", type=float, default=1.025, help="stand-in buffer: std of the modes")
    ap.add_argument("--components", type=int, default=None, help="M (default 2K)")
    ap.add_argument("--inflate", type=float, default=1.2)
    ap.add_argument("--cov", default="full", choices=["full", "tied", "diag", "spherical"],
                    help="covariance family of the fitted components")
    ap.add_argument("--shrink", type=float, default=0.0, help="shrink each covariance towards tr(C)/d I")
    ap.add_argument("--denoise", action="store_true")
    ap.add_argument("--chains", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--eval-every", type=int, default=10, help="metrics logged to W&B every n MH steps")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    M = args.components or 2 * args.K

    torch.manual_seed(args.seed)
    p = GMM(dim=args.dim, K=args.K, seed=args.target_seed, layout="spread",
            weight_law="loguniform", weight_ratio=args.weight_ratio, radius=args.radius)
    p.to(args.device)
    energy = DistEnergy(p, device=args.device)
    evaluator = GMMEvaluator(energy, w2_samples=500)

    model_samples = None
    if args.buffer:
        buf, model_samples = load_samples(args.buffer, args.buffer_key, args.device)
        buf = buf.reshape(-1, args.dim)
    else:
        w = p.weights ** args.power
        k = torch.multinomial(w / w.sum(), args.buffer_size, replacement=True)
        buf = p.means[k] + args.model_std * torch.randn(args.buffer_size, args.dim, device=args.device)
    n = min(len(buf), args.chains)
    print(f"device {args.device}; target d={args.dim} K={args.K}: weights min {p.weights.min():.4f} max {p.weights.max():.4f}; "
          f"fit M={M}, cov {args.cov} shrink {args.shrink}, inflate {args.inflate}, denoise {args.denoise}")
    print("  exact-sample floor:", fmt(evaluate(evaluator, p.sample([n,]).to(args.device))))
    print("  buffer:            ", fmt(evaluate(evaluator, buf[:n])))
    if model_samples is not None:
        print("  AS model samples:  ", fmt(evaluate(evaluator, model_samples.reshape(-1, args.dim)[:n])))

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project="adjoint_sampler_cv",
            name=f"imh_fitted_gmm_d{args.dim}_K{args.K}_wr{args.weight_ratio:g}_rad{args.radius:g}_M{M}_{args.cov}{args.shrink}_i{args.inflate}_dn{int(args.denoise)}",
            config=vars(args),
        )
        wandb.define_metric("step")
        for prefix in ("oracle", "fitted", "buffer", "floor"):
            wandb.define_metric(f"{prefix}/*", step_metric="step")

    def make_log(prefix):
        return lambda step, d: run.log({"step": step, **{f"{prefix}/{k}": v for k, v in d.items()}}) if run else None

    if run:
        run.log({"step": 0, **{f"floor/{k}": v for k, v in evaluate(evaluator, p.sample([n,]).to(args.device)).items()},
                 **{f"buffer/{k}": v for k, v in evaluate(evaluator, buf[:n]).items()}})

    report = {1, 5, 20, 100, args.steps}
    print("  chain, oracle r (true modes, equal weights):")
    x_oracle, acc = run_chain(p.uniform_weights(), energy, evaluator, args.chains, args.steps, report,
                              make_log("oracle"), args.eval_every)
    print(f"    accept (2nd half) {acc:.3f}")

    init = _kmeans(buf + p.score(buf), M, iters=20) if args.denoise else None
    w, mu, cov = fit_gmm(buf, M, init_means=init, cov_type=args.cov, shrink=args.shrink)
    r = FittedGMM(torch.full_like(w, 1 / M), mu, cov * args.inflate)
    assigned = torch.bincount(p.assign(mu), minlength=args.K)
    print(f"  fitted r: {(assigned > 0).sum().item()}/{args.K} true modes have a component, "
          f"eigenvalues of the covariances "
          f"min {torch.linalg.eigvalsh(cov).min().item():.3f} max {torch.linalg.eigvalsh(cov).max().item():.3f}")
    print("  chain, fitted r:")
    x_fit, acc = run_chain(r, energy, evaluator, args.chains, args.steps, report,
                           make_log("fitted"), args.eval_every)
    print(f"    accept (2nd half) {acc:.3f}")

    # per-mode proportions: share of the samples in mode k / target weight (1: right)
    if args.K <= 64:
        rows = {"AS buffer": mode_ratios(evaluator, buf[:n])}
        if model_samples is not None:
            rows["AS model samples"] = mode_ratios(evaluator, model_samples.reshape(-1, args.dim)[:n])
        rows["IMH oracle r"] = mode_ratios(evaluator, x_oracle)
        rows["IMH fitted r"] = mode_ratios(evaluator, x_fit)
        order = p.weights.argsort() # lightest first
        print("\n=== per-mode proportion / target weight, lightest modes first")
        print(f"  {'target w':>9s} " + " ".join(f"{k:>16s}" for k in rows))
        for j in order.tolist():
            print(f"  {p.weights[j].item():9.4f} " + " ".join(f"{v[j].item():16.2f}" for v in rows.values()))
        lightest = order[: max(1, args.K // 4)]
        heaviest = order[-max(1, args.K // 4):]
        print(f"  mean ratio, lightest {len(lightest)} modes / heaviest {len(heaviest)} modes:")
        for name, v in rows.items():
            print(f"    {name:<18s} {v[lightest].mean().item():6.2f} / {v[heaviest].mean().item():6.2f}")

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * args.K + 3), 4))
        width = 0.8 / len(rows)
        xs = torch.arange(args.K).float()
        for i, (name, v) in enumerate(rows.items()):
            ax.bar((xs + i * width).tolist(), v[order].tolist(), width, label=name)
        ax.axhline(1.0, color="k", ls="--")
        ax.set_xticks((xs + 0.4).tolist())
        ax.set_xticklabels([f"{w:.3f}" for w in p.weights[order].tolist()], rotation=60, fontsize=7)
        ax.set_xlabel("target weight (lightest first)")
        ax.set_ylabel("share / target weight")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "mode_ratios.png", dpi=120)
        print("saved", out / "mode_ratios.png")
        if run:
            import wandb
            run.log({"mode_ratios": wandb.Image(str(out / "mode_ratios.png"))})
            for name, v in rows.items():
                key = name.replace(" ", "_")
                run.summary[f"ratio_lightest/{key}"] = v[lightest].mean().item()
                run.summary[f"ratio_heaviest/{key}"] = v[heaviest].mean().item()
    if run:
        run.summary["fitted/accept_2nd_half"] = acc
        run.summary["fitted/modes_with_component"] = (assigned > 0).sum().item()
        run.finish()


if __name__ == "__main__":
    main()
