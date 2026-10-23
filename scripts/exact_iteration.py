""" One outer iteration from a FROZEN buffer p1^u (mode masses ŵ ≠ w), inner regression solved exactly, GMM target with
    unit-variance modes (posteriors in closed form). Per-mode transfer ratio  r_j = Δŵ_j / e_j,  e = w - ŵ,
    Δŵ = terminal mode masses of the new controller minus those of the bridge to p1^u (same Brownian increments).

    arms (drift = σ²(t) u(x, t), u in closed form, dX = σ² u dt + σ dW from X_0 = 0, up to t_stop):
      bridge   u = (E_{π^u}[X1 | x] - x) / ν_{1|t}                    baseline: reproduces ŵ
      A        bridge + E_{π^u}[∇log(p1/p1^u)(X1) | x]                AS regression only (γ = 0): exact RAM minimizer
      B(γ)     A + γ E_{π^u}[h | x] / ν_{1|t}                         Barker term (AdjointVEBarkerMatcher.mh_term, true-mean centres)
      C(ts)    u* (bridge to the TRUE target) for t < ts, A for ts <= t < t_stop, for ts in --t-smalls (0.07, 0.15, 0.3)
                                                                       best any correction restricted to t < ts can do
      D        u* everywhere (masses read with the true bridge)       sanity: must recover w
    Past t_stop the arms A, B, C follow the bridge to p1^u (masses = E[P^u(k | X_{t_stop})]); D follows the true bridge.
    NOTE: AS's own correction beyond t_stop is therefore neglected in A, B, C (reasonable: ∇log(p1 / p1^u) ≈ 0 once the
    posterior over modes is unimodal, but an approximation).

    python scripts/exact_iteration.py --dim 36 --K 36 --sigma-max 2.4509 --samples K36_g2_oracle.pt --focus 22   # cluster / GPU
    python scripts/exact_iteration.py --dim 6 --K 4 --sigma-max 2.0 --N 1500 --n-mc 4                             # smoke test
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_mh_matcher import kappa_nu, make_matcher, posterior, sample_posterior  # noqa: E402
from adjoint_samplers.components.matcher import AdjointVEBarkerMatcher  # noqa: E402
from adjoint_samplers.utils.dist_utils import GMM  # noqa: E402


def with_weights(gmm, weights):
    other = copy.copy(gmm)
    other.weights = weights / weights.sum()
    return other


def chunked_eval(fn, x, chunk):
    return torch.cat([fn(x[i:i + chunk]) for i in range(0, len(x), chunk)])


class Sim:
    def __init__(self, target, model, matcher, args, device):
        self.target, self.model, self.matcher, self.args, self.device = target, model, matcher, args, device
        self.energy = matcher.grad_term_cost.energy
        self.ref = matcher.sde.ref_sde
        self.timesteps = torch.linspace(0, 1, args.steps + 1)

    # --- closed-form pieces -------------------------------------------------------------------------------
    def bridge(self, gmm, x, kappa, nu1, nu1t):
        probs, means, _ = posterior(gmm, kappa, nu1, x)
        return ((probs.unsqueeze(-1) * means).sum(1) - x) / nu1t

    def as_step(self, x, kappa, nu1):
        """ E_{π^u}[∇log p1(X1) - ∇log p1^u(X1) | x], by exact posterior draws """
        x1 = sample_posterior(self.model, kappa, nu1, x, self.args.n_mc)
        N, n, D = x1.shape
        flat = x1.reshape(-1, D)
        d = chunked_eval(lambda z: self.target.score(z) - self.model.score(z), flat, self.args.chunk)
        return d.view(N, n, D).mean(1)

    def barker_term(self, x, kappa, nu1, t):
        """ γ E_{π^u}[h | x] / ν_{1|t} through the real mh_term (all M - 1 translates, true-mean centres) """
        m = self.matcher
        x1 = sample_posterior(self.model, kappa, nu1, x, self.args.n_mc)
        N, n, D = x1.shape
        out = []
        per = max(self.args.chunk // (n * m.n_clusters), 1)
        for i in range(0, N, per):
            xc, x1c = x[i:i + per], x1[i:i + per].reshape(-1, D)
            xt = xc.repeat_interleave(n, 0)
            jumps, valid = m.propose(x1c)
            EY = torch.zeros_like(valid)
            mask = valid.bool()
            EY[mask] = self.energy.eval((x1c.unsqueeze(1) + jumps)[mask]).reshape(-1)
            E1 = self.energy.eval(x1c).reshape(-1, 1)
            tv = torch.full((len(x1c), 1), t, device=x.device)
            term, _ = m.mh_term(tv, torch.zeros_like(x1c), xt, x1c, E1, jumps, EY, valid)
            out.append(term.view(-1, n, D).mean(1))
        return torch.cat(out)

    # --- one simulation ----------------------------------------------------------------------------------
    @torch.no_grad()
    def run(self, kind, gamma=0.0, t_small=0.0):
        a = self.args
        self.matcher.gamma = gamma
        gen = torch.Generator().manual_seed(6) # same Brownian increments for every arm
        x = torch.zeros(a.N, self.target.dim, device=self.device)
        t = 0.0
        for i in range(len(self.timesteps) - 1):
            t, dt = self.timesteps[i].item(), (self.timesteps[i + 1] - self.timesteps[i]).item()
            if t >= a.t_stop:
                break
            kappa, nu1t, nu1 = kappa_nu(self.matcher, t)
            sigma2 = self.ref.diff(torch.tensor(t)).item() ** 2
            small = t < t_small
            if kind == "D" or (kind == "C" and small):
                u = self.bridge(self.target, x, kappa, nu1, nu1t)
            else:
                u = self.bridge(self.model, x, kappa, nu1, nu1t)
                if kind in ("A", "B", "C"):
                    u = u + self.as_step(x, kappa, nu1)
                if kind == "B":
                    u = u + self.barker_term(x, kappa, nu1, t)
            noise = torch.randn(a.N, self.target.dim, generator=gen).to(self.device)
            x = x + sigma2 * u * dt + math.sqrt(sigma2 * dt) * noise
        kappa, _, nu1 = kappa_nu(self.matcher, t)
        gmm = self.target if kind == "D" else self.model
        return posterior(gmm, kappa, nu1, x)[0].mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=36)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--radius", type=float, default=4.25)
    ap.add_argument("--weight-ratio", type=float, default=20.0)
    ap.add_argument("--sigma-max", type=float, required=True)
    ap.add_argument("--samples", default=None, help="samples_final.pt: frozen buffer = its mode masses (Laplace-smoothed)")
    ap.add_argument("--imbalance", type=float, default=0.7, help="without --samples: ŵ ∝ w exp(imbalance · N(0, 1))")
    ap.add_argument("--N", type=int, default=20000)
    ap.add_argument("--n-mc", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=100000)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--t-stop", type=float, default=0.3)
    ap.add_argument("--t-smalls", type=float, nargs="+", default=[0.07, 0.15, 0.3])
    ap.add_argument("--gammas", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    ap.add_argument("--focus", type=int, nargs="*", default=[])
    ap.add_argument("--out", default="results/exact_iteration")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    K = args.K or args.dim
    target = GMM(dim=args.dim, K=K, seed=0, layout="spread", weight_law="loguniform", weight_ratio=args.weight_ratio, radius=args.radius)
    target.K = K
    w = target.weights.clone()
    if args.samples:
        buf = torch.load(args.samples, map_location="cpu")["buffer_x1"].float()
        counts = torch.bincount(target.assign(buf), minlength=K).float()
        w_hat = (counts + 1) / (counts.sum() + K)
        src = f"masses of the buffer in {args.samples}"
    else:
        w_hat = w * torch.exp(args.imbalance * torch.randn(K))
        w_hat = w_hat / w_hat.sum()
        src = f"synthetic: ŵ ∝ w exp({args.imbalance} N(0, 1))"
    target.to(device)
    model = with_weights(target, w_hat.to(device))
    matcher = make_matcher(target, AdjointVEBarkerMatcher, sigma_max=args.sigma_max, gamma=1.0, n_clusters=K)
    matcher.centres = target.means.clone()
    sim = Sim(target, model, matcher, args, device)
    w, w_hat = w.to(device), w_hat.to(device)
    e = w - w_hat
    print(f"d = {args.dim}, K = {K}, device {device}, N = {args.N}, n_mc = {args.n_mc}, t_stop = {args.t_stop}; frozen buffer: {src}")
    print("note: after t_stop the arms A, B, C follow the bridge to p1^u, i.e. AS's own correction beyond t_stop is neglected")

    arms = ([("bridge", "bridge", 0.0, 0.0), ("A", "A", 0.0, 0.0)] + [(f"B{g:g}", "B", g, 0.0) for g in args.gammas]
            + [(f"C{ts:g}", "C", 0.0, ts) for ts in args.t_smalls] + [("D", "D", 0.0, 0.0)])
    masses = {}
    for name, kind, g, ts in arms:
        t0 = time.time()
        masses[name] = sim.run(kind, g, ts)
        print(f"  arm {name:7s} done in {time.time() - t0:6.1f} s", flush=True)

    base = masses["bridge"]
    print(f"\nbridge alone reproduces ŵ: max err {(base - w_hat).abs().max():.5f}; D (u* everywhere) recovers w: max err {(masses['D'] - w).abs().max():.5f}")
    delta = {n: masses[n] - base for n in masses if n != "bridge"}
    names = list(delta)

    mbar = (w[:, None] * target.means).sum(0)
    F = e @ target.means
    cos = ((target.means - mbar) @ F) / ((target.means - mbar).norm(dim=1) * F.norm())
    dist = torch.cdist(target.means, target.means).fill_diagonal_(1e9).min(1).values
    order = torch.argsort(w)
    q = K // 4
    light, heavy = order[:q], order[-q:]

    print(f"\n{'arm':8s}{'global Δŵ·e/e·e':>16}{'cos(Δŵ, e)':>12}{'light quarter':>15}{'heavy quarter':>15}{'corr(r_j, cos_j)':>18}")
    summary = {}
    for n in names:
        d = delta[n]
        r = d / e
        g_ = float(d @ e / (e @ e))
        c_ = float(d @ e / (d.norm() * e.norm()))
        lq, hq = float(d[light].sum() / e[light].sum()), float(d[heavy].sum() / e[heavy].sum())
        cc = float(np.corrcoef(r.cpu().numpy(), cos.cpu().numpy())[0, 1]) if K > 2 else float("nan")
        summary[n] = dict(global_transfer=g_, cos=c_, light=lq, heavy=hq, corr_cos=cc, ratio=r.cpu().tolist(), delta=d.cpu().tolist())
        print(f"{n:8s}{g_:16.3f}{c_:12.3f}{lq:15.3f}{hq:15.3f}{cc:18.3f}")

    print(f"\nper-mode transfer ratio Δŵ_j / e_j, modes sorted by weight (first = lightest); cos = cos(m_j - m_bar, Σ_i e_i m_i)")
    print(f"{'mode':>5}{'w_j':>9}{'ŵ_j':>9}{'e_j':>9}{'cos':>7}" + "".join(f"{n:>8s}" for n in names))
    for j in order.tolist():
        mark = " <--" if j in args.focus else ""
        print(f"{j:5d}{w[j]:9.5f}{w_hat[j]:9.5f}{e[j]:9.5f}{cos[j]:7.2f}" + "".join(f"{(delta[n][j] / e[j]):8.2f}" for n in names) + mark)
    for j in args.focus:
        print(f"\nmode {j}: " + ", ".join(f"{n} {float(delta[n][j] / e[j]):.2f}" for n in names)
              + f"  (w {w[j]:.5f}, ŵ {w_hat[j]:.5f}, nearest mode at {dist[j]:.2f}, cos {cos[j]:.2f})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(args=vars(args), w=w.cpu().tolist(), w_hat=w_hat.cpu().tolist(), cos=cos.cpu().tolist(), dist=dist.cpu().tolist(),
                   summary=summary), open(out / "exact_iteration.json", "w"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shown = [n for n in names if n in ("A", "D") or n[0] in "BC"]
        fig, axes = plt.subplots(2, len(shown), figsize=(3.2 * len(shown), 6), squeeze=False)
        for k, n in enumerate(shown):
            r = (delta[n] / e).cpu().numpy()
            for row, (xv, lab) in enumerate(((cos.cpu().numpy(), "cos(m_j - m_bar, Σ e_i m_i)"), (np.log10(w.cpu().numpy()), "log10 w_j"))):
                ax = axes[row, k]
                ax.scatter(xv, r, c=np.log10(w.cpu().numpy()), s=18, cmap="viridis")
                for j in args.focus:
                    ax.annotate(str(j), (xv[j], r[j]), color="red")
                ax.axhline(1, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.5)
                ax.set_title(n); ax.set_xlabel(lab); ax.set_ylabel("Δŵ_j / e_j")
        fig.tight_layout(); fig.savefig(out / "exact_iteration.png", dpi=110)
        print(f"\nsaved {out}/exact_iteration.json and .png")
    except Exception as ex: # plotting is optional
        print("figure skipped:", ex)


if __name__ == "__main__":
    main()
