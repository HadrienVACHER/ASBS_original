"""Phase-0 checks (CPU, < 1 min). Run: python test_gmm40_darting.py"""
import math, time
import torch
from gmm40_darting import (make_gmm40, energy, sample_exact, true_proportions, proportions, metrics, tv_floor,
                           discover_minima, darting_cv, girsanov_logw, ess_frac, K)

t0 = time.time()
torch.manual_seed(0)
for wmode in ["equal", "unequal"]:
    gmm = make_gmm40(weights=wmode)
    E = lambda z: energy(z, gmm)
    p = true_proportions(gmm, n=1_000_000)
    print(f"\n== GMM-40 {wmode}: std={gmm['std']:.4f}, w range [{gmm['logw'].exp().min():.4f}, "
          f"{gmm['logw'].exp().max():.4f}], cells with p<0.005: {(p < 0.005).sum().item()}")
    print("TV floor n=5000:", {k: round(v, 4) for k, v in tv_floor(gmm, p, 5000, reps=50).items()})

    # minima from exact samples (stand-in for a buffer)
    xb = sample_exact(4000, gmm)
    mins, H = discover_minima(xb, E, steps=300)
    print(f"minima found: {len(mins)} (40 means; close pairs may merge)")

    # zero-mean test of the darting term under pi(.|x_t), with a positive control (wrong v)
    n = 20000
    x1 = sample_exact(n, gmm)
    for kap in [0.1, 0.3, 0.5, 0.7]:
        nu1 = 50.0 ** 2
        kappa = torch.full((n,), kap); v = nu1 * kappa * (1 - kappa); nuc = nu1 * (1 - kappa)
        xt = kappa[:, None] * x1 + v.sqrt()[:, None] * torch.randn(n, 2)
        c = darting_cv(x1, xt, kappa, v, nuc, mins, H, E)                                   # [n,2]
        cbad = darting_cv(x1, xt, kappa, 1.2 * v, nuc, mins, H, E)
        psi = torch.stack([torch.ones(n), xt[:, 0] / 50, xt[:, 1] / 50, torch.sin(xt[:, 0] / 10)], 1)   # [n,4]
        def zmax(cc):
            m = (cc[:, :, None] * psi[:, None, :]).reshape(n, -1)                           # [n, 8]
            return (m.mean(0) / (m.std(0) / math.sqrt(n) + 1e-30)).abs().max().item()
        move = (c.norm(dim=1) > 0).float().mean().item()
        print(f"kappa={kap}: max|z| zero-mean = {zmax(c):.2f} (control v*1.2: {zmax(cbad):.2f}); "
              f"frac samples with nonzero term = {move:.3f}")

# Girsanov sign test: target = base law N(0, nu1 I), constant drift u = (1, 0)
sig = lambda t: 2.0
nu1 = 4.0
Eg = lambda z: 0.5 * (z ** 2).sum(-1) / nu1
for sign in [1, -1]:
    u = lambda x, t: sign * torch.tensor([1.0, 0.0]).expand_as(x)
    x, lw = girsanov_logw(u, sig, Eg, 20000, 100)
    w = (lw - lw.max()).exp(); w = w / w.sum()
    m = (w[:, None] * x).sum(0)
    print(f"Girsanov drift sign {sign:+d}: IS mean of X1 = {m.numpy().round(3)} (target 0), ESS = {ess_frac(lw):.3f}")
# flipped Girsanov term must fail: recompute with the sign of the path term reversed
u = lambda x, t: torch.tensor([1.0, 0.0]).expand_as(x)
x, lw = girsanov_logw(u, sig, Eg, 20000, 100)
logp_base = -0.5 * (x ** 2).sum(-1) / nu1
path_term = lw - (-Eg(x) - logp_base).double() - math.log(2 * math.pi * nu1) * 0
lw_bad = (-Eg(x) - logp_base).double() - path_term
w = (lw_bad - lw_bad.max()).exp(); w = w / w.sum()
print(f"positive control (path term sign flipped): IS mean = {(w[:, None] * x).sum(0).numpy().round(3)} (should be far from 0)")
print(f"\ntotal time {time.time() - t0:.1f}s")
