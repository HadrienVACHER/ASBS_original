""" kappa_max of the division control variate, chosen from the ORACLE only (frozen d = K = 36 isotropic GMM, exact posterior under the target), before any run.

      python scripts/division_kappa_max.py

    lambda(t) = 1{kappa_t <= kappa_max} multiplies the term (any function of (x_t, t) keeps the exact zero mean). At large kappa the mode posterior given x_t is ~one-hot and the exact
    correction carries no weight information. Rule 1 (absolute, NO SOLUTION: floor 2.1e-2 at kappa = 1): smallest kappa with E_{x_t}[1 - max_k pi*_k(x_t)] < 1e-3 under the target. Rule 2 (adopted): smallest kappa with excess over the kappa = 1 floor < 1% of (value at kappa = 0 minus floor) (x1 ~ p, x_t = kappa x1 + sqrt(v) eps,
    pi*_k(x_t) ∝ w_k N(x_t; kappa mu_k, (kappa^2 + v) I)). Prints the curve and the value.
"""
import math

import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils.dist_utils import GMM  # noqa: E402

D = K = 36
torch.set_default_dtype(torch.float64)


def ambiguity(kappa, mu, w, nu1, n, gen):
    v = nu1 * kappa * (1 - kappa)
    k = torch.multinomial(w, n, replacement=True, generator=gen)
    xt = kappa * (mu[k] + torch.randn(n, D, generator=gen)) + math.sqrt(v) * torch.randn(n, D, generator=gen)
    logr = torch.log(w)[None] - 0.5 * ((xt[:, None, :] - kappa * mu[None]) ** 2).sum(-1) / (v + kappa ** 2)
    pi = torch.softmax(logr, dim=-1)
    a = 1 - pi.max(-1).values
    return float(a.mean()), float(a.std() / math.sqrt(n))


def main(n=200000):
    g = GMM(dim=D, K=K, seed=0, layout="spread", weight_law="loguniform", weight_ratio=20.0, radius=4.25)
    mu, w = g.means.double(), g.weights.double()
    nu1 = (2 * GMM.coord_std(D, 4.25)) ** 2
    gen = torch.Generator().manual_seed(0)
    print(f"frozen GMM d = K = {D}, nu_1 = {nu1:.4f}, n = {n} per kappa\\n kappa    E[1 - max pi*]   (SE)")
    grid = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95]
    vals = []
    for k in grid:
        a, se = ambiguity(k, mu, w, nu1, n, gen)
        vals.append((k, a))
        print(f"{k:6.2f}   {a:14.3e}   ({se:.1e})")
    floor, se = ambiguity(1.0 - 1e-12, mu, w, nu1, n, gen)                           # kappa = 1 (v = 0): ambiguity of the mode posterior given x1 itself, intrinsic to the frozen GMM
    print(f"{1.0:6.2f}   {floor:14.3e}   ({se:.1e})   <- intrinsic floor")
    lo = next((k for k, a in vals if a < 1e-3), None)
    print(f"\nrule as pre-registered (E[1 - max pi*] < 1e-3): {'kappa_max = ' + str(lo) if lo is not None else 'NO SOLUTION (floor %.3e >= 1e-3)' % floor}")
    ex = [(k, a - floor) for k, a in vals]
    print("excess over the floor: " + ", ".join(f"{k:.2f}: {e:.2e}" for k, e in ex))
    for k in (0.97, 0.99, 0.995, 0.998, 0.999):
        a, _ = ambiguity(k, mu, w, nu1, n, gen)
        print(f"  kappa {k}: ambiguity {a:.3e}, excess {a - floor:.2e}")
    a0 = 1 - float(w.max())                                                  # kappa = 0: x_t independent of the mode, posterior = w
    thr = 0.01 * (a0 - floor)
    print(f"\nRULE (relative excess, adopted after the absolute rule had no solution): kappa_max = smallest kappa with (ambiguity - floor) < 1% of (ambiguity(0) - floor) = {thr:.3e}  [ambiguity(0) = {a0:.4f}, floor = {floor:.4f}]")
    fine = [0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995]
    sel = None
    for k in fine:
        a, _ = ambiguity(k, mu, w, nu1, n, gen)
        print(f"  kappa {k}: excess {a - floor:.3e}  {'< thr' if a - floor < thr else ''}")
        if sel is None and a - floor < thr:
            sel = k
    print(f"kappa_max = {sel}")
    a_hi = sel
    return a_hi


if __name__ == "__main__":
    main()
