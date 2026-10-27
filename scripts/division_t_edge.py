""" Finiteness of the division term at the edges of t (CPU, float64): t = 0 (torch.rand in float32 returns exactly 0.0 with p = 2^-24 ~ 6e-8 per sample), smallest positive float32, 1e-12, 1e-8, 1e-6,
    kappa_max and t = 1. Intermediates reported: log N_k, log C_k, log Q_k, m_k, softmax(-L + log Q), log-weights, term.   python scripts/division_t_edge.py   (PROXY buffer: exact samples of the frozen GMM)
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.components.sde import VESDE  # noqa: E402
from adjoint_samplers.energies.dist_energy import DistEnergy  # noqa: E402
from adjoint_samplers.utils.dist_utils import GMM  # noqa: E402
from adjoint_samplers.utils.division_cv import _lg, division_term, fit_modes  # noqa: E402

D = K = 36


def fin(a):
    return "ok" if bool(torch.isfinite(a).all()) else f"NON-FINITE ({int((~torch.isfinite(a)).sum())}/{a.numel()})"


def main(kappa_max=0.99, n=2000):
    g = GMM(dim=D, K=K, seed=0, layout="spread", weight_law="loguniform", weight_ratio=20.0, radius=4.25)
    energy, gen = DistEnergy(g), torch.Generator().manual_seed(0)
    k = torch.multinomial(g.weights, 10000, replacement=True, generator=gen)
    x1all = g.means[k] + torch.randn(10000, D, generator=gen)
    par, _ = fit_modes(energy, x1all, None, 1.0, 12, 0.8, 1.0)
    x1 = x1all[:n]
    E1 = energy.eval(x1).reshape(-1)
    ref = VESDE(sigma_min=0.001, sigma_max=2 * GMM.coord_std(D, 4.25))
    nu = lambda t: ref._pt_gauss_param(t, torch.zeros_like(t))[1]
    nu1 = nu(torch.ones(1, 1, dtype=torch.float64))
    tiny = float(torch.finfo(torch.float32).tiny)
    eps32 = 2.0 ** -24
    # t such that kappa == kappa_max exactly (bisection in float64)
    a, b = 0.0, 1.0
    for _ in range(80):
        m = 0.5 * (a + b)
        a, b = (m, b) if float(nu(torch.tensor([[m]], dtype=torch.float64)) / nu1) < kappa_max else (a, m)
    print("t        | kappa        v            nu_{1|t}    | logN logC logQ m softmax logw term")
    for t in (0.0, eps32, 1e-12, 1e-8, 1e-6, b, 1.0):
        t32 = torch.full((n, 1), t, dtype=torch.float32)
        nut, n1 = nu(t32.double()), nu1.expand(n, 1)
        kap, nu1t = nut / n1, (n1 - nut).clamp(min=1e-12)
        v = n1 * kap * (1 - kap)
        x0 = torch.zeros(n, D)
        xt = ref.sample_posterior(t32, x0, x1)                                      # the training sampler (float32)
        x1d, xtd = x1.double(), xt.double() - (1 - kap) * x0.double()
        mu, lam, V, L, s = par["mu"], par["lam"], par["V"], par["L"], par["shrink"]
        y1 = torch.einsum("kdj,bkd->bkj", V, x1d[:, None, :] - mu[None])
        yt = torch.einsum("kdj,bkd->bkj", V, xtd[:, None, :] - kap.reshape(-1, 1, 1) * mu[None])
        logN = _lg(y1, s * lam[None])
        if True:
            logC = _lg(yt, kap.reshape(-1, 1, 1) ** 2 * s * lam[None] + v.reshape(-1, 1, 1))
            varQ = kap.reshape(-1, 1, 1) ** 2 * lam[None] + v.reshape(-1, 1, 1)
            logQ = _lg(yt, varQ)
            z = lam[None] / varQ * yt
            mm = mu[None] + kap.reshape(-1, 1, 1) * torch.einsum("kdj,bkj->bkd", V, z)
            sm = torch.softmax(-L[None] + logQ, -1)
            term, logs, _, _ = division_term(xtd, x1, E1, kap, v, nu1t, par, 1.0, terms=True)
        print(f"{t:<9.3g}| {float(kap[0]):<12.4g} {float(v[0]):<12.4g} {float(nu1t[0]):<11.4g} | " + " ".join(fin(a) for a in (logN, logC, logQ, mm, sm, logs, term)))
    print(f"(t with kappa = kappa_max = {kappa_max}: {b:.6f})")


if __name__ == "__main__":
    main()
