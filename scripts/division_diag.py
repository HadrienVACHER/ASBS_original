""" Offline diagnosis of the division run (CPU, no training) -- hypothesis: an estimated N_k wider than the true mode in some direction makes N_k / p unbounded
    (overflow for outlying x1, infinite variance of r_k).

      python scripts/division_diag.py                         # PROXY buffers (exact samples / wrong-weight samples of the frozen GMM): checks the fit procedure only
      python scripts/division_diag.py --buffer results/gmm_as_division/seed0/samples_final.pt   # the real final buffer of the run

    1. every fit: max eigenvalue of s * Sigma_true^{-1/2} Sigma_hat_k Sigma_true^{-1/2} per mode k (s = shrink = 1 - eta); N_k / p_k is bounded iff it is <= 1 (Sigma_true = I for this GMM).
    2. max_k (-L_k + log N_k(x1) + E(x1)/tau) per sample (relative to its median; float64 overflows at +709) and Pareto k_hat of u_k(x1) = exp(-L_k + log N_k(x1) + E(x1)/tau) per mode
       (samples whose nearest centre is k; GPD fit of the top 20 %; k_hat > 0.5: infinite variance).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import genpareto

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.energies.dist_energy import DistEnergy  # noqa: E402
from adjoint_samplers.utils.dist_utils import GMM  # noqa: E402
from adjoint_samplers.utils.division_cv import _lg, fit_modes  # noqa: E402

D = K = 36


def khat(u):
    u = np.sort(np.asarray(u))
    n = len(u)
    m = max(int(0.2 * n), 20)
    if n < 40:
        return float("nan")
    thr = u[-m - 1]
    ex = u[-m:] - thr
    if ex.max() <= 0:
        return float("nan")
    return float(genpareto.fit(ex, floc=0)[0])


def report(name, x1, gmm, energy, shrink=0.8):
    E1 = energy.eval(x1).reshape(-1)
    par, info = fit_modes(energy, x1, E1, 1.0, 12, shrink, 1.0)
    mu, lam, V, L = par["mu"], par["lam"], par["V"], par["L"]
    Kf = len(mu)
    # 1. eigenvalue ratio: Sigma_true = I, Sigma_hat_k = V diag(lam) V^T
    r = shrink * lam.max(-1).values
    near = torch.cdist(mu, gmm.means.double()).argmin(1)
    print(f"[{name}] K found {Kf}; centre error mean {float((mu - gmm.means.double()[near]).norm(dim=1).mean()):.2e}")
    print(f"  1. shrink * maxeig(Sigma_true^-1/2 Sigma_hat_k Sigma_true^-1/2): min {float(r.min()):.5f}  median {float(r.median()):.5f}  max {float(r.max()):.5f}   (> 1: unbounded ratio)"
          f"   | unshrunk max {float(lam.max()):.5f}, min eig {float(lam.min()):.5f}")
    # 2. log weights
    y = torch.einsum("kdj,bkd->bkj", V, x1.double()[:, None, :] - mu[None])
    logu = -L[None] + _lg(y, shrink * lam[None]) + E1.double()[:, None]
    mx = logu.max(1).values
    print(f"  2. max_k log u_k(x1) over samples: median {float(mx.median()):.2f}  p99.9 {float(mx.quantile(0.999)):.2f}  max {float(mx.max()):.2f}  (float64 overflow at 709); spread max - median {float(mx.max() - mx.median()):.2f}")
    owner = torch.cdist(x1.double(), mu).argmin(1)
    ks = []
    for k in range(Kf):
        sel = owner == k
        if int(sel.sum()) >= 40:
            ks.append(khat(torch.exp(logu[sel, k] - logu[sel, k].max()).numpy()))
    ks = np.array([k for k in ks if np.isfinite(k)])
    print(f"     Pareto k_hat of u_k over modes ({len(ks)} modes with >= 40 samples): median {np.median(ks):.3f}  max {ks.max():.3f}  n(k_hat > 0.5) = {int((ks > 0.5).sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", default=None, help="samples_final.pt of the run (key buffer_x1)")
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()
    g = GMM(dim=D, K=K, seed=0, layout="spread", weight_law="loguniform", weight_ratio=20.0, radius=4.25)
    energy = DistEnergy(g)
    gen = torch.Generator().manual_seed(0)
    if a.buffer:
        x1 = torch.load(a.buffer)["buffer_x1"][-a.n:].float()
        report("real final buffer, epoch 2000", x1, g, energy)
        return
    w = g.weights
    for name, ww in (("PROXY exact samples", w), ("PROXY wrong weights (w^0.5)", (w ** 0.5) / (w ** 0.5).sum())):
        k = torch.multinomial(ww, a.n, replacement=True, generator=gen)
        report(name, g.means[k] + torch.randn(a.n, D, generator=gen), g, energy)


if __name__ == "__main__":
    main()
