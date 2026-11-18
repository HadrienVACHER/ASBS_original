""" t_focus / t_focus_max of the AS regression times and the resulting kappa distribution (report only; seconds).
    kappa_t = nu_t / nu_1 with nu_t = int_0^t sigma^2 (the repo's GraphVESDE, sigma_max = 2, sigma_min = 0.001). Regression times: t ~ U(0,1); a fraction t_focus of the times is replaced by t * t_focus_max (uniform on [0, t_focus_max]).
      python scripts/dw4_tfocus_check.py """
import inspect
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.components import matcher as M  # noqa: E402
from adjoint_samplers.components.sde import GraphVESDE  # noqa: E402

print("AdjointVEMatcher.__init__ signature:", inspect.signature(M.AdjointVEMatcher.__init__))
print("AdjointVEDW4DartingMatcher inherits sample_t from AdjointVEMatcher:", M.AdjointVEDW4DartingMatcher.sample_t is M.AdjointVEMatcher.sample_t, "| prepare_target of the darting matcher calls super().prepare_target (which calls self.sample_t):", "super().prepare_target(data, device)" in inspect.getsource(M.AdjointVEDW4DartingMatcher.prepare_target))
sde = GraphVESDE(n_particles=4, spatial_dim=2, sigma_max=2, sigma_min=0.001)
nu = lambda t: sde._pt_gauss_param(t, torch.zeros_like(t))[1].double()
tt = torch.linspace(0, 1, 2_000_001)[:, None]
kap = (nu(tt) / nu(torch.ones(1, 1))).squeeze(-1).numpy()
tt = tt.squeeze(-1).numpy()
def t_of(k):
    return float(np.interp(k, kap, tt))
print(f"kappa(t): t(kappa = 0.1) = {t_of(0.1):.4f}, t(0.5) = {t_of(0.5):.4f}, t(0.7) = {t_of(0.7):.4f}, t(0.9) = {t_of(0.9):.4f}; kappa(0.08) = {np.interp(0.08, tt, kap):.4f}, kappa(0.15) = {np.interp(0.15, tt, kap):.4f}")
def frac(lo, hi, tmax, f):
    """ P(kappa in [lo, hi]) under t ~ (1 - f) U(0,1) + f U(0, tmax) """
    a, b = t_of(lo), t_of(hi)
    pu = b - a
    pf = (min(b, tmax) - min(a, tmax)) / tmax
    return (1 - f) * pu + f * pf
print("\nfraction of regression times with kappa in [0.1, 0.7] (and in [0, 0.1], [0.7, 1]):")
print(f"  t_focus = 0 (uniform t, today): [0,0.1] {frac(0, 0.1, 1, 0):.3f}  [0.1,0.7] {frac(0.1, 0.7, 1, 0):.3f}  [0.7,1] {frac(0.7, 1, 1, 0):.3f}")
for tmax in (0.05, 0.08, 0.15):
    print(f"  t_focus_max = {tmax}: all focused times have kappa < {np.interp(tmax, tt, kap):.3f};  within the focus interval: [0,0.1] {frac(0, 0.1, tmax, 1):.3f}  [0.1,0.7] {frac(0.1, 0.7, tmax, 1):.3f}")
    for f in (0.25, 0.5, 0.7, 0.9):
        print(f"      t_focus = {f}: [0,0.1] {frac(0, 0.1, tmax, f):.3f}  [0.1,0.7] {frac(0.1, 0.7, tmax, f):.3f}  [0.7,1] {frac(0.7, 1, tmax, f):.3f}")

print("\nfraction of regression times with kappa in [0.1, 0.5] (explore plan: t_focus = 0.5, t_focus_max = 0.05):")
print(f"  default (t_focus = 0, uniform t): [0.1,0.5] {frac(0.1, 0.5, 1, 0):.4f}  ([0,0.1] {frac(0, 0.1, 1, 0):.4f}, [0.5,1] {frac(0.5, 1, 1, 0):.4f})")
for tmax in (0.05, 0.08, 0.15):
    print(f"  t_focus = 0.5, t_focus_max = {tmax}: [0.1,0.5] {frac(0.1, 0.5, tmax, 0.5):.4f}  ([0,0.1] {frac(0, 0.1, tmax, 0.5):.4f}, [0.5,1] {frac(0.5, 1, tmax, 0.5):.4f}, kappa(t_focus_max) = {np.interp(tmax, tt, kap):.3f})")
