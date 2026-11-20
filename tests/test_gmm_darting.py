""" T1' (CPU, < 1 min): zero mean of the GMM darting term c under pi(. | x_t), minima DISCOVERED from a buffer of the trained (32,8,6) x 0.5 seed-0 controller (no oracle), with the positive control v x 1.2.
    python tests/test_gmm_darting.py """
import math, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from adjoint_samplers.energies.dist_energy import DistEnergy
from adjoint_samplers.utils.dist_utils import GMMDK
from adjoint_samplers.utils import gmm_darting as gd
import darting_vs_is_stepB as B

t0 = time.time()
torch.manual_seed(0)
d, K, DELTA = 32, 8, 6.0
energy = DistEnergy(GMMDK(dim=d, K=K, delta=DELTA), device="cpu")
x_buf, _ = B.simulate("gmmdk_d32_K8_D6_s0_sx0.5", 2000)                      # a stand-in for the training buffer: end points of the trained controller
print(f"buffer stand-in: {len(x_buf)} end points of the trained controller ({time.time() - t0:.1f} s)")
M, H = gd.discover_minima(torch.tensor(x_buf), lambda z: energy.eval(z))
print(f"discovered minima: {len(M)} of {K}; max |H - I| = {float((H - torch.eye(d, dtype=H.dtype)).abs().max()):.2e}")
maps = gd.make_maps(M, H)
sigma_max = 1.25                                                            # the x 0.5 value of this cell
nu1 = sigma_max ** 2 * (1 - (0.001 / sigma_max) ** 2)
n, chunk = (int(sys.argv[1]) if len(sys.argv) > 1 else 20000), 4000
x1_all = energy.dist.sample((n,)).double()
ok0, ok1 = True, True
for kap in (0.1, 0.3, 0.5, 0.7):
    kappa = torch.full((n,), kap, dtype=torch.float64); v = nu1 * kappa * (1 - kappa); nuc = nu1 * (1 - kappa)
    xt = kappa[:, None] * x1_all + v.sqrt()[:, None] * torch.randn(n, d, dtype=torch.float64)
    out = {}
    for tag, vs in (("zero-mean", 1.0), ("control v*1.2", 1.2)):
        c = torch.cat([gd.darting_term(x1_all[i:i + chunk], xt[i:i + chunk], kappa[i:i + chunk], v[i:i + chunk], nuc[i:i + chunk], maps, lambda z: energy.eval(z), vs)[0] for i in range(0, n, chunk)])
        s = math.sqrt(nu1)
        psi = torch.stack([torch.ones(n, dtype=torch.float64), xt[:, 0] / s, xt[:, 1] / s, torch.sin(xt[:, 0] / s), (xt[:, :K] / s).sum(1) / math.sqrt(K)], 1)   # [n, 5]
        m = (c[:, :, None] * psi[:, None, :]).reshape(n, -1)
        z = (m.mean(0) / (m.std(0) / math.sqrt(n) + 1e-30)).abs().max().item()
        out[tag] = (z, float((c.norm(dim=1) > 0).double().mean()))
    ok0 &= out["zero-mean"][0] < 4.0
    ok1 &= out["control v*1.2"][0] > 4.0
    print(f"kappa={kap}: max|z| zero-mean = {out['zero-mean'][0]:.2f}; control (v x 1.2) = {out['control v*1.2'][0]:.2f}; fraction of samples with a nonzero term = {out['zero-mean'][1]:.3f}")
print(f"\nT1' zero mean (max|z| < 4 at every kappa): {'PASS' if ok0 else 'FAIL'};  positive control (max|z| > 4 at every kappa): {'PASS' if ok1 else 'FAIL'};  total {time.time() - t0:.1f} s")
sys.exit(0 if (ok0 and ok1) else 1)
