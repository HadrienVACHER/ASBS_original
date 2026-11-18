""" Torch darting kernel (adjoint_samplers/utils/dw4_darting.py): exactness checks. Needs `results/pt_samples_T1.pt` (scripts/dw4_pt.py, 52 s): its T = 1 states are samples of e^{-E}/Z (independent parallel tempering).
      python tests/test_dw4_darting.py
    Thresholds fixed in advance:
      1. round trip: for 300 valid proposals, the reverse proposal from y (same element) returns x (|diff| < 1e-6) with base' = -base (|diff| < 1e-6);
      2. lumped flux balance on 40 000 PT states (kappa = 0): for every pair of types with both fluxes > 1e-3, |log(F_ab / F_ba)| < 3 bootstrap SE (the offline T1' statistic, here with the torch kernel);
      3. END-TO-END exact zero mean under pi: x1 ~ PT states, x_t = kappa x1 + sqrt(v) xi (xi in the zero-CoM subspace), kappa in 8 values: z of mean <h, psi(x_t)> for psi = x_t and x_t |x_t|^2, max |z| < 3.5;
         POSITIVE CONTROL: the same with v multiplied by 1.2 inside the Barker Gaussian factor only (breaks detailed balance): max |z| > 5.
"""
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils import dw4_darting as dd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
torch.manual_seed(0)
pt = torch.load(ROOT / "results" / "pt_samples_T1.pt", map_location="cpu", weights_only=False)["states"].double().reshape(-1, 8)
x = pt[torch.randperm(len(pt))[:40000]]
t0 = time.time()
pr = dd.proposals(x)
print(f"proposals for {len(x)} PT states: {time.time() - t0:.1f} s; valid proposals per state {float((pr['weight'] > 0).sum(1).double().mean()):.2f}; types -1: {int((pr['type'] < 0).sum())}")

# 0. far-from-minimum configurations (the polar chart can fail there): everything stays finite, invalid moves have weight 0 and finite placeholders
xf = x[:3000] + 1.5 * torch.randn(3000, 8, dtype=torch.float64)
prf = dd.proposals(xf)
nonf = int((~torch.isfinite(prf["base"])).sum() + (~torch.isfinite(prf["y"])).sum() + (~torch.isfinite(prf["weight"])).sum())
print(f"0. 3000 PT states perturbed by N(0, 1.5^2): non-finite entries {nonf}; valid proposals per state {float((prf['weight'] > 0).sum(1).double().mean()):.2f}; types -1: {int((prf['type'] < 0).sum())}")
assert nonf == 0

# 1. round trip
xs = dd.recentre(x.reshape(-1, 4, 2)).reshape(-1, 8)
rows = (pr["weight"][:300] > 0).nonzero()
ymax, bmax, n_ok = 0.0, 0.0, 0
for n, s in rows[:300].tolist() if False else [(int(i), int(j)) for i, j in rows[:300]]:
    y = pr["y"][n, s][None]
    back = dd.proposals(y)
    b_src = int(pr["type"][n])
    sl = back["weight"][0, b_src * dd.SLOTS:(b_src + 1) * dd.SLOTS] > 0
    if not bool(sl.any()):
        raise AssertionError("no reverse proposal to the type of x")
    yy = back["y"][0, b_src * dd.SLOTS:(b_src + 1) * dd.SLOTS][sl]
    d = (yy - xs[n]).abs().amax(-1)
    k = int(d.argmin())
    ymax = max(ymax, float(d[k]))
    bb = back["base"][0, b_src * dd.SLOTS:(b_src + 1) * dd.SLOTS][sl][k]
    bmax = max(bmax, abs(float(bb + pr["base"][n, s])))
    n_ok += 1
print(f"1. round trip on {n_ok} valid proposals: max |x_back - x| {ymax:.2e}, max |base' + base| {bmax:.2e}")
assert ymax < 1e-6 and bmax < 1e-6

# 2. flux balance
om = torch.sigmoid(pr["base"])
F = torch.zeros(len(x), 5, 5, dtype=torch.float64)
for b in range(5):
    F[torch.arange(len(x)), pr["type"].clamp(min=0), b] = (pr["weight"] * om)[:, b * dd.SLOTS:(b + 1) * dd.SLOTS].sum(1)
ok_t = pr["type"] >= 0
oneh = torch.nn.functional.one_hot(pr["type"].clamp(min=0), 5).double() * ok_t[:, None]
flux = lambda idx: torch.einsum("ni,nij->ij", oneh[idx], F[idx]) / len(idx)
Fm = flux(torch.arange(len(x)))
g = torch.Generator().manual_seed(1)
boots = torch.stack([flux(torch.randint(len(x), (len(x),), generator=g)) for _ in range(300)])
worst = 0.0
for i in range(5):
    for j in range(i + 1, 5):
        if Fm[i, j] > 1e-3 and Fm[j, i] > 1e-3:
            lr = math.log(float(Fm[i, j] / Fm[j, i]))
            se = float(torch.log(boots[:, i, j] / boots[:, j, i].clamp(min=1e-12)).std())
            worst = max(worst, abs(lr) / se)
            print(f"   pair {i}-{j}: {float(Fm[i, j]):.5f} vs {float(Fm[j, i]):.5f} log-ratio {lr:+.3f} SE {se:.3f} (z {lr / se:+.1f})")
print(f"2. flux balance on PT states: max |z| = {worst:.2f}")
assert worst < 3.0

# 3. end-to-end zero mean
nu1 = 4.0
def zs(vscale):
    out = []
    for kap in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.97):
        v = nu1 * kap * (1 - kap)
        xi = torch.randn(len(x), 8, dtype=torch.float64)
        xi = (xi.reshape(-1, 4, 2) - xi.reshape(-1, 4, 2).mean(1, keepdim=True)).reshape(-1, 8)
        xt = kap * xs + math.sqrt(v) * xi
        h, _ = dd.darting_h(xs, xt, torch.full((len(x), 1), kap), torch.full((len(x), 1), v * vscale), pr["y"], pr["base"], pr["weight"])
        for psi in (xt, xt * (xt ** 2).sum(-1, keepdim=True)):
            f = (h * psi).sum(-1)
            out.append(float(f.mean() / (f.std() / math.sqrt(len(f)))))
    return out
z_ok, z_bad = zs(1.0), zs(1.2)
print(f"3. exact zero mean: max |z| = {max(abs(z) for z in z_ok):.2f} (exact pi); positive control (v x 1.2 in the Barker factor): max |z| = {max(abs(z) for z in z_bad):.1f}")
assert max(abs(z) for z in z_ok) < 3.5 and max(abs(z) for z in z_bad) > 5
print("all passed")
