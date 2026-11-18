""" Flux of the darting kernel between the 5 basins at kappa = 0, 0.5 (and 0.7), on PT samples (T = 1, stratified) and on AS samples (ed8bb67 model_samples) (report only; local, seconds).
    Out-flux of a sample x1 of type a at (kappa, x_t): m_b = sum over the proposal slots whose target type is b of w_p * omega_p, omega = sigmoid(base + (|x_t - kappa x1|^2 - |x_t - kappa y|^2) / (2 v)), x_t = kappa x1 + sqrt(v) xi
    (x0 = 0, v = nu_1 kappa (1 - kappa), xi standard normal projected on the zero-centre-of-mass subspace; averaged over 20 draws of xi); at kappa = 0, x_t = 0 and omega = sigmoid(base).
    F[a, b] = mean of m_b over the samples of type a.   out_a = sum_b F[a, b]   (probability, per step, that a sample of basin a is moved: the matcher's `outflux`).
    in_b = sum_a p_a F[a, b] / p_b : probability per unit mass of basin b of being fed by the other basins; p = the proportions of the sample set (PT: p_PT; AS: its own basin frequencies).
    net_b = in_b - out_b: > 0 means that the kernel brings more mass to b than it takes (the AS samples are far from pi, so the flux is NOT balanced there; at pi, the exact balance is at kappa = 0).
      python scripts/dw4_darting_flux.py """
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils import dw4_darting as dd  # noqa: E402
import dw4_reeval_pt as rp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["triangle + pendant", "path of 2", "2 disjoint pairs", "K4 minus an edge", "star"]
SIGMA_MAX, SIGMA_MIN = 2.0, 0.001
NU1 = (SIGMA_MAX ** 2 - SIGMA_MIN ** 2) / (2 * np.log(SIGMA_MAX / SIGMA_MIN))
mdl = dd.model()
ID_OF_TYPE = [int((mdl["type_of_id"] == k).nonzero()[0]) for k in range(5)]
g = torch.Generator().manual_seed(0)


def flux(x8, kappa, ptype, p):
    pr = dd.proposals(x8)
    typ = pr["type"]
    keep = typ >= 0
    N = len(x8)
    x1 = dd.recentre(x8.double().reshape(N, 4, 2)).reshape(N, 8)
    F = np.zeros((5, 5))
    cnt = np.zeros(5)
    for rep in range(20):
        if kappa == 0:
            om = torch.sigmoid(pr["base"])
        else:
            v = NU1 * kappa * (1 - kappa)
            xi = torch.randn(N, 8, generator=g, dtype=torch.float64)
            xi = dd.recentre(xi.reshape(N, 4, 2)).reshape(N, 8)
            xt = kappa * x1 + np.sqrt(v) * xi
            om = dd.darting_h(x1, xt, torch.full((N, 1), kappa), torch.full((N, 1), v), pr["y"], pr["base"], pr["weight"])[1]
        m = (pr["weight"] * om).reshape(N, 5, dd.SLOTS).sum(-1)                                 # (N, 5): mass towards target type b
        for a in range(5):
            sel = (typ == a)
            if sel.any():
                F[a] += m[sel].mean(0).numpy()
                cnt[a] += 1
    F /= np.maximum(cnt, 1)[:, None]
    out = F.sum(1)
    inn = (p[:, None] * F).sum(0) / p
    return F, out, inn


def report(title, x8, ptype, p):
    print(f"\n=== {title} (basin proportions used: " + ", ".join(f"{NAMES[a]} {p[a]:.4f}" for a in range(5)) + ") ===")
    for kappa in (0.0, 0.5, 0.7):
        F, out, inn = flux(x8, kappa, ptype, p)
        print(f"  kappa = {kappa}:")
        print(f"    {'basin':20} {'out':>8} {'in':>8} {'net=in-out':>11} | F[a -> b] for b = " + ", ".join(NAMES[b][:9] for b in range(5)))
        for a in range(5):
            print(f"    {NAMES[a]:20} {out[a]:8.4f} {inn[a]:8.4f} {inn[a] - out[a]:+11.4f} | " + " ".join(f"{F[a, b]:8.4f}" for b in range(5)))


# --- PT samples, stratified by basin (all the states of the small basins, up to 1500 of the others), proportions = p_PT
d = torch.load(ROOT / "results" / "pt_samples_T1.pt", map_location="cpu", weights_only=False)
st, ids = d["states"].reshape(-1, 8), d["basin_state_id"].reshape(-1)
tr = rp.truth()["Q"][0]
pPT = np.array([tr[i] for i in ID_OF_TYPE])
rng = np.random.default_rng(0)
sel = []
for k in range(5):
    idx = (ids == ID_OF_TYPE[k]).nonzero().flatten().numpy()
    sel.append(rng.choice(idx, size=min(1500, len(idx)), replace=False))
sel = np.concatenate(sel)
print(f"PT: {len(sel)} states (stratified), counts per basin " + ", ".join(f"{int((ids[sel] == ID_OF_TYPE[k]).sum())}" for k in range(5)))
report("PT samples", st[sel].double(), None, pPT / pPT.sum())
# --- AS samples ed8bb67 (model_samples), own basin frequencies from the quench
ms = torch.load(ROOT / "results" / "dw4_e1500_ed8bb67_samples.pt", map_location="cpu", weights_only=False)["model_samples"].double()
pr0 = dd.proposals(ms)
pAS = np.array([(pr0["type"] == k).double().mean().item() for k in range(5)])
print(f"\nAS ed8bb67: {len(ms)} samples, untyped {(pr0['type'] < 0).double().mean():.4f}")
report("AS samples (ed8bb67 model_samples, all 10000)", ms, None, pAS / pAS.sum())
