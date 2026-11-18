""" AdjointVEDW4DartingMatcher wiring and END-TO-END exact zero mean, under the law of x_t that TRAINING uses (the matcher's own populate_buffer with injected endpoints and the matcher's own prepare_target, gamma = 1:
    term = target + adjoint1).
    x1 ~ exact pi (T = 1 states of the independent parallel tempering, results/pt_samples_T1.pt, 30 000 states); for 8 values of kappa, S_psi = mean <term, psi(x_t)>, psi1 = x_t, psi2 = x_t |x_t|^2:
    PASS iff |S| / SE < 3.5 for every (kappa, psi). POSITIVE CONTROL: v_t x 1.2 inside the Gaussian factor only -> max |z| > 5. Also: the stored proposals are used (keys dart_y / dart_base / dart_w), pop_stats works.
      python tests/test_dw4_darting_matcher.py
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adjoint_samplers.components.matcher as matcher_module  # noqa: E402
from adjoint_samplers.components.buffer import BatchBuffer  # noqa: E402
from adjoint_samplers.components.matcher import AdjointVEDW4DartingMatcher  # noqa: E402
from adjoint_samplers.components.sde import ControlledSDE, GraphVESDE  # noqa: E402
from adjoint_samplers.components.state_cost import ZeroGradStateCost  # noqa: E402
from adjoint_samplers.components.term_cost import GraphScoreGradTermCost  # noqa: E402
from adjoint_samplers.energies.double_well_energy import DoubleWellEnergy  # noqa: E402
from adjoint_samplers.utils.dist_utils import Delta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def build(gamma=1.0):
    energy = DoubleWellEnergy(dim=8, n_particles=4)
    source = Delta(dim=8)
    ref = GraphVESDE(n_particles=4, spatial_dim=2, sigma_min=0.001, sigma_max=2.0)
    term = GraphScoreGradTermCost(source=source, ref_sde=ref, energy=energy, max_grad_E_norm=100)
    sde = ControlledSDE(ref, torch.nn.Identity())
    return AdjointVEDW4DartingMatcher(gamma=gamma, grad_term_cost=term, grad_state_cost=ZeroGradStateCost(), sde=sde, buffer=BatchBuffer(100000), resample_size=512, duplicates=1, stats_every=1)


def insert(m, x1):
    real = matcher_module.sdeint
    matcher_module.sdeint = lambda sde, x0, ts, only_boundary=True: (x0, x1)
    try:
        m.populate_buffer(torch.zeros(len(x1), 8), torch.linspace(0, 1, 100), False)
    finally:
        matcher_module.sdeint = real


def t_of_kappa(m, kappa):
    sde = m.sde.ref_sde
    nu = kappa * sde.total_var
    return float(-torch.log(1 - torch.tensor(nu / sde.sigma_max ** 2, dtype=torch.float64)) / (2 * torch.log(torch.tensor(sde.sigma_diff, dtype=torch.float64))))


def run(v_scale, x1):
    m = build(1.0)
    m.v_scale = v_scale
    for lo in range(0, len(x1), 2000):
        insert(m, x1[lo:lo + 2000])
    d = {k: torch.cat(m.buffer.batches[k]) for k in ("x0", "x1", "adjoint1", "dart_y", "dart_base", "dart_w")}
    zs = []
    for kappa in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.97):
        t = torch.full((len(x1), 1), t_of_kappa(m, kappa), dtype=torch.float32)
        m.sample_t = lambda x, _t=t: _t
        torch.manual_seed(5)
        (tt, xt), target = m.prepare_target(d, "cpu")
        term = (target + d["adjoint1"]).double()
        xt = xt.double()
        for psi in (xt, xt * (xt ** 2).sum(-1, keepdim=True)):
            f = (term * psi).sum(-1)
            zs.append(float(f.mean() / (f.std() / math.sqrt(len(f)))))
    return zs, m


torch.manual_seed(0)
pt = torch.load(ROOT / "results" / "pt_samples_T1.pt", map_location="cpu", weights_only=False)["states"].float().reshape(-1, 8)
x1 = pt[torch.randperm(len(pt))[:30000]]
z_ok, m = run(1.0, x1)
z_bad, _ = run(1.2, x1)
st = m.pop_stats()
print(f"exact pi endpoints: max |z| = {max(abs(z) for z in z_ok):.2f}; positive control (v x 1.2 in the Gaussian factor): max |z| = {max(abs(z) for z in z_bad):.1f}")
print("stats:", {k: round(v, 4) for k, v in st.items() if "kappa" in k and "outflux" in k or k.startswith("dart/") and "kappa" not in k})
assert max(abs(z) for z in z_ok) < 3.5 and max(abs(z) for z in z_bad) > 5
print("all passed")
