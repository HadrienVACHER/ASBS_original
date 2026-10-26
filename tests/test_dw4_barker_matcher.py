""" AdjointVEDW4BarkerMatcher (closed-form Barker control variate in prepare_target), two unit tests:
      1. detailed balance THROUGH THE MATCHER's own κ_t, v_t, ν_{1|t}: π~(x) ω_r(x) = π~(T_r x) ω_r(T_r x) for every r  =>  E_π[h | x_t] = 0 exactly (the fixed point is unchanged);
      2. target construction: γ = 0 gives exactly the AS target; the extra term equals γ h / ν_{1|t} computed independently; it is mean-free and equivariant under rotations,
         reflections and permutations of (x1, x_t); finite, and the stats are logged.
    python tests/test_dw4_barker_matcher.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.components.buffer import BatchBuffer  # noqa: E402
from adjoint_samplers.components.matcher import AdjointVEDW4BarkerMatcher  # noqa: E402
from adjoint_samplers.components.sde import ControlledSDE, GraphVESDE  # noqa: E402
from adjoint_samplers.components.state_cost import ZeroGradStateCost  # noqa: E402
from adjoint_samplers.components.term_cost import GraphScoreGradTermCost  # noqa: E402
from adjoint_samplers.energies.double_well_energy import DoubleWellEnergy  # noqa: E402
from adjoint_samplers.utils import dw4_reflections as dwr  # noqa: E402
from adjoint_samplers.utils.dist_utils import Delta  # noqa: E402


def build(gamma=3.0, sigma_max=2.0):
    energy = DoubleWellEnergy(dim=8, n_particles=4)
    source = Delta(dim=8)
    ref = GraphVESDE(n_particles=4, spatial_dim=2, sigma_min=0.001, sigma_max=sigma_max)
    term = GraphScoreGradTermCost(source=source, ref_sde=ref, energy=energy, max_grad_E_norm=100)
    sde = ControlledSDE(ref, torch.nn.Identity())
    return AdjointVEDW4BarkerMatcher(gamma=gamma, grad_term_cost=term, grad_state_cost=ZeroGradStateCost(), sde=sde, buffer=BatchBuffer(1000), resample_size=512, duplicates=1, stats_every=1)


def reference(n):
    return torch.tensor(np.load(Path(__file__).resolve().parents[1] / "data" / "test_split_DW4.npy", allow_pickle=True)).float()[:n]


def test_detailed_balance_through_the_matcher():
    torch.manual_seed(0)
    m = build()
    x1 = dwr.recentre(reference(60).reshape(-1, 4, 2)).reshape(-1, 8).double()
    t = torch.rand(60, 1).double() * 0.9 + 0.05
    x0 = torch.zeros(60, 8).double()
    nut, nu1 = m._nu(t), m._nu(torch.ones_like(t))
    kappa, v = nut / nu1, nu1 * (nut / nu1) * (1 - nut / nu1)
    xt = kappa * x1 + v.sqrt() * dwr.recentre(torch.randn(60, 4, 2)).reshape(-1, 8).double()
    imgs, _ = dwr.images(x1)
    logpi = lambda z: -dwr.energy(z) - ((xt - kappa * z) ** 2).sum(-1) / (2 * v.squeeze(-1))
    worst = 0.0
    for r in range(24):
        y = imgs[:, r]
        # ω from the matcher; use the logit for numerical safety of the log
        lx = torch.nn.functional.logsigmoid(dwr.barker_logit(x1, xt, kappa, v)[0][:, r])
        ly = torch.nn.functional.logsigmoid(dwr.barker_logit(y, xt, kappa, v)[0][:, r])
        worst = max(worst, float(((logpi(x1) + lx) - (logpi(y) + ly)).abs().max()))
    assert worst < 1e-5, worst                                            # float32 κ_t, v_t from the SDE buffers
    print(f"ok  1. detailed balance with the matcher's own κ_t, v_t: max |log π~ω(x) - log π~ω(T_r x)| = {worst:.1e} for the 24 maps -> E_π[h | x_t] = 0 exactly, the fixed point is unchanged")


def group(kind, gen):
    if kind == "permutation":
        p = torch.randperm(4, generator=gen)
        return lambda x: x.reshape(-1, 4, 2)[:, p].reshape(-1, 8)
    th = float(torch.rand(1, generator=gen) * 6.28)
    Q = torch.tensor([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=torch.float64)
    if kind == "reflection":
        Q = Q @ torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
    return lambda x: (x.reshape(-1, 4, 2) @ Q.T.to(x.dtype)).reshape(-1, 8)


def test_target_construction():
    m3, m0 = build(3.0), build(0.0)
    x1 = reference(256)
    x0 = torch.zeros(256, 8)
    data = {"x0": x0, "x1": x1, "adjoint1": m3.grad_term_cost(x1)}
    torch.manual_seed(7)
    (t3, xt3), tg3 = m3.prepare_target(data, "cpu")
    torch.manual_seed(7)
    (t0, xt0), tg0 = m0.prepare_target(data, "cpu")
    assert torch.equal(t3, t0) and torch.equal(xt3, xt0)
    assert torch.allclose(tg0, -data["adjoint1"]), "gamma = 0 must be the AS target"
    extra = tg3 - tg0
    nut, nu1 = m3._nu(t3), m3._nu(torch.ones_like(t3))
    kappa, nu1t = nut / nu1, (nu1 - nut).clamp(min=1e-12)                    # same floors as the matcher (κ_t = 1 exactly in float32 at t ~ 1)
    h, _ = dwr.barker_h(x1.double(), xt3.double(), kappa.double(), (nu1 * kappa * (1 - kappa)).clamp(min=1e-12).double())
    ref_term = (3.0 * h / nu1t.double()).float()
    # float32 matcher vs the float64 formula: rows in the transition region of the Barker weights amplify rounding (1/ν_{1|t}), so a per-row relative bound
    rel = (extra - ref_term).norm(dim=-1) / (ref_term.norm(dim=-1) + 1e-3)
    assert float(rel.max()) < 0.5 and float(rel.mean()) < 5e-3, (float(rel.max()), float(rel.mean()))
    # the formula itself, in float64 through the matcher's own barker_term (tight)
    td, xtd, x1d, x0d = t3.double(), xt3.double(), x1.double(), x0.double()
    e64, _ = m3.barker_term(td, x0d, xtd, x1d)
    nutd, nu1d = m3._nu(td), m3._nu(torch.ones_like(td))
    kd, nu1td = nutd / nu1d, (nu1d - nutd).clamp(min=1e-12)
    h64, _ = dwr.barker_h(x1d, xtd, kd, (nu1d * kd * (1 - kd)).clamp(min=1e-12))
    assert torch.allclose(e64, 3.0 * h64 / nu1td, rtol=1e-9, atol=1e-12), (e64 - 3.0 * h64 / nu1td).abs().max()
    assert torch.isfinite(tg3).all()
    assert float(e64.reshape(-1, 4, 2).sum(1).abs().max()) < 1e-9 * (1 + float(e64.abs().max())), "the term must be mean-free"
    # equivariance of the term under g acting on (x1, xt)
    gen = torch.Generator().manual_seed(1)
    xtd0, x1d0 = xt3.double(), x1.double()
    for kind in ("rotation", "reflection", "permutation"):
        g = group(kind, gen)
        t_ = torch.rand(256, 1).double() * 0.9 + 0.05
        e1, _ = m3.barker_term(t_, x0d, xtd0, x1d0)
        e2, _ = m3.barker_term(t_, x0d, g(xtd0), g(x1d0))
        assert torch.allclose(e2, g(e1), rtol=1e-7, atol=1e-9), (kind, (e2 - g(e1)).abs().max())
    st = m3.pop_stats()
    assert "barker/gamma_eff" in st and any("term_norm_inter" in k for k in st) and abs(sum(v for k, v in st.items() if k.endswith("/frac_t")) - 1.0) < 1e-6, st
    print(f"ok  2. gamma = 0 is exactly the AS target; the extra term = γ h/ν_(1|t) (independent computation), mean-free, equivariant under rotations / reflections / permutations; "
          f"stats {sorted(st)[:3]} ...; mean |term|/|∇g| = {float(extra.norm(dim=-1).mean() / tg0.norm(dim=-1).mean()):.3f}")


if __name__ == "__main__":
    test_detailed_balance_through_the_matcher()
    test_target_construction()
    print("all passed")
