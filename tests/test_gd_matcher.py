""" Checks of the Gelfand-Dey control variate (AdjointVEGelfandDeyMatcher) on the 1D demo target:

    python tests/test_gd_matcher.py
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adjoint_samplers.components.buffer import BatchBuffer
from adjoint_samplers.components.matcher import AdjointVEGelfandDeyMatcher
from adjoint_samplers.components.model import FourierMLP
from adjoint_samplers.components.sde import ControlledSDE, VESDE
from adjoint_samplers.components.state_cost import ZeroGradStateCost
from adjoint_samplers.components.term_cost import ScoreGradTermCost
from adjoint_samplers.energies.dist_energy import DistEnergy
from adjoint_samplers.utils.dist_utils import GMM1D, Delta

SIGMA_MIN, SIGMA_MAX = 0.001, 4.0
N = 400_000


def make_matcher(**kwargs):
    ref_sde = VESDE(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX)
    sde = ControlledSDE(ref_sde, FourierMLP(1, activation=torch.nn.GELU()))
    term_cost = ScoreGradTermCost(source=Delta(1), ref_sde=ref_sde, energy=DistEnergy(GMM1D()))
    return AdjointVEGelfandDeyMatcher(
        grad_term_cost=term_cost, grad_state_cost=ZeroGradStateCost(), sde=sde,
        buffer=BatchBuffer(10**6), resample_size=256, duplicates=1, **kwargs,
    )


def t_at_kappa(kappa):
    # κ_t = 1 - (σ_max/σ_min)^{-2t}
    return - math.log(1 - kappa) / (2 * math.log(SIGMA_MAX / SIGMA_MIN))


def sample_x1(weights, n):
    k = torch.multinomial(torch.tensor(weights), n, replacement=True)
    return torch.tensor([-3.0, 3.0])[k].reshape(-1, 1) + 0.5 * torch.randn(n, 1)


def cv_at(matcher, kappa, x1):
    n = len(x1)
    t = torch.full((n, 1), t_at_kappa(kappa))
    x0 = torch.zeros(n, 1)
    xt = matcher.sde.sample_base_posterior(t, x0, x1)
    E1 = matcher.grad_term_cost.energy.eval(x1).reshape(-1, 1)
    return matcher.control_variate(t, x0, xt, x1, E1)


def z_score(c):
    return (c.mean() / (c.std() / math.sqrt(len(c)))).item()


def main():
    torch.manual_seed(0)
    matcher = make_matcher()

    # 1. Laplace fit: μ_k = ±3, Σ_k = sigma_scale × 0.25
    matcher.fit_modes(sample_x1([.5, .5], 2000))
    mu, var = matcher.modes
    assert torch.allclose(mu, torch.tensor([-3.0, 3.0], dtype=mu.dtype), atol=1e-3), mu
    assert torch.allclose(var, torch.full_like(var, 0.25 * matcher.sigma_scale), rtol=1e-3), var
    print("fit ok:", mu.tolist(), var.tolist())

    # 2. zero mean under the target p1, bounded
    for kappa in [0.15, 0.4, 0.6]:
        cv = cv_at(matcher, kappa, sample_x1([.1, .9], N))
        z = z_score(cv)
        assert abs(z) < 5, (kappa, z)
        assert cv.abs().max() < 15, cv.abs().max()
        print(f"E_p[cv] kappa={kappa}: z={z:+.2f}, max|cv|={cv.abs().max():.2f}")

    # 3. under q = (.5, .5): E_q[cv] = +1.129 at κ = 0.6 (quadrature, Σ_k = 0.2); sign flips with q_1 < 0.1
    cv = cv_at(matcher, 0.6, sample_x1([.5, .5], N))
    se = cv.std() / math.sqrt(N)
    assert abs(cv.mean() - 1.129) < 5 * se + 0.01, (cv.mean(), se)
    print(f"E_q[cv] q=(.5,.5) kappa=0.6: {cv.mean():+.3f} (quadrature +1.129)")
    cv = cv_at(matcher, 0.15, sample_x1([.05, .95], N))
    assert cv.mean() < 0 and z_score(cv) < -5, cv.mean()
    print(f"E_q[cv] q=(.05,.95) kappa=0.15: {cv.mean():+.3f} < 0")

    # 4. prepare_target: target = - (adjoint1 + c cv), through the buffer
    x0 = torch.zeros(512, 1)
    matcher.populate_buffer(x0, torch.linspace(0, 1, 11), is_asbs_init_stage=False)
    data = next(iter(matcher.build_dataloader(256)))
    (t, xt), target = matcher.prepare_target(data, "cpu")
    assert target.shape == xt.shape == (256, 1) and torch.isfinite(target).all()
    print("prepare_target ok")

    # 5. tracking: source and parameters of N_k, statistics of cv
    stats = matcher.log_stats()
    assert stats["gd/source"] == 1 and stats["gd/active"] == 1.0, stats
    assert abs(stats["gd/mu_0"] + 3) < 1e-3 and abs(stats["gd/var_1"] - 0.2) < 1e-3, stats
    assert {"gd/cv_mean", "gd/cv_std", "gd/cv_absmax", "gd/shift_over_target"} <= set(stats), stats
    assert "gd/cv_mean" not in matcher.log_stats() # reset after each call
    print("tracking laplace ok:", {k: round(v, 3) for k, v in stats.items()})

    # 6. manual modes (unsorted on purpose): used as given, no fit, source = 2
    manual = make_matcher(modes=[{"mu": 3.0, "var": 0.3}, {"mu": -3.0, "var": 0.1}])
    assert manual.mode_source == "manual"
    manual.populate_buffer(x0, torch.linspace(0, 1, 11), is_asbs_init_stage=False)
    assert manual.mode_source == "manual" and manual.modes[0].tolist() == [-3.0, 3.0]
    assert manual.modes[1].tolist() == [0.1, 0.3] and manual.log_stats()["gd/source"] == 2
    print("tracking manual ok")

    # 7. fit failure (flat energy-free case emulated): source = 0, cv = 0
    dis = make_matcher()
    dis.fit_modes(torch.full((100, 1), 3.0)) # all seeds in the right basin -> 1 mode
    assert dis.mode_source == "disabled" and dis.log_stats()["gd/source"] == 0
    print("tracking disabled ok")


if __name__ == "__main__":
    main()
