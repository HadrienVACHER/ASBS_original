""" Girsanov path weights: E_{P^u}[w~] = Z = ∫ e^{-E} for ANY controller (zero, or an arbitrary one), on a small ManyWell, within Monte Carlo error.

    python tests/test_girsanov.py
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.components.sde import ControlledSDE, VESDE  # noqa: E402
from adjoint_samplers.energies.dist_energy import DistEnergy  # noqa: E402
from adjoint_samplers.utils.dist_utils import ManyWell  # noqa: E402
from adjoint_samplers.utils.girsanov import ess_and_logz, log_path_weights, mode_weight_ess_frac  # noqa: E402


class Const(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, t, x):
        return self.f(t, x)


def run(dim, u, n, sigma_max=1.4, steps=100, seed=0):
    torch.manual_seed(seed)
    g = ManyWell(dim=dim)
    sde = ControlledSDE(VESDE(0.001, sigma_max), u)
    ts = torch.linspace(0, 1, steps)
    lw, x1 = log_path_weights(sde, DistEnergy(g), torch.zeros(n, dim), ts)
    return ess_and_logz(lw, g.log_Z()), g.log_Z(), lw


def test_log_z():
    g = ManyWell(dim=32)
    assert abs(g.log_Z() - 16 * (math.log(11784.50927) + 0.5 * math.log(2 * math.pi))) < 1e-4
    print(f"ok  exact log Z of ManyWell-32 = {g.log_Z():.4f} (FAB's TARGET_Z per well)")


def test_unbiased_for_any_controller():
    for name, u in (("u = 0", Const(lambda t, x: torch.zeros_like(x))),
                    ("u = 0.4 tanh(x) - 0.2 x t", Const(lambda t, x: 0.4 * torch.tanh(x) - 0.2 * x * t))):
        r, lz, lw = run(2, u, 400000)
        # standard error of log mean exp(lw)
        w = torch.exp(lw.double() - lw.double().max())
        se = float(w.std() / w.mean() / math.sqrt(len(w)))
        assert abs(r["log_zhat"] - lz) < 4 * se + 1e-3, (name, r, lz, se)
        print(f"ok  {name}: log Z_hat = {r['log_zhat']:.4f}, exact {lz:.4f}, |err| = {abs(r['log_zhat'] - lz):.4f} (4 se = {4 * se:.4f}), ESS fraction {r['ess_frac']:.3f}")


def test_mode_weight_ess():
    w = 0.156
    assert abs(mode_weight_ess_frac(torch.full((16,), w), w) - 1.0) < 1e-9
    one = (w ** 2 / 0.5 + (1 - w) ** 2 / 0.5) ** -1                       # one coordinate with f = 0.5
    assert abs(mode_weight_ess_frac(torch.full((16,), 0.5), w) - one ** 16) < 1e-12 * one
    assert mode_weight_ess_frac(torch.tensor([0.0] + [w] * 15), w) == 0.0
    print(f"ok  mode-weight ESS: 1 at f = w, {one:.4f} per coordinate at f = 0.5 ({100 * one ** 16:.1e} % for 16), 0 if a well is missing")


if __name__ == "__main__":
    test_mode_weight_ess()
    test_log_z()
    test_unbiased_for_any_controller()
    print("all passed")
