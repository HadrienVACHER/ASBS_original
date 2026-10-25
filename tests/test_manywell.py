""" Checks of ManyWell (FAB definition) and of its evaluator:  python tests/test_manywell.py
    Definition against the FAB code (fab-torch: DoubleWellEnergy a = -0.5, b = -6, c = 1, TARGET_Z = 11784.50927): quadrature, light-well mass.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adjoint_samplers.components.evaluator import ManyWellEvaluator
from adjoint_samplers.energies.dist_energy import DistEnergy
from adjoint_samplers.utils.dist_utils import ManyWell


def test_definition_matches_fab():
    g = ManyWell(dim=32)
    x = torch.linspace(-6, 6, 400001, dtype=torch.float64)
    p = torch.exp(-x ** 4 + 6 * x ** 2 + 0.5 * x)
    Z = float(p.sum() * (x[1] - x[0]))
    assert abs(Z - 11784.50927) < 1e-3, Z # FAB: TARGET_Z
    assert abs(g.light_mass - 0.15569) < 2e-4 and abs(g.well_split + 0.0417) < 2e-3, (g.light_mass, g.well_split)
    z = torch.randn(5, 32)
    want = sum(-z[:, 2 * i] ** 4 + 6 * z[:, 2 * i] ** 2 + 0.5 * z[:, 2 * i] - 0.5 * z[:, 2 * i + 1] ** 2 for i in range(16))
    assert torch.allclose(g.log_prob(z), want, atol=1e-4) and torch.allclose(DistEnergy(g).eval(z), -want, atol=1e-4)
    zz = z.clone().requires_grad_(True)
    assert torch.allclose(torch.autograd.grad(g.log_prob(zz).sum(), zz)[0], g.score(z), atol=1e-3)
    print(f"ok  ManyWell-32 = FAB: Z_1D = {Z:.5f} (FAB 11784.50927), light well mass {g.light_mass:.5f}, split {g.well_split:.4f}, log p~ and score checked")


def test_exact_sampler_and_evaluator():
    torch.manual_seed(0)
    g = ManyWell(dim=32)
    x = g.sample([40000])
    left = (x[:, 0::2] < g.well_split).float().mean()
    assert abs(left - g.light_mass) < 0.004, left
    assert abs(x[:, 1::2].mean()) < 0.01 and abs(x[:, 1::2].var() - 1) < 0.03
    assert abs(x[:, 0::2].mean() - 1.188) < 0.02 and abs(x[:, 0::2].var() - 1.5486) < 0.04
    ev = ManyWellEvaluator(DistEnergy(g))
    out = ev(x[:10000])
    assert out["wells_visited_frac"] == 1.0 and out["well_logratio_mean"] < 0.1 and out["energy_w2"] < 0.3, out
    stuck = x[:2000].clone()
    stuck[:, 0::2] = stuck[:, 0::2].abs() # every coordinate in the right well: the light well is never visited
    m = ev.well_metrics(stuck)
    assert m["wells_visited_frac"] == 0.0 and m["well_ratio_mean"] == 0.0
    print(f"ok  exact sampler (left share {left:.4f}) and evaluator (R mean|log| {out['well_logratio_mean']:.3f}, energy_w2 {out['energy_w2']:.3f}); a sampler stuck in the right wells: visited 0, R = 0")


if __name__ == "__main__":
    test_definition_matches_fab()
    test_exact_sampler_and_evaluator()
    print("all passed")
