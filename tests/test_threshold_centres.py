""" Checks of the converged-point clustering used for the Barker / MH centres (centre_method: threshold):

    python tests/test_threshold_centres.py

    _bb_denoise (per-point Barzilai-Borwein descent on E, no global state), _converged_centres (leader clustering of
    converged points, radius = median width, barrier merge). Isotropic GMM, ANISOTROPIC GMM (unequal curvatures),
    and the points of an untrained model (epoch 0).
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_mh_matcher import make_gmm, with_weights  # noqa: E402
from adjoint_samplers.components.matcher import _barrier_merge, _bb_denoise, _converged_centres  # noqa: E402
from adjoint_samplers.energies.dist_energy import DistEnergy  # noqa: E402


class AnisoEnergy:
    """ E = - log Σ_k w_k N(x; m_k, diag(σ_k²)): unequal curvatures per mode and per direction """
    def __init__(self, means, weights, scales):
        self.means, self.logw, self.scales = means, weights.log(), scales

    def eval(self, x):
        z = (x.unsqueeze(-2) - self.means) / self.scales
        logp = self.logw - 0.5 * z.pow(2).sum(-1) - self.scales.log().sum(-1) - 0.5 * x.shape[-1] * math.log(2 * math.pi)
        return - torch.logsumexp(logp, -1)

    def grad_E(self, x):
        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)
            return torch.autograd.grad(self.eval(x).sum(), x)[0]

    def sample(self, n):
        k = torch.multinomial(self.logw.exp(), n, replacement=True)
        return self.means[k] + self.scales[k] * torch.randn(n, self.means.shape[1])


def coverage(means, centres, tol):
    cov = (torch.cdist(means, centres).min(1).values < tol).float().mean().item()
    spur = int((torch.cdist(centres, means).min(1).values >= tol).sum())
    return cov, spur


def run(energy, x, means, tol_cov, **kw):
    E0, s0 = energy.eval(x).reshape(-1), energy.grad_E(x)
    xd, conv, lam, steps, n_grad, n_E = _bb_denoise(energy, x, E0, s0, **kw)
    centres, counts, info = _converged_centres(energy, xd, conv, lam)
    cov, spur = coverage(means, centres, tol_cov)
    return xd, conv, lam, steps, n_grad, n_E, centres, info, cov, spur


def test_isotropic_gmm():
    torch.manual_seed(0)
    for dim in (24, 36):
        g = make_gmm(dim=dim, K=dim, radius=4.25)
        w = g.weights.clone()
        w[w <= w.quantile(0.3)] /= 3 # light modes under-sampled, as in the runs
        x = with_weights(g, w).sample([10000])
        xd, conv, lam, steps, n_grad, n_E, centres, info, cov, spur = run(DistEnergy(g), x, g.means, 1.0)
        assert conv.float().mean() > 0.99, conv.float().mean()
        assert len(centres) == dim and cov == 1.0 and spur == 0, (len(centres), cov, spur)
        assert abs(info["width"] - 1.0) < 0.05, info # unit-covariance modes: width 1
        print(f"ok  isotropic d={dim}: {len(centres)} centres, coverage {cov:.2f}, {spur} spurious, converged {conv.float().mean():.3f},"
              f" mean steps {steps.mean():.2f}, ∇E/sample {n_grad / len(x):.2f}, E/sample {n_E / len(x):.2f}, width {info['width']:.3f}, merged {info['n_merged']}")


def test_anisotropic_gmm():
    torch.manual_seed(1)
    dim = 12
    g = make_gmm(dim=dim, K=dim, radius=4.25)
    scales = torch.exp(torch.empty(dim, dim).uniform_(math.log(0.6), math.log(1.6))) # curvature 1/σ² in [0.4, 2.8]
    energy = AnisoEnergy(g.means, g.weights, scales)
    x = energy.sample(8000)
    xd, conv, lam, steps, n_grad, n_E, centres, info, cov, spur = run(energy, x, g.means, 0.3, max_steps=12)
    assert conv.float().mean() > 0.9, conv.float().mean()
    assert len(centres) == dim and cov == 1.0 and spur == 0, (len(centres), cov, spur)
    assert 0.3 < info["radius"] < 3.0, info
    print(f"ok  anisotropic d={dim} (σ in [0.6, 1.6]): {len(centres)} centres, coverage {cov:.2f} (tol 0.3), {spur} spurious,"
          f" converged {conv.float().mean():.3f}, mean steps {steps.mean():.2f}, radius {info['radius']:.2f}")


def test_epoch_zero_points():
    """ the untrained model samples N(0, ν_1 I), ν_1 ~ 10: no mode structure, 512 points """
    torch.manual_seed(2)
    for dim in (24, 36):
        g = make_gmm(dim=dim, K=dim, radius=4.25)
        x = torch.randn(512, dim) * 10 ** 0.5
        xd, conv, lam, steps, n_grad, n_E, centres, info, cov, spur = run(DistEnergy(g), x, g.means, 1.0)
        assert len(centres) == dim and cov == 1.0 and spur == 0, (len(centres), cov, spur, conv.float().mean())
        print(f"ok  epoch 0, d={dim}: {len(centres)} centres from 512 prior points, coverage {cov:.2f}, converged {conv.float().mean():.3f},"
              f" mean steps {steps.mean():.2f}")


def test_barrier_merge_drops_saddle_centres():
    g = make_gmm(dim=3, K=2, radius=4.25)
    g.means = torch.tensor([[-4.0, 0, 0], [4.0, 0, 0]])
    g.weights = torch.tensor([0.5, 0.5])
    energy = DistEnergy(g)
    mid = torch.zeros(1, 3)
    centres = torch.cat([g.means, mid, g.means[:1] + torch.tensor([[0.0, 1.0, 0.0]])]) # 2 modes, a saddle, a near-duplicate
    counts = torch.tensor([50, 40, 1, 2])
    kept, kc, merged = _barrier_merge(energy, centres, counts, radius=1.0, factor=10.0)
    assert len(kept) == 2 and merged == 2 and int(kc.sum()) == 93, (len(kept), merged, kc)
    assert torch.cdist(kept, g.means).min(1).values.max() < 1.01
    print(f"ok  barrier merge: saddle centre and near-duplicate merged, the 2 modes kept (counts {kc.tolist()})")


def test_unconverged_points_are_excluded():
    torch.manual_seed(3)
    g = make_gmm(dim=12, K=12, radius=4.25)
    energy = DistEnergy(g)
    x = g.sample([4000])
    E0, s0 = energy.eval(x).reshape(-1), energy.grad_E(x)
    # two steps: the points whose first step lands on the mode have converged, the tails have not
    xd, conv, lam, steps, _, _ = _bb_denoise(energy, x, E0, s0, max_steps=2)
    centres, counts, info = _converged_centres(energy, xd, conv, lam)
    assert 0.5 < info["conv_frac"] < 1.0 and int(counts.sum()) <= int(conv.sum()), info
    # one step: nothing has converged (the first step is never small): no centre, no crash
    xd1, conv1, lam1, _, _, _ = _bb_denoise(energy, x, E0, s0, max_steps=1)
    c0, _, i0 = _converged_centres(energy, xd1, conv1, lam1)
    assert len(c0) == 0 and i0["conv_frac"] == 0.0
    print(f"ok  two steps: converged fraction {info['conv_frac']:.3f}, only those are clustered ({int(counts.sum())} points);"
          f" one step: none converged -> no centre, no crash")


if __name__ == "__main__":
    test_isotropic_gmm()
    test_anisotropic_gmm()
    test_epoch_zero_points()
    test_barrier_merge_drops_saddle_centres()
    test_unconverged_points_are_excluded()
    print("all passed")
