""" Check of the independent MH chain of AdjointVEIMHBufferMatcher: it samples the target p1, not r.

    python tests/test_imh_buffer.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adjoint_samplers.components.matcher import AdjointVEIMHBufferMatcher
from adjoint_samplers.energies.dist_energy import DistEnergy
from adjoint_samplers.utils.dist_utils import GMM, GMM1D


def chain_mode_share(dist, n_steps, B=20000):
    matcher = AdjointVEIMHBufferMatcher.__new__(AdjointVEIMHBufferMatcher)
    matcher.grad_term_cost = type("C", (), {"energy": DistEnergy(dist)})()
    matcher.r = dist.uniform_weights()
    matcher.chain, matcher.burn_in = None, 0
    matcher._stats = {"accepted": 0.0, "proposals": 0.0}
    x = matcher._advance(torch.zeros(B, dist.dim), n_steps)
    return dist.assign(x) if hasattr(dist, "assign") else (x[:, 0] > 0).long()


def test_demo():
    torch.manual_seed(0)
    k = chain_mode_share(GMM1D(), n_steps=60)
    share = k.float().mean().item() # mass of the right mode: 0.9 (r: 0.5)
    assert abs(share - 0.9) < 0.01, share
    print(f"ok  GMM1D right-mode share {share:.3f} (target 0.9, r 0.5)")


def test_gmm():
    torch.manual_seed(0)
    g = GMM(dim=12, K=12, seed=0, layout="spread", weight_law="loguniform", radius=4.25)
    k = chain_mode_share(g, n_steps=300)
    w = torch.bincount(k, minlength=g.K).float() / k.numel()
    tv = 0.5 * (w - g.weights).abs().sum().item()
    assert tv < 0.03, tv
    print(f"ok  GMM d=K=12 mode-weight TV to the target {tv:.3f}")


if __name__ == "__main__":
    test_demo()
    test_gmm()
