""" Checks of adjoint_samplers/components/mode_correction.py (CPU, under a minute):

    python tests/test_mode_correction.py
"""
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adjoint_samplers.components.mode_correction import (
    EnergyCounter, TiedMixture, jump_mala_cycle, refine_centres, run_imh, run_jump_mala, snis_mode_weights,
)


def simplex_gmm(d, K, weights, sep=8.0):
    """ p = sum_k w_k N(sep e_k, I), K <= d: returns means, log_prob and exact sampler """
    means = sep * torch.eye(d)[:K]
    logw = weights.log()

    def log_prob(x):
        sq = (x.unsqueeze(-2) - means).pow(2).sum(-1)
        return torch.logsumexp(logw - 0.5 * sq - 0.5 * d * math.log(2 * math.pi), -1)

    def sample(n):
        k = torch.multinomial(weights, n, replacement=True)
        return means[k] + torch.randn(n, d)

    return means, log_prob, sample


def counter_of(log_prob):
    return EnergyCounter(lambda x: -log_prob(x))


def test_imh_oracle():
    """ equal weights and the exact centres: r = p, so every proposal is accepted """
    torch.manual_seed(0)
    d, K = 6, 6
    means, log_prob, sample = simplex_gmm(d, K, torch.full((K,), 1 / K))
    r = TiedMixture(means, torch.eye(d))
    ec = counter_of(log_prob)
    _, _, diag = run_imh(ec, r, sample(500), T=20, burn=0, assign=lambda x: torch.zeros(len(x), dtype=torch.long), n_modes=1)
    assert diag["acc"] > 0.98, diag
    print(f"ok  IMH with the oracle proposal: acceptance {diag['acc']:.3f}")


def test_jump_acceptance():
    """ exact centres, states from p: the jump acceptance is E[min(1, pi_k' / pi_k)] over k ~ pi, k' uniform != k """
    torch.manual_seed(0)
    d, K = 12, 12
    pi = torch.distributions.Dirichlet(torch.full((K,), 2.0)).sample()
    means, log_prob, sample = simplex_gmm(d, K, pi)
    r = TiedMixture(means, torch.eye(d))
    ec = counter_of(log_prob)
    x = sample(40000)
    _, diag = jump_mala_cycle(ec, r, x, n_mala=0, h=0.1)
    ratio = (pi[None, :] / pi[:, None]).clamp(max=1.0)
    ratio.fill_diagonal_(0.0)
    theory = (pi * ratio.sum(1) / (K - 1)).sum().item()
    assert abs(diag["acc_jump"] - theory) < 0.02, (diag, theory)
    print(f"ok  jump acceptance {diag['acc_jump']:.3f} (theory {theory:.3f})")


def test_jump_mala_invariance():
    """ 1D, N(-4, 1) and N(4, 1) with weights 0.2 / 0.8, all chains start in the light mode """
    torch.manual_seed(0)
    pi = torch.tensor([0.2, 0.8])
    means = torch.tensor([[-4.0], [4.0]])
    logw = pi.log()

    def log_prob(x):
        return torch.logsumexp(logw - 0.5 * (x.unsqueeze(-2) - means).pow(2).sum(-1) - 0.5 * math.log(2 * math.pi), -1)

    r = TiedMixture(means, torch.eye(1))
    ec = counter_of(log_prob)
    x0 = -4.0 + torch.randn(4000, 1)
    assign = lambda x: (x[:, 0] > 0).long()
    _, counts, _ = run_jump_mala(ec, r, x0, cycles=120, burn=40, n_mala=2, h=0.5, assign=assign, n_modes=2)
    share = (counts.sum(0) / counts.sum())[1].item()
    assert abs(share - 0.8) < 0.02, share
    print(f"ok  jump + MALA mode share {share:.3f} (target 0.8), chains started in the light mode")


def test_refine():
    """ twice the centres, noisy: refinement recovers the K exact centres """
    torch.manual_seed(0)
    d, K = 12, 12
    means, log_prob, _ = simplex_gmm(d, K, torch.full((K,), 1 / K))
    raw = torch.cat([means, means]) + 0.5 * torch.randn(2 * K, d)
    ec = counter_of(log_prob)
    out, n_grad = refine_centres(ec, raw, merge_tol=1.0)
    assert out.shape == (K, d), out.shape
    err = torch.cdist(out, means).min(-1).values.max().item()
    assert err < 1e-3, err
    assert n_grad == ec.n_grad > 0
    print(f"ok  refinement: {2 * K} -> {len(out)} centres, max error {err:.1e}, {n_grad} gradient evaluations")


def test_counters():
    torch.manual_seed(0)
    d, K, C, T, n_mala = 4, 4, 100, 7, 3
    means, log_prob, sample = simplex_gmm(d, K, torch.full((K,), 1 / K))
    r = TiedMixture(means, torch.eye(d))
    assign = lambda x: torch.zeros(len(x), dtype=torch.long)

    ec = counter_of(log_prob)
    run_imh(ec, r, sample(C), T=T, burn=0, assign=assign, n_modes=1)
    assert (ec.n_E, ec.n_grad) == (C * (1 + T), 0), (ec.n_E, ec.n_grad)

    ec = counter_of(log_prob)
    snis_mode_weights(ec, r, B=1234, assign=assign, n_modes=1, chunk=500)
    assert (ec.n_E, ec.n_grad) == (1234, 0), (ec.n_E, ec.n_grad)

    ec = counter_of(log_prob)
    jump_mala_cycle(ec, r, sample(C), n_mala=n_mala, h=0.1)
    assert (ec.n_E, ec.n_grad) == (C * (n_mala + 2), C * (n_mala + 1)), (ec.n_E, ec.n_grad)
    print("ok  evaluation counters match the call counts of IMH, SNIS and a jump + MALA cycle")


if __name__ == "__main__":
    t0 = time.time()
    for test in (test_imh_oracle, test_jump_acceptance, test_jump_mala_invariance, test_refine, test_counters):
        test()
    print(f"all passed in {time.time() - t0:.1f}s")
