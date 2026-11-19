"""Target of the (d, K, Delta) study of Adjoint Sampling: ONE definition, used by the training config (`GMMDK` in dist_utils), the metrics and the tests.

Target: K isotropic Gaussians of variance 1 in dimension d (K <= d),
  means mu_k = (Delta / sqrt 2) e_k on orthogonal axes  -> every pair at distance exactly Delta (in standard deviations),
  normalised separation (distance / (sigma_i + sigma_j)) = Delta / 2;
  geometric weights w_k proportional to rho^(k-1), rho = 10^(-1/(K-1)) -> max / min = 10 for every K.
Truth: p_k = mass of the Voronoi cell k (nearest mean) under the target, estimated with 2e6 exact draws.
Voronoi cell = nearest mean; with equal norms this is argmax_k x_k, but the general formula x.mu_k - |mu_k|^2/2 is used.

sigma_max rule (same for every cell): sigma_max = 2 sqrt(1 + max_k |mu_k|^2 / d) = 2 sqrt(1 + Delta^2 / (2 d)),
i.e. the repository's convention "2 x the per-coordinate std of the target" (gmm_as: `gmm_std:dim,2,radius`) with radius = max |mu_k| = Delta / sqrt 2.
"""
import math

import numpy as np
import torch

SQRT2 = math.sqrt(2.0)


def means(d, K, delta):
    assert 2 <= K <= d
    m = torch.zeros(K, d)
    m[torch.arange(K), torch.arange(K)] = delta / SQRT2
    return m                                                  # [K, d]


def weights(K):
    rho = 10 ** (-1.0 / (K - 1))
    w = rho ** torch.arange(K, dtype=torch.float64)
    return (w / w.sum()).float()                              # [K], w_1 / w_K = 10


def sigma_max_rule(d, delta):
    return 2.0 * math.sqrt(1.0 + delta ** 2 / (2.0 * d))


def voronoi(x, mu):
    """cell index = nearest mean. x [n, d'], mu [K, d'] -> [n]"""
    return (x @ mu.T - 0.5 * (mu ** 2).sum(-1)).argmax(-1)


def sample_exact(n, d, K, delta, generator=None, full=False):
    """Exact draws of the target. By default only the first K coordinates are drawn (all the other coordinates are N(0, 1) noise, identical for every mean:
    they change neither the Voronoi cell nor anything used here); full=True draws all d."""
    mu = means(d, K, delta)
    k = torch.multinomial(weights(K), n, replacement=True, generator=generator)
    if full:
        return mu[k] + torch.randn(n, d, generator=generator), k
    return mu[k][:, :K] + torch.randn(n, K, generator=generator), k


def true_proportions(d, K, delta, n=2_000_000, seed=123):
    """p_k = target mass of Voronoi cell k, by n exact draws (chunks of 200 000)."""
    g = torch.Generator().manual_seed(seed)
    mu = means(d, K, delta)[:, :K]
    counts = torch.zeros(K, dtype=torch.float64)
    done = 0
    while done < n:
        m = min(200_000, n - done)
        x, _ = sample_exact(m, d, K, delta, g)
        counts += torch.bincount(voronoi(x, mu), minlength=K).double()
        done += m
    return counts / n                                         # [K]


def proportions(x, d, K, delta, w=None):
    """cell proportions of samples x [n, d] (or [n, >= K]); optional self-normalised weights w [n]"""
    mu = means(d, K, delta)[:, :x.shape[1]] if x.shape[1] < d else means(d, K, delta)
    a = voronoi(x.double(), mu.double())
    ww = torch.ones(len(x), dtype=torch.float64) if w is None else w.double()
    return torch.zeros(K, dtype=torch.float64).index_add_(0, a, ww) / ww.sum()


def tv(p_hat, p):
    return 0.5 * (p_hat - p).abs().sum().item()


def tv_floor(d, K, delta, p_true, n=5000, reps=200, seed=7):
    """TV of an exact sampler with n samples: median and p5-p95 (computed once per (K, Delta))."""
    g = torch.Generator().manual_seed(seed)
    mu = means(d, K, delta)[:, :K]
    out = []
    for _ in range(reps):
        x, _ = sample_exact(n, d, K, delta, g)
        ph = torch.bincount(voronoi(x, mu), minlength=K).double() / n
        out.append(tv(ph, p_true))
    t = torch.tensor(out)
    return {"median": t.median().item(), "p5": t.quantile(0.05).item(), "p95": t.quantile(0.95).item()}


def coverage(p_hat, p, frac=0.25):
    cov = p_hat >= frac * p
    lost = [(int(k), float(p[k])) for k in torch.nonzero(~cov).flatten()]   # (mode index, true weight p_k)
    return int(cov.sum()), lost, cov


def shrink_beta(p_hat, p, cov):
    """OLS slope of log(p_hat/p) on log p over the covered modes, with its SE. beta = 0: right weights; beta = -1: uniform weights.
    SE is nan with fewer than 3 covered modes (a line through 2 points has no residual)."""
    m = cov & (p_hat > 0)
    x = p[m].log().numpy()
    y = (p_hat[m] / p[m]).log().numpy()
    if len(x) < 2:
        return float("nan"), float("nan")
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    b = ((x - xm) * (y - ym)).sum() / sxx
    if len(x) < 3:
        return float(b), float("nan")
    res = y - (ym + b * (x - xm))
    se = math.sqrt((res ** 2).sum() / (len(x) - 2) / sxx)
    return float(b), float(se)
