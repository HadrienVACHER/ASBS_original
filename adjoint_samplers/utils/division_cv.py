""" Division ("Gelfand-Dey") control variate for Adjoint Sampling on GMM targets: building blocks (no matcher yet).

    Target p(x1) = e^{-E(x1)} (normalised, Z = 1), Dirac source X0 = 0, VE bridge x_t | x1 ~ N(kappa_t x1, v_t I), kappa_t = nu_t / nu_1, v_t = nu_1 kappa_t (1 - kappa_t).
    For each mode k a Gaussian N_k = N(mu_k, s_k I) (Laplace approximation of the mode, s_k < 2 sigma^2) and its exact smoothed marginal

        C_k(x_t) = ∫ N_k(x1) N(x_t; kappa x1, v I) dx1 = N(x_t; kappa mu_k, (v + kappa^2 s_k) I),
        r_k(x1, x_t) = N_k(x1) e^{E(x1)} / C_k(x_t).

    Under the exact posterior pi(x1 | x_t) ∝ p(x1) N(x_t; kappa x1, v I):  E_pi[r_k | x_t] = 1 / p_t(x_t) for EVERY k (the integrand p N(x_t; .) r_k = N_k N(x_t; .) / C_k integrates to 1).
    Hence for any vectors a_k(x_t) with sum_k a_k = 0 (they may depend on x_t) and any function h(x_t):

        term(x1, x_t) = h(x_t) sum_k r_k(x1, x_t) a_k      has   E_pi[term | x_t] = 0   EXACTLY.

    Carrier used here: constant vectors a_k = mbar - mu_k, mbar = mean_k mu_k (sum_k a_k = 0): with r_k ∝ q/p smoothed around mode k, the term is -Cov_k(r_k, mu_k), the same structure as the exact
    posterior-mean correction (E_p[x1 | x_t] - E_q[x1 | x_t]) ∝ sum_k (p_k - q_k)(x_t) mu_k. It pushes the drift away from over-sampled modes. Stabiliser h(x_t) = 1 / sum_j C_j(x_t)^{-1}
    (h = 1 gives E[r_k^2] = infinity for every s_k; with this h, h r_k <= sup_x1 N_k e^E). Everything is computed in log space.
"""
import math

import torch


def log_gauss_iso(x, mean, var, d):
    """ log N(x; mean, var I_d): x (..., d), mean broadcastable, var broadcastable to the batch (..., 1) """
    return -0.5 * ((x - mean) ** 2).sum(-1) / var.squeeze(-1) - 0.5 * d * torch.log(2 * math.pi * var.squeeze(-1))


def log_r(xt, x1, logp1, kappa, v, mu, s, s_c=None):
    """ log r_k(x1, x_t), (B, K). xt, x1 (B, d); logp1 = log p(x1) (B,) with p normalised; mu (K, d); s (K,) variances of the N_k;
        kappa, v: scalars or (B, 1). s_c: variances used inside C_k only (positive control: s_c != s breaks the zero mean). """
    d = x1.shape[-1]
    s_c = s if s_c is None else s_c
    kappa, v = torch.as_tensor(kappa, dtype=x1.dtype), torch.as_tensor(v, dtype=x1.dtype)
    kappa = kappa.reshape(-1, 1, 1) if kappa.dim() > 0 else kappa
    vk = (v.reshape(-1, 1) if v.dim() > 0 else v) + (kappa.reshape(-1, 1) if kappa.dim() > 0 else kappa) ** 2 * s_c   # (B, K) or (K,)
    kap = kappa.reshape(-1, 1, 1) if kappa.dim() > 0 else kappa
    logN = log_gauss_iso(x1[:, None, :], mu[None], s[None, :, None], d)                       # (B, K)
    logC = log_gauss_iso(xt[:, None, :], kap * mu[None], (vk if vk.dim() == 2 else vk[None])[..., None], d)
    return logN - logp1[:, None] - logC, logC


def division_term(xt, x1, logp1, kappa, v, mu, s, s_c=None, scale=1.0):
    """ scale * h(x_t) sum_k r_k a_k, (B, d), and the stabilised weights h r_k (B, K) """
    lr, logC = log_r(xt, x1, logp1, kappa, v, mu, s, s_c)
    logh = -torch.logsumexp(-logC, dim=-1, keepdim=True)
    rt = torch.exp(lr + logh)                                                                  # h r_k, (B, K)
    a = mu.mean(0, keepdim=True) - mu                                                          # (K, d), sum_k a_k = 0
    return scale * rt @ a, rt
