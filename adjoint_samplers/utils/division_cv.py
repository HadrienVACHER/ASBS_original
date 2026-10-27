""" Division ("Gelfand-Dey") control variate for Adjoint Sampling, GMM-type targets: parameter fit and term (no training code).

    Target p(x1) = e^{-E(x1)} / Z, Dirac source X0 = 0, VE bridge x_t | x1 ~ N(kappa x1, v I), kappa = nu_t / nu_1, v = nu_1 kappa (1 - kappa), nu_{1|t} = nu_1 - nu_t.
    Frozen parameters (a "version"): modes q_k = N(mu_k, Sigma_k), log-weights L_k (exp(-L_k) = beta_k ~ Z w_k), shrink s < 2, tau = 1.

        N_k = N(mu_k, s Sigma_k) (shrunk),    C_k(x_t) = N(x_t; kappa mu_k, kappa^2 s Sigma_k + v I)        r_k = N_k(x1) e^{E(x1)/tau} / C_k(x_t)
        Q_k(x_t) = N(x_t; kappa mu_k, kappa^2 Sigma_k + v I),    m_k(x_t) = mu_k + kappa Sigma_k (kappa^2 Sigma_k + v I)^{-1} (x_t - kappa mu_k)     (posterior mean of x1 under q_k)
        pi_hat_k = softmax_k(-L_k + log Q_k),    M_hat = sum_k pi_hat_k m_k
        term = sum_k exp(-L_k + log Q_k - log C_k + log N_k(x1) + E(x1)/tau) (M_hat - m_k) / nu_{1|t}          target = -grad g + gamma * term

    Zero mean under pi(x1 | x_t) for ANY frozen parameters: E_pi[r_k | x_t] = 1 / (Z p_t(x_t)) for every k, so E_pi[term | x_t] ∝ sum_k e^{-L_k} Q_k (M_hat - m_k) = 0 (pi_hat_k ∝ e^{-L_k} Q_k).
    For disjoint modes with exact shapes and e^{-L_k} = Z w_k: E_q[term | x_t] = (M* - M^q) / nu_{1|t}, the exact Tweedie correction, at every kappa (no kappa-dependent scale).
    Computed in log space, in float64, in the eigenbasis of each Sigma_k.
"""
import math

import torch


def make_params(mu, cov, L, shrink=0.8):
    """ mu (K, D), cov (K, D, D) SPD, L (K,) -> dict with the eigendecomposition of each Sigma_k """
    lam, V = torch.linalg.eigh(0.5 * (cov + cov.transpose(-1, -2)))
    return dict(mu=mu, lam=lam.clamp(min=1e-10), V=V, L=L, shrink=float(shrink))


def _lg(y, var):
    """ log N in the eigenbasis: y (B, K, D) centred and rotated, var (B or 1, K, D) -> (B, K) """
    return -0.5 * (y ** 2 / var).sum(-1) - 0.5 * torch.log(2 * math.pi * var).sum(-1)


def _bk(a, like):
    a = torch.as_tensor(a, dtype=like.dtype, device=like.device)
    return a.reshape(-1, 1, 1) if a.dim() > 0 else a


def division_term(xt, x1, E1, kappa, v, nu1t, par, tau=1.0, terms=False, c_var_scale=1.0):
    """ term (B, D) (and, with terms=True, the log-weights, pi_hat and M_hat) for samples (x1, x_t); kappa, v, nu1t: scalars or (B, 1).
        `par` holds ONE version (mu (K, D), lam (K, D), V (K, D, D), L (K,)) or one version PER SAMPLE (leading batch dimension: mu (B, K, D), lam (B, K, D), V (B, K, D, D), L (B, K);
        modes padded with L = +inf have zero weight): the same formulas, no Python loop over versions. """
    mu, lam, V, L, s = par["mu"], par["lam"], par["V"], par["L"], par["shrink"]
    per_sample = mu.dim() == 3
    xt, x1, E1 = xt.double(), x1.double(), E1.double()                                            # float64 whatever the training dtype
    kap, vv = _bk(kappa, x1), _bk(v, x1)
    nu = torch.as_tensor(nu1t, dtype=x1.dtype, device=x1.device)
    nu = nu.reshape(-1, 1) if nu.dim() > 0 else nu
    if per_sample:
        mu_b, lam_b, L_b = mu, lam, L
        y1 = torch.einsum("bkdj,bkd->bkj", V, x1[:, None, :] - mu_b)
        yt = torch.einsum("bkdj,bkd->bkj", V, xt[:, None, :] - kap * mu_b)
    else:
        mu_b, lam_b, L_b = mu[None], lam[None], L[None]
        y1 = torch.einsum("kdj,bkd->bkj", V, x1[:, None, :] - mu_b)
        yt = torch.einsum("kdj,bkd->bkj", V, xt[:, None, :] - kap * mu_b)
    logN = _lg(y1, s * lam_b)
    logC = _lg(yt, kap ** 2 * s * c_var_scale * lam_b + vv)                                       # c_var_scale != 1 only in the positive control of the tests
    varQ = kap ** 2 * lam_b + vv
    logQ = _lg(yt, varQ)
    z = lam_b / varQ * yt
    m = mu_b + kap * (torch.einsum("bkdj,bkj->bkd", V, z) if per_sample else torch.einsum("kdj,bkj->bkd", V, z))   # (B, K, D)
    logw = -L_b + logQ
    pi = torch.softmax(logw, dim=-1)
    Mhat = (pi[..., None] * m).sum(1)
    logs = logw - logC + logN + E1.reshape(-1, 1) / tau
    term = (torch.exp(logs)[..., None] * (Mhat[:, None, :] - m)).sum(1) / nu
    return (term, logs, pi, Mhat) if terms else term


def stack_versions(versions, ids):
    """ stack the parameter dicts of the versions `ids` into padded tensors (n_versions, Kmax, ...) for per-sample gathering; padded modes get L = +inf (zero weight) """
    ps = [versions[i] for i in ids]
    Kmax, D = max(len(p["mu"]) for p in ps), ps[0]["mu"].shape[1]
    dev, dt = ps[0]["mu"].device, ps[0]["mu"].dtype
    mu = torch.zeros(len(ps), Kmax, D, dtype=dt, device=dev)
    lam = torch.ones(len(ps), Kmax, D, dtype=dt, device=dev)
    V = torch.eye(D, dtype=dt, device=dev).expand(len(ps), Kmax, D, D).clone()
    L = torch.full((len(ps), Kmax), float("inf"), dtype=dt, device=dev)
    for i, p in enumerate(ps):
        K = len(p["mu"])
        mu[i, :K], lam[i, :K], V[i, :K], L[i, :K] = p["mu"], p["lam"], p["V"], p["L"]
    return dict(mu=mu, lam=lam, V=V, L=L, shrink=ps[0]["shrink"])


def gather_versions(stack, idx):
    """ per-sample parameters from the stacked versions: idx (B,) indices into the stack """
    return dict(mu=stack["mu"][idx], lam=stack["lam"][idx], V=stack["V"][idx], L=stack["L"][idx], shrink=stack["shrink"])


def _logq_fn(mu, cov):
    lam, V = torch.linalg.eigh(0.5 * (cov + cov.transpose(-1, -2)))
    lam = lam.clamp(min=1e-10)
    return lambda z: _lg(torch.einsum("kdj,bkd->bkj", V, z[:, None, :] - mu[None]), lam[None])      # (B, K): log q_k(z_b)


def fit_log_beta(x, E, mu, cov, tau=1.0, iters=100):
    """ log-space regression  min_{log beta} mean_i ( logsumexp_k(log beta_k + log q_k(x_i)) + E_i / tau )^2,  q_k = N(mu_k, Sigma_k) (unshrunk); returns L_k = -log beta_k (K,).
        e^{-E/tau} = sum_k beta_k q_k(x) is the model; exp(-L_k) = beta_k ~ Z w_k. Initialised from the buffer point nearest to each mu_k. """
    logq = _logq_fn(mu, cov)
    lq = logq(x)                                                                                  # (N, K)
    near = torch.cdist(mu, x).argmin(dim=1)
    logbeta = (-E[near] / tau - lq[near, torch.arange(len(mu))]).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([logbeta], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = ((torch.logsumexp(logbeta[None] + lq, dim=1) + E / tau) ** 2).mean()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    return -logbeta.detach()


def coverage(x, E, par, tau=1.0):
    """ xi(x) = sum_k exp(-L_k) q_k(x) e^{E(x)/tau}  (~ 1 where the fitted mixture reproduces e^{-E/tau}), (N,) """
    logq = _logq_fn(par["mu"], torch.einsum("kdj,kj,kej->kde", par["V"], par["lam"], par["V"]))
    return torch.exp(torch.logsumexp(-par["L"][None] + logq(x), dim=1) + E / tau)


def fit_modes(energy, x1, E1=None, radius_scale=1.0, denoise_steps=12, shrink=0.8, tau=1.0, xd=None, conv=None, lam=None, generator=None):
    """ Threshold centres of the buffer (frozen Barker method: Barzilai-Borwein basin minima, leader clustering at radius_scale x median width, barrier merge),
        Sigma_k by Laplace (Hessian of E at the centre), L_k by log_beta regression. Returns (params dict or None, info dict).
        xd / conv / lam: cached denoised points (otherwise computed here). """
    from adjoint_samplers.components.matcher import _bb_denoise, _converged_centres, _laplace_factors
    if E1 is None:
        E1 = energy.eval(x1).reshape(-1)
    if xd is None:
        s0 = energy.grad_E(x1)
        xd, conv, lam, *_ = _bb_denoise(energy, x1, E1, s0, denoise_steps, 0.05)
    centres, counts, info = _converged_centres(energy, xd, conv, lam, radius_scale, generator=generator)
    info["K_found"] = len(centres)
    if len(centres) < 2:
        return None, info
    Lf, _, _, cond = _laplace_factors(energy, centres)
    centres, Lf, x1d, E1d = centres.double(), Lf.double(), x1.double(), E1.double()
    cov = Lf @ Lf.transpose(-1, -2)
    Lk = fit_log_beta(x1d, E1d, centres, cov, tau)
    par = make_params(centres, cov, Lk, shrink)
    info["counts"] = counts
    return par, info
