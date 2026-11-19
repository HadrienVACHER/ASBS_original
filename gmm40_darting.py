"""GMM-40 target, darting kernel control variate and path-space (Girsanov) IS for Adjoint Sampling.

Conventions (AS, VE bridge): X_0 = 0, dX = sigma_t u dt + sigma_t dB,
x_t | x_1 ~ N(kappa_t x_1, v_t I), nu_{1|t} = nu_1 (1 - kappa_t), p_1^base = N(0, nu_1 I).

Contents
  make_gmm40 / energy / sample_exact          target (FAB spec: 40 means U[-40,40]^2, std softplus(1))
  assign_voronoi / true_proportions / metrics mode-proportion metrics (Voronoi cells of the means)
  discover_minima                            minima + Hessians from buffer samples (no oracle)
  darting_cv                                 Barker darting term c(x_1, x_t), zero mean under pi(.|x_t)
  girsanov_logw                              log importance weights of simulated AS paths
"""
import math
import torch
import torch.nn.functional as F

K = 40


# ----------------------------------------------------------------------------- target
def make_gmm40(seed=0, loc_scaling=40.0, log_var_scaling=1.0, weights="equal", weight_scale=1.0):
    """FAB GMM-40 (Midgley et al. 2023). weights: 'equal' (standard) or 'unequal' (log w ~ scale*N(0,1), seed+1)."""
    torch.manual_seed(seed)                                   # FAB draws the means right after manual_seed
    means = (torch.rand(K, 2) - 0.5) * 2 * loc_scaling        # [K, 2]
    std = F.softplus(torch.tensor(log_var_scaling)).item()    # scalar, ~1.3133
    if weights == "equal":
        logw = torch.full((K,), -math.log(K))                 # [K]
    else:
        g = torch.Generator().manual_seed(seed + 1)
        logw = weight_scale * torch.randn(K, generator=g)
        logw = logw - torch.logsumexp(logw, 0)                # [K]
    return {"means": means, "std": std, "logw": logw}


def log_prob(x, gmm):
    """Normalised log density. x: [..., 2] -> [...]"""
    mu, s, lw = gmm["means"].to(x), gmm["std"], gmm["logw"].to(x)
    d2 = ((x[..., None, :] - mu) ** 2).sum(-1)                # [..., K]
    comp = -0.5 * d2 / s ** 2 - 2 * math.log(s) - math.log(2 * math.pi)
    return torch.logsumexp(lw + comp, -1)                     # [...]


def energy(x, gmm):
    """E = -log pi (tau = 1). x: [..., 2] -> [...]"""
    return -log_prob(x, gmm)


def sample_exact(n, gmm, generator=None):
    k = torch.multinomial(gmm["logw"].exp(), n, replacement=True, generator=generator)   # [n]
    eps = torch.randn(n, 2, generator=generator)                                       # [n, 2]
    return gmm["means"][k] + gmm["std"] * eps                                          # [n, 2]


# ----------------------------------------------------------------------------- metrics
def assign_voronoi(x, gmm):
    """Mode index = nearest mean. x: [n, 2] -> [n] long"""
    return torch.cdist(x, gmm["means"].to(x)).argmin(-1)


def true_proportions(gmm, n=2_000_000, seed=123):
    """p_k = pi(Voronoi cell k), by exact sampling (cells, not w_k: close means leak)."""
    g = torch.Generator().manual_seed(seed)
    x = sample_exact(n, gmm, g)
    return torch.bincount(assign_voronoi(x, gmm), minlength=K).double() / n            # [K]


def proportions(x, gmm, weights=None):
    """Empirical (or self-normalised weighted) cell proportions. x: [n,2], weights: [n] or None -> [K]"""
    a = assign_voronoi(x, gmm)
    w = torch.ones(len(x), dtype=torch.float64) if weights is None else weights.double()
    return torch.zeros(K, dtype=torch.float64).index_add_(0, a, w) / w.sum()


def metrics(p_hat, p_true, cover_frac=0.25):
    """TV, coverage (#cells with p_hat >= cover_frac * p_true), max |log ratio| on covered cells."""
    tv = 0.5 * (p_hat - p_true).abs().sum().item()
    cov = p_hat >= cover_frac * p_true
    lr = (p_hat[cov] / p_true[cov]).log().abs()
    return {"tv": tv, "coverage": int(cov.sum()), "max_abs_log_ratio": lr.max().item() if cov.any() else float("nan")}


def tv_floor(gmm, p_true, n, reps=200, seed=7):
    """TV of an exact sampler with n samples: median, p5, p95."""
    g = torch.Generator().manual_seed(seed)
    tvs = torch.tensor([metrics(proportions(sample_exact(n, gmm, g), gmm), p_true)["tv"] for _ in range(reps)])
    return {"median": tvs.median().item(), "p5": tvs.quantile(0.05).item(), "p95": tvs.quantile(0.95).item()}


# ----------------------------------------------------------------------------- darting
def discover_minima(x, energy_fn, steps=500, lr=0.1, merge_tol=0.5, min_count=5):
    """Quench buffer samples x [n,2] by gradient descent, merge minima within merge_tol,
    keep minima reached by >= min_count samples. Returns (minima [M,2], hessians [M,2,2])."""
    y = x.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([y], lr=lr)
    for _ in range(steps):
        opt.zero_grad(); energy_fn(y).sum().backward(); opt.step()
    y = y.detach()
    mins, counts = [], []
    for p in y:                                               # greedy merge (n is small after dedup)
        for i, m in enumerate(mins):
            if (p - m).norm() < merge_tol:
                counts[i] += 1; break
        else:
            mins.append(p.clone()); counts.append(1)
    mins = torch.stack([m for m, c in zip(mins, counts) if c >= min_count])               # [M, 2]
    H = torch.stack([torch.autograd.functional.hessian(lambda z: energy_fn(z[None])[0], m) for m in mins])
    return mins, H                                            # [M,2], [M,2,2]


def _sqrtm_sym(H):
    """Symmetric square root and inverse square root of SPD matrices [M,d,d]."""
    lam, V = torch.linalg.eigh(H)
    lam = lam.clamp_min(1e-8)
    return (V * lam.sqrt()[..., None, :]) @ V.transpose(-1, -2), (V * lam.rsqrt()[..., None, :]) @ V.transpose(-1, -2)


def darting_cv(x1, xt, kappa, v, nu_cond, minima, H, energy_fn):
    """Barker darting term c [n,2] with E_{pi(.|x_t)}[c] = 0.

    x1, xt: [n,2]; kappa, v, nu_cond: [n] (bridge kappa_t, v_t, nu_{1|t}); minima [M,2], H [M,2,2].
    From x1 in cell k (nearest minimum), propose y_j = m_j + A_jk (x1 - m_k) for every j != k,
    A_jk = H_j^{-1/2} H_k^{1/2} (involution, |det A_jk| = sqrt(det H_k / det H_j)).
    A proposal is valid only if y_j lands in cell j (the reverse move then exists); invalid -> weight 0.
    omega_j = pi~(y_j)|J| / (pi~(x1) + pi~(y_j)|J|),  pi~(y) = exp(-E(y)) N(x_t; kappa y, v I).
    c = (1/nu_{1|t}) * mean_j omega_j (y_j - x1).
    """
    n, M = x1.shape[0], minima.shape[0]
    Hs, Hi = _sqrtm_sym(H)                                                  # [M,2,2] each
    k = torch.cdist(x1, minima).argmin(-1)                                  # [n]
    A = Hi[None, :] @ Hs[k][:, None]                                        # [n, M, 2, 2]  A_jk
    logdet = 0.5 * (torch.logdet(H)[k][:, None] - torch.logdet(H)[None, :]) # [n, M]
    y = minima[None] + (A @ (x1 - minima[k])[:, None, :, None]).squeeze(-1) # [n, M, 2]
    valid = (torch.cdist(y.reshape(-1, 2), minima).argmin(-1).reshape(n, M) == torch.arange(M)[None])
    valid &= torch.arange(M)[None] != k[:, None]                            # [n, M]

    def log_post(z):                                                        # z: [..., 2] -> [...]
        d2 = ((xt.view(n, *([1] * (z.dim() - 2)), 2) - kappa.view(n, *([1] * (z.dim() - 1))) * z) ** 2).sum(-1)
        return -energy_fn(z) - 0.5 * d2 / v.view(n, *([1] * (z.dim() - 2)))

    lx = log_post(x1)                                                       # [n]
    ly = log_post(y) + logdet                                               # [n, M]
    omega = torch.sigmoid(ly - lx[:, None]) * valid                         # [n, M]   Barker
    disp = (omega[..., None] * (y - x1[:, None])).sum(1) / (M - 1)          # [n, 2]
    return disp / nu_cond[:, None]


# ----------------------------------------------------------------------------- path-space IS
@torch.no_grad()
def girsanov_logw(u_fn, sigma_fn, energy_fn, n, n_steps, dim=2, t0=0.0, t1=1.0, generator=None):
    """Simulate n AS paths (Euler-Maruyama from X_0 = 0) and return (X_1 [n,dim], log w [n]).

    log w = -E(X_1) - log p_1^base(X_1) - sum_k [u_k . dB_k + 0.5 |u_k|^2 dt],
    with p_1^base = N(0, sum_k sigma_k^2 dt I) (exact for the discretised base chain).
    u_fn(x [n,dim], t float) -> [n,dim]; sigma_fn(t float) -> float.
    """
    dt = (t1 - t0) / n_steps
    x = torch.zeros(n, dim)
    lw = torch.zeros(n, dtype=torch.float64)
    nu1 = 0.0
    for i in range(n_steps):
        t = t0 + i * dt
        s = sigma_fn(t)
        u = u_fn(x, t)                                                      # [n, dim]
        dB = math.sqrt(dt) * torch.randn(n, dim, generator=generator)       # [n, dim]
        lw -= ((u * dB).sum(-1) + 0.5 * (u ** 2).sum(-1) * dt).double()
        x = x + s * u * dt + s * dB
        nu1 += s ** 2 * dt
    logp_base = -0.5 * (x ** 2).sum(-1) / nu1 - 0.5 * dim * math.log(2 * math.pi * nu1)
    lw += (-energy_fn(x) - logp_base).double()
    return x, lw


def ess_frac(lw):
    w = (lw - lw.max()).exp()
    return (w.sum() ** 2 / (w ** 2).sum() / len(w)).item()
