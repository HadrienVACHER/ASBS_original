""" Mode-weight correction of an Adjoint Sampling buffer: independent MH, self-normalised IS, and a
    mode-jump + MALA sampler, all driven by an equal-weight tied-covariance Gaussian mixture r fitted on the
    buffer (centres optionally refined by descent on E).

    Target p = e^{-E} / Z. Energy cost is counted through `EnergyCounter` so that methods compare at equal budget.
"""
import math
from contextlib import contextmanager

import torch
from torch import distributions





def structure_cov(covs, Nk, cov_type, shrink, reg):
    """ covs (M, D, D) of the components -> the covariance family, then shrunk towards tr(C)/D I, then + reg I
          full       as is
          tied       the pooled covariance Σ_k N_k C_k / N, the same for every component
          diag       diag(C_k)
          spherical  tr(C_k) / D I
    """
    M, D, _ = covs.shape
    eye = torch.eye(D, device=covs.device, dtype=covs.dtype)
    if cov_type == "tied":
        covs = ((Nk[:, None, None] * covs).sum(0) / Nk.sum()).expand(M, D, D)
    elif cov_type == "diag":
        covs = torch.diag_embed(torch.diagonal(covs, dim1=-2, dim2=-1))
    elif cov_type == "spherical":
        covs = torch.diagonal(covs, dim1=-2, dim2=-1).mean(-1)[:, None, None] * eye
    else:
        assert cov_type == "full", cov_type
    scale = torch.diagonal(covs, dim1=-2, dim2=-1).mean(-1)[:, None, None]
    return (1 - shrink) * covs + shrink * scale * eye + reg * eye


def fit_gmm(x, M, iters=100, reg=1e-4, tol=1e-6, init_means=None, cov_type="full", shrink=0.0):
    """ EM for a Gaussian mixture on x (N, D) from k-means++ (Lloyd) centres (or from `init_means`);
        covariance family `cov_type` (see structure_cov), shrunk by `shrink` in [0, 1]
    """
    N, D = x.shape
    from adjoint_samplers.components.matcher import _kmeans # lazy: matcher is heavy
    means = _kmeans(x, M, iters=20) if init_means is None else init_means.clone()
    a = torch.cdist(x, means).argmin(-1)
    counts = torch.bincount(a, minlength=M).float().clamp(min=1)
    weights = counts / N
    eye = torch.eye(D, device=x.device, dtype=x.dtype)
    covs = torch.stack([
        torch.cov(x[a == k].T).reshape(D, D) if (a == k).sum() > 1 else eye
        for k in range(M)
    ])
    covs = structure_cov(covs, counts, cov_type, shrink, reg)
    prev = -float("inf")
    for _ in range(iters):
        comp = distributions.MultivariateNormal(means, covariance_matrix=covs)
        logp = comp.log_prob(x.unsqueeze(1)) + weights.log() # (N, M)
        ll = torch.logsumexp(logp, -1)
        resp = (logp - ll.unsqueeze(-1)).exp()
        Nk = resp.sum(0).clamp(min=1e-8)
        weights = Nk / N
        means = (resp.T @ x) / Nk[:, None]
        diff = x.unsqueeze(1) - means # (N, M, D)
        covs = torch.einsum("nk,nkd,nke->kde", resp, diff, diff) / Nk[:, None, None]
        covs = structure_cov(covs, Nk, cov_type, shrink, reg)
        if abs(ll.mean().item() - prev) < tol:
            break
        prev = ll.mean().item()
    return weights, means, covs


class EnergyCounter:
    """ Wraps E to count energy and gradient evaluations. `energy`: an object with `.eval(x) -> (n,)` or a callable. """

    def __init__(self, energy):
        self.fn = energy.eval if hasattr(energy, "eval") else energy
        self.n_E, self.n_grad = 0, 0

    @torch.no_grad()
    def E(self, x):                                   # (n, d) -> (n,)
        self.n_E += x.shape[0]
        return self.fn(x).reshape(-1)

    def E_and_grad(self, x):                          # (n, d) -> (n,), (n, d); counts one E and one gradient
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            e = self.fn(x).reshape(-1)
            g = torch.autograd.grad(e.sum(), x)[0]
        self.n_E += x.shape[0]
        self.n_grad += x.shape[0]
        return e.detach(), g.detach()

    def snapshot(self):
        return self.n_E, self.n_grad


def refine_centres(ec, m, n_steps=200, lr=0.1, max_step=0.5, gtol=1e-4, merge_tol=1.0):
    """ Gradient descent on E from each GMM mean (step norm clipped at max_step), then greedy merge of the minima
        closer than merge_tol. m: (M, d) -> (M', d), and the number of gradient evaluations used.
        The distance is euclidean: replace it by a symmetry-aware one (Kabsch + permutation) for LJ / DW-4.
    """
    x = m.clone()
    active = torch.ones(len(x), dtype=torch.bool, device=x.device)
    n0 = ec.n_grad
    for _ in range(n_steps):
        if not active.any():
            break
        _, g = ec.E_and_grad(x[active])
        step = -lr * g
        scale = (max_step / step.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
        idx = active.nonzero(as_tuple=True)[0]
        x[idx] = x[idx] + scale * step
        active[idx[g.norm(dim=-1) <= gtol]] = False
    keep = []
    for i in range(len(x)):
        if all((x[i] - x[j]).norm() > merge_tol for j in keep):
            keep.append(i)
    return x[keep], ec.n_grad - n0


class TiedMixture:
    """ Equal-weight Gaussian mixture with a shared covariance. """

    def __init__(self, means, cov):                   # (M, d), (d, d)
        self.m = means
        self.M, self.d = means.shape
        self.L = torch.linalg.cholesky(cov)
        self.mz = torch.linalg.solve_triangular(self.L, means.T, upper=False).T
        self.logdet = 2 * self.L.diagonal().log().sum()

    def comp_logpdf(self, x):                         # (n, d) -> (n, M)
        z = torch.linalg.solve_triangular(self.L, x.T, upper=False).T
        sq = (z ** 2).sum(-1, keepdim=True) - 2 * z @ self.mz.T + (self.mz ** 2).sum(-1)
        return -0.5 * (sq + self.logdet + self.d * math.log(2 * math.pi))

    def logpdf(self, x):                              # (n, d) -> (n,)
        return torch.logsumexp(self.comp_logpdf(x), -1) - math.log(self.M)

    def log_resp(self, x):                            # (n, d) -> (n, M), log rho_k(x)
        c = self.comp_logpdf(x)
        return c - torch.logsumexp(c, -1, keepdim=True)

    def sample(self, n):                              # -> (n, d)
        j = torch.randint(self.M, (n,), device=self.m.device)
        return self.m[j] + torch.randn(n, self.d, device=self.m.device, dtype=self.m.dtype) @ self.L.T


def add_uncovered_centres(ec, x, centres, cov, quantile=0.99, max_outliers=1000, n_steps=200, lr=0.1,
                          max_step=0.5, gtol=1e-4, merge_tol=1.0):
    """ Add the modes that no centre explains. Buffer points x (N, d) whose squared Mahalanobis distance (tied covariance
        `cov`) to every centre exceeds the chi2_d quantile are relaxed by descent on E (at most `max_outliers`, the
        farthest first), and each minimum farther than merge_tol from every existing centre becomes a new centre.
        Returns the centres (M + added, d), the number added, the number of outliers relaxed, and the gradients used.
        A mode with a handful of buffer samples is enough: a single outlier relaxed into it adds its minimum.
    """
    from scipy.stats import chi2
    L = torch.linalg.cholesky(cov)
    z = torch.linalg.solve_triangular(L, x.T, upper=False).T
    zc = torch.linalg.solve_triangular(L, centres.T, upper=False).T
    d2 = torch.cdist(z, zc).min(-1).values ** 2
    out = (d2 > float(chi2.ppf(quantile, x.shape[1]))).nonzero(as_tuple=True)[0]
    if len(out) == 0:
        return centres, 0, 0, 0
    if len(out) > max_outliers:
        out = out[d2[out].argsort(descending=True)[:max_outliers]]
    minima, n_grad = refine_centres(ec, x[out], n_steps=n_steps, lr=lr, max_step=max_step, gtol=gtol, merge_tol=merge_tol)
    new = minima[(torch.cdist(minima, centres).min(-1).values > merge_tol)]
    return torch.cat([centres, new]), len(new), len(out), n_grad


def merge_centres(ec, existing, new, tol, gtol=1e-2):
    """ The memory of the minima found so far: `existing` (M, d) plus the centres of `new` that are converged
        (gradient norm <= gtol, so no stalled relaxation becomes a spurious mode) and farther than tol from every
        centre kept. Returns the centres, the number added and the number of unconverged ones skipped.
    """
    if len(new) == 0:
        return existing, 0, 0
    _, g = ec.E_and_grad(new)
    ok = g.norm(dim=-1) <= gtol
    kept, added = existing, 0
    for x in new[ok]:
        if len(kept) == 0 or (x - kept).norm(dim=-1).min() > tol:
            kept = torch.cat([kept, x[None]])
            added += 1
    return kept, added, int((~ok).sum())


def _label_counts(labels, n_modes):
    return torch.bincount(labels, minlength=n_modes).float()


@torch.no_grad()
def run_imh(ec, r, x0, T, burn, assign, n_modes):
    """ Independent MH with proposal r from the states x0 (C, d). Returns the final states, the mode-label counts of
        every chain over the post-burn-in states (C, K), and diagnostics (acceptance, max log(p/r) over the proposals).
    """
    x = x0.clone()
    C = x.shape[0]
    lw = -ec.E(x) - r.logpdf(x)
    counts = torch.zeros(C, n_modes, device=x.device)
    acc, max_lw = 0.0, -float("inf")
    for t in range(T):
        y = r.sample(C)
        lwy = -ec.E(y) - r.logpdf(y)
        max_lw = max(max_lw, lwy.max().item())
        a = torch.rand_like(lw).log() < lwy - lw
        x[a], lw[a] = y[a], lwy[a]
        acc += a.float().mean().item() / T
        if t >= burn:
            counts.scatter_add_(1, assign(x)[:, None], torch.ones(C, 1, device=x.device))
    return x, counts, dict(acc=acc, max_log_w=max_lw)


@torch.no_grad()
def snis_mode_weights(ec, r, B, assign, n_modes, chunk=50000):
    """ Self-normalised importance sampling with B draws of r. Returns the mode weights (K,), the normalised ESS
        1 / (B sum w^2), and the resampling weights w (B,) with the draws' labels (B,).
    """
    lws, labels = [], []
    for i in range(0, B, chunk):
        y = r.sample(min(chunk, B - i))
        lws.append(-ec.E(y) - r.logpdf(y))
        labels.append(assign(y))
    lw, labels = torch.cat(lws), torch.cat(labels)
    w = torch.softmax(lw, 0)
    ess = 1.0 / (B * (w ** 2).sum())
    pk = torch.zeros(n_modes, device=w.device).index_add_(0, labels, w)
    return pk, ess.item(), w, labels


def mala_steps(ec, x, n, h):
    """ n MALA steps on p from x (C, d), step h. Returns the new x and the mean acceptance. """
    C = x.shape[0]
    e, g = ec.E_and_grad(x)
    acc = 0.0
    for _ in range(n):
        y = x - h * g + math.sqrt(2 * h) * torch.randn_like(x)
        ey, gy = ec.E_and_grad(y)
        lq_fw = -((y - x + h * g) ** 2).sum(-1) / (4 * h)
        lq_bw = -((x - y + h * gy) ** 2).sum(-1) / (4 * h)
        a = torch.rand(C, device=x.device).log() < (e - ey + lq_bw - lq_fw)
        x, e, g = torch.where(a[:, None], y, x), torch.where(a, ey, e), torch.where(a[:, None], gy, g)
        acc += a.float().mean().item() / max(n, 1)
    return x, acc


def jump_mala_cycle(ec, r, x, n_mala, h):
    """ One cycle of the augmented chain (x, k) with target p(x) rho_k(x), rho = responsibilities of r: n_mala MALA
        steps on p, Gibbs k | x, then the translation x' = x + m_k' - m_k to a uniformly chosen other centre.
        Tied covariance: the Jacobian is 1. Returns the new x and the MALA and jump acceptance rates.
    """
    C = x.shape[0]
    assert r.M >= 2
    e, g = ec.E_and_grad(x)
    acc_m = 0.0
    for _ in range(n_mala):
        y = x - h * g + math.sqrt(2 * h) * torch.randn_like(x)
        ey, gy = ec.E_and_grad(y)
        lq_fw = -((y - x + h * g) ** 2).sum(-1) / (4 * h)
        lq_bw = -((x - y + h * gy) ** 2).sum(-1) / (4 * h)
        a = torch.rand(C, device=x.device).log() < (e - ey + lq_bw - lq_fw)
        x, e, g = torch.where(a[:, None], y, x), torch.where(a, ey, e), torch.where(a[:, None], gy, g)
        acc_m += a.float().mean().item() / max(n_mala, 1)
    with torch.no_grad():
        lr_x = r.log_resp(x)
        k = torch.distributions.Categorical(logits=lr_x).sample()
        kp = (k + torch.randint(1, r.M, (C,), device=x.device)) % r.M
        y = x + r.m[kp] - r.m[k]
        ey = ec.E(y)
        ar = torch.arange(C, device=x.device)
        log_a = -ey + r.log_resp(y)[ar, kp] + e - lr_x[ar, k]
        a = torch.rand(C, device=x.device).log() < log_a
        x = torch.where(a[:, None], y, x)
    return x, dict(acc_mala=acc_m, acc_jump=a.float().mean().item())


def run_jump_mala(ec, r, x0, cycles, burn, n_mala, h, assign, n_modes):
    """ `cycles` of jump_mala_cycle from x0 (C, d); mode-label counts of every chain over the post-burn-in cycles. """
    x = x0.clone()
    C = x.shape[0]
    counts = torch.zeros(C, n_modes, device=x.device)
    acc_m = acc_j = 0.0
    for t in range(cycles):
        x, d = jump_mala_cycle(ec, r, x, n_mala, h)
        acc_m += d["acc_mala"] / cycles
        acc_j += d["acc_jump"] / cycles
        if t >= burn:
            with torch.no_grad():
                counts.scatter_add_(1, assign(x)[:, None], torch.ones(C, 1, device=x.device))
    return x, counts, dict(acc_mala=acc_m, acc_jump=acc_j)


def mode_tv_from_counts(counts, pi):
    """ total variation between the pooled mode shares of the chains' states and the target weights pi (K,) """
    share = counts.sum(0) / counts.sum()
    return 0.5 * (share - pi).abs().sum().item()


def chain_ess(counts, pi):
    """ effective sample size of the pooled mode-share estimate, from the spread of the per-chain shares:
            N_eff = pi_k (1 - pi_k) C / Var_c(share_c,k),   median over the modes k.
        Chains are independent, so this accounts for the autocorrelation along each chain.
    """
    C = counts.shape[0]
    share_c = counts / counts.sum(1, keepdim=True).clamp_min(1)
    var = share_c.var(0, unbiased=True).clamp_min(1e-12)
    n_eff = pi * (1 - pi) * C / var
    return n_eff.median().item()


def iid_mode_tv(pi, n, draws=20, generator=None):
    """ expected mode_tv of n iid samples of the target (the floor at that sample size) """
    pi = pi.cpu()
    tv = []
    for _ in range(draws):
        c = torch.multinomial(pi, int(n), replacement=True, generator=generator)
        share = torch.bincount(c, minlength=len(pi)).float() / n
        tv.append(0.5 * (share - pi).abs().sum().item())
    return sum(tv) / len(tv)


@contextmanager
def preserve_rng():
    """ restore the global (CPU and CUDA) random state on exit, so that an observer leaves the run it watches unchanged """
    cpu = torch.get_rng_state()
    cuda = None
    try:
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    except Exception:
        cuda = None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            try:
                torch.cuda.set_rng_state_all(cuda)
            except Exception:
                pass


def buffer_correction_diagnostic(energy, dist, x1, M=None, chains=2000, steps=60, burn=15, inflate=1.0, merge_tol_mult=1.0,
                                 cover_quantile=0.99, max_outliers=1000, refine_steps=200, refine_lr=0.1,
                                 memory=None, use_memory=True):
    """ What the correction would do to the buffer x1 (N, d), without changing it: fit (EM, M = 2K, tied), refine the centres,
        add the uncovered modes, run independent MH from buffer states. `dist` is the GMM target (K, dim, weights, assign).
        The global random state is restored. Returns (metrics, memory): floats (mode_tv of the buffer and of the chain
        pooled over all post-burn-in states and of the final states, acceptance, ESS, centres, coverage, evaluation
        cost) and the memory of the minima found so far. With `use_memory`, a second chain uses the centres of
        `memory` (the minima of the earlier calls) together with the new ones: keys with the suffix `_mem`.
    """
    K, d = dist.K, dist.dim
    pi = dist.weights.to(x1.device)
    assign = dist.assign
    M = M or 2 * K
    q = max(1, K // 4)
    order = pi.argsort()
    with preserve_rng():
        _, mu, cov = fit_gmm(x1, M, cov_type="tied")
        cov_t = cov[0]
        ec_setup = EnergyCounter(energy)
        merge_tol = merge_tol_mult * (cov_t.trace() / d).sqrt().item()
        centres, _ = refine_centres(ec_setup, mu, n_steps=refine_steps, lr=refine_lr, merge_tol=merge_tol)
        centres_cov, n_added, n_out, _ = add_uncovered_centres(
            ec_setup, x1, centres, cov_t, quantile=cover_quantile, max_outliers=max_outliers,
            n_steps=refine_steps, lr=refine_lr, merge_tol=merge_tol)
        r = TiedMixture(centres_cov, cov_t * inflate)
        ec = EnergyCounter(energy)
        x0 = x1[torch.randint(len(x1), (chains,), device=x1.device)]
        x, counts, diag = run_imh(ec, r, x0, steps, burn, assign, K)
        mem_out, extra = memory, {}
        if use_memory:
            empty = centres_cov.new_zeros(0, d)
            mem_out, n_new_mem, n_unconv = merge_centres(ec_setup, empty if memory is None else memory, centres_cov, merge_tol)
            ec_mem = EnergyCounter(energy)
            x_m, counts_m, diag_m = run_imh(ec_mem, TiedMixture(mem_out, cov_t * inflate), x0, steps, burn, assign, K)
            share_m = counts_m.sum(0) / counts_m.sum()
            ratio_m = share_m / pi
            extra = dict(
                imh_mode_tv_mem=0.5 * (share_m - pi).abs().sum().item(), accept_mem=diag_m["acc"], ess_mem=chain_ess(counts_m, pi),
                max_log_w_mem=diag_m["max_log_w"], M_mem=len(mem_out), centres_new_in_memory=n_new_mem, unconverged_skipped=n_unconv,
                modes_with_centre_mem=torch.unique(assign(mem_out)).numel(),
                ratio_lightest_imh_mem=ratio_m[order[:q]].mean().item(), n_under_half_imh_mem=int((ratio_m < 0.5).sum()),
                n_E_mem=ec_mem.n_E, n_grad_mem=ec_mem.n_grad,
            )
    share_buf = torch.bincount(assign(x1), minlength=K).float() / len(x1)
    share = counts.sum(0) / counts.sum()
    share_final = torch.bincount(assign(x), minlength=K).float() / len(x)
    ratio_buf, ratio = share_buf / pi, share / pi
    tv = lambda s_: 0.5 * (s_ - pi).abs().sum().item()
    n_E_mem, n_grad_mem = extra.get("n_E_mem", 0), extra.get("n_grad_mem", 0)
    metrics = dict(
        buffer_mode_tv=tv(share_buf), imh_mode_tv=tv(share), imh_mode_tv_final=tv(share_final),
        accept=diag["acc"], ess=chain_ess(counts, pi), max_log_w=diag["max_log_w"],
        M_fit=M, M_cover=len(centres_cov), centres_added=n_added, outliers=n_out,
        modes_with_centre_raw=torch.unique(assign(mu)).numel(), modes_with_centre_cover=torch.unique(assign(centres_cov)).numel(),
        ratio_lightest_buffer=ratio_buf[order[:q]].mean().item(), ratio_lightest_imh=ratio[order[:q]].mean().item(),
        ratio_heaviest_buffer=ratio_buf[order[-q:]].mean().item(), ratio_heaviest_imh=ratio[order[-q:]].mean().item(),
        n_under_half_buffer=int((ratio_buf < 0.5).sum()), n_under_half_imh=int((ratio < 0.5).sum()),
        n_E_total=ec.n_E + ec_setup.n_E + n_E_mem, n_grad_total=ec.n_grad + ec_setup.n_grad + n_grad_mem, **extra,
    )
    return metrics, mem_out
