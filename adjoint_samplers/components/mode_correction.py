""" Mode-weight correction of an Adjoint Sampling buffer: independent MH, self-normalised IS, and a
    mode-jump + MALA sampler, all driven by an equal-weight tied-covariance Gaussian mixture r fitted on the
    buffer (centres optionally refined by descent on E).

    Target p = e^{-E} / Z. Energy cost is counted through `EnergyCounter` so that methods compare at equal budget.
"""
import math

import torch


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
