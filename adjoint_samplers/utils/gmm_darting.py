""" Minimum-to-minimum DARTING control variate for isotropic GMM targets in dimension d (GMM-dk study), the d-dimensional counterpart of `gmm40_darting.darting_cv` and of the DW-4 kernel
    (`dw4_darting.py`, same convention: term = gamma * h / nu_{1|t}, h = sum_j omega_j (y_j - x1) / (M - 1), Barker weights, zero mean under pi(. | x_t) for ANY fixed set of minima and maps).

    Conventions (AS, VE bridge from a Dirac at X_0 = 0): x_t | x_1 ~ N(kappa_t x_1, v_t I), kappa_t = nu_t / nu_1, v_t = nu_1 kappa_t (1 - kappa_t) = nu_t (1 - kappa_t), nu_{1|t} = nu_1 - nu_t.
    From x1 in cell k (nearest minimum m_k) the proposal towards every other minimum j is  y_j = m_j + A_jk (x1 - m_k),  A_jk = H_j^{-1/2} H_k^{1/2}  (an involution: A_kj = A_jk^{-1}), |J_jk| = sqrt(det H_k / det H_j);
    it is VALID only if y_j lands in cell j (the reverse move then exists), else omega_j = 0;  omega_j = sigmoid(log pi~(y_j) + log|J| - log pi~(x1)),  pi~(z) = pi(z) N(x_t; kappa z, v I).
    Minima and Hessians come from the buffer only (no oracle): quench by gradient descent of the energy from buffer samples, greedy merge, minima reached by >= min_count samples, Hessians by autograd. """
import math

import torch


def quench(x, energy_fn, steps=400, lr=0.7, tol=1e-3):
    """ gradient descent on the energy from x [n, d] (step lr, adapted to unit-variance Gaussian modes: Hessian ~ I) until >= 95 % of the samples have |grad| < tol. """
    y = x.detach().clone()
    for it in range(steps):
        y.requires_grad_(True)
        g = torch.autograd.grad(energy_fn(y).sum(), y)[0]
        y = (y - lr * g).detach()
        if it >= 20 and it % 20 == 0 and bool((g.norm(dim=1) < tol).double().mean() >= 0.95):
            break
    return y


def discover_minima(x, energy_fn, n_samples=500, merge_tol=0.5, min_count=5, refine=50):
    """ minima [M, d] and Hessians [M, d, d] (float64) discovered from the samples x (the first n_samples): quench, greedy merge at merge_tol, keep minima reached by >= min_count samples. """
    y = quench(x[:n_samples].double(), energy_fn)
    mins, cnt = [], []
    for p in y:
        if mins:
            dist = torch.stack(mins).sub(p).norm(dim=1)
            i = int(dist.argmin())
            if float(dist[i]) < merge_tol:
                cnt[i] += 1
                continue
        mins.append(p.clone()); cnt.append(1)
    keep = [m for m, c in zip(mins, cnt) if c >= min_count]
    if not keep:
        return None, None
    M = torch.stack(keep)
    for _ in range(refine):                                                            # refinement of the kept minima
        M.requires_grad_(True)
        g = torch.autograd.grad(energy_fn(M).sum(), M)[0]
        M = (M - 0.7 * g).detach()
    Hs = torch.stack([torch.autograd.functional.hessian(lambda z: energy_fn(z[None])[0], m) for m in M])
    return M, 0.5 * (Hs + Hs.transpose(-1, -2))


def make_maps(minima, H):
    """ maps A[j, k] = H_j^{-1/2} H_k^{1/2} [M, M, d, d] and log|J|[j, k] = 0.5 (logdet H_k - logdet H_j). """
    ev, V = torch.linalg.eigh(H)
    assert bool((ev > 0).all()), "Hessian not positive definite at a discovered minimum"
    Hh = (V * ev.sqrt()[:, None, :]) @ V.transpose(-1, -2)                             # H^{1/2}  [M, d, d]
    Hmh = (V / ev.sqrt()[:, None, :]) @ V.transpose(-1, -2)                            # H^{-1/2}
    A = Hmh[:, None] @ Hh[None, :]                                                     # [M(j), M(k), d, d]
    logdet = ev.log().sum(-1)
    logJ = 0.5 * (logdet[None, :] - logdet[:, None])                                   # [j, k]
    return dict(minima=minima, A=A, logJ=logJ)


def darting_term(x1, xt, kappa, v, nu_cond, maps, energy_fn, v_scale=1.0):
    """ term c [n, d] (division by nu_{1|t} included, gamma NOT included) and the Barker weights omega [n, M].
        x1, xt [n, d]; kappa, v, nu_cond [n] (float64 recommended). v_scale multiplies v inside the Gaussian factor ONLY (test hook, 1.0 in all runs). """
    m, A, logJ = maps["minima"], maps["A"], maps["logJ"]
    n, d = x1.shape
    M = m.shape[0]
    k = torch.cdist(x1, m).argmin(-1)                                                  # [n]
    y = torch.empty(n, M, d, dtype=x1.dtype, device=x1.device)
    for k0 in range(M):
        idx = (k == k0).nonzero(as_tuple=True)[0]
        if len(idx):
            y[idx] = m[None] + torch.einsum("jde,ne->njd", A[:, k0], x1[idx] - m[k0])
    jj = torch.arange(M, device=x1.device)[None]
    valid = (torch.cdist(y.reshape(-1, d), m).argmin(-1).reshape(n, M) == jj) & (jj != k[:, None])
    vv = (v * v_scale)

    def log_post(z):                                                                   # z [n, d] or [n, M, d]
        sh = (n,) + (1,) * (z.dim() - 1)
        diff = xt.reshape((n,) + (1,) * (z.dim() - 2) + (d,)) - kappa.reshape(sh) * z
        return -energy_fn(z) - 0.5 * diff.pow(2).sum(-1) / vv.reshape((n,) + (1,) * (z.dim() - 2))

    lx = log_post(x1)                                                                  # [n]
    ly = log_post(y) + logJ[:, k].T                                                    # [n, M]
    omega = torch.sigmoid(ly - lx[:, None]) * valid
    disp = (omega[..., None] * (y - x1[:, None])).sum(1) / max(M - 1, 1)
    return disp / nu_cond[:, None], omega
