""" Girsanov path-weight importance sampling for a trained Adjoint Sampling controller.

    The sampler is the Euler-Maruyama chain X_{k+1} = X_k + m_k + sqrt(v_k) eps_k, m_k = g(t_k)^2 u(t_k, X_k) dt_k, v_k = g(t_k)^2 dt_k, from X_0 (a Dirac).
    The reference chain (u = 0) has Gaussian increments N(0, v_k), hence the terminal law N(X_0, V I), V = Σ_k v_k. With the terminal tilt
    g(x) = E(x) + log N(x; X_0, V I), the path measure P* ∝ P^ref e^{-g(X_1)} has terminal marginal ∝ e^{-E} (memoryless), so

        w(path) = dP*/dP^u = e^{-g(X_1)} dP^ref/dP^u / Z,   Z = ∫ e^{-E},
        log dP^u/dP^ref = Σ_k [ |m_k|^2 / (2 v_k) + <eps_k, m_k> / sqrt(v_k) ]   (for the paths sampled from P^u),

    and E_{P^u}[e^{log w~}] = Z for ANY controller u (the importance weights of the discretised chain are exact): log Z_hat = logmeanexp(log w~), and the
    ESS of the normalised weights measures how far the model is from the target (in path space).
"""
import math

import torch


@torch.no_grad()
def log_path_weights(sde, energy, x0, timesteps):
    """ unnormalised log weights log w~ (N,) of N paths sampled from the controlled chain, and their endpoints """
    N, D = x0.shape
    sde.train(False)
    x = x0.clone()
    log_ratio = torch.zeros(N, device=x0.device, dtype=x0.dtype)
    V = 0.0
    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        dt = timesteps[i + 1] - t
        v = sde.diff(t) ** 2 * dt
        m = sde.drift(t, x) * dt
        eps = sde.randn_like(x)
        log_ratio += 0.5 * (m ** 2).sum(-1) / v + (eps * m).sum(-1) / v.sqrt()
        x = sde.propagate(x, m + v.sqrt() * eps)
        V += float(v)
    log_ref_end = -0.5 * ((x - x0) ** 2).sum(-1) / V - 0.5 * D * math.log(2 * math.pi * V)
    return - log_ratio - energy.eval(x) - log_ref_end, x


def ess_and_logz(log_w, log_z_exact=None):
    """ ESS (absolute and fraction) of the normalised weights, log Z_hat and |log Z_hat - log Z| """
    lw = log_w.double()
    w = torch.softmax(lw, 0)
    ess = float(1.0 / (w ** 2).sum())
    log_zhat = float(torch.logsumexp(lw, 0) - math.log(len(lw)))
    out = {"ess": ess, "ess_frac": ess / len(lw), "log_zhat": log_zhat}
    if log_z_exact is not None:
        out["log_z_err"] = abs(log_zhat - log_z_exact)
    return out
