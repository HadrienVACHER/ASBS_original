""" Converged quench and basin assignment for the MoS / MoS-w targets (rules_mosw_step3.md): descent on the energy -log pi with a MONOTONE backtracking step, explicit convergence criterion
    ||grad log pi||_2 < TOL, basin = polished minimum within MERGE of the converged point. Used for the quality criterion Q (Q1: distance to the nearest minimum, Q2: basins reached) and for the basin proportions. """
import math
import numpy as np
from scipy.special import logsumexp

TOL = 1e-3
MERGE = 0.5
MAXIT = 3000
NU = 2.0


def _grad_and_logp(pi, X):
    """ log pi(x), grad log pi(x) and the index of the component of largest responsibility (pi: dart_lib.Mix 'S'). """
    lc = pi.logw + pi.comp(X)
    lp = logsumexp(lc, axis=1)
    r = np.exp(lc - lp[:, None])
    g = np.zeros_like(X)
    for k in range(pi.K):
        u = X - pi.mu[k]
        g += r[:, k:k + 1] * (-(NU + 1) * u / (NU + u * u))
    return lp, g, lc.argmax(1)


def quench(pi, X, maxit=MAXIT, tol=TOL):
    """ returns (Y converged points [n, d], converged mask [n], n iterations). Direction P grad log pi with P = diag((NU + u^2)/(NU + 1)), u = x - mu of the most responsible component (positive diagonal: a descent direction of -log pi);
        each sample has its own step scale s_i (starts at 0.7): a move is accepted only if log pi increases, else s_i is halved (monotone); after an accepted move s_i = min(0.7, 1.5 s_i). """
    Y = X.astype(np.float64).copy(); n = len(Y)
    s = np.full(n, 0.7); active = np.arange(n); conv = np.zeros(n, bool)
    lp, g, ks = _grad_and_logp(pi, Y)
    it = 0
    for it in range(maxit):
        gn = np.linalg.norm(g, axis=1)
        done = gn < tol
        conv[active[done]] = True
        keep = ~done
        if not keep.any():
            break
        active = active[keep]; lp, g, ks, s = lp[keep], g[keep], ks[keep], s[keep]
        Ya = Y[active]
        u = Ya - pi.mu[ks]
        P = (NU + u * u) / (NU + 1)
        Yn = Ya + s[:, None] * P * g
        lpn, gn_, ksn = _grad_and_logp(pi, Yn)
        ok = lpn > lp
        Y[active[ok]] = Yn[ok]
        lp = np.where(ok, lpn, lp); g = np.where(ok[:, None], gn_, g); ks = np.where(ok, ksn, ks)
        s = np.where(ok, np.minimum(0.7, 1.5 * s), 0.5 * s)
        stuck = s < 1e-12                                                                 # no progress possible: left unconverged
        if stuck.any():
            keep = ~stuck
            active = active[keep]; lp, g, ks, s = lp[keep], g[keep], ks[keep], s[keep]
            if len(active) == 0:
                break
    else:
        pass
    return Y, conv, it + 1


def polish_minima(pi):
    """ the K minima of the energy, from the means (quench of the means, converged, within MERGE of the mean). """
    Y, conv, _ = quench(pi, pi.mu.copy(), maxit=5000, tol=1e-8)
    assert conv.all() and (np.linalg.norm(Y - pi.mu, axis=1) < MERGE).all(), "minima not polished"
    return Y


def basins(pi, X, minima, tol=TOL):
    """ basin index of each sample (-1 if the quench did not converge or ended farther than MERGE from every minimum); returns (basin [n], fraction converged, fraction assigned). """
    Y, conv, _ = quench(pi, X, tol=tol)
    d = np.linalg.norm(Y[:, None, :] - minima[None], axis=2)
    b = d.argmin(1); near = d[np.arange(len(Y)), b] < MERGE
    ok = conv & near
    return np.where(ok, b, -1), float(conv.mean()), float(ok.mean())


def basin_props(b, K, w=None):
    """ proportions among the assigned samples (b >= 0), weighted if w is given. """
    m = b >= 0
    if m.sum() == 0:
        return np.full(K, float("nan"))
    ww = None if w is None else w[m]
    return np.bincount(b[m], weights=ww, minlength=K) / (m.sum() if ww is None else ww.sum())


def dist_to_nearest_minimum(X, minima):
    return np.sqrt(np.maximum(((X ** 2).sum(1)[:, None] - 2 * X @ minima.T + (minima ** 2).sum(1)[None]), 0)).min(1)


def sample_exact(pi, n, rng):
    """ exact draws from the MoS(-w): component ~ weights, then mu_k + z / sqrt(Gamma(NU/2, rate NU/2)) per coordinate (product of t_NU of scale 1). """
    pw = np.asarray(pi.w, float); comp = rng.choice(pi.K, size=n, p=pw / pw.sum())   # weights come from float32 tensors: renormalise in float64
    z = rng.standard_normal((n, pi.d)); chi = rng.gamma(NU / 2, 2.0 / NU, size=(n, pi.d))
    return pi.mu[comp] + z / np.sqrt(chi)
