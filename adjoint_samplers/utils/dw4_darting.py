""" DW-4 minimum-to-minimum DARTING kernel, batched in torch (float64), for the Barker control variate. The numpy reference with the derivation and the offline tests is scripts/dw4_darting_test.py.

    For x in basin type a and a target type b != a (5 types: triangle + pendant, path of 2, two disjoint pairs, K4 minus an edge, star):
      1. m = quench(x) to ||grad|| < 1e-8 (deterministic function of x); the type a is the state of m.
      2. For every element g0 = (perm, flip) of the 48-element group such that m lies on the rotation orbit of g0(src_ab) (n_a elements = the stabiliser of the type), the angle theta_x is the PROCRUSTES angle of g0(src_ab) onto x itself
         (not onto m), g = (g0, theta_x), z = g^{-1} x, d = z - src_ab, d' = M_ab d, y = g(tgt_ab + d'). Jacobian of x -> y: |det M_ab| f_tgt(d') / f_src(d), f_mu(d) = |mu| + <d, mu> / |mu|.
      3. The move is valid only if quench(y) lies on the orbit of g0(tgt_ab) AND the Procrustes angle of y onto g0(tgt_ab) equals theta_x (and both quenches converge to ||grad|| < 1e-8); invalid moves have weight 0.
      4. Barker weight with the Hastings ratio, at the bridge x_t | x ~ N(kappa x, v I) in the zero-CoM subspace:
            log R = -E(y) + E(x) + log Jac + log(n_a / n_b) + (|x_t - kappa x|^2 - |x_t - kappa y|^2) / (2 v),    omega = sigmoid(log R)
         proposal probability 1 / ((K - 1) n_a) per (b, g0).   h(x) = sum_{b, g0} omega / ((K - 1) n_a) (y - x),   term = gamma h / nu_{1|t}.
    All that is heavy (quench, alignment, Jacobian) depends on x only: `proposals(x)` is computed ONCE per sample (at buffer insertion) and stored; `darting_h` only needs the Gaussian factor.
    Layout of the stored proposals: slot = b * SLOTS + rank, SLOTS = 6 (the largest stabiliser, the star), P = 5 * 6 = 30 slots (the slots of b = a, of non-matching elements and of invalid moves have weight 0).
"""
import itertools
import json
import math
from pathlib import Path

import torch

from adjoint_samplers.utils.dw4_quench import energy_grad_hess
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, state_ids

K_TYPES, SLOTS = 5, 6
P_SLOTS = K_TYPES * SLOTS
_PERMS = torch.tensor(list(itertools.permutations(range(4))))                                   # (24, 4)
PERM48 = _PERMS.repeat_interleave(2, 0)                                                         # (48, 4): element e = (perm e // 2, flip e % 2), the order of the numpy reference
FLIP48 = torch.arange(48) % 2
INV48 = torch.argsort(PERM48, dim=1)
_MODEL = {}


def model(dtype=torch.float64):
    if dtype not in _MODEL:
        d = json.loads(Path(__file__).with_name("dw4_darting_model.json").read_text())
        T = lambda k: torch.tensor([[d["pairs"][f"{a},{b}"][k] if a != b else d["pairs"]["0,1"][k] for b in range(K_TYPES)] for a in range(K_TYPES)], dtype=dtype)
        names = d["names"]
        id_of = {NAMES[k]: i for i, k in enumerate(KEYS)}
        type_of_id = torch.full((len(KEYS),), -1, dtype=torch.long)
        for t, nm in enumerate(names):
            type_of_id[id_of[nm]] = t
        _MODEL[dtype] = dict(src=T("src"), tgt=T("tgt"), M8=T("M8"), logdet=T("logdet"), n_tgt=T("n_tgt"), type_of_id=type_of_id, names=names)
    return _MODEL[dtype]


def recentre(x):
    return x - x.mean(-2, keepdim=True)


def rot(Y, theta):
    """ Y (..., 4, 2) rotated by theta (...) (counter-clockwise: Y @ R(theta)^T) """
    c, s = torch.cos(theta)[..., None], torch.sin(theta)[..., None]
    return torch.stack([c * Y[..., 0] - s * Y[..., 1], s * Y[..., 0] + c * Y[..., 1]], -1)


def elem_images(X):
    """ the 48 images (perm, flip) of X (..., 4, 2) -> (..., 48, 4, 2) (flip: y -> -y after the permutation) """
    Y = X[..., PERM48, :]
    sign = (1 - 2 * FLIP48).to(X.dtype)
    return torch.stack([Y[..., 0], Y[..., 1] * sign[:, None]], -1)


def fit_angle(Y, m):
    """ angle theta maximising <rot(Y, theta), m> """
    return torch.atan2((Y[..., 0] * m[..., 1] - Y[..., 1] * m[..., 0]).sum(-1), (Y * m).sum((-1, -2)))


def g_apply(e, theta, X):
    Yp = torch.gather(X, 1, PERM48[e][:, :, None].expand(-1, -1, 2))                              # X[n, perm_e[n], :]
    sign = (1 - 2 * FLIP48[e]).to(X.dtype)
    Yp = torch.stack([Yp[..., 0], Yp[..., 1] * sign[:, None]], -1)
    return rot(Yp, theta)


def g_inv_apply(e, theta, X):
    Y = rot(X, -theta)
    sign = (1 - 2 * FLIP48[e]).to(X.dtype)
    Y = torch.stack([Y[..., 0], Y[..., 1] * sign[:, None]], -1)
    return torch.gather(Y, 1, INV48[e][:, :, None].expand(-1, -1, 2))


def f_vol(mu, d):
    nm = mu.flatten(-2).norm(dim=-1)
    return nm + (d * mu).sum((-1, -2)) / nm


def energy(x8):
    return energy_grad_hess(x8)[0]


@torch.no_grad()
def quench_precise(x8, tol=1e-10, max_iter=200, lam_min=0.05, max_step=0.5):
    """ saddle-free Newton quench in float64 (the repo's `quench` returns float32): returns the minimised configurations (N, 8) and the final gradient norms """
    xb = x8.double().clone()
    for _ in range(max_iter):
        E, g, H = energy_grad_hess(xb)
        act = g.norm(dim=-1) >= tol
        if not act.any():
            break
        lam, V = torch.linalg.eigh(H[act])
        c = torch.einsum("nij,ni->nj", V, g[act])
        step = -torch.einsum("nij,nj->ni", V, c / lam.abs().clamp(min=lam_min))
        step = step * torch.clamp(max_step / step.norm(dim=-1, keepdim=True).clamp(min=1e-30), max=1.0)
        idx = act.nonzero().flatten()
        alpha = torch.ones(len(idx), dtype=xb.dtype)
        pending = torch.ones(len(idx), dtype=torch.bool)
        for _ in range(10):
            En = energy_grad_hess(xb[idx] + alpha[:, None] * step)[0]
            pending = ~(En <= E[idx] + 1e-12)
            if not pending.any():
                break
            alpha = torch.where(pending, alpha * 0.5, alpha)
        move = ~pending
        xb[idx[move]] = xb[idx[move]] + alpha[move, None] * step[move]
    return xb, energy_grad_hess(xb)[1].norm(dim=-1)


@torch.no_grad()
def proposals(x8, tol_orbit=1e-4, tol_angle=1e-8, gtol=1e-8):
    """ x8 (N, 8) (any translation) -> dict with y (N, P, 8), base (N, P) = -E(y) + E(x) + log Jac + log(n_a / n_b), weight (N, P) = valid / ((K - 1) n_a), type (N,) (-1: not among the 5 types or quench not converged: all weights 0) """
    mdl = model()
    N = len(x8)
    x = recentre(x8.double().reshape(N, 4, 2))
    q8, gq = quench_precise(x.reshape(N, 8))
    q = q8.reshape(N, 4, 2)
    a = mdl["type_of_id"][state_ids(q8)]
    a = torch.where(gq < gtol, a, torch.full_like(a, -1))
    Ex = energy(x.reshape(N, 8))
    y_out = torch.zeros(N, P_SLOTS, 8, dtype=torch.float64)
    base_out = torch.zeros(N, P_SLOTS, dtype=torch.float64)
    w_out = torch.zeros(N, P_SLOTS, dtype=torch.float64)
    for b in range(K_TYPES):
        idx = ((a >= 0) & (a != b)).nonzero().flatten()
        if len(idx) == 0:
            continue
        ai = a[idx]
        src, tgt, M8 = mdl["src"][ai, b], mdl["tgt"][ai, b], mdl["M8"][ai, b]
        logdet, n_b = mdl["logdet"][ai, b], mdl["n_tgt"][ai, b]
        Ysrc = elem_images(src)                                                                  # (n, 48, 4, 2)
        th_m = fit_angle(Ysrc, q[idx][:, None])
        match = (rot(Ysrc, th_m) - q[idx][:, None]).abs().amax((-1, -2)) < tol_orbit             # (n, 48): m on the orbit of g0(src)
        n_a = match.sum(1).double()
        j, e = match.nonzero(as_tuple=True)
        rank = (torch.cumsum(match.long(), 1) - 1)[j, e]
        assert int(rank.max()) < SLOTS, "more matching elements than slots"
        xs, ss, ts, Ms = x[idx][j], src[j], tgt[j], M8[j]
        th_x = fit_angle(Ysrc[j, e], xs)                                                         # Procrustes angle of g0(src) onto x itself
        z = recentre(g_inv_apply(e, th_x, xs))
        d8 = z - ss
        dp8 = (Ms @ d8.reshape(-1, 8, 1)).reshape(-1, 4, 2)
        y = recentre(g_apply(e, th_x, ts + dp8))
        logjac = logdet[j] + torch.log(f_vol(ts, dp8)) - torch.log(f_vol(ss, d8))
        qy8, gy = quench_precise(y.reshape(-1, 8))
        qy = qy8.reshape(-1, 4, 2)
        Yt = elem_images(ts)[torch.arange(len(e)), e]                                            # g0(tgt)
        th_q = fit_angle(Yt, qy)
        on_orbit = (rot(Yt, th_q) - qy).abs().amax((-1, -2)) < tol_orbit
        th_y = fit_angle(Yt, y)
        same_angle = torch.atan2(torch.sin(th_y - th_x), torch.cos(th_y - th_x)).abs() < tol_angle
        valid = on_orbit & same_angle & (gy < gtol)
        Ey = energy(y.reshape(-1, 8))
        base = -Ey + Ex[idx][j] + logjac + torch.log(n_a[j] / n_b[j])
        slot = b * SLOTS + rank
        gi = idx[j]
        y_out[gi, slot] = y.reshape(-1, 8)
        base_out[gi, slot] = base
        w_out[gi, slot] = valid.double() / ((K_TYPES - 1) * n_a[j])
    return dict(y=y_out, base=base_out, weight=w_out, type=a)


def darting_h(x1, xt, kappa, v, y, base, weight):
    """ h(x1 | x_t) = sum_p weight_p omega_p (y_p - x1), omega_p = sigmoid(base_p + (|x_t - kappa x1|^2 - |x_t - kappa y_p|^2) / (2 v)); x1, xt (N, 8) (xt already minus (1 - kappa) x0), kappa and v (N, 1);
        returns h (N, 8) and omega (N, P) """
    x1, xt, y, base, weight = (t.double() for t in (x1, xt, y, base, weight))
    kappa, v = kappa.double(), v.double().clamp(min=1e-12)
    g = ((xt - kappa * x1) ** 2).sum(-1, keepdim=True) - ((xt[:, None, :] - kappa[:, None, :] * y) ** 2).sum(-1)
    omega = torch.sigmoid(base + g / (2 * v))
    return ((weight * omega)[..., None] * (y - x1[:, None, :])).sum(1), omega
