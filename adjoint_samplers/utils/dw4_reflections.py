""" Symmetry-compatible Barker kernel for DW-4 (4 particles in 2D, pairwise double well E = Σ_{a<b} e(d_ab)): no clustering, no centres.

    Maps T_{i;jk} (12 choices of the particle i and of the pair {j, k} among the three others; l is the remaining particle):
        line reflection   : reflect particle i across the line through particles j and k, the others unchanged. d_ij and d_ik are preserved, only d_il changes, so
                            E(Tx) - E(x) = e(d'_il) - e(d_il);
        point reflection  : x_i -> x_j + x_k - x_i (DEFAULT together with the line reflections: R = 24), which swaps d_ij and d_ik and changes d_il: the same closed form for the energy change.
    Everything lives in the zero-CoM subspace S (dim 6): the map is applied and the result is recentred. T commutes with translations, so the recentred map is still an
    involution (T_r ∘ T_r = id) and |det J| = 1 on S (the full Jacobian is block-triangular with a reflection block, det = ±1, and a translation-commuting map decomposes into
    (s, c) -> (f(s), c + m(s))).

    Barker control variate (for the later integration into the matcher):
        h = (1/R) Σ_r ω_r (T_r x1 - x1),  ω_r = π~(T_r x1 | x_t) / (π~(x1 | x_t) + π~(T_r x1 | x_t)),  π~(y | x_t) ∝ exp(-E(y)) N(x_t; κ_t y, v_t),  R = 24 (12 with variant=False, line reflections only; the point reflections connect states that the line ones do not, e.g. path-3 -> two disjoint pairs).
    Because every T_r is an involution preserving the volume of S and R does not depend on the position, E_π[h | x_t] = 0 exactly (detailed balance of each r).
"""
import itertools

import torch

A, B, O = 0.9, -4.0, 4.0                       # e(d) = A (d - O)^4 + B (d - O)^2  (c = 0), the benchmark's DoubleWellEnergy
TRIPLES = [(i, j, k) for i in range(4) for (j, k) in itertools.combinations([m for m in range(4) if m != i], 2)]   # 12 choices of (i; j, k)


def pair_energy(d):
    s = d - O
    return A * s ** 4 + B * s ** 2


def _dist(a, b):
    return torch.sqrt(((a - b) ** 2).sum(-1) + 1e-8)       # same regularisation as the benchmark's compute_distances


def recentre(z):
    return z - z.mean(-2, keepdim=True)


def energy(x):
    """ DW-4 energy of (N, 8) configurations (same as DoubleWellEnergy.eval) """
    z = x.reshape(-1, 4, 2)
    return sum(pair_energy(_dist(z[:, a], z[:, b])) for a, b in itertools.combinations(range(4), 2))


def reflect(z, i, j, k, point=False):
    """ image of (N, 4, 2) configurations under T_{i;jk} (not recentred) """
    out = z.clone()
    if point:
        out[:, i] = z[:, j] + z[:, k] - z[:, i]
    else:
        u = z[:, k] - z[:, j]
        u = u / torch.sqrt((u ** 2).sum(-1, keepdim=True) + 1e-30)
        w = z[:, i] - z[:, j]
        out[:, i] = z[:, j] + 2 * (w * u).sum(-1, keepdim=True) * u - w
    return out


def images(x, variant=True):
    """ all images T_r x (recentred), (N, R, 8), and the energy changes E(T_r x) - E(x) from the single changed distance d_il, (N, R) """
    z = recentre(x.reshape(-1, 4, 2))
    imgs, dE = [], []
    for point in ((False, True) if variant else (False,)):
        for (i, j, k) in TRIPLES:
            l = ({0, 1, 2, 3} - {i, j, k}).pop()
            y = reflect(z, i, j, k, point)
            imgs.append(recentre(y).reshape(len(z), 8))
            dE.append(pair_energy(_dist(y[:, i], z[:, l])) - pair_energy(_dist(z[:, i], z[:, l])))
    return torch.stack(imgs, 1), torch.stack(dE, 1)


def barker_logit(x1, xt, kappa, v, variant=True):
    """ log π~(T_r x1 | x_t) - log π~(x1 | x_t) = -(E(Tx) - E(x)) - (|x_t - κ Tx|^2 - |x_t - κ x|^2) / (2 v), (N, R), with the recentred x1 and its images (N, R, 8) """
    z = recentre(x1.reshape(-1, 4, 2)).reshape(-1, 8)
    imgs, dE = images(z, variant)
    k, vv = torch.as_tensor(kappa, dtype=z.dtype, device=z.device), torch.as_tensor(v, dtype=z.dtype, device=z.device)
    k3, v2 = (k.reshape(-1, 1, 1), vv.reshape(-1, 1)) if k.dim() > 0 else (k, vv)         # per-sample kappa_t, v_t (N, 1) or scalars
    k2 = k.reshape(-1, 1) if k.dim() > 0 else k
    # |x_t - κ y|^2 - |x_t - κ x|^2 = κ^2 |Δ|^2 - 2 κ <Δ, r>, Δ = y - x, r = x_t - κ x: no cancellation between two large numbers (float32)
    delta = imgs - z[:, None, :]
    r = xt - k2 * z
    kk = k3.reshape(-1, 1) if k3.dim() > 0 else k3
    gauss = -(kk ** 2 * (delta ** 2).sum(-1) - 2 * kk * (delta * r[:, None, :]).sum(-1)) / (2 * v2)
    return -dE + gauss, z, imgs


def barker_h(x1, xt, kappa, v, variant=True):
    """ h (N, 8) and the weights ω = sigmoid(logit) (N, R); x1, xt (N, 8), kappa and v the posterior parameters (x_t | x_1 ~ N(κ x_1, v) in the zero-CoM subspace) """
    logit, z, imgs = barker_logit(x1, xt, kappa, v, variant)
    omega = torch.sigmoid(logit)
    h = (omega[:, :, None] * (imgs - z[:, None, :])).mean(1)
    return h, omega


def kernel_step(x, variant=True, generator=None):
    """ t = 0 kernel (no Gaussian factor): r uniform, move to T_r x with probability π(T_r x) / (π(x) + π(T_r x)). Returns the new configurations and the chosen r """
    imgs, dE = images(x, variant)
    R = imgs.shape[1]
    r = torch.randint(R, (len(x),), generator=generator)
    idx = torch.arange(len(x))
    accept = torch.rand(len(x), generator=generator) < torch.sigmoid(-dE[idx, r])
    new = torch.where(accept[:, None], imgs[idx, r], recentre(x.reshape(-1, 4, 2)).reshape(-1, 8))
    return new, r


def rb_terms(x, label, variant=True, n_classes=None):
    """ Rao-Blackwellised, paired one-step statistics of the t = 0 kernel (no sampling of r, no accept draw), for a labelling function `label` (configurations (M, 8) -> integer class ids):

            D[n, b] = (1/R) Σ_r w_r(x_n) [1{label(T_r x_n) = b} - 1{label(x_n) = b}],   w_r = π(T_r x) / (π(x) + π(T_r x))  (energy only, κ = 0),
            Fl[n, a, b] = 1{label(x_n) = a} (1/R) Σ_r w_r(x_n) 1{label(T_r x_n) = b}                                         (flux, a = b included)

        Returns D (N, K), Fl (N, K, K), the labels of x (N,) and the number of classes K. The mean of D over samples is the expected one-step change of the class frequencies. """
    imgs, dE = images(x, variant)
    N, Rn, _ = imgs.shape
    w = torch.sigmoid(-dE)
    lab_x = label(recentre(x.reshape(-1, 4, 2)).reshape(-1, 8))
    lab_i = label(imgs.reshape(-1, 8)).reshape(N, Rn)
    K = n_classes or int(max(lab_x.max(), lab_i.max())) + 1
    oh_i = torch.nn.functional.one_hot(lab_i, K).to(w.dtype)                # (N, R, K)
    oh_x = torch.nn.functional.one_hot(lab_x, K).to(w.dtype)                # (N, K)
    to = (w[:, :, None] * oh_i).mean(1)                                      # (N, K): (1/R) Σ_r w_r 1{label(T_r x) = b}
    D = to - w.mean(1)[:, None] * oh_x
    Fl = oh_x[:, :, None] * to[:, None, :]
    return D, Fl, lab_x, K
