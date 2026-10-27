""" Division (Gelfand-Dey) control variate on a GMM, building blocks (adjoint_samplers/utils/division_cv.py). No matcher, no training.

    1. exact identity by 2D quadrature: ∫ pi(x1 | x_t) r_k dx1 p_t(x_t) = 1 for every mode k, hence E_pi[term | x_t] = 0 at fixed x_t;
    2. Monte Carlo under the training law (x1 ~ p, x_t ~ N(kappa x1, v I)): S_psi(t) = mean <term, psi(x_t)> compatible with 0 (psi1 = x_t, psi2 = x_t |x_t|^2), and a positive control
       (variances inside C_k multiplied by 1.5) that must be detected;
    3. boundedness of h r_k against r_k alone (second moment as the sample size grows);
    4. detection of mode-weight errors with an EXACT conditional mean: buffer q = same components with other weights; for x_t ~ q, E_q[term | x_t] (exact posterior sampling of x1 | x_t under q) against the exact
       correction c*(x_t) = (E_p[x1 | x_t] - E_q[x1 | x_t]) / nu_{1|t}: alignment (cosine, correlation) per kappa.

    python tests/test_division_cv.py
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils.division_cv import division_term, log_r  # noqa: E402

torch.set_default_dtype(torch.float64)
D, K = 2, 3
MU = torch.tensor([[3.0, 0.0], [-1.5, 2.6], [-1.5, -2.6]])            # regular triangle, radius 3, unit covariance modes
W_TRUE = torch.tensor([0.5, 0.3, 0.2])
W_BUF = torch.tensor([1 / 3, 1 / 3, 1 / 3])
S = torch.full((K,), 0.8)                                               # N_k variance (< 2 sigma^2 = 2)
NU1 = 16.0


def kv(kappa):
    return kappa, NU1 * kappa * (1 - kappa)


def logp(x, w):
    return torch.logsumexp(torch.log(w)[None] - 0.5 * ((x[:, None, :] - MU[None]) ** 2).sum(-1) - 0.5 * D * math.log(2 * math.pi), dim=1)


def sample(n, w, gen):
    k = torch.multinomial(w, n, replacement=True, generator=gen)
    return MU[k] + torch.randn(n, D, generator=gen)


def test_exact_identity(mu=None, s=None, tag="oracle modes"):
    mu = MU if mu is None else mu
    s = S if s is None else s
    g = torch.linspace(-11, 11, 441)
    X1 = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    dx2 = float((g[1] - g[0]) ** 2)
    lp = logp(X1, W_TRUE)
    worst, worst_term = 0.0, 0.0
    for kappa in (0.05, 0.2, 0.5, 0.8):
        kap, v = kv(kappa)
        for xt in (torch.tensor([[1.0, 0.5]]) * kappa, torch.tensor([[-2.0, 1.0]]) * kappa, torch.tensor([[0.3, -3.0]]) * kappa):
            XT = xt.expand(len(X1), 2)
            lr, logC = log_r(XT, X1, lp, kap, v, mu, s)
            logpost = lp - 0.5 * ((XT - kap * X1) ** 2).sum(-1) / v                                       # log p(x1) N(x_t; kappa x1, v) up to a constant
            post = torch.exp(logpost - logpost.max())
            Z = float((post * dx2).sum())
            Ik = (post[:, None] * torch.exp(lr - lr.max(0).values[None]) * torch.exp(lr.max(0).values)[None] * dx2).sum(0) / Z   # E_pi[r_k | x_t]
            ratio = Ik / Ik[0]
            worst = max(worst, float((ratio - 1).abs().max()))
            term, _ = division_term(XT, X1, lp, kap, v, mu, s)
            Eterm = (post[:, None] * term * dx2).sum(0) / Z
            worst_term = max(worst_term, float(Eterm.abs().max()) / float(term.abs().max()))
    assert worst < 1e-6 and worst_term < 1e-6, (worst, worst_term)
    print(f"ok  1. [{tag}] quadrature: E_pi[r_k | x_t] is the same for the 3 modes (max relative spread {worst:.1e}) and E_pi[term | x_t] = 0 (relative {worst_term:.1e}) at 12 (kappa, x_t) points")


def psis(xt):
    return [xt, xt * (xt ** 2).sum(-1, keepdim=True)]


def test_monte_carlo(n=400000):
    gen = torch.Generator().manual_seed(0)
    x1 = sample(n, W_TRUE, gen)
    lp = logp(x1, W_TRUE)
    out = []
    for kappa in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        kap, v = kv(kappa)
        xt = kap * x1 + math.sqrt(v) * torch.randn(n, D, generator=gen)
        res = []
        for s_c in (None, S * 1.5):
            term, _ = division_term(xt, x1, lp, kap, v, MU, S, s_c=s_c)
            res.append(max(abs(float((term * p).sum(-1).mean() / ((term * p).sum(-1).std() / n ** 0.5))) for p in psis(xt)))
        out.append((kappa, res[0], res[1]))
    zmax, zctl = max(o[1] for o in out), max(o[2] for o in out)
    assert zmax < 3.5 and zctl > 5, (out,)
    print("ok  2. Monte Carlo (400k): max |z| of S_psi over kappa = " + ", ".join(f"{k:g}:{a:.1f}" for k, a, _ in out) + f" ; positive control (C_k variance x1.5): max |z| {zctl:.0f}")


def test_boundedness():
    gen = torch.Generator().manual_seed(1)
    kap, v = kv(0.5)
    rows = []
    for n in (10000, 100000, 1000000):
        x1 = sample(n, W_TRUE, gen)
        xt = kap * x1 + math.sqrt(v) * torch.randn(n, D, generator=gen)
        lr, logC = log_r(xt, x1, logp(x1, W_TRUE), kap, v, MU, S)
        h = torch.exp(-torch.logsumexp(-logC, dim=-1, keepdim=True))
        rows.append((n, float(torch.exp(lr).pow(2).mean()), float((h * torch.exp(lr)).pow(2).mean()), float((h * torch.exp(lr)).max())))
    assert rows[-1][3] < 50 and rows[-1][2] < 10 * rows[0][2]
    print("ok  3. boundedness at kappa 0.5 (n = 1e4 / 1e5 / 1e6): E[r^2] = " + " / ".join(f"{r[1]:.2f}" for r in rows) + " ; E[(h r)^2] = " + " / ".join(f"{r[2]:.2f}" for r in rows)
          + f" ; max h r = {rows[-1][3]:.1f}")


def posterior_components(xt, w, kappa, v):
    """ component responsibilities (B, K) and posterior means (B, K, d) of x1 | x_t under the mixture with weights w; posterior variance of every component """
    var_post = 1.0 / (1.0 + kappa ** 2 / v)
    mean_post = var_post * (MU[None] + kappa * xt[:, None, :] / v)
    logr = torch.log(w)[None] - 0.5 * ((xt[:, None, :] - kappa * MU[None]) ** 2).sum(-1) / (v + kappa ** 2) - 0.5 * D * math.log(2 * math.pi * (v + kappa ** 2))
    return torch.softmax(logr, dim=-1), mean_post, var_post


def test_detection_and_alignment(n_xt=300, n_post=3000):
    gen = torch.Generator().manual_seed(2)
    rows = []
    for kappa in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
        kap, v = kv(kappa)
        nu1t = NU1 * (1 - kappa)
        x1q = sample(n_xt, W_BUF, gen)
        xt = kap * x1q + math.sqrt(v) * torch.randn(n_xt, D, generator=gen)
        rq, mp, vp = posterior_components(xt, W_BUF, kap, v)
        rp, _, _ = posterior_components(xt, W_TRUE, kap, v)
        cstar = ((rp[..., None] * mp).sum(1) - (rq[..., None] * mp).sum(1)) / nu1t                         # (E_p[x1 | x_t] - E_q[x1 | x_t]) / nu_{1|t}
        m = torch.zeros(n_xt, D)
        for i in range(n_xt):
            comp = torch.multinomial(rq[i], n_post, replacement=True, generator=gen)
            x1 = mp[i, comp] + math.sqrt(vp) * torch.randn(n_post, D, generator=gen)
            term, _ = division_term(xt[i].expand(n_post, D), x1, logp(x1, W_TRUE), kap, v, MU, S)
            m[i] = term.mean(0)
        cos = float((m * cstar).sum() / (m.norm() * cstar.norm()))
        corr = float(((m - m.mean(0)) * (cstar - cstar.mean(0))).sum() / ((m - m.mean(0)).norm() * (cstar - cstar.mean(0)).norm()))
        beta = float((m * cstar).sum() / (cstar * cstar).sum())
        rows.append((kappa, float(cstar.pow(2).sum(-1).mean().sqrt()), float(m.pow(2).sum(-1).mean().sqrt()), cos, beta))
    print("    kappa  rms|c*|  rms|E_q[term]|  cosine(E_q[term], c*)  beta (unit scale)")
    for k, a, b, c, be in rows:
        print(f"    {k:5.2f}  {a:7.4f}  {b:13.4f}  {c:+19.3f}  {be:+16.4f}")
    assert all(r[3] > 0.3 for r in rows if 0.1 <= r[0] <= 0.7), rows
    print("ok  4. E_q[term | x_t] (exact conditional sampling) is positively aligned with the exact correction c*(x_t) for kappa in [0.1, 0.7]")


if __name__ == "__main__":
    test_exact_identity()
    g_ = torch.Generator().manual_seed(7)                                  # any fitted (mu_k, s_k): the zero mean does not depend on the fit
    test_exact_identity(MU + 0.7 * torch.randn(K, D, generator=g_), torch.tensor([0.3, 1.1, 1.8]), tag="random (mu_k, s_k)")
    test_monte_carlo()
    test_boundedness()
    test_detection_and_alignment()
    print("all passed")
