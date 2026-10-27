""" Division control variate, new carrier (adjoint_samplers/utils/division_cv.py): STEP 1 unit tests. No training.

    PRE-REGISTERED THRESHOLDS (fixed in this header before any of these tests was run):
      (a) zero mean. Quadrature (2D, K = 3, random frozen parameters: random mu_k, SPD Sigma_k, L_k, shrink): |E_pi[term | x_t]| / max|term| < 1e-6 at every (kappa, x_t);
          Monte Carlo under the training law (x1 ~ p, x_t ~ N(kappa x1, v I)): max |z| < 3.5 of S_psi = mean <term, psi(x_t)> (psi1 = x_t, psi2 = x_t |x_t|^2) over
          kappa in {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9}; positive control (variance inside C_k x 2.5): max |z| > 5.
          AMENDED ONCE after the first run (positive control only, zero-mean thresholds unchanged): the first control (x 1.5, parameters drawn far from the truth) was not detected because the term is
          nearly zero for overlapping random modes; the Monte Carlo test now uses parameters drawn around the truth (mu +- 0.5, Sigma in [1, 1.3] I, L +- 0.5) and a x 2.5 control.
      (b) oracle (true mu_k, Sigma_k = I, L_k = -log w_k), separation >= 6 sd, buffer with wrong weights, exact posterior sampling of x1 | x_t under the buffer:
          cos(E_q[term | x_t], c*(x_t)) >= 0.95 and scale <m, c*> / <c*, c*> in [0.8, 1.25], for kappa in {0.05, 0.1, 0.2, 0.3, 0.5, 0.7}. Separations 4 and 3 sd: report only.
      (c) same thresholds as (b) at d = 24 and 36 (isotropic GMM of the frozen setup: K = d, spread layout, radius 4.25, loguniform weights ratio 20, seed 0; buffer weights ∝ w^0.5).
      (d) estimated parameters (threshold centres of the frozen method, Laplace Sigma_k, L_k by log-space regression) on a 10 000-sample buffer drawn with weights ∝ w^0.5, d = 24 and 36:
          K_found == K; softmax(-L) within 25 % of the true weight for every mode of weight >= 1 %; coverage xi on the buffer: mean in [0.8, 1.25] and >= 90 % of the points in [0.5, 2];
          then (b)'s statistics with the ESTIMATED parameters: cos >= 0.9 and scale in [0.7, 1.4] for kappa in {0.1, 0.2, 0.3, 0.5, 0.7} (kappa = 0.05 reported).
    c*(x_t) = (E_p[x1 | x_t] - E_q[x1 | x_t]) / nu_{1|t}: the exact Tweedie correction (closed form for the isotropic GMM).

    python tests/test_division_cv.py [--quick]
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils.division_cv import coverage, division_term, fit_modes, make_params  # noqa: E402
from adjoint_samplers.utils.dist_utils import GMM  # noqa: E402
from adjoint_samplers.energies.dist_energy import DistEnergy  # noqa: E402

torch.set_default_dtype(torch.float64)
KAPPAS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
RESULTS = {}


def kv(kappa, nu1):
    return kappa, nu1 * kappa * (1 - kappa), nu1 * (1 - kappa)


def logp(x, w, mu):
    d = x.shape[-1]
    return torch.logsumexp(torch.log(w)[None] - 0.5 * ((x[:, None, :] - mu[None]) ** 2).sum(-1) - 0.5 * d * math.log(2 * math.pi), dim=1)


def sample_gmm(n, w, mu, gen):
    k = torch.multinomial(w, n, replacement=True, generator=gen)
    return mu[k] + torch.randn(n, mu.shape[1], generator=gen)


def posterior(xt, w, mu, kappa, v):
    """ isotropic unit-covariance GMM: responsibilities (B, K) and per-component posterior means (B, K, D) / variance of x1 | x_t """
    d = mu.shape[1]
    vp = 1.0 / (1.0 + kappa ** 2 / v)
    mp = vp * (mu[None] + kappa * xt[:, None, :] / v)
    lr = torch.log(w)[None] - 0.5 * ((xt[:, None, :] - kappa * mu[None]) ** 2).sum(-1) / (v + kappa ** 2) - 0.5 * d * math.log(2 * math.pi * (v + kappa ** 2))
    return torch.softmax(lr, dim=-1), mp, vp


def oracle_params(mu, w, shrink=0.8):
    K, d = mu.shape
    return make_params(mu, torch.eye(d).expand(K, d, d).clone(), -torch.log(w), shrink)


# ---------------------------------------------------------------- (a)
def test_a_quadrature(tag, par, mu, w, nu1=16.0):
    g = torch.linspace(-12, 12, 481)
    X1 = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    dx2 = float((g[1] - g[0]) ** 2)
    lp = logp(X1, w, mu)
    worst = 0.0
    for kappa in (0.05, 0.2, 0.5, 0.8):
        kap, v, nu1t = kv(kappa, nu1)
        for xt0 in ((1.0, 0.5), (-2.0, 1.0), (0.3, -3.0)):
            xt = torch.tensor([xt0]) * kappa
            XT = xt.expand(len(X1), 2)
            logpost = lp - 0.5 * ((XT - kap * X1) ** 2).sum(-1) / v
            post = torch.exp(logpost - logpost.max())
            Z = float((post * dx2).sum())
            term = division_term(XT, X1, -lp, kap, v, nu1t, par)
            worst = max(worst, float(((post[:, None] * term * dx2).sum(0) / Z).abs().max()) / float(term.abs().max()))
    assert worst < 1e-6, (tag, worst)
    print(f"ok  (a) quadrature [{tag}]: max |E_pi[term | x_t]| / max|term| = {worst:.1e} at 12 (kappa, x_t) points")


def random_params(gen, K=3, d=2):
    mu = 3 * torch.randn(K, d, generator=gen)
    A = torch.randn(K, d, d, generator=gen)
    cov = A @ A.transpose(-1, -2) * 0.3 + 0.4 * torch.eye(d)
    return make_params(mu, cov, 2 * torch.randn(K, generator=gen), shrink=float(0.4 + torch.rand(1, generator=gen)))


def noisy_params(mu, w, gen):
    K, d = mu.shape
    mu_hat = mu + 0.5 * torch.randn(K, d, generator=gen)
    cov = torch.eye(d)[None] * (1.0 + 0.3 * torch.rand(K, 1, 1, generator=gen))
    return make_params(mu_hat, cov.expand(K, d, d).clone(), -torch.log(w) + 0.5 * torch.randn(K, generator=gen), shrink=0.8)


def test_a_monte_carlo(par, mu, w, n=400000, nu1=16.0):
    gen = torch.Generator().manual_seed(0)
    x1 = sample_gmm(n, w, mu, gen)
    E1 = -logp(x1, w, mu)
    out = []
    for kappa in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        kap, v, nu1t = kv(kappa, nu1)
        xt = kap * x1 + math.sqrt(v) * torch.randn(n, 2, generator=gen)
        zs = []
        for cvs in (1.0, 2.5):
            term = division_term(xt, x1, E1, kap, v, nu1t, par, c_var_scale=cvs)
            zs.append(max(abs(float((term * p).sum(-1).mean() / ((term * p).sum(-1).std() / n ** 0.5))) for p in (xt, xt * (xt ** 2).sum(-1, keepdim=True))))
        out.append((kappa, zs[0], zs[1]))
    zmax, zctl = max(o[1] for o in out), max(o[2] for o in out)
    assert zmax < 3.5 and zctl > 5, out
    print("ok  (a) Monte Carlo (400k, trained-like parameters): max |z| per kappa " + ", ".join(f"{k:g}:{a:.1f}" for k, a, _ in out) + f" ; positive control: max |z| {zctl:.0f}")


# ---------------------------------------------------------------- (b), (c), (d): alignment with the exact Tweedie correction
def alignment(par, mu, w_true, w_buf, nu1, kappas=KAPPAS, n_xt=120, n_post=2000, seed=2):
    gen = torch.Generator().manual_seed(seed)
    res = {}
    for kappa in kappas:
        kap, v, nu1t = kv(kappa, nu1)
        x1q = sample_gmm(n_xt, w_buf, mu, gen)
        xt = kap * x1q + math.sqrt(v) * torch.randn(n_xt, mu.shape[1], generator=gen)
        rq, mp, vp = posterior(xt, w_buf, mu, kap, v)
        rp, _, _ = posterior(xt, w_true, mu, kap, v)
        cstar = ((rp[..., None] * mp).sum(1) - (rq[..., None] * mp).sum(1)) / nu1t
        m = torch.zeros_like(xt)
        for i in range(n_xt):
            comp = torch.multinomial(rq[i], n_post, replacement=True, generator=gen)
            x1 = mp[i, comp] + math.sqrt(vp) * torch.randn(n_post, mu.shape[1], generator=gen)
            m[i] = division_term(xt[i].expand(n_post, -1), x1, -logp(x1, w_true, mu), kap, v, nu1t, par).mean(0)
        cos = float((m * cstar).sum() / (m.norm() * cstar.norm()))
        scale = float((m * cstar).sum() / (cstar * cstar).sum())
        res[kappa] = (cos, scale, float(cstar.pow(2).sum(-1).mean().sqrt()))
    return res


def show(title, res):
    print(f"    {title}: " + "  ".join(f"k={k:g}: cos {c:+.3f} scale {s:.3f}" for k, (c, s, _) in res.items()))


def check(res, cos_min, lo, hi, kappas=KAPPAS):
    bad = [(k, res[k][:2]) for k in kappas if k in res and not (res[k][0] >= cos_min and lo <= res[k][1] <= hi)]
    return bad


def test_b_oracle():
    w = torch.tensor([0.5, 0.3, 0.2])
    wb = torch.full((3,), 1 / 3)
    for sep in (6.0, 4.0, 3.0):
        r = sep / math.sqrt(3)
        mu = r * torch.tensor([[1.0, 0.0], [-0.5, math.sqrt(3) / 2], [-0.5, -math.sqrt(3) / 2]])
        res = alignment(oracle_params(mu, w), mu, w, wb, nu1=(2 * math.sqrt(1 + r ** 2 / 2)) ** 2)
        show(f"(b) separation {sep:g} sd", res)
        if sep >= 6:
            bad = check(res, 0.95, 0.8, 1.25)
            RESULTS["b"] = not bad
            print(f"{'ok ' if not bad else 'FAIL'} (b) oracle at 6 sd: cos >= 0.95 and scale in [0.8, 1.25] at all kappa" + ("" if not bad else f" -> violations {bad}") + " (4 and 3 sd: reported above)")


def test_c_noise_check(d=24):
    """ REPORT ONLY: is the cosine of (c) limited by the Monte Carlo noise of E_q[term | x_t] (n_post draws per x_t)? Same statistic with more draws. """
    g, mu, w, nu1 = frozen_gmm(d)
    wb = w ** 0.5
    wb = wb / wb.sum()
    for n_post in (1500, 6000, 24000):
        res = alignment(oracle_params(mu, w), mu, w, wb, nu1, kappas=(0.05, 0.3), n_xt=40, n_post=n_post)
        show(f"(c, noise check) d = {d}, {n_post} posterior draws per x_t, 40 x_t", res)


class F32:
    """ the repo's BB denoising / Laplace helpers assume the float32 default dtype """
    def __enter__(self):
        self.old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)

    def __exit__(self, *a):
        torch.set_default_dtype(self.old)


def test_c_confirmation():
    """ REPORT ONLY (added after the first run, not a pre-registered threshold): (c) with 24000 posterior draws per x_t, all kappa, d = 24 and 36 """
    for d in (24, 36):
        g, mu, w, nu1 = frozen_gmm(d)
        wb = w ** 0.5
        wb = wb / wb.sum()
        res = alignment(oracle_params(mu, w), mu, w, wb, nu1, n_xt=40, n_post=24000)
        show(f"(c, confirmation) d = {d}, oracle, 24000 draws per x_t, 40 x_t", res)


def frozen_gmm(d):
    with F32():                                                       # built as in train.py (float32 default): the float32 / float64 random streams give different means
        g = GMM(dim=d, K=d, seed=0, layout="spread", weight_law="loguniform", weight_ratio=20.0, radius=4.25)
    return g, g.means.double(), g.weights.double(), (2 * GMM.coord_std(d, 4.25)) ** 2


def test_c_high_dim():
    for d in (24, 36):
        g, mu, w, nu1 = frozen_gmm(d)
        wb = w ** 0.5
        wb = wb / wb.sum()
        res = alignment(oracle_params(mu, w), mu, w, wb, nu1, n_xt=100, n_post=1500)
        show(f"(c) d = {d}, K = {d}, oracle", res)
        bad = check(res, 0.95, 0.8, 1.25)
        RESULTS[f"c{d}"] = not bad
        print(f"{'ok ' if not bad else 'FAIL'} (c) oracle at d = {d} (isotropic frozen geometry): cos >= 0.95 and scale in [0.8, 1.25] at all kappa" + ("" if not bad else f" -> violations (kappa, (cos, scale)): {[(k, tuple(round(x, 3) for x in v)) for k, v in bad]}"))


def test_d_estimated():
    ok = True
    for d in (24, 36):
        g, mu, w, nu1 = frozen_gmm(d)
        wb = w ** 0.5
        wb = wb / wb.sum()
        gen = torch.Generator().manual_seed(11)
        x1 = sample_gmm(10000, wb, mu, gen)
        torch.manual_seed(0)
        with F32():
            energy = DistEnergy(g)
            x1f = x1.float()
            E1f = energy.eval(x1f)
            par, info = fit_modes(energy, x1f, E1f)
        E1 = E1f.double()
        K_found = len(par["mu"])
        near = torch.cdist(par["mu"], mu).argmin(1)
        cerr = (par["mu"] - mu[near]).norm(dim=1)
        what = torch.softmax(-par["L"], 0)
        wtrue = w[near]
        rel = (what / wtrue - 1).abs()
        big = wtrue >= 0.01
        xi = coverage(x1, E1, par)
        print(f"    (d) d = {d}: K_found {K_found} / {d} (distinct true modes matched: {len(set(near.tolist()))}), unexplained fraction {1 - info['explained']:.3f}, centre error mean {cerr.mean():.3f} max {cerr.max():.3f}")
        print(f"        softmax(-L) vs true weight: max relative error {rel[big].max():.3f} on modes >= 1 % ({int(big.sum())} modes), median {rel.median():.3f}; sum of estimated weights {float(torch.exp(-par['L']).sum()):.3f}")
        print(f"        coverage xi: mean {xi.mean():.3f}, std {xi.std():.3f}, 5 % / 95 % quantiles {xi.quantile(0.05):.3f} / {xi.quantile(0.95):.3f}, share in [0.5, 2] {float(((xi > 0.5) & (xi < 2)).double().mean()):.3f}")
        res = alignment(par, mu, w, wb, nu1, kappas=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7), n_xt=100, n_post=1500)
        show(f"(d) d = {d}, estimated parameters", res)
        c1 = K_found == d
        c2 = bool((rel[big] <= 0.25).all())
        c3 = 0.8 <= float(xi.mean()) <= 1.25 and float(((xi > 0.5) & (xi < 2)).double().mean()) >= 0.9
        c4 = not check(res, 0.9, 0.7, 1.4, kappas=(0.1, 0.2, 0.3, 0.5, 0.7))
        print(f"        thresholds: K_found == K {c1} ; weights within 25 % {c2} ; coverage {c3} ; cos/scale {c4}")
        ok &= c1 and c2 and c3 and c4
        if "--confirm" in sys.argv:
            res = alignment(par, mu, w, wb, nu1, kappas=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7), n_xt=40, n_post=24000)
            show(f"(d, confirmation, REPORT ONLY) d = {d}, estimated parameters, 24000 draws per x_t, 40 x_t", res)
    return ok


if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else "abcd"
    if "d" in only and len(only) == 1:
        RESULTS["d"] = test_d_estimated()
        print("\nSUMMARY:", RESULTS)
        sys.exit(0)
    gen = torch.Generator().manual_seed(7)
    mu2 = torch.tensor([[3.0, 0.0], [-1.5, 2.6], [-1.5, -2.6]])
    w2 = torch.tensor([0.5, 0.3, 0.2])
    for i in range(3):
        test_a_quadrature(f"random parameters {i}", random_params(gen), mu2, w2)
    test_a_monte_carlo(noisy_params(mu2, w2, torch.Generator().manual_seed(3)), mu2, w2)
    test_b_oracle()
    test_c_high_dim()
    test_c_noise_check()
    test_c_confirmation()
    RESULTS["d"] = test_d_estimated()
    print("\nSUMMARY of the pre-registered thresholds:", RESULTS)
