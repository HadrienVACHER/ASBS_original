""" Checks of the heterogeneous GMM (HeteroGMM: unequal covariances, non-equidistant modes) and of the AFFINE orbit

        T_{k<-c}(x) = μ_k + L_k L_c^{-1} (x - μ_c),    Barker weights π(Y_k | xt) |det L_k|

    (AdjointVEBarkerMatcher(orbit="affine")), and of orbit_size = s (a uniform random subset of s - 1 other clusters):

    python tests/test_hetero_barker.py
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_mh_matcher import kappa_nu, make_matcher, t_at_kappa, zscores  # noqa: E402
from adjoint_samplers.components.matcher import (  # noqa: E402
    AdjointVEBarkerMatcher, _bb_denoise, _converged_centres, _laplace_factors,
)
from adjoint_samplers.energies.dist_energy import DistEnergy  # noqa: E402
from adjoint_samplers.utils.dist_utils import HeteroGMM  # noqa: E402

Z_MAX = 5.0


def make_target(dim, K, seed=0, double=False, **kw):
    g = HeteroGMM(dim=dim, K=K, seed=seed, **kw)
    if double:
        g.means, g.weights, g.L = g.means.double(), g.weights.double(), g.L.double()
        g.to("cpu") # recomputes P, logdet in double
    return g


def with_weights(g, w):
    import copy
    o = copy.copy(g)
    o.weights = w / w.sum()
    return o


def hetero_matcher(g, **kw):
    m = make_matcher(g, AdjointVEBarkerMatcher, sigma_max=2.0 * g.coord_std(), gamma=1.0, n_clusters=g.K, orbit="affine", centre_method="oracle", **kw)
    m.centres, m.factors, m.prec, m.logdet = g.means.clone(), g.L.clone(), g.P.clone(), g.logdet.clone()
    return m


def posterior_sample(g, kappa, nu1, xt, n):
    """ x1 ~ π(x1 | xt) ∝ p1(x1) N(xt; κ x1, v I), v = ν_1 κ (1 - κ), p1 the mixture of N(μ_k, Σ_k), exactly """
    v = nu1 * kappa * (1 - kappa)
    D = g.dim
    eye = torch.eye(D, dtype=g.means.dtype)
    Sig = g.L @ g.L.transpose(-1, -2)
    Sinv = g.P.transpose(-1, -2) @ g.P
    S_post = torch.linalg.inv(Sinv + (kappa ** 2 / v) * eye)                                   # (K, D, D)
    m_post = torch.einsum("kde,ke->kd", S_post, torch.einsum("kde,ke->kd", Sinv, g.means) + (kappa / v) * xt)
    cov_marg = kappa ** 2 * Sig + v * eye
    diff = xt - kappa * g.means
    logw = g.weights.log() - 0.5 * torch.einsum("kd,kde,ke->k", diff, torch.linalg.inv(cov_marg), diff) - 0.5 * torch.linalg.slogdet(cov_marg)[1]
    k = torch.multinomial(torch.softmax(logw, 0), n, replacement=True)
    Lp = torch.linalg.cholesky(S_post)
    return m_post[k] + torch.einsum("nde,ne->nd", Lp[k], torch.randn(n, D, dtype=g.means.dtype))


def test_hetero_gmm_basics():
    torch.manual_seed(0)
    for D, K in ((6, 4), (24, 24)):
        g = make_target(D, K)
        dist = torch.cdist(g.means, g.means).fill_diagonal_(1e9)
        assert dist.min() < 0.8 * dist[dist < 1e8].max(), "modes must not be equidistant"
        sv = torch.linalg.svdvals(g.L)
        assert sv.min() >= 0.59 and sv.max() <= 1.61
        acc = torch.stack([(g.assign(g.means[k] + torch.einsum("de,ne->nd", g.L[k], torch.randn(1000, D))) == k).float().mean() for k in range(K)])
        assert acc.min() > 0.995, acc.min()
        x = g.sample([40])
        xx = x.clone().requires_grad_(True)
        gr = torch.autograd.grad(g.log_prob(xx).sum(), xx)[0]
        assert torch.allclose(gr, g.score(x), atol=1e-4), (gr - g.score(x)).abs().max()
        k = 0
        xs = g.means[k] + torch.einsum("de,ne->nd", g.L[k], torch.randn(100000, D))
        Sig = g.L[k] @ g.L[k].T
        assert ((torch.cov(xs.T) - Sig).norm() / Sig.norm()) < 0.03
        print(f"ok  HeteroGMM d={D} K={K}: distances between modes {dist.min():.1f}..{dist[dist < 1e8].max():.1f}, widths {sv.min():.2f}..{sv.max():.2f}, "
              f"assignment accuracy {acc.min():.4f}, analytic score = autograd, covariance within 3%")


def test_affine_map_and_group_property():
    torch.manual_seed(1)
    g = make_target(6, 4, double=True)
    c, k, j = 0, 1, 2
    A = lambda a, b: g.L[a] @ g.P[b]                      # L_a L_b^{-1}
    T = lambda x, a, b: g.means[a] + torch.einsum("de,ne->nd", A(a, b), x - g.means[b])  # T_{a<-b}
    xs = g.means[c] + torch.einsum("de,ne->nd", g.L[c], torch.randn(200000, 6, dtype=torch.float64))
    y = T(xs, k, c)
    Sk = g.L[k] @ g.L[k].T
    assert (y.mean(0) - g.means[k]).norm() < 0.02 and ((torch.cov(y.T) - Sk).norm() / Sk.norm()) < 0.02
    assert torch.allclose(T(T(xs, k, c), j, k), T(xs, j, c), atol=1e-9) # group property: the orbit is the same set from any member
    assert torch.allclose(T(T(xs, k, c), c, k), xs, atol=1e-9)         # inverse
    print("ok  T_{k<-c} maps N(μ_c, Σ_c) onto N(μ_k, Σ_k); T_{j<-k} ∘ T_{k<-c} = T_{j<-c}; T_{c<-k} ∘ T_{k<-c} = id")


def check_orbits(g, matcher, kappas=(0.03, 0.3, 0.7), n_per=25):
    """ Σ_{z in O} p_O(z) h(z) = 0, p_O(z_j) ∝ π(z_j | xt) |det L_j|; every member sees the same orbit. Float64, arbitrary x, xt,
        arbitrary centres and factors """
    D = g.dim
    worst, n_checked = 0.0, 0
    for kappa0 in kappas:
        t = t_at_kappa(matcher, kappa0)
        t64 = torch.tensor([[t]], dtype=torch.float64)
        nut, nu1 = matcher._nu(t64), matcher._nu(torch.ones_like(t64))
        kappa, nu1t = (nut / nu1).item(), (nu1 - nut).item()
        v = nu1.item() * kappa * (1 - kappa)
        for _ in range(n_per):
            x = g.sample([1]) + torch.randn(1, D, dtype=torch.float64)
            xt = kappa * x[0] + 2 * torch.randn(D, dtype=torch.float64)
            jumps, valid, ld = matcher._orbit(x)
            members = [x[0]] + [x[0] + jumps[0, i] for i in range(jumps.shape[1]) if valid[0, i] > 0.5]
            z = torch.stack(members)
            ref = z.round(decimals=7).sort(dim=0).values
            for zi in z:
                j2, v2, _ = matcher._orbit(zi[None])
                oz = torch.stack([zi] + [zi + j2[0, i] for i in range(j2.shape[1]) if v2[0, i] > 0.5])
                assert oz.shape == z.shape and torch.equal(oz.round(decimals=7).sort(dim=0).values, ref)
            cl = matcher._assign(z)
            logp = g.log_prob(z) - 0.5 * (xt - kappa * z).pow(2).sum(-1) / v + matcher.logdet[cl]
            p = torch.softmax(logp, 0)
            terms = []
            for zi in z:
                jj, vv, ldd = matcher._orbit(zi[None])
                EY = torch.zeros_like(vv)
                mask = vv.bool()
                energy = matcher.grad_term_cost.energy
                EY[mask] = energy.eval((zi[None, None] + jj)[mask]).reshape(-1)
                E1 = energy.eval(zi[None]).reshape(-1, 1)
                term, _ = matcher.mh_term(t64, torch.zeros(1, D, dtype=torch.float64), xt[None], zi[None], E1, jj, EY, vv, logdet=ldd)
                terms.append(term[0])
            err = (p.unsqueeze(-1) * torch.stack(terms)).sum(0).abs().max().item() * nu1t
            worst, n_checked = max(worst, err), n_checked + 1
    return worst, n_checked


def test_affine_orbit_enumeration_is_exact():
    """ exact zero mean by orbit enumeration, for the TRUE (μ_k, L_k) and for ARBITRARY centres and factors """
    torch.manual_seed(2)
    torch.set_default_dtype(torch.float64)
    try:
        g = make_target(3, 4, double=True, radius_min=7.0, radius_max=10.0)
        res = {}
        m = hetero_matcher(g)
        res["true (μ_k, L_k)"] = check_orbits(g, m)
        m2 = hetero_matcher(g)
        m2.centres = g.means + 1.5 * torch.randn(4, 3)
        m2.factors = torch.stack([torch.linalg.qr(torch.randn(3, 3))[0] * torch.exp(0.5 * torch.randn(3)) for _ in range(4)])
        m2.prec, m2.logdet = torch.linalg.inv(m2.factors), torch.linalg.slogdet(m2.factors)[1]
        res["arbitrary centres and factors"] = check_orbits(g, m2)
        for name, (worst, n) in res.items():
            assert worst < 1e-10, (name, worst)
    finally:
        torch.set_default_dtype(torch.float32)
    print("ok  affine orbit, orbit enumeration (same orbit from every member, exact zero mean): "
          + ", ".join(f"{k}: max|Σ p_O h| = {w:.1e} ({n} orbits)" for k, (w, n) in res.items()))


def test_zero_mean_under_target_posterior_incl_orbit_size(n=60000):
    torch.manual_seed(3)
    g = make_target(6, 4, double=True, radius_min=8.0, radius_max=11.0)
    model = with_weights(g, g.weights.flip(0))
    for label, kw in (("all clusters", {}), ("orbit_size = 3", {"orbit_size": 3}), ("orbit_size = 2", {"orbit_size": 2})):
        matcher = hetero_matcher(g, **kw)
        for kappa in (0.05, 0.3):
            t = t_at_kappa(matcher, kappa)
            _, _, nu1 = kappa_nu(matcher, t)
            xt = kappa * g.sample([1])[0] + (nu1 * kappa * (1 - kappa)) ** 0.5 * torch.randn(6, dtype=torch.float64)
            out = []
            for tgt in (g, model):
                x1 = posterior_sample(tgt, kappa, nu1, xt, n)
                jumps, valid, ld = matcher._orbit(x1)
                energy = matcher.grad_term_cost.energy
                EY = torch.zeros_like(valid)
                mask = valid.bool()
                EY[mask] = energy.eval((x1.unsqueeze(1) + jumps)[mask]).reshape(-1)
                tv = torch.full((n, 1), t, dtype=torch.float64)
                term, _ = matcher.mh_term(tv, torch.zeros_like(x1), xt.expand(n, -1), x1, energy.eval(x1).reshape(-1, 1), jumps, EY, valid, logdet=ld)
                out.append(zscores(term * kappa_nu(matcher, t)[1]).abs().max().item())
            assert out[0] < Z_MAX, f"{label} κ={kappa}: |z| = {out[0]:.1f} under π"
            assert out[1] > 10, f"{label} κ={kappa}: the term is not detecting wrong weights (|z| = {out[1]:.1f})"
            print(f"    {label:14s} κ={kappa}: max|z| under π = {out[0]:.2f}, under a wrong-weight model = {out[1]:6.1f}")
    print("ok  zero mean under the target posterior (exact draws), also with orbit_size subsets")


def test_laplace_factors_recover_the_covariances():
    torch.manual_seed(4)
    g = make_target(12, 6, radius_min=9.0, radius_max=12.0)
    L, P, ld, cond = _laplace_factors(DistEnergy(g), g.means)
    Sig = L @ L.transpose(-1, -2)
    true = g.L @ g.L.transpose(-1, -2)
    err = ((Sig - true).flatten(1).norm(dim=1) / true.flatten(1).norm(dim=1)).max().item()
    assert err < 0.02 and torch.allclose(ld, torch.linalg.slogdet(g.L)[1], atol=0.05), err
    print(f"ok  Laplace factors at the modes recover Σ_k: max relative error {err:.4f}, log|det| within 0.05, condition numbers {cond.min():.1f}..{cond.max():.1f}")


def test_threshold_affine_in_the_matcher():
    """ threshold centres + Laplace factors from the buffer, valid affine translates, finite target, stats """
    torch.manual_seed(5)
    g = make_target(6, 4, radius_min=8.0, radius_max=11.0)
    matcher = make_matcher(g, AdjointVEBarkerMatcher, sigma_max=2.0 * g.coord_std(), gamma=1.0, max_centres=8, stats_every=1,
                           centre_method="threshold", orbit="affine")
    timesteps = torch.linspace(0, 1, 50)
    for _ in range(3):
        matcher.populate_buffer(torch.zeros(256, 6), timesteps, False)
    assert matcher.centres is not None and len(matcher.centres) == 4, len(matcher.centres)
    assert torch.cdist(g.means, matcher.centres).min(1).values.max() < 0.5
    Sig = matcher.factors @ matcher.factors.transpose(-1, -2)
    near = torch.cdist(matcher.centres, g.means).argmin(1)
    true = (g.L @ g.L.transpose(-1, -2))[near]
    err = ((Sig - true).flatten(1).norm(dim=1) / true.flatten(1).norm(dim=1)).max().item()
    assert err < 0.05, err
    b = matcher.buffer.batches
    assert b["jumps"][2].shape == (256, 7, 6) and b["logdet"][2].shape == (256, 7) and b["valid"][2].sum() > 0
    data = next(iter(matcher.build_dataloader(512)))
    (t, xt), target_ = matcher.prepare_target(data, "cpu")
    assert torch.isfinite(target_).all()
    stats = matcher.pop_stats()
    assert "centres/cond_median" in stats and stats["barker/centre_coverage"] > 0
    print(f"ok  threshold + affine in the matcher: 4 centres, covariances within {err:.3f} of the truth, logdet stored, finite target, "
          f"median condition number {stats['centres/cond_median']:.1f}")


def test_orbit_size_bounds_the_energy_calls():
    torch.manual_seed(6)
    g = make_target(24, 24)
    timesteps = torch.linspace(0, 1, 50)
    evals = {}
    for s in (None, 8, 4):
        matcher = make_matcher(g, AdjointVEBarkerMatcher, sigma_max=2.0 * g.coord_std(), gamma=1.0, stats_every=1,
                               centre_method="oracle", orbit="affine", n_clusters=24, orbit_size=s)
        for _ in range(3):
            matcher.populate_buffer(torch.zeros(256, 24), timesteps, False)
        st = matcher.pop_stats()
        evals[s] = st["mh/energy_evals_per_sample"]
        J = matcher.buffer.batches["jumps"][2].shape[1]
        assert J == (23 if s is None else s - 1)
    assert evals[4] < evals[8] < evals[None]
    print("ok  orbit_size: energy evaluations per sample " + ", ".join(f"{'all' if k is None else k}: {v:.2f}" for k, v in evals.items()))


def test_simplex_layout_keeps_the_isotropic_positions_and_weights():
    """ layout = "simplex": the mode positions and weights of the isotropic GMM (same seed, spread, radius 4.25), covariances heterogeneous """
    from adjoint_samplers.utils.dist_utils import GMM
    iso = GMM(dim=12, K=12, seed=0, layout="spread", weight_law="loguniform", weight_ratio=20, radius=4.25)
    g = HeteroGMM(dim=12, K=12, seed=0, layout="simplex", simplex_radius=4.25, scale_min=1.0, scale_max=1.4, weight_ratio=20)
    assert torch.allclose(g.means, iso.means, atol=1e-5) and torch.allclose(g.weights, iso.weights, atol=1e-6)
    sv = torch.linalg.svdvals(g.L)
    assert sv.min() >= 0.99 and sv.max() <= 1.41
    r = []
    for i in range(12):
        for j in range(i + 1, 12):
            u = g.means[j] - g.means[i]
            d = u.norm()
            u = u / d
            r.append((d / ((g.L[i].T @ u).norm() + (g.L[j].T @ u).norm())).item())
    assert max(r) <= 3.06, max(r)
    acc = torch.stack([(g.assign(g.means[k] + torch.einsum("de,ne->nd", g.L[k], torch.randn(2000, 12))) == k).float().mean() for k in range(12)])
    print(f"ok  simplex layout: positions and weights of the isotropic GMM, widths {sv.min():.2f}..{sv.max():.2f}, normalised separation "
          f"<= {max(r):.2f} (isotropic: 3.05), exact assignment accuracy {acc.mean():.3f} (worst {acc.min():.3f})")


if __name__ == "__main__":
    test_hetero_gmm_basics()
    test_simplex_layout_keeps_the_isotropic_positions_and_weights()
    test_affine_map_and_group_property()
    test_affine_orbit_enumeration_is_exact()
    test_zero_mean_under_target_posterior_incl_orbit_size()
    test_laplace_factors_recover_the_covariances()
    test_threshold_affine_in_the_matcher()
    test_orbit_size_bounds_the_energy_calls()
    print("all passed")
