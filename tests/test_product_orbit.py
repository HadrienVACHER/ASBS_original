""" Checks of the SUBGROUP ORBIT for product-structured targets (AdjointVEBarkerMatcher(orbit="product", centre_method="wells")) on ManyWell:
    per sample a subset S of s double-well coordinates drawn independently of the position, the orbit = the 2^s well combinations on S, each
    coordinate with its 1D affine map between its two wells (Laplace centre and width per well) and the Jacobian in the Barker weights.

    python tests/test_product_orbit.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_mh_matcher import kappa_nu, make_matcher, t_at_kappa, zscores  # noqa: E402
from adjoint_samplers.components.matcher import AdjointVEBarkerMatcher, _bb_denoise, _product_wells  # noqa: E402
from adjoint_samplers.energies.dist_energy import DistEnergy  # noqa: E402
from adjoint_samplers.utils.dist_utils import ManyWell  # noqa: E402

Z_MAX = 5.0


def product_matcher(g, s, wells=None, **kw):
    m = make_matcher(g, AdjointVEBarkerMatcher, sigma_max=2.0 * g.coord_std(), gamma=1.0, n_clusters=2, orbit="product",
                     centre_method="wells", product_subset=s, **kw)
    m.wells = wells
    return m


def arbitrary_wells(idx, double=False, seed=0):
    """ arbitrary centres, widths and splits: the exactness does not depend on them """
    g = torch.Generator().manual_seed(seed)
    J = len(idx)
    dt = torch.float64 if double else torch.float32
    mu = torch.stack([-1.7 + 0.4 * torch.randn(J, generator=g), 1.7 + 0.4 * torch.randn(J, generator=g)], 1).to(dt)
    sigma = torch.exp(0.5 * torch.randn(J, 2, generator=g)).to(dt) * 0.5
    split = (0.3 * torch.randn(J, generator=g)).to(dt)
    return dict(idx=torch.tensor(idx), mu=mu, sigma=sigma, split=split)


def test_wells_are_found():
    torch.manual_seed(0)
    g = ManyWell(dim=8)
    energy = DistEnergy(g)
    x = g.sample([6000])
    xd, conv, lam, steps, _, _ = _bb_denoise(energy, x, energy.eval(x).reshape(-1), energy.grad_E(x))
    wells, n = _product_wells(energy, xd, conv, lam)
    assert n == 4 and wells["idx"].tolist() == [0, 2, 4, 6], (n, wells["idx"])
    roots = np.sort(np.roots([4, 0, -12, -0.5]).real)                    # E' = 4x^3 - 12x - 0.5 = 0: left well, barrier top, right well
    xl, xt_, xr = roots
    sig = lambda xs: 1 / math.sqrt(12 * xs ** 2 - 12)
    assert torch.allclose(wells["mu"][:, 0], torch.full((4,), float(xl)), atol=0.05) and torch.allclose(wells["mu"][:, 1], torch.full((4,), float(xr)), atol=0.05)
    assert torch.allclose(wells["sigma"][:, 0], torch.full((4,), sig(xl)), rtol=0.06) and torch.allclose(wells["sigma"][:, 1], torch.full((4,), sig(xr)), rtol=0.06)
    assert (wells["split"] - float(xt_)).abs().max() < 0.1
    print(f"ok  wells found from the basin minima: coordinates {wells['idx'].tolist()} (the 4 double wells; the Gaussian coordinates have one cluster), "
          f"centres {float(wells['mu'][0, 0]):.3f} / {float(wells['mu'][0, 1]):.3f} (true {xl:.3f} / {xr:.3f}), widths {float(wells['sigma'][0, 0]):.3f} / "
          f"{float(wells['sigma'][0, 1]):.3f} (true {sig(xl):.3f} / {sig(xr):.3f}), split {float(wells['split'][0]):.3f} (barrier top {xt_:.3f})")


def check_orbits(g, m, subsets, kappas=(0.05, 0.4), n_per=20):
    """ Σ_{z in O} p_O(z) h(z) = 0 with p_O(z) ∝ π(z | xt) Π_{j in S} σ_{w_j(z)}, for a FIXED subset S per orbit; every member sees the same orbit """
    D = g.dim
    w = m.wells
    worst, n = 0.0, 0
    for kappa0 in kappas:
        t = t_at_kappa(m, kappa0)
        t64 = torch.tensor([[t]], dtype=torch.float64)
        nut, nu1 = m._nu(t64), m._nu(torch.ones_like(t64))
        kappa, nu1t = (nut / nu1).item(), (nu1 - nut).item()
        v = nu1.item() * kappa * (1 - kappa)
        for S in subsets:
            S = torch.tensor([S])
            for _ in range(n_per):
                x = g.sample([1]).double() + 0.5 * torch.randn(1, D, dtype=torch.float64)
                xt = kappa * x[0] + 2 * torch.randn(D, dtype=torch.float64)
                jumps, valid, ld = m._orbit_product(x, subset=S)
                z = torch.stack([x[0]] + [x[0] + jumps[0, i] for i in range(jumps.shape[1]) if valid[0, i] > 0.5])
                ref = z.round(decimals=7).sort(dim=0).values
                terms, logw = [], []
                for zi in z:
                    j2, v2, l2 = m._orbit_product(zi[None], subset=S)
                    oz = torch.stack([zi] + [zi + j2[0, i] for i in range(j2.shape[1]) if v2[0, i] > 0.5])
                    assert oz.shape == z.shape and torch.equal(oz.round(decimals=7).sort(dim=0).values, ref), "different orbits"
                    energy = m.grad_term_cost.energy
                    EY = torch.zeros_like(v2)
                    mask = v2.bool()
                    EY[mask] = energy.eval((zi[None, None] + j2)[mask]).reshape(-1)
                    E1 = energy.eval(zi[None]).reshape(-1, 1)
                    term, _ = m.mh_term(t64, torch.zeros(1, D, dtype=torch.float64), xt[None], zi[None], E1, j2, EY, v2, logdet=l2)
                    terms.append(term[0])
                    sides = (zi[w["idx"][S[0]]] >= w["split"][S[0]]).long()
                    logw.append(g.log_prob(zi[None])[0] - 0.5 * (xt - kappa * zi).pow(2).sum() / v
                                + torch.log(w["sigma"][S[0]].gather(1, sides[:, None]).squeeze(1)).sum())
                p = torch.softmax(torch.stack(logw), 0)
                err = (p.unsqueeze(-1) * torch.stack(terms)).sum(0).abs().max().item() * nu1t
                worst, n = max(worst, err), n + 1
    return worst, n


def test_orbit_enumeration_is_exact():
    """ exact zero mean by orbit enumeration for ARBITRARY centres, widths and splits; s = 2 and s = 3; the 2^s combinations form a group """
    torch.manual_seed(1)
    torch.set_default_dtype(torch.float64)
    try:
        g = ManyWell(dim=6)
        res = {}
        for s, subsets in ((2, [[0, 1], [1, 2], [0, 2]]), (3, [[0, 1, 2]])):
            m = product_matcher(g, s, arbitrary_wells([0, 2, 4], double=True, seed=s))
            res[s] = check_orbits(g, m, subsets)
            assert res[s][0] < 1e-10, res[s]
    finally:
        torch.set_default_dtype(torch.float32)
    print("ok  product orbit, orbit enumeration (same orbit from every member = group Z_2^S, exact zero mean): "
          + ", ".join(f"s = {s}: max|Σ p_O h| = {w:.1e} ({n} orbits)" for s, (w, n) in res.items()))


def posterior_sample(g_post, kappa, nu1, xt, n):
    """ x1 ~ π(x1 | xt) for the product target: independent 1D problems, p1_j(x) exp(-(xt_j - κ x)^2 / (2 v)), by inverse CDF on a grid """
    v = nu1 * kappa * (1 - kappa)
    D = g_post.dim
    grid = torch.linspace(-8, 8, 16001, dtype=torch.float64)
    cdfs = []
    for j in range(D):
        if j % 2 == 0:
            lp = -(g_post.a * grid + g_post.b * grid ** 2 + g_post.c * grid ** 4)
        else:
            lp = -0.5 * grid ** 2
        lp = lp - 0.5 * (xt[j] - kappa * grid) ** 2 / v
        p = torch.exp(lp - lp.max())
        cdfs.append(torch.cumsum(p, 0) / p.sum())
    cdf = torch.stack(cdfs)                                              # (D, G)
    u = torch.rand(D, n, dtype=torch.float64)
    idx = torch.searchsorted(cdf, u).clamp(1, cdf.shape[1] - 1)
    c0, c1 = cdf.gather(1, idx - 1), cdf.gather(1, idx)
    tt = ((u - c0) / (c1 - c0).clamp(min=1e-300)).clamp(0, 1)
    return (grid[idx - 1] + tt * (grid[idx] - grid[idx - 1])).T


def test_zero_mean_under_the_exact_posterior_with_random_subsets(n=60000):
    torch.manual_seed(2)
    torch.set_default_dtype(torch.float64)
    try:
        g = ManyWell(dim=6)
        wrong = ManyWell(dim=6, a=-1.4)                                  # different well weights
        m = product_matcher(g, 2, arbitrary_wells([0, 2, 4], double=True, seed=5))
        for kappa in (0.05, 0.3):
            t = t_at_kappa(m, kappa)
            nu1 = kappa_nu(m, t)[2]
            xt = kappa * g.sample([1])[0].double() + (nu1 * kappa * (1 - kappa)) ** 0.5 * torch.randn(6, dtype=torch.float64)
            out = []
            for post in (g, wrong):
                x1 = posterior_sample(post, kappa, nu1, xt, n)
                jumps, valid, ld = m._orbit_product(x1)                  # random subsets, independent of x1
                energy = m.grad_term_cost.energy
                EY = torch.zeros_like(valid)
                mask = valid.bool()
                EY[mask] = energy.eval((x1.unsqueeze(1) + jumps)[mask]).reshape(-1)
                tv = torch.full((n, 1), t, dtype=torch.float64)
                term, _ = m.mh_term(tv, torch.zeros_like(x1), xt.expand(n, -1), x1, energy.eval(x1).reshape(-1, 1), jumps, EY, valid, logdet=ld)
                tt = term * kappa_nu(m, t)[1]
                tt = tt[:, tt.std(0) > 0]                                # Gaussian coordinates never move: h = 0 exactly
                out.append(zscores(tt).abs().max().item())
            assert out[0] < Z_MAX, f"κ={kappa}: |z| = {out[0]:.1f} under π"
            assert out[1] > 10, f"κ={kappa}: the term does not detect the wrong well weights (|z| = {out[1]:.1f})"
            print(f"    κ={kappa}: max|z| under π = {out[0]:.2f}, under a model with other well weights = {out[1]:.1f}")
    finally:
        torch.set_default_dtype(torch.float32)
    print("ok  zero mean under the exact posterior with random subsets S (independent of the position), not under wrong well weights")


def test_product_orbit_in_the_matcher():
    """ populate with wells in place, stored translates and logdet, finite target, stats; energy calls <= 2^s - 1.
        (The wells are fitted on exact target samples, as the early samples of an untrained controller are too broad to denoise in a test.) """
    torch.manual_seed(3)
    g = ManyWell(dim=8)
    energy = DistEnergy(g)
    x = g.sample([4000])
    xd, conv, lam, _, _, _ = _bb_denoise(energy, x, energy.eval(x).reshape(-1), energy.grad_E(x))
    wells, n = _product_wells(energy, xd, conv, lam)
    assert n == 4
    m = product_matcher(g, 3, wells, stats_every=1)
    timesteps = torch.linspace(0, 1, 50)
    m.populate_buffer(torch.zeros(256, 8), timesteps, False)
    b = m.buffer.batches
    assert b["jumps"][0].shape == (256, 7, 8) and b["logdet"][0].shape == (256, 7) and b["valid"][0].sum() > 0
    data = next(iter(m.build_dataloader(512)))
    (t, xt), target = m.prepare_target(data, "cpu")
    assert torch.isfinite(target).all()
    st = m.pop_stats()
    assert 0 < st["mh/energy_evals_per_sample"] <= 7
    print(f"ok  product orbit in the matcher: 7 translates stored, energy calls per sample {st['mh/energy_evals_per_sample']:.2f} (<= 7)")


if __name__ == "__main__":
    test_wells_are_found()
    test_orbit_enumeration_is_exact()
    test_zero_mean_under_the_exact_posterior_with_random_subsets()
    test_product_orbit_in_the_matcher()
    print("all passed")
