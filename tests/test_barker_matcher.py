""" Checks of the Barker orbit control variate (AdjointVEBarkerMatcher), in the same setting as test_mh_matcher.py:

    python tests/test_barker_matcher.py            # light tests, a few seconds each
    python tests/test_barker_matcher.py --heavy    # + exact outer iteration (minutes: run on the cluster)

    Targets are GMMs with unit-variance modes, so π(x1 | xt) is a GMM too and is sampled exactly (never from a
    buffer). Monte Carlo statements are tested with |z-score| < 5.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_mh_matcher import (  # noqa: E402
    Z_MAX, h_of, kappa_nu, make_gmm, make_matcher, mh_inputs, posterior, sample_posterior, t_at_kappa,
    test_drift_to_weights, with_weights, zscores,
)
from adjoint_samplers.components.matcher import (  # noqa: E402
    AdjointVEBarkerMatcher, AdjointVEMatcher, AdjointVEMHMatcher, _kmeans,
)


def test_zero_mean_under_target_posterior(n=60000):
    """ E_π[h | Xt] = 0 for any centres, M = K or M = 2K; and ≠ 0 under a wrong-weight model """
    torch.manual_seed(0)
    target = make_gmm(dim=6, K=4, radius=4.25)
    model = with_weights(target, target.weights.flip(0))
    centre_sets = {
        "means + noise, M=K": target.means + 0.5 * torch.randn_like(target.means),
        "k-means, M=2K": _kmeans(target.sample([20000]), 8),
        "random, M=K": 2 * torch.randn(4, 6),
    }
    for name, centres in centre_sets.items():
        matcher = make_matcher(target, AdjointVEBarkerMatcher, gamma=1.0, n_clusters=len(centres))
        matcher.centres = centres
        for kappa in (0.05, 0.2, 0.5):
            t = t_at_kappa(matcher, kappa)
            _, _, nu1 = kappa_nu(matcher, t)
            xt = kappa * target.sample([1])[0] + (nu1 * kappa * (1 - kappa)) ** 0.5 * torch.randn(6)
            h, w = h_of(matcher, t, xt, sample_posterior(target, kappa, nu1, xt, n))
            h_model, _ = h_of(matcher, t, xt, sample_posterior(model, kappa, nu1, xt, n))
            z, z_model = zscores(h).abs().max().item(), zscores(h_model).abs().max().item()
            assert z < Z_MAX, f"{name} κ={kappa}: |z| = {z:.1f} under π"
            print(f"    {name:<20} κ={kappa}: max|z| under π = {z:.2f}, under π^u = {z_model:6.1f}, mean ω_own = {1 - w.sum(1).mean():.3f}")
    print("ok  h has zero mean under the target posterior (exact draws), not under a wrong-weight model")


def test_orbit_enumeration_is_exact():
    """ Σ_{z in O} p_O(z) h(z) = 0 to machine precision, p_O ∝ π(z | xt) on the orbit; every member of the
        orbit sees the same orbit (same set of points). Float64, arbitrary x and xt, M = K and M = 2K.
    """
    torch.manual_seed(1)
    torch.set_default_dtype(torch.float64)
    try:
        dim = 3
        target = make_gmm(dim=dim, K=4, radius=4.25)
        target.means, target.weights = target.means.double(), target.weights.double()
        worst, n_checked = 0.0, 0
        for M in (4, 8):
            centres = torch.randn(M, dim) * 3
            matcher = make_matcher(target, AdjointVEBarkerMatcher, gamma=1.0, n_clusters=M)
            matcher.centres = centres
            for kappa0 in (0.03, 0.3, 0.7):
                t = t_at_kappa(matcher, kappa0)
                # κ_t, ν_{1|t} in float64, as the matcher computes them (kappa_nu rounds t to float32)
                t64 = torch.tensor([[t]], dtype=torch.float64)
                nut, nu1 = matcher._nu(t64), matcher._nu(torch.ones_like(t64))
                kappa, nu1t = (nut / nu1).item(), (nu1 - nut).item()
                v = nu1.item() * kappa * (1 - kappa)
                for _ in range(30):
                    x = target.sample([1]) + torch.randn(1, dim)
                    xt = kappa * x[0] + 2 * torch.randn(dim)
                    jumps, valid = matcher.propose(x)
                    z = torch.cat([x, x + jumps[0][valid[0].bool()]]) # orbit, own point first
                    # every member recomputes its own orbit: same set
                    ref = z.round(decimals=8).sort(dim=0).values
                    for zi in z:
                        j, val = matcher.propose(zi[None])
                        oz = torch.cat([zi[None], zi[None] + j[0][val[0].bool()]])
                        assert oz.shape == z.shape and torch.equal(oz.round(decimals=8).sort(dim=0).values, ref)
                    logp = target.log_prob(z) - 0.5 * (xt - kappa * z).pow(2).sum(-1) / v
                    p = torch.softmax(logp, 0)
                    term, _ = matcher.mh_term(**mh_inputs(matcher, t, xt, z))
                    err = (p.unsqueeze(-1) * term).sum(0).abs().max().item() * nu1t
                    worst, n_checked = max(worst, err), n_checked + 1
        assert worst < 1e-10, worst
    finally:
        torch.set_default_dtype(torch.float32)
    print(f"ok  orbit enumeration: max |Σ p_O h| = {worst:.1e} over {n_checked} orbits (M = K and 2K, same orbit from every member)")


def test_flux_is_weight_error_at_t0(n=150000):
    """ t = 0, equal-shape separated modes, centres = means: E_{p1^u}[h] = Σ_k (w_k - ŵ_k) m_k (flat spectrum),
        whereas MH-RB gives Σ_k c_k m_k with c = - G^T e (printed for contrast)
    """
    torch.manual_seed(2)
    target = make_gmm(dim=6, K=4, radius=6.0)
    model = with_weights(target, target.weights * torch.exp(0.5 * torch.randn(4)))
    samples = model.sample([n])
    want = (target.weights - model.weights) @ target.means
    errs = {}
    for cls in (AdjointVEBarkerMatcher, AdjointVEMHMatcher):
        matcher = make_matcher(target, cls, gamma=1.0, n_clusters=4, rao_blackwell=True)
        matcher.centres = target.means.clone()
        h, _ = h_of(matcher, 0.0, torch.zeros(6), samples)
        errs[cls.__name__] = ((h.mean(0) - want).norm() / want.norm()).item()
        if cls is AdjointVEBarkerMatcher:
            z = zscores(h - want).abs().max().item()
            assert errs[cls.__name__] < 0.03 and z < Z_MAX, f"rel err {errs[cls.__name__]:.3f}, |z| {z:.1f}"
    assert errs["AdjointVEMHMatcher"] > 2 * errs["AdjointVEBarkerMatcher"]
    print(f"ok  at t = 0, E[h] = Σ (w - ŵ) m: rel err Barker {errs['AdjointVEBarkerMatcher']:.4f}"
          f" (MH-RB: {errs['AdjointVEMHMatcher']:.3f}, not the weight error)")


def test_own_weight_goes_to_one(n=20000):
    torch.manual_seed(3)
    target = make_gmm(dim=6, K=4, radius=4.25)
    matcher = make_matcher(target, AdjointVEBarkerMatcher, gamma=1.0, n_clusters=4)
    matcher.centres = target.means.clone()
    own = {}
    for kappa in (0.02, 0.5, 0.9):
        t = t_at_kappa(matcher, kappa)
        _, _, nu1 = kappa_nu(matcher, t)
        x1 = target.sample([n])
        xt = kappa * x1 + (nu1 * kappa * (1 - kappa)) ** 0.5 * torch.randn_like(x1)
        _, w = matcher.mh_term(**mh_inputs(matcher, t, xt, x1))
        own[kappa] = (1 - w.sum(1)).mean().item()
    assert own[0.9] > 0.95 and own[0.02] < own[0.5] < own[0.9], own
    print("ok  ω_own = " + ", ".join(f"{v:.3f} (κ={k})" for k, v in own.items()))


def test_gamma_zero_recovers_adjoint_ve_matcher():
    target = make_gmm(dim=6, K=4, radius=4.25)
    matchers = [make_matcher(target, AdjointVEMatcher), make_matcher(target, AdjointVEBarkerMatcher, gamma=0.0, n_clusters=4)]
    matchers[1].centres = target.means.clone()
    x1 = target.sample([64])
    data = {"x0": torch.zeros(64, 6), "x1": x1, "adjoint1": matchers[0].grad_term_cost(x1)}
    data.update({k: v for k, v in mh_inputs(matchers[1], 0.1, torch.zeros(6), x1).items()
                 if k in ("E1", "jumps", "EY", "valid")})
    targets = []
    for m in matchers:
        torch.manual_seed(3)
        targets.append(m.prepare_target(data, "cpu")[1])
    assert torch.equal(*targets)
    print("ok  γ = 0 gives the AdjointVEMatcher target")


def test_no_centres_or_no_valid_jump_is_finite():
    torch.manual_seed(4)
    target = make_gmm(dim=6, K=4, radius=4.25)
    matcher = make_matcher(target, AdjointVEBarkerMatcher, gamma=1.0, n_clusters=4)
    x1 = target.sample([128])
    inputs = mh_inputs(matcher, 0.1, torch.zeros(6), x1) # centres is None: no valid jump
    term, w = matcher.mh_term(**inputs)
    assert inputs["valid"].sum() == 0 and torch.isfinite(term).all() and term.abs().max() == 0 and w.abs().max() == 0
    print("ok  no centres -> no valid jump -> zero, finite term")


def test_oracle_centres():
    """ oracle_centres: centres = the true means from the first batch, never refitted; coverage diagnostic = 1 """
    torch.manual_seed(6)
    target = make_gmm(dim=6, K=4, radius=4.25)
    matcher = make_matcher(target, AdjointVEBarkerMatcher, n_clusters=4, stats_every=1, oracle_centres=True)
    timesteps = torch.linspace(0, 1, 50)
    for _ in range(3):
        matcher.populate_buffer(torch.zeros(128, 6), timesteps, False)
        assert torch.equal(matcher.centres, target.means)
    assert matcher.buffer.batches["valid"][0].sum() > 0, "valid jumps from the first batch on"
    data = next(iter(matcher.build_dataloader(256)))
    matcher.prepare_target(data, "cpu")
    stats = matcher.pop_stats()
    assert stats["barker/centre_coverage"] == 1.0 and stats["barker/centres_spurious"] == 0.0
    # k-means centres do not necessarily cover the modes of an untrained model's buffer
    plain = make_matcher(target, AdjointVEBarkerMatcher, n_clusters=4)
    assert plain.centres is None
    print("ok  oracle centres: used from the first batch, never refitted; coverage 1.0, no spurious centre")


def test_flux_diagnostics_point_to_underweighted_modes(n=30000):
    """ per-mode net flux (t < 0.1) correlates with the buffer weight error e = w - ŵ, and the keys exist """
    torch.manual_seed(7)
    target = make_gmm(dim=6, K=4, radius=6.0)
    model = with_weights(target, target.weights * torch.exp(0.7 * torch.randn(4)))
    matcher = make_matcher(target, AdjointVEBarkerMatcher, n_clusters=4, stats_every=1)
    matcher.centres = target.means.clone()
    x1 = model.sample([n])
    inputs = mh_inputs(matcher, 0.0, torch.zeros(6), x1)
    data = {"x0": inputs["x0"], "x1": x1, "adjoint1": matcher.grad_term_cost(x1), "E1": inputs["E1"],
            "jumps": inputs["jumps"], "EY": inputs["EY"], "valid": inputs["valid"],
            "mode1": target.assign(x1), "modeY": target.assign(x1.unsqueeze(1) + inputs["jumps"])}
    matcher.prepare_target(data, "cpu")
    stats = matcher.pop_stats()
    assert stats["barker/flux_err_corr"] > 0.5, stats["barker/flux_err_corr"]
    assert all(f"barker/flux_net/{j}" in stats and f"barker/err/{j}" in stats for j in range(4))
    print(f"ok  flux diagnostics: corr(net flux, weight error) = {stats['barker/flux_err_corr']:.2f}")


def test_buffer_to_loss():
    """ populate (no centres, then k-means on the previous buffer) -> dataloader -> target -> stats """
    torch.manual_seed(5)
    target = make_gmm(dim=6, K=4, radius=4.25)
    matcher = make_matcher(target, AdjointVEBarkerMatcher, n_clusters=4, stats_every=1)
    timesteps = torch.linspace(0, 1, 50)

    matcher.populate_buffer(torch.zeros(256, 6), timesteps, False)
    assert matcher.centres is None
    matcher.populate_buffer(torch.zeros(256, 6), timesteps, False)
    assert matcher.centres.shape == (4, 6) and matcher.buffer.batches["jumps"][1].shape == (256, 3, 6)
    assert matcher.buffer.batches["EY"][1].shape == (256, 3)

    data = next(iter(matcher.build_dataloader(512)))
    (t, xt), target_ = matcher.prepare_target(data, "cpu")
    loss = (matcher.sde.u(t, xt) - target_).pow(2).mean()
    loss.backward()
    stats = matcher.pop_stats()
    assert torch.isfinite(loss) and torch.isfinite(target_).all()
    assert "mh/valid_frac" in stats and "mh/energy_evals_per_sample" in stats
    assert any(k.startswith("barker/own_weight/") for k in stats) and any(k.startswith("barker/orbit_ess/") for k in stats)
    assert not any(k.startswith("mh/acc") for k in stats)
    # the untrained model samples far from the modes: few valid translates; all M - 1 are stored and evaluated when valid
    assert stats["mh/energy_evals_per_sample"] > 0
    print(f"ok  buffer -> target -> loss (loss {loss.item():.3f}); evals/sample {stats['mh/energy_evals_per_sample']:.2f}; "
          + ", ".join(f"{k.split('/')[-1]} {v:.2f}" for k, v in stats.items() if k.startswith("barker/own_weight")))


if __name__ == "__main__":
    test_gamma_zero_recovers_adjoint_ve_matcher()
    test_no_centres_or_no_valid_jump_is_finite()
    test_buffer_to_loss()
    test_oracle_centres()
    test_flux_diagnostics_point_to_underweighted_modes()
    test_own_weight_goes_to_one()
    test_orbit_enumeration_is_exact()
    test_zero_mean_under_target_posterior()
    test_flux_is_weight_error_at_t0()
    if "--heavy" in sys.argv:
        print("exact outer iteration, Barker (gamma = 1):")
        test_drift_to_weights(gamma=1.0, matcher_cls=AdjointVEBarkerMatcher)
    print("all passed")
