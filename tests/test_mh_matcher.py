""" Checks of the Metropolis-Hastings control variate (AdjointVEMHMatcher):

    python tests/test_mh_matcher.py

    Targets are GMMs with unit-variance modes, so π(x1 | xt) ∝ p1(x1) p^base(xt | x1) is a GMM too and is
    sampled exactly (never from a buffer). Monte Carlo statements are tested with |z-score| < 5.
"""
import copy
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adjoint_samplers.components.buffer import BatchBuffer
from adjoint_samplers.components.matcher import AdjointVEMatcher, AdjointVEMHMatcher, _kmeans
from adjoint_samplers.components.model import FourierMLP
from adjoint_samplers.components.sde import ControlledSDE, VESDE
from adjoint_samplers.components.state_cost import ZeroGradStateCost
from adjoint_samplers.components.term_cost import ScoreGradTermCost
from adjoint_samplers.energies.dist_energy import DistEnergy
from adjoint_samplers.utils.dist_utils import GMM, Delta

Z_MAX = 5.0


def make_gmm(dim, K, radius, seed=0):
    return GMM(dim=dim, K=K, seed=seed, layout="spread", weight_law="loguniform", radius=radius)


def with_weights(gmm, weights):
    other = copy.copy(gmm)
    other.weights = weights / weights.sum()
    return other


def make_matcher(target, matcher_cls=AdjointVEMHMatcher, sigma_max=None, **kwargs):
    dim = target.dim
    sigma_max = sigma_max or 2 * GMM.coord_std(dim, target.means.norm(dim=-1).mean().item())
    ref_sde = VESDE(sigma_min=1e-3, sigma_max=sigma_max)
    sde = ControlledSDE(ref_sde, FourierMLP(dim, activation=torch.nn.GELU()))
    term_cost = ScoreGradTermCost(source=Delta(dim), ref_sde=ref_sde, energy=DistEnergy(target))
    return matcher_cls(
        grad_term_cost=term_cost,
        grad_state_cost=ZeroGradStateCost(),
        sde=sde,
        buffer=BatchBuffer(10**6),
        resample_size=256,
        duplicates=1,
        **kwargs,
    )


def kappa_nu(matcher, t):
    t = torch.as_tensor([t], dtype=torch.float)
    nut, nu1 = matcher._nu(t), matcher._nu(torch.ones_like(t))
    return (nut / nu1).item(), (nu1 - nut).item(), nu1.item()


def t_at_kappa(matcher, kappa):
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if kappa_nu(matcher, mid)[0] < kappa else (lo, mid)
    return (lo + hi) / 2


def posterior(gmm, kappa, nu1, xt):
    """ component probabilities (.., K) and means (.., K, D), variance s2 of π(x1 | xt) ∝ gmm(x1) N(xt; κ x1, ν_1 κ (1 - κ)) """
    if kappa == 0:
        return gmm.weights.expand(*xt.shape[:-1], -1), gmm.means.expand(*xt.shape[:-1], -1, -1), 1.0
    v = nu1 * kappa * (1 - kappa)
    s2 = v / (v + kappa**2)
    means = (v * gmm.means + kappa * xt.unsqueeze(-2)) / (v + kappa**2)
    logits = gmm.weights.log() - 0.5 * (xt.unsqueeze(-2) - kappa * gmm.means).pow(2).sum(-1) / (v + kappa**2)
    return torch.softmax(logits, -1), means, s2


def sample_posterior(gmm, kappa, nu1, xt, n):
    """ X1 ~ π(x1 | xt), exactly; xt (D,) -> (n, D), or xt (N, D) -> (N, n, D) """
    probs, means, s2 = posterior(gmm, kappa, nu1, xt)
    if xt.dim() == 1:
        k = torch.multinomial(probs, n, replacement=True)
        return means[k] + math.sqrt(s2) * torch.randn(n, gmm.dim, dtype=means.dtype)
    k = torch.multinomial(probs, n, replacement=True) # (N, n)
    x1 = torch.gather(means, 1, k.unsqueeze(-1).expand(-1, -1, gmm.dim))
    return x1 + math.sqrt(s2) * torch.randn_like(x1)


def mh_inputs(matcher, t, xt, x1):
    """ what populate_buffer stores, then the arguments of mh_term """
    n = x1.shape[0]
    jumps, valid = matcher.propose(x1)
    energy = matcher.grad_term_cost.energy
    EY = torch.zeros_like(valid)
    mask = valid.bool()
    EY[mask] = energy.eval((x1.unsqueeze(1) + jumps)[mask])
    return dict(
        t=torch.full((n, 1), t, dtype=x1.dtype),
        x0=torch.zeros_like(x1),
        xt=xt.expand(n, -1) if xt.dim() == 1 else xt,
        x1=x1,
        E1=energy.eval(x1).reshape(-1, 1),
        jumps=jumps,
        EY=EY,
        valid=valid,
    )


def h_of(matcher, t, xt, x1):
    # h itself (term = γ h / ν_{1|t}, with γ = 1 here)
    term, alpha = matcher.mh_term(**mh_inputs(matcher, t, xt, x1))
    return term * kappa_nu(matcher, t)[1], alpha


def zscores(samples):
    return samples.mean(0) / (samples.std(0) / math.sqrt(samples.shape[0]))


def test_zero_mean_under_target_posterior(n=200000):
    """ E_π[h | Xt] = 0 for any centres, all M - 1 jumps or one; and ≠ 0 under a wrong-weight model """
    torch.manual_seed(0)
    target = make_gmm(dim=6, K=4, radius=4.25)
    model = with_weights(target, target.weights.flip(0))
    centre_sets = {
        "means + noise, M=K": target.means + 0.5 * torch.randn_like(target.means),
        "k-means, M=2K": _kmeans(target.sample([20000]), 8),
    }
    for name, centres in centre_sets.items():
        for rb in (True, False):
            matcher = make_matcher(target, gamma=1.0, n_clusters=len(centres), rao_blackwell=rb)
            matcher.centres = centres
            for kappa in (0.05, 0.2, 0.5):
                t = t_at_kappa(matcher, kappa)
                _, _, nu1 = kappa_nu(matcher, t)
                xt = kappa * target.sample([1])[0] + math.sqrt(nu1 * kappa * (1 - kappa)) * torch.randn(6)

                h, alpha = h_of(matcher, t, xt, sample_posterior(target, kappa, nu1, xt, n))
                h_model, _ = h_of(matcher, t, xt, sample_posterior(model, kappa, nu1, xt, n))
                z, z_model = zscores(h).abs().max().item(), zscores(h_model).abs().max().item()
                label = f"{name:<18} {'all jumps' if rb else 'one jump':<9} κ={kappa}"
                assert z < Z_MAX, f"{label}: |z| = {z:.1f} under π"
                print(f"    {label}  max|z| under π = {z:.2f}, under π^u = {z_model:6.1f}, mean α = {alpha.mean():.3f}")
    print("ok  h has zero mean under the target posterior (exact draws), not under a wrong-weight model")


def test_flux_direction_two_modes(n=200000):
    torch.manual_seed(1)
    target = with_weights(make_gmm(dim=6, K=2, radius=4.25), torch.tensor([0.5, 0.5]))
    model = with_weights(target, torch.tensor([0.8, 0.2])) # mode 1 under-weighted
    matcher = make_matcher(target, gamma=1.0, n_clusters=2, rao_blackwell=True)
    matcher.centres = target.means.clone()
    toward = target.means[1] - target.means[0]
    for kappa in (0.0, 0.1, 0.3):
        t = t_at_kappa(matcher, kappa) if kappa > 0 else 0.0
        _, _, nu1 = kappa_nu(matcher, t)
        xt = torch.zeros(6)
        h, _ = h_of(matcher, t, xt, sample_posterior(model, kappa, nu1, xt, n))
        proj = h @ toward / toward.norm()
        z = (proj.mean() / (proj.std() / math.sqrt(n))).item()
        assert z > 10, f"κ={kappa}: flux not toward the under-weighted mode (z = {z:.1f})"
        print(f"    κ={kappa}: E[h] along m_1 - m_0 = {proj.mean():.3f} (z = {z:.0f})")
    print("ok  under the model, h points toward the under-weighted mode")


def generator(w, M):
    # rates min(1, w_k / w_j) / (M - 1) of the cluster-level MH chain
    Q = torch.minimum(w.new_ones(()), w[None, :] / w[:, None]) / (M - 1)
    Q.fill_diagonal_(0)
    return Q - torch.diag(Q.sum(1))


def test_flux_is_generator_at_t0(n=400000):
    """ t = 0, equal-shape separated modes, centres = means: E_{p1^u}[h] = Σ_k c_k m_k, c = - G^T (w - ŵ) """
    torch.manual_seed(2)
    target = make_gmm(dim=6, K=4, radius=6.0)
    model = with_weights(target, target.weights * torch.exp(0.5 * torch.randn(4)))
    matcher = make_matcher(target, gamma=1.0, n_clusters=4, rao_blackwell=True)
    matcher.centres = target.means.clone()
    h, _ = h_of(matcher, 0.0, torch.zeros(6), model.sample([n]))

    c = - generator(target.weights, 4).T @ (target.weights - model.weights)
    want = c @ target.means
    err = ((h.mean(0) - want).norm() / want.norm()).item()
    z = zscores(h - want).abs().max().item()
    assert err < 0.03 and z < Z_MAX, f"rel err {err:.3f}, |z| {z:.1f}"
    print(f"ok  at t = 0, E[h] = Σ c_k m_k with c = -G^T e (rel err {err:.4f}, max|z| {z:.2f})")


def test_gamma_zero_recovers_adjoint_ve_matcher():
    target = make_gmm(dim=6, K=4, radius=4.25)
    matchers = [make_matcher(target, AdjointVEMatcher), make_matcher(target, gamma=0.0, n_clusters=4)]
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


def test_t_focus():
    target = make_gmm(dim=6, K=4, radius=4.25)
    t = make_matcher(target, AdjointVEMatcher, t_focus=0.3).sample_t(torch.zeros(200000, 6)).squeeze(-1)
    # P(t < 0.15) = 0.3 + 0.7 * 0.15
    frac = (t < 0.15).float().mean().item()
    assert abs(frac - (0.3 + 0.7 * 0.15)) < 0.005 and t.min() >= 0 and t.max() < 1, frac
    print(f"ok  t_focus = 0.3 puts {frac:.3f} of the regression times in [0, 0.15] (expected 0.405)")


def test_denoised_kmeans_finds_light_modes():
    """ d = K = 24: light modes under-sampled 3x, as in the runs. Raw k-means misses them,
        k-means on x + ∇log p1(x) puts a centre on every mode
    """
    torch.manual_seed(7)
    target = make_gmm(dim=24, K=24, radius=4.25)
    w = target.weights.clone()
    light = w <= w.quantile(0.3)
    w[light] /= 3
    x = with_weights(target, w).sample([10000])
    found = lambda pts: (torch.cdist(target.means, _kmeans(pts, 48)).min(1).values < 1.0)
    raw, den = found(x), found(x + target.score(x))
    assert den.all() and raw[light].float().mean() < 0.5
    print(f"ok  d = 24: k-means finds {raw[light].float().mean():.0%} of the light modes on raw points,"
          f" {den[light].float().mean():.0%} on x + ∇log p1 (all modes: {raw.float().mean():.0%} vs {den.float().mean():.0%})")


def test_buffer_to_loss():
    """ populate (no centres, then k-means on the previous buffer) -> dataloader -> target -> stats """
    torch.manual_seed(4)
    target = make_gmm(dim=6, K=4, radius=4.25)
    matcher = make_matcher(target, n_clusters=4, stats_every=1)
    timesteps = torch.linspace(0, 1, 50)

    matcher.populate_buffer(torch.zeros(256, 6), timesteps, False)
    assert matcher.centres is None and matcher.buffer.batches["valid"][0].sum() == 0
    matcher.populate_buffer(torch.zeros(256, 6), timesteps, False)
    assert matcher.centres.shape == (4, 6) and matcher.buffer.batches["valid"][1].sum() > 0
    assert set(matcher.buffer.batches) == {"x0", "x1", "adjoint1", "E1", "jumps", "EY", "valid", "same_mode"}

    data = next(iter(matcher.build_dataloader(512)))
    (t, xt), target_ = matcher.prepare_target(data, "cpu")
    loss = (matcher.sde.u(t, xt) - target_).pow(2).mean()
    loss.backward()
    stats = matcher.pop_stats()
    assert torch.isfinite(loss) and "mh/valid_frac" in stats and "mh/energy_evals_per_sample" in stats
    print(f"ok  buffer -> target -> loss (loss {loss.item():.3f}); stats: "
          + ", ".join(f"{k} {v:.3g}" for k, v in stats.items() if "/acc" not in k))


def test_drift_to_weights(N=40000, n_mc=16, gamma=0.5, t_stop=0.3):
    """ One outer iteration from a frozen buffer p1^u (wrong weights), with the inner regression solved
        exactly: drift σ²(t) [u_bridge + γ E_{π^u}[h | x] / ν_{1|t}], u_bridge = (E_{π^u}[X1 | x] - x) / ν_{1|t}
        the bridge to p1^u. Its change of terminal mode masses vs the bridge alone (same noise) is compared
        with (a) the first-order prediction conditional on x_t, Σ_t dt E[<δb(X_t), ∇_x P^u(k | X_t)>], and
        (b) the cluster-level prediction - γ G^T (w - ŵ). For reference, AS's own exact step,
        u_bridge + E_{π^u}[∇log(p1 / p1^u)(X1) | x], is simulated the same way.
        Past t_stop every jump is rejected, so both runs follow the bridge to p1^u, whose terminal mode
        probabilities given X_t are P^u(k | X_t): masses = E[P^u(k | X_t_stop)] (no stiff steps near t = 1).
    """
    torch.manual_seed(5)
    torch.set_default_dtype(torch.float64)
    try:
        dim, K = 6, 4
        target = make_gmm(dim, K, radius=4.25)
        target.means, target.weights = target.means.double(), target.weights.double()
        model = with_weights(target, target.weights * torch.exp(0.4 * torch.randn(K)))
        matcher = make_matcher(target, gamma=gamma, n_clusters=K, rao_blackwell=True, sigma_max=2 * GMM.coord_std(dim, 4.25))
        matcher.centres = target.means.clone()
        ref = matcher.sde.ref_sde
        timesteps = torch.linspace(0, 1, 201)

        def h_given_xt(t, x, kappa, nu1):
            # E_{π^u}[h | x] by exact posterior draws, all M - 1 jumps
            x1 = sample_posterior(model, kappa, nu1, x, n_mc) # (N, n_mc, D)
            term, _ = matcher.mh_term(**mh_inputs(matcher, t, x.repeat_interleave(n_mc, 0), x1.reshape(-1, dim)))
            return term.view(-1, n_mc, dim).mean(1) # γ h / ν_{1|t}

        def as_step(x, kappa, nu1):
            # AS's correction E_{π^u}[∇log(p1 / p1^u)(X1) | x], by exact posterior draws
            x1 = sample_posterior(model, kappa, nu1, x, n_mc)
            return (target.score(x1) - model.score(x1)).mean(1)

        def simulate(perturb):
            gen = torch.Generator().manual_seed(6) # same Brownian increments for both runs
            x = torch.zeros(N, dim)
            lin = torch.zeros(K)
            for i in range(len(timesteps) - 1):
                t, dt = timesteps[i].item(), (timesteps[i + 1] - timesteps[i]).item()
                if t >= t_stop:
                    break
                kappa, nu1t, nu1 = kappa_nu(matcher, t)
                probs, means, _ = posterior(model, kappa, nu1, x)
                u = ((probs.unsqueeze(-1) * means).sum(1) - x) / nu1t
                sigma2 = ref.diff(torch.tensor(t)).item() ** 2
                db = sigma2 * h_given_xt(t, x, kappa, nu1) if perturb != "as" else None
                if perturb == "as":
                    u = u + as_step(x, kappa, nu1)
                elif perturb:
                    u = u + db / sigma2
                elif kappa > 0:
                    # first-order effect on P(X1 in mode k): <δb, ∇_x P^u(k | x)> dt
                    v = nu1 * kappa * (1 - kappa)
                    mbar = probs @ model.means
                    grad_P = probs.unsqueeze(-1) * kappa * (model.means - mbar.unsqueeze(1)) / (v + kappa**2)
                    lin += (grad_P * db.unsqueeze(1)).sum(-1).mean(0) * dt
                noise = torch.randn(N, dim, generator=gen)
                x = x + sigma2 * u * dt + math.sqrt(sigma2 * dt) * noise
            kappa, _, nu1 = kappa_nu(matcher, t)
            return posterior(model, kappa, nu1, x)[0].mean(0), lin

        base, lin = simulate(perturb=False)
        pert, _ = simulate(perturb=True)
        as_mass, _ = simulate(perturb="as")
        measured = pert - base
        e = target.weights - model.weights
        naive = - gamma * generator(target.weights, K).T @ e

        cos = lambda a, b: (a @ b / (a.norm() * b.norm())).item()
        print(f"    target w {target.weights.numpy().round(3)}, model ŵ {model.weights.numpy().round(3)}, bridge masses {base.numpy().round(3)}")
        print(f"    e = w - ŵ             {e.numpy().round(4)}")
        print(f"    measured Δŵ          {measured.numpy().round(4)}   (bridge alone reproduces ŵ: max err {(base - model.weights).abs().max():.4f})")
        print(f"    linear, cond. on x_t {lin.numpy().round(4)}   cos {cos(measured, lin):.3f}, |measured|/|linear| {measured.norm() / lin.norm():.2f}")
        print(f"    naive -γ G^T e       {naive.numpy().round(4)}   cos {cos(measured, naive):.3f}, |measured|/|naive| {measured.norm() / naive.norm():.2f}")
        print(f"    cos(measured, e) {cos(measured, e):.3f}: the step moves the masses toward the target,"
              f" by {measured @ e / (e @ e):.2f} e")
        as_change = as_mass - base
        print(f"    AS's own step Δŵ     {as_change.numpy().round(4)}   = {as_change @ e / (e @ e):.3f} e")
        assert cos(measured, e) > 0
    finally:
        torch.set_default_dtype(torch.float32)
    print("ok  drift-to-weights (numbers above)")


if __name__ == "__main__":
    test_gamma_zero_recovers_adjoint_ve_matcher()
    test_t_focus()
    test_denoised_kmeans_finds_light_modes()
    test_buffer_to_loss()
    test_flux_is_generator_at_t0()
    test_flux_direction_two_modes()
    test_zero_mean_under_target_posterior()
    test_drift_to_weights()
    print("all passed")
