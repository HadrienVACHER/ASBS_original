""" Checks of the Stein term of ram_stein.tex (AdjointVESteinMatcher):

    python tests/test_stein_matcher.py

    The target p1 is a GMM, so π(x1 | xt) ∝ p1(x1) p^base(xt | x1) is a GMM too and is sampled
    exactly; the zero-mean and bias statements are tested by Monte Carlo (|z-score| < 5).
"""
import math
import sys
from pathlib import Path

import torch
from torch.autograd.functional import jacobian

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adjoint_samplers.components.buffer import BatchBuffer
from adjoint_samplers.components.matcher import AdjointVEMatcher, AdjointVESteinMatcher
from adjoint_samplers.components.model import FourierMLP, SteinMatrixNet
from adjoint_samplers.components.sde import BrownianMotionSDE, ControlledSDE, VESDE
from adjoint_samplers.components.state_cost import ZeroGradStateCost
from adjoint_samplers.components.term_cost import ScoreGradTermCost
from adjoint_samplers.energies.dist_energy import DistEnergy
from adjoint_samplers.utils.dist_utils import GMM, Delta

DIM = 3
Z_MAX = 5.0
SDES = {
    "ve": lambda: VESDE(sigma_min=0.01, sigma_max=3.0),
    "bm": lambda: BrownianMotionSDE(sigma=2.0), # constant σ: the formulas of ram_stein.tex
}


def make_gmm(seed=0, means_shift=0.0, flip_weights=False):
    gmm = GMM(dim=DIM, K=3, seed=seed, layout="spread", weight_law="loguniform", radius=2.5)
    gmm.means = gmm.means + means_shift
    if flip_weights:
        gmm.weights = gmm.weights.flip(0)
    return gmm


def make_stein_net(scale=0.5, **kwargs):
    # random last layer, so that A = B B^T is far from I and depends on (t, xt, x1, E(x1))
    net = SteinMatrixNet(DIM, activation=torch.nn.GELU(), num_layers=3, channels=32, **kwargs)
    with torch.no_grad():
        net.net.out_layer.weight.mul_(scale)
    return net


def make_matcher(sde_name, target, matcher_cls=AdjointVESteinMatcher, **kwargs):
    ref_sde = SDES[sde_name]()
    sde = ControlledSDE(ref_sde, FourierMLP(DIM, activation=torch.nn.GELU()))
    term_cost = ScoreGradTermCost(source=Delta(DIM), ref_sde=ref_sde, energy=DistEnergy(target))
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
    nut, nu1 = matcher._nu(t), matcher._nu(torch.ones_like(t))
    return nut / nu1, nu1 - nut, nu1


def t_at_kappa(matcher, kappa):
    # κ_t = ν_t / ν_1 is increasing: bisection (VE reaches κ = 0.9 at t ≈ 0.2)
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if kappa_nu(matcher, torch.tensor([mid]))[0] < kappa else (lo, mid)
    return (lo + hi) / 2


def sample_posterior(matcher, gmm, t, xt, n):
    """ X1 ~ π(x1 | xt) ∝ gmm(x1) N(xt; κ x1, ν_1 κ (1 - κ)) (X0 = 0), exactly
    """
    kappa, _, nu1 = kappa_nu(matcher, torch.tensor([t]))
    v = nu1 * kappa * (1 - kappa)
    s2 = 1 / (1 + kappa**2 / v) # posterior variance of each unit-variance component
    means = s2 * (gmm.means + kappa * xt / v)
    logits = gmm.weights.log() - 0.5 * (xt - kappa * gmm.means).pow(2).sum(-1) / (v + kappa**2)
    k = torch.multinomial(torch.softmax(logits, 0), n, replacement=True)
    return means[k] + s2.sqrt() * torch.randn(n, DIM)


def typical_xt(matcher, target, t):
    # a point where the optimal process passes at time t
    kappa, _, nu1 = kappa_nu(matcher, torch.tensor([t]))
    x1 = target.sample([1,])[0]
    return kappa * x1 + (nu1 * kappa * (1 - kappa)).sqrt() * torch.randn(DIM)


def stein_inputs(matcher, t, xt, x1):
    n = x1.shape[0]
    tc = matcher.grad_term_cost
    return dict(
        t=torch.full((n, 1), t),
        x0=torch.zeros(n, DIM),
        xt=xt.expand(n, DIM),
        x1=x1,
        adjoint1=tc(x1),
        E1=tc.energy.eval(x1).reshape(-1, 1),
    )


def chunked_stein(matcher, t, xt, x1, chunk=20000):
    out = []
    for x1c in x1.split(chunk):
        with torch.no_grad():
            out.append(matcher.stein_term(**stein_inputs(matcher, t, xt, x1c)))
    return torch.cat(out)


def zscores(samples):
    return samples.mean(0) / (samples.std(0) / math.sqrt(samples.shape[0]))


def brute_force_A_div(matcher, t, xt, x1):
    """ A and ∇_{x1}·A with E(x1) differentiated by autograd through the energy, per sample
    """
    energy = matcher.grad_term_cost.energy
    tt = torch.tensor([[t]])

    def A_of(x):
        B = matcher.stein_net(tt, xt[None], x[None], energy.eval(x[None]).reshape(1, 1))[0]
        return B @ B.T

    A = torch.stack([A_of(x) for x in x1])
    J = torch.stack([jacobian(A_of, x) for x in x1]) # (n, i, j, l) = ∂A_ij / ∂x1_l
    return A, torch.einsum("nijj->ni", J)


def test_tex_formula_constant_sigma():
    """ constant σ, X0 = 0: T F = c [A (Xt - t X1) + σ²(1 - t) (A ∇log p1(X1) + ∇·A)] """
    torch.manual_seed(0)
    target = make_gmm()
    c, sigma = -0.7, 2.0
    for stein_net in (None, make_stein_net()):
        matcher = make_matcher("bm", target, c=c, stein_net=stein_net)
        for t in (0.1, 0.5, 0.9):
            xt, x1 = torch.randn(DIM), 3 * torch.randn(16, DIM)
            got = matcher.stein_term(**stein_inputs(matcher, t, xt, x1))
            score1 = target.score(x1)
            if stein_net is None:
                A, div = torch.eye(DIM).expand(16, DIM, DIM), torch.zeros(16, DIM)
            else:
                A, div = brute_force_A_div(matcher, t, xt, x1)
            Av = lambda v: (A @ v.unsqueeze(-1)).squeeze(-1)
            want = c * (Av(xt - t * x1) + sigma**2 * (1 - t) * (Av(score1) + div))
            err = (got - want).abs().max() / want.abs().max()
            assert err < 1e-4, f"A={'I' if stein_net is None else 'learned'} t={t}: rel err {err:.2e}"
    print("ok  T F matches the formula of ram_stein.tex (constant σ), A = I and learned A")


def test_zero_mean_under_target_posterior(n=200000):
    """ E_π[T F | Xt] = 0 """
    torch.manual_seed(1)
    target = make_gmm()
    for sde_name in SDES:
        for stein_net in (None, make_stein_net()):
            matcher = make_matcher(sde_name, target, c=-1.0, stein_net=stein_net)
            for kappa in (0.2, 0.6, 0.9):
                t = t_at_kappa(matcher, kappa)
                xt = typical_xt(matcher, target, t)
                x1 = sample_posterior(matcher, target, t, xt, n)
                stein = chunked_stein(matcher, t, xt, x1)
                z = zscores(stein).abs().max().item()
                name = f"{sde_name} A={'I' if stein_net is None else 'learned'} t={t:.3f}"
                assert z < Z_MAX, f"{name}: |z| = {z:.1f}"
                assert stein.std(0).min() > 0.05, f"{name}: T F is trivially small"
                print(f"    {name:<22} max|z| = {z:.2f}   std(T F) = {stein.std(0).mean():.3f}")
    print("ok  T F has zero mean under π (target posterior)")


def test_bias_under_model_posterior(n=200000):
    """ E_{π^u}[T F | Xt] = E_{π^u}[F ∇log(p1 / p1^u)(X1) | Xt] ≠ 0, and the drift shift
        - E[T F] points along E[∇log(p1 / p1^u)] (A = I)
    """
    torch.manual_seed(2)
    target = make_gmm()
    model = make_gmm(means_shift=0.4, flip_weights=True) # p1^u: wrong weights, shifted modes
    for sde_name in SDES:
        for stein_net in (None, make_stein_net()):
            matcher = make_matcher(sde_name, target, c=-1.0, stein_net=stein_net)
            for kappa in (0.2, 0.6):
                t = t_at_kappa(matcher, kappa)
                xt = typical_xt(matcher, target, t)
                x1 = sample_posterior(matcher, model, t, xt, n)
                stein = chunked_stein(matcher, t, xt, x1)

                ratio_score = target.score(x1) - model.score(x1) # ∇log(p1 / p1^u)
                _, nu1t, _ = kappa_nu(matcher, torch.tensor([t]))
                if stein_net is None:
                    bias = matcher.c * nu1t * ratio_score
                else:
                    inp = stein_inputs(matcher, t, xt, x1)
                    with torch.no_grad():
                        B = matcher.stein_net(inp["t"], inp["xt"], inp["x1"], inp["E1"])
                    bias = matcher.c * nu1t * (B @ (B.mT @ ratio_score.unsqueeze(-1))).squeeze(-1)

                name = f"{sde_name} A={'I' if stein_net is None else 'learned'} t={t:.3f}"
                z_bias = zscores(stein).abs().max().item()
                z_diff = zscores(stein - bias).abs().max().item()
                assert z_bias > 10, f"{name}: bias not detected (|z| = {z_bias:.1f})"
                assert z_diff < Z_MAX, f"{name}: E[T F] != E[F ∇log(p1/p1^u)] (|z| = {z_diff:.1f})"
                if stein_net is None:
                    assert (- stein.mean(0) @ ratio_score.mean(0)) > 0
                print(f"    {name:<22} max|z| of E[T F] = {z_bias:6.1f}, of E[T F - bias] = {z_diff:.2f}")
    print("ok  under π^u, E[T F] is the bias c ν_{1|t} E[A ∇log(p1/p1^u)], with the sign of c < 0")


def test_hutchinson_unbiased(n_probes=20000):
    torch.manual_seed(3)
    target = make_gmm()
    matcher = make_matcher("ve", target, stein_net=make_stein_net())
    inp = stein_inputs(matcher, 0.4, torch.randn(DIM), 2 * torch.randn(4, DIM))
    score1 = matcher.grad_term_cost.base_score(inp["x1"]) - inp["adjoint1"]
    B = lambda x1, E1: matcher.stein_net(inp["t"], inp["xt"], x1, E1)
    with torch.no_grad():
        Bx = B(inp["x1"], inp["E1"])
        exact = matcher._div_A(B, Bx, inp["x1"], inp["E1"], - score1)
        matcher.div_estimator = "hutchinson"
        probes = torch.stack([
            matcher._div_A(B, Bx, inp["x1"], inp["E1"], - score1) for _ in range(n_probes)
        ])
    z = ((probes.mean(0) - exact) / (probes.std(0) / math.sqrt(n_probes))).abs().max().item()
    assert z < Z_MAX, f"|z| = {z:.1f}"
    print(f"ok  Hutchinson ∇·A is unbiased (max|z| = {z:.2f})")


def test_c_zero_recovers_adjoint_ve_matcher():
    target = make_gmm()
    data = {"x0": torch.zeros(64, DIM), "x1": 3 * torch.randn(64, DIM)}
    data["adjoint1"] = make_matcher("ve", target).grad_term_cost(data["x1"])
    targets = []
    for cls, kwargs in ((AdjointVEMatcher, {}), (AdjointVESteinMatcher, {"c": 0.0})):
        torch.manual_seed(4)
        (_, _), target_ = make_matcher("ve", target, matcher_cls=cls, **kwargs).prepare_target(data, "cpu")
        targets.append(target_)
    assert torch.equal(*targets)
    print("ok  c = 0 gives the AdjointVEMatcher target")


def test_gradient_wrt_stein_net():
    """ d loss / dθ through the JVP divergence = finite differences (float64) """
    torch.manual_seed(5)
    target = make_gmm()
    stein_net = make_stein_net().double()
    matcher = make_matcher("ve", target, c=-0.5, stein_net=stein_net)
    inp = stein_inputs(matcher, 0.3, torch.randn(DIM), 2 * torch.randn(32, DIM))
    inp = {k: v.double() for k, v in inp.items()}

    loss = lambda: (inp["adjoint1"] + matcher.stein_term(**inp)).pow(2).mean()
    params = list(stein_net.parameters())
    grads = torch.autograd.grad(loss(), params)
    direction = [torch.randn_like(p) for p in params]
    analytic = sum((g * d).sum() for g, d in zip(grads, direction)).item()

    eps = 1e-5
    with torch.no_grad():
        for p, d in zip(params, direction):
            p.add_(eps * d)
        up = loss().item()
        for p, d in zip(params, direction):
            p.sub_(2 * eps * d)
        down = loss().item()
        for p, d in zip(params, direction):
            p.add_(eps * d)
    numeric = (up - down) / (2 * eps)
    assert abs(analytic - numeric) < 1e-6 * max(1.0, abs(numeric)), f"{analytic} vs {numeric}"
    print(f"ok  gradient w.r.t. the Stein network matches finite differences ({analytic:.6f})")


def test_buffer_to_loss():
    """ populate_buffer -> dataloader -> prepare_target -> loss.backward, with A learned """
    torch.manual_seed(6)
    target = make_gmm()
    stein_net = SteinMatrixNet(DIM, activation=torch.nn.GELU()) # default init, A ≠ I
    matcher = make_matcher("ve", target, stein_net=stein_net)
    timesteps = torch.linspace(0, 1, 50)
    matcher.populate_buffer(torch.zeros(128, DIM), timesteps, False)
    assert set(matcher.buffer.batches) == {"x0", "x1", "adjoint1", "E1"}

    data = next(iter(matcher.build_dataloader(64)))
    (t, xt), target_ = matcher.prepare_target(data, "cpu")
    u = matcher.sde.u
    loss = (u(t, xt) - target_).pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in stein_net.parameters())
    assert any(p.grad.abs().sum() > 0 for p in stein_net.parameters())
    print(f"ok  buffer -> loss -> backward reaches the Stein network (loss {loss.item():.3f})")


if __name__ == "__main__":
    test_tex_formula_constant_sigma()
    test_c_zero_recovers_adjoint_ve_matcher()
    test_hutchinson_unbiased()
    test_gradient_wrt_stein_net()
    test_buffer_to_loss()
    test_zero_mean_under_target_posterior()
    test_bias_under_model_posterior()
    print("all passed")
