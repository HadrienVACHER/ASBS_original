""" Symmetry-compatible Barker kernel for DW-4 (adjoint_samplers/utils/dw4_reflections.py).

    Test 1 (unit): T_r ∘ T_r = id; |det J| = 1 on the zero-CoM subspace (numerical Jacobian); the energy change from the single distance d_il equals the full energy
                   difference; equivariance of h under rotations, reflections, translations and particle permutations; E_π[h | x_t] = 0 under the exact posterior of a
                   small system is replaced by the exact orbit identity (Σ over a pair {x, T x} of ω (Tx - x) π = 0).
    Test 2 (invariance): the t = 0 kernel (1, 5, 20 steps) applied to half of the reference MCMC samples leaves the state frequencies and the energy histogram unchanged
                   within the floor (two independent halves of the reference), and the state-to-state transition counts of one step are reported.

    python tests/test_dw4_reflections.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.energies.double_well_energy import DoubleWellEnergy  # noqa: E402
from adjoint_samplers.utils import dw4_reflections as R  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, state_freq, state_ids, state_tv  # noqa: E402

torch.set_default_dtype(torch.float64)


def zero_com(n, scale=2.0):
    return R.recentre((scale * torch.randn(n, 4, 2))).reshape(n, 8)


def subspace_basis():
    P = torch.eye(8) - torch.kron(torch.ones(4, 4) / 4, torch.eye(2))
    lam, V = torch.linalg.eigh(P)
    return V[:, lam > 0.5]                                   # (8, 6) orthonormal basis of the zero-CoM subspace


def test_involution_and_energy():
    torch.manual_seed(0)
    en = DoubleWellEnergy(dim=8, n_particles=4)
    x = zero_com(50)
    for variant in (False, True):
        imgs, dE = R.images(x, variant)
        n_r = imgs.shape[1]
        assert n_r == (24 if variant else 12)
        for r in range(n_r):
            y = imgs[:, r]
            back = R.images(y, variant)[0][:, r]
            assert torch.allclose(back, x, atol=1e-9), (variant, r, (back - x).abs().max())
            assert torch.allclose(R.energy(y) - R.energy(x), dE[:, r], atol=1e-7), "closed-form energy change"
            assert torch.allclose(en.eval(y) - en.eval(x), dE[:, r], atol=1e-5), "repo energy"
    print("ok  T_r ∘ T_r = id for the 12 line reflections and the 12 point reflections; E(Tx) - E(x) = e(d'_il) - e(d_il) (closed form, own energy and the repo's)")


def test_volume():
    torch.manual_seed(1)
    Bm = subspace_basis()
    worst = 0.0
    for variant in (False, True):
        n_r = 24 if variant else 12
        for r in range(n_r):
            for _ in range(3):
                z = torch.randn(6) * 2
                f = lambda v: Bm.T @ R.images((Bm @ v)[None], variant)[0][0, r]
                J = torch.autograd.functional.jacobian(f, z)
                worst = max(worst, abs(abs(float(torch.linalg.det(J))) - 1.0))
    assert worst < 1e-7, worst
    print(f"ok  |det J| = 1 on the 6-dimensional zero-CoM subspace for all 12 + 12 maps (max deviation {worst:.1e}, numerical Jacobian)")


def group_element(kind, gen):
    """ (linear map on (N, 8) configurations, label) """
    if kind == "rotation":
        th = float(torch.rand(1, generator=gen) * 6.28)
        Q = torch.tensor([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return (lambda x: (x.reshape(-1, 4, 2) @ Q.T).reshape(-1, 8))
    if kind == "reflection":
        Q = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
        th = float(torch.rand(1, generator=gen) * 6.28)
        Rm = torch.tensor([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return (lambda x: (x.reshape(-1, 4, 2) @ (Rm @ Q).T).reshape(-1, 8))
    if kind == "permutation":
        p = torch.randperm(4, generator=gen)
        return (lambda x: x.reshape(-1, 4, 2)[:, p].reshape(-1, 8))
    raise ValueError(kind)


def test_equivariance():
    gen = torch.Generator().manual_seed(2)
    torch.manual_seed(2)
    x1, xt = zero_com(40), zero_com(40, 1.5)
    kappa, v = 0.3, 0.8
    out = []
    for variant in (False, True):
        h, _ = R.barker_h(x1, xt, kappa, v, variant)
        for kind in ("rotation", "reflection", "permutation"):
            for _ in range(3):
                g = group_element(kind, gen)
                hg, _ = R.barker_h(g(x1), g(xt), kappa, v, variant)
                err = float((hg - g(h)).abs().max())
                assert err < 1e-9, (variant, kind, err)
        a = torch.randn(1, 2).repeat(4, 1).reshape(1, 8)
        ht, _ = R.barker_h(x1 + a, xt, kappa, v, variant)
        assert float((ht - h).abs().max()) < 1e-9
        out.append(variant)
    print("ok  h(g x1; g x_t) = g h(x1; x_t) for random rotations, reflections and particle permutations; h unchanged by a translation of x1 (line and point variants)")


def test_zero_mean_identity():
    """ E_π[h | x_t] = 0: for a fixed map T_r, ω(x) π(x) (Tx - x) is odd under x -> Tx (ω(x) π(x) = π(x)π(Tx) / (π(x) + π(Tx)) is symmetric), so the pair {x, Tx} contributes 0 """
    torch.manual_seed(3)
    ref = torch.tensor(np.load(Path(__file__).resolve().parents[1] / "data" / "test_split_DW4.npy", allow_pickle=True)).double()
    x1 = R.recentre(ref[:60].reshape(-1, 4, 2)).reshape(-1, 8)                    # configurations near the wells: the weights are not saturated
    kappa, v = 0.4, 0.5
    xt = kappa * x1 + (v ** 0.5) * R.recentre(torch.randn(60, 4, 2)).reshape(-1, 8)
    imgs, _ = R.images(x1)
    worst = 0.0
    for r in range(12):
        y = imgs[:, r]
        lx = torch.nn.functional.logsigmoid(R.barker_logit(x1, xt, kappa, v)[0][:, r])
        ly = torch.nn.functional.logsigmoid(R.barker_logit(y, xt, kappa, v)[0][:, r])
        # pi~(x) w(x) = pi~(Tx) w(Tx) : log pi~(x) + log w_r(x) vs log pi~(y) + log w_r(y), w_r of y uses the same r (T_r is an involution)
        logpi = lambda z: -R.energy(z) - ((xt - kappa * z) ** 2).sum(-1) / (2 * v)
        lhs = logpi(x1) + lx
        rhs = logpi(y) + ly
        worst = max(worst, float((lhs - rhs).abs().max()))
    assert worst < 1e-8, worst
    print(f"ok  detailed balance of each r: π~(x) ω_r(x) = π~(T_r x) ω_r(T_r x) (max deviation {worst:.1e}) -> E_π[h | x_t] = 0 exactly (odd integrand, volume preserved, R fixed)")


def test_invariance_on_reference(n_steps=(1, 5, 20), repeats=25):
    root = Path(__file__).resolve().parents[1]
    ref = torch.tensor(np.load(root / "data" / "test_split_DW4.npy", allow_pickle=True)).double()
    N = len(ref) // 2
    g = torch.Generator().manual_seed(4)
    edges = np.quantile(R.energy(ref).numpy(), np.linspace(0, 1, 21))
    edges[0], edges[-1] = -1e9, 1e9
    hist = lambda x: torch.tensor(np.histogram(R.energy(x).numpy(), bins=edges)[0], dtype=torch.float64) / len(x)
    floor_s, floor_e, tv_s, tv_e = [], [], {s: [] for s in n_steps}, {s: [] for s in n_steps}
    for _ in range(repeats):
        perm = torch.randperm(len(ref), generator=g)
        A, Bset = ref[perm[:N]], ref[perm[N:2 * N]]
        fa, fb = state_freq(state_ids(A.float())), state_freq(state_ids(Bset.float()))
        floor_s.append(state_tv(fa, fb))
        floor_e.append(0.5 * float((hist(A) - hist(Bset)).abs().sum()))
        x = A.clone()
        done = 0
        for s in n_steps:
            for _ in range(s - done):
                x, _r = R.kernel_step(x, generator=g)
            done = s
            tv_s[s].append(state_tv(state_freq(state_ids(x.float())), fb))
            tv_e[s].append(0.5 * float((hist(x) - hist(Bset)).abs().sum()))
    p95s, p95e = float(np.percentile(floor_s, 95)), float(np.percentile(floor_e, 95))
    print(f"    floor (two independent halves of {N}): state TV mean {np.mean(floor_s):.4f}, p95 {p95s:.4f}; energy-histogram TV mean {np.mean(floor_e):.4f}, p95 {p95e:.4f}")
    for s in n_steps:
        print(f"    after {s:2d} step(s): state TV to the other half {np.mean(tv_s[s]):.4f}, energy-histogram TV {np.mean(tv_e[s]):.4f}")
        assert np.mean(tv_s[s]) < p95s and np.mean(tv_e[s]) < p95e, f"kernel changes the reference distribution after {s} steps"
    print("ok  the t = 0 kernel (1, 5, 20 steps) leaves the reference state frequencies and energy histogram unchanged within the floor")
    # connectivity of the state graph under one step (accepted moves, from the reference samples)
    x = ref.clone()
    new, r = R.kernel_step(x, generator=g)
    a, b = state_ids(x.float()), state_ids(new.float())
    moved = (new - R.recentre(x.reshape(-1, 4, 2)).reshape(-1, 8)).abs().sum(1) > 1e-9
    C = torch.zeros(len(KEYS), len(KEYS), dtype=torch.long)
    for i, j in zip(a[moved].tolist(), b[moved].tolist()):
        C[i, j] += 1
    print(f"    one step: {int(moved.sum())} of {len(x)} moves accepted ({100 * float(moved.double().mean()):.1f} %); state-to-state counts of accepted moves (rows: before, columns: after):")
    used = [k for k in range(len(KEYS)) if C[k].sum() + C[:, k].sum() > 0]
    print("    " + " " * 26 + "".join(f"{NAMES[KEYS[k]][:9]:>10}" for k in used))
    for i in used:
        print(f"    {NAMES[KEYS[i]][:25]:26}" + "".join(f"{int(C[i, j]):10d}" for j in used))
    return C


def test_rao_blackwell_matches_the_kernel(draws=300):
    """ the RB statistic D_b equals the expected one-step change of the class frequencies of the sampled kernel; on the reference it is compatible with 0 """
    from adjoint_samplers.utils.dw4_states import state_ids as sid
    root = Path(__file__).resolve().parents[1]
    ref = torch.tensor(np.load(root / "data" / "test_split_DW4.npy", allow_pickle=True)).double()[:2000]
    label = lambda z: sid(z.float())
    D, Fl, lab_x, K = R.rb_terms(ref, label, n_classes=len(KEYS))
    g = torch.Generator().manual_seed(5)
    acc = torch.zeros(len(KEYS), dtype=torch.float64)
    lab0 = torch.nn.functional.one_hot(lab_x, len(KEYS)).double()
    for _ in range(draws):
        new, _r = R.kernel_step(ref, True, g)
        acc += (torch.nn.functional.one_hot(label(new), len(KEYS)).double() - lab0).mean(0)
    emp = acc / draws
    rb = D.mean(0)
    se = D.std(0) / np.sqrt(len(D))
    sel = se > 0
    assert (torch.abs(emp - rb)[sel] < 4 * (se[sel] / 1 + 1e-4)).all(), (emp, rb)
    z = (rb[sel] / se[sel]).abs()
    assert float(z.max()) < 3, z
    assert abs(float(D.sum(1).abs().max())) < 1e-12
    print(f"ok  RB statistic = expected one-step change of the sampled kernel ({draws} draws; max |emp - RB| {float((emp - rb).abs().max()):.1e}); on the reference |Delta_b|/SE <= {float(z.max()):.1f} (< 3)")


if __name__ == "__main__":
    test_involution_and_energy()
    test_volume()
    test_equivariance()
    test_zero_mean_identity()
    test_invariance_on_reference()
    test_rao_blackwell_matches_the_kernel()
    print("all passed")
