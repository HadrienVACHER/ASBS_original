""" Tests of AdjointVENewtonSteinMatcher and AdjointVEDampedMatcher (rules_ns_damping.md), CPU, < 1 min.
    python tests/test_newton_stein_matcher.py
    T1  h_hat = 1/nu_1 gives exactly the AS target (same random draws).
    T2  algebraic identity Y_hat = Y + (Lambda_t - 1) (Y - (X1 - X_t)/nu_{1|t}) (float64, t <= 0.95).
    T3  zero mean under the exact target (GMM-dk (32, 8, 4), sigma_max rule): (X1, X_t) drawn from pi x base bridge; 192 statistics E[(Y_hat - Y)_j f(t, X_t)] with
        f in {1, X_t,j, four time bins}; PASS if max |z| < 4.5.
    T4  positive control: X1 from a shifted law (means + 0.3): max |z| > 10 (the test detects a non-zero mean).
    T5  damped matcher: the frozen copy does not move when the controller does; target = (1 - eta) u_prev + eta Y.
"""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gmm_dk
from adjoint_samplers.components.matcher import AdjointVEMatcher, AdjointVENewtonSteinMatcher, AdjointVEDampedMatcher
from adjoint_samplers.components.sde import VESDE, ControlledSDE
from adjoint_samplers.components.state_cost import ZeroGradStateCost
from adjoint_samplers.components.term_cost import ScoreGradTermCost
from adjoint_samplers.components.model import FourierMLP
from adjoint_samplers.energies.dist_energy import DistEnergy
from adjoint_samplers.utils.dist_utils import GMMDK, Delta

D, K, DELTA = 32, 8, 4.0
SIG = gmm_dk.sigma_max_rule(D, DELTA)
ref = VESDE(sigma_min=0.001, sigma_max=SIG)
src = Delta(D)
energy = DistEnergy(GMMDK(D, K, DELTA))
tc = ScoreGradTermCost(source=src, ref_sde=ref, energy=energy)
ctrl = FourierMLP(dim=D, activation=torch.nn.GELU(), num_layers=4, channels=128)
sde = ControlledSDE(ref, ctrl)
nu1 = float(ref.total_var)
ok_all = True


def make(cls, **kw):
    return cls(sde=sde, buffer=None, grad_term_cost=tc, grad_state_cost=ZeroGradStateCost(), **kw)


def data_from(x1):
    return {"x0": torch.zeros_like(x1), "x1": x1, "adjoint1": tc(x1)}


def report(name, ok, msg):
    global ok_all
    ok_all &= ok
    print(f"{name}: {'PASS' if ok else 'FAIL'}  {msg}", flush=True)


# T1
x1 = gmm_dk.sample_exact(4000, D, K, DELTA, torch.Generator().manual_seed(1), full=True)[0].float()
d = data_from(x1)
torch.manual_seed(3); (t_a, xt_a), y_as = make(AdjointVEMatcher).prepare_target(d, "cpu")
torch.manual_seed(3); (t_n, xt_n), y_ns = make(AdjointVENewtonSteinMatcher, h_hat=1.0 / nu1).prepare_target(d, "cpu")
err = float((y_as - y_ns).abs().max() / y_as.abs().max())
report("T1 h_hat = 1/nu_1 -> AS", bool(torch.equal(t_a, t_n) and torch.equal(xt_a, xt_n) and err < 1e-5), f"max rel diff {err:.2e}")

# T2
torch.manual_seed(4); (t, xt), y_hat = make(AdjointVENewtonSteinMatcher, h_hat=1.0).prepare_target(d, "cpu")
m = (t[:, 0] <= 0.95)
kap = (ref._diffsquare_integral(t.double()) / nu1)
lam = 1.0 / (nu1 * (1 - kap) * 1.0 + kap)
Y = -d["adjoint1"].double()
y_bm = (x1.double() - xt.double()) / (nu1 * (1 - kap))
rhs = Y + (lam - 1) * (Y - y_bm)
err = float(((y_hat.double() - rhs).abs().max(1).values[m]).max() / rhs[m].abs().max())
report("T2 algebraic identity", err < 1e-5, f"max rel diff {err:.2e} on {int(m.sum())} rows with t <= 0.95")


def zstats(x1, seed):
    torch.manual_seed(seed)
    dd = data_from(x1)
    (t, xt), y_hat = make(AdjointVENewtonSteinMatcher, h_hat=1.0).prepare_target(dd, "cpu")
    diff = (y_hat + dd["adjoint1"]).double()                       # Y_hat - Y
    fs = [torch.ones_like(t), xt.double()] + [((t >= a) & (t < a + 0.25)).double() for a in (0.0, 0.25, 0.5, 0.75)]
    zs = []
    for f in fs:
        v = diff * f.double()
        zs.append(v.mean(0) / (v.std(0) / len(v) ** 0.5))
    return torch.cat(zs).abs().max().item(), len(torch.cat(zs))


# T3
N = 200_000
x1 = gmm_dk.sample_exact(N, D, K, DELTA, torch.Generator().manual_seed(11), full=True)[0].float()
zmax, ns = zstats(x1, 12)
report("T3 zero mean under exact pi", zmax < 4.5, f"max |z| = {zmax:.2f} over {ns} statistics (N = {N})")

# T4
zmax, _ = zstats(x1 + 0.3, 13)
report("T4 positive control (shifted law)", zmax > 10, f"max |z| = {zmax:.1f}")

# T5
dm = make(AdjointVEDampedMatcher, eta=0.5)
dm.set_progress(0, 10)
with torch.no_grad():
    for p in ctrl.parameters():
        p.add_(0.05 * torch.randn_like(p))
d = data_from(x1[:2000])
torch.manual_seed(5); (t, xt), y_d = dm.prepare_target(d, "cpu")
torch.manual_seed(5); (t2, xt2), y_a = make(AdjointVEMatcher).prepare_target(d, "cpu")
with torch.no_grad():
    expect = 0.5 * dm.u_prev(t, xt) + 0.5 * y_a
    moved = float((dm.u_prev(t, xt) - ctrl(t, xt)).abs().max())
err = float((y_d - expect).abs().max())
frozen = all(not p.requires_grad for p in dm.u_prev.parameters())
report("T5 damped target", err < 1e-5 and moved > 1e-3 and frozen, f"max diff {err:.1e}; frozen copy differs from the moved controller by {moved:.3f}")

print("ALL PASS" if ok_all else "SOME FAIL")
sys.exit(0 if ok_all else 1)
