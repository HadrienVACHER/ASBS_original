""" AdjointVEOracleDSMMatcher (rules_oracle_buffer.md, arm o-DSM), CPU, seconds.
    T1 the conditional mean of the DSM target given x_t is the exact optimal controller of a 1-d Gaussian target (closed form), and the weight is Var(x_t | x1);
    T2 simulating the SDE (200 Euler steps, as the sampler) with that exact controller gives the target mean and variance;
    T3 the buffer holds exact draws, refreshed at every call, adjoint1 unused (zeros); T4 the controller-regression weights are bounded. """
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.components.matcher import AdjointVEOracleDSMMatcher
from adjoint_samplers.components.buffer import BatchBuffer
from adjoint_samplers.components.sde import VESDE, ControlledSDE, sdeint
from adjoint_samplers.components.state_cost import ZeroGradStateCost
from adjoint_samplers.components.term_cost import ScoreGradTermCost
from adjoint_samplers.components.model import FourierMLP
from adjoint_samplers.energies.dist_energy import DistEnergy
from adjoint_samplers.utils.dist_utils import GMMDK, Delta
torch.manual_seed(0); ok = True
D = 8; ref = VESDE(0.001, 1.602001); V = ref.total_var; src = Delta(D); en = DistEnergy(GMMDK(D, 8, 6.0)); tc = ScoreGradTermCost(source=src, ref_sde=ref, energy=en)
sde = ControlledSDE(ref, FourierMLP(dim=D, activation=torch.nn.GELU(), num_layers=2, channels=32))
m = AdjointVEOracleDSMMatcher(sde=sde, buffer=BatchBuffer(1000), grad_term_cost=tc, grad_state_cost=ZeroGradStateCost(), resample_size=64, duplicates=1)
ts = torch.linspace(0, 1, 201)
m.populate_buffer(torch.zeros(500, D), ts, False); m.populate_buffer(torch.zeros(500, D), ts, False)
b = {k: torch.cat(v) for k, v in m.buffer.batches.items()}
ok &= len(b["x1"]) == 1000 and float(b["adjoint1"].abs().max()) == 0.0 and not torch.equal(b["x1"][:500], b["x1"][500:]); print("T3 buffer", b["x1"].shape, "adjoint1 zeros, two distinct refreshes")
# 1-d Gaussian target N(mu, tau2): exact controller and weights
mu, tau2 = 2.0, 0.25; N = 400000
x1 = mu + tau2 ** 0.5 * torch.randn(N, 1); x0 = torch.zeros(N, 1)
class G1(torch.nn.Module): pass
m1 = AdjointVEOracleDSMMatcher.__new__(AdjointVEOracleDSMMatcher); m1.sde = ControlledSDE(ref, torch.nn.Identity()); m1.t_focus = 0.0; m1.t_focus_max = 0.15
t0 = 0.05
class FixT(AdjointVEOracleDSMMatcher):
    def sample_t(self, x): return torch.full((x.shape[0], 1), t0)
m1.__class__ = FixT
(t, xt), tg = m1.prepare_target({"x0": x0, "x1": x1}, "cpu"); w = m1.sample_weight
r = float(ref._diffsquare_integral(torch.tensor(t0)) / V); var = V * r * (1 - r)
ok &= abs(float(w[0]) - var) < 1e-5; print("T1a weight", float(w[0]), "Var(x_t|x1)", var)
for xv in (0.5, 1.0, 1.5):                                                                  # E[target | x_t ~ xv] vs closed form
    sel = (xt[:, 0] - xv).abs() < 0.03; est = float(tg[sel].mean())
    Ex1 = mu + (r * tau2) / (r * r * tau2 + var) * (xv - r * mu); star = (Ex1 - xv) / (V * (1 - r))
    ok &= abs(est - star) < 0.12 * (1 + abs(star)); print(f"T1b x={xv}: E[target|x] {est:.3f} vs exact u* {star:.3f}")
class Star(torch.nn.Module):
    def forward(self, t, x):
        rr = ref._diffsquare_integral(t) / V; vv = V * rr * (1 - rr)
        return (mu + (rr * tau2) / (rr * rr * tau2 + vv).clamp_min(1e-12) * (x - rr * mu) - x) / (V * (1 - rr).clamp_min(1e-7))
cs = ControlledSDE(ref, Star()); x = torch.zeros(100000, 1)
_, xe = sdeint(cs, x, ts, only_boundary=True)
ok &= abs(float(xe.mean()) - mu) < 0.02 and abs(float(xe.var()) - tau2) < 0.01; print("T2 sampled mean/var", float(xe.mean()), float(xe.var()), "target", mu, tau2)
ok &= float(var) <= V / 4 + 1e-9; print("T4 max weight V/4 =", V / 4)
print("ALL PASS" if ok else "SOME FAIL"); sys.exit(0 if ok else 1)
