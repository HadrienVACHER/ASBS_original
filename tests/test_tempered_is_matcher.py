""" AdjointVETemperedISMatcher (rules_train_weightfree.md), CPU, seconds.
    T1 the regression weights average to 1 over the whole buffer (self-normalisation); T2 alpha = 0 gives weights 1 (plain AS); T3 after an insertion the normalisation is recomputed
    on the truncated buffer; T4 the weights follow exp(alpha log W) up to the constant. """
import math, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.components.matcher import AdjointVETemperedISMatcher
from adjoint_samplers.components.buffer import BatchBuffer
from adjoint_samplers.components.sde import VESDE, ControlledSDE
from adjoint_samplers.components.state_cost import ZeroGradStateCost
from adjoint_samplers.components.term_cost import ScoreGradTermCost
from adjoint_samplers.components.model import FourierMLP
from adjoint_samplers.energies.dist_energy import DistEnergy
from adjoint_samplers.utils.dist_utils import GMMDK, Delta
D = 8; ref = VESDE(0.001, 2.236068); src = Delta(D); en = DistEnergy(GMMDK(D, 8, 8.0)); tc = ScoreGradTermCost(source=src, ref_sde=ref, energy=en)
sde = ControlledSDE(ref, FourierMLP(dim=D, activation=torch.nn.GELU(), num_layers=2, channels=32))
ok = True
def mk(alpha, size=1000):
    return AdjointVETemperedISMatcher(alpha=alpha, sde=sde, buffer=BatchBuffer(size), grad_term_cost=tc, grad_state_cost=ZeroGradStateCost(), resample_size=64, duplicates=1)
g = torch.Generator().manual_seed(0)
x1 = torch.randn(1000, D, generator=g) * 3; lw = torch.randn(1000, generator=g) * 2
m = mk(0.5); m.add_initial(x1, lw, torch.zeros(1000, D))
data = {k: torch.cat(v) for k, v in m.buffer.batches.items()}
m.prepare_target(data, "cpu"); w = m.sample_weight
ok &= abs(float(w.mean()) - 1) < 1e-6; print("T1 mean weight over the buffer", float(w.mean()))
r = torch.exp(0.5 * lw.double()); r = r / r.mean(); ok &= float((w.double() - r).abs().max()) < 1e-5; print("T4 max |w - exp(a lw)/mean|", float((w.double() - r).abs().max()))
m0 = mk(0.0); m0.add_initial(x1, lw, torch.zeros(1000, D)); m0.prepare_target(data, "cpu"); ok &= float((m0.sample_weight - 1).abs().max()) < 1e-6; print("T2 alpha=0 max |w - 1|", float((m0.sample_weight - 1).abs().max()))
ts = torch.linspace(0, 1, 21)
m.populate_buffer(torch.zeros(200, D), ts, False)                                          # buffer 1000 + 200 -> truncated to the last 1000
lwb = torch.cat(m.buffer.batches["logw"])[-1000:].double(); ref_norm = float(torch.logsumexp(0.5 * lwb, 0) - math.log(1000))
ok &= abs(m._lognorm - ref_norm) < 1e-9; print("T3 normalisation after insertion", m._lognorm, ref_norm, "stats", m.pop_stats())
print("ALL PASS" if ok else "SOME FAIL"); sys.exit(0 if ok else 1)
