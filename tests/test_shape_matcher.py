""" AdjointVEShapeAS/DSMMatcher (rules_shape_correction.md), CPU, seconds. T1 (V) the buffer variance per mode is 1.44, means exact, masses exact; T2 (M) means shifted by 0.3 along the fixed direction, variance 1, masses exact;
    T3 the measure shape_measures recovers rho_V = 1.44 and rho_M = 1 on the buffers, 1 and 0 on exact draws; T4 fresh draws at every call; T5 AS adjoint = TRUE energy gradient at the buffer points; DSM adjoint unused. """
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gmm_dk
from adjoint_samplers.components.matcher import AdjointVEShapeASMatcher, AdjointVEShapeDSMMatcher
from adjoint_samplers.components.buffer import BatchBuffer
from adjoint_samplers.components.sde import VESDE, ControlledSDE
from adjoint_samplers.components.state_cost import ZeroGradStateCost
from adjoint_samplers.components.term_cost import ScoreGradTermCost
from adjoint_samplers.components.model import FourierMLP
from adjoint_samplers.energies.dist_energy import DistEnergy
from adjoint_samplers.utils.dist_utils import GMMDK, Delta
torch.manual_seed(0); ok = True; D = 128
ref = VESDE(0.001, 1.602001); src = Delta(D); en = DistEnergy(GMMDK(D, 8, 6.0)); tc = ScoreGradTermCost(source=src, ref_sde=ref, energy=en)
sde = ControlledSDE(ref, FourierMLP(dim=D, activation=torch.nn.GELU(), num_layers=2, channels=32)); ts = torch.linspace(0, 1, 11)
def mk(cls, mode, n=20000):
    return cls(mode=mode, sde=sde, buffer=BatchBuffer(n), grad_term_cost=tc, grad_state_cost=ZeroGradStateCost(), resample_size=64, duplicates=1)
res = {}
for mode in ("V", "M"):
    m = mk(AdjointVEShapeASMatcher, mode)
    for _ in range(4): m.populate_buffer(torch.zeros(5000, D), ts, False)
    b = {k: torch.cat(v) for k, v in m.buffer.batches.items()}; x = b["x1"].double().numpy(); r = gmm_dk.shape_measures(x, D, 8, 6.0); res[mode] = r
    print(mode, "rho_V", round(r["rho_V"], 3), "rho_M", round(r["rho_M"], 3), "counts", r["counts"])
    ok &= (abs(r["rho_V"] - 1.44) < 0.03 and abs(r["rho_M"]) < 0.25) if mode == "V" else (abs(r["rho_V"] - 1.0) < 0.03 and abs(r["rho_M"] - 1.0) < 0.25)
    gt = en.dist.score(b["x1"][:50]) if hasattr(en.dist, "score") else None
    ok &= float((b["adjoint1"][:50] - tc(b["x1"][:50])).abs().max()) < 1e-4; print("T5 adjoint1 = the term-cost gradient (true target) at the buffer points")
    ok &= not torch.equal(b["x1"][:5000], b["x1"][5000:10000])
mu = gmm_dk.means(D, 8, 6.0); w = gmm_dk.weights(8); comp = torch.multinomial(w, 20000, replacement=True); xe = (mu[comp] + torch.randn(20000, D)).double().numpy()
re = gmm_dk.shape_measures(xe, D, 8, 6.0); print("exact rho_V", round(re["rho_V"], 3), "rho_M", round(re["rho_M"], 3)); ok &= abs(re["rho_V"] - 1) < 0.02 and abs(re["rho_M"]) < 0.25
md = mk(AdjointVEShapeDSMMatcher, "M", 1000); md.populate_buffer(torch.zeros(500, D), ts, False); ok &= float(torch.cat(md.buffer.batches["adjoint1"]).abs().max()) == 0
print("ALL PASS" if ok else "SOME FAIL"); sys.exit(0 if ok else 1)
