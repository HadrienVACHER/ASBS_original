""" AdjointVEDartBankMatcher (rules_selftrain_dartbank.md), CPU, seconds.
    T1 the bank initialised by quench from the base law holds the K true minima (GMM-dk (8, 8, 8)); T2 the end points entering the buffer are the DARTED points (not the raw SDE end points) and are
    unchanged in mean-field terms only through darting (they differ from the raw ones); T3 the stored adjoint1 is the energy gradient (grad_term_cost) recomputed at the STORED (moved) points;
    T4 the cost counter grows by the darting cost of each batch; T5 a batch from one single mode ends up covering the other modes (jumps between bank minima) with S = 10 sweeps. """
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adjoint_samplers.components.matcher as mm
from adjoint_samplers.components.matcher import AdjointVEDartBankMatcher
from adjoint_samplers.components.buffer import BatchBuffer
from adjoint_samplers.components.sde import VESDE, ControlledSDE
from adjoint_samplers.components.state_cost import ZeroGradStateCost
from adjoint_samplers.components.term_cost import ScoreGradTermCost
from adjoint_samplers.components.model import FourierMLP
from adjoint_samplers.energies.dist_energy import DistEnergy
from adjoint_samplers.utils.dist_utils import GMMDK, Delta
D = 8; ref = VESDE(0.001, 2.236068); src = Delta(D); dist = GMMDK(D, 8, 8.0); en = DistEnergy(dist); tc = ScoreGradTermCost(source=src, ref_sde=ref, energy=en)
sde = ControlledSDE(ref, FourierMLP(dim=D, activation=torch.nn.GELU(), num_layers=2, channels=32))
m = AdjointVEDartBankMatcher(sweeps=10, n_bank_starts=400, sde=sde, buffer=BatchBuffer(2000), grad_term_cost=tc, grad_state_cost=ZeroGradStateCost(), resample_size=256, duplicates=1)
ok = True
mu = dist.means.double().numpy()
# a raw batch concentrated on mode 0 (the controller is irrelevant: sdeint is replaced by a fixed batch)
g = torch.Generator().manual_seed(0); raw = (torch.tensor(mu[0]) + torch.randn(256, D, generator=g, dtype=torch.float64)).float()
real = mm.sdeint
mm.sdeint = lambda sde_, x0_, ts_, only_boundary=True: (x0_, raw)
try:
    m.populate_buffer(torch.zeros(256, D), torch.linspace(0, 1, 21), False)
finally:
    mm.sdeint = real
ok &= m.bank.shape[0] == 8; print("T1 bank size", m.bank.shape[0])
d_ = np.linalg.norm(m.bank[:, None, :] - mu[None, :, :], axis=2).min(1); ok &= float(d_.max()) < 0.5; print("T1 max distance bank -> nearest true mean", float(d_.max()))
x1 = torch.cat(m.buffer.batches["x1"]); a1 = torch.cat(m.buffer.batches["adjoint1"])
ok &= float((x1 - raw).abs().max()) > 1.0; print("T2 max |stored - raw|", float((x1 - raw).abs().max()))
ok &= float((a1 - tc(x1)).abs().max()) < 1e-5; print("T3 max |adjoint1 - grad at stored points|", float((a1 - tc(x1)).abs().max()))
c0 = sum(m.cost.values()); ok &= m.cost["dart"] > 0 and m.cost["bank_init"] > 0 and c0 > 0; print("T4 cost", m.cost)
cells = (-2 * x1.double().numpy() @ mu.T + (mu ** 2).sum(1)[None, :]).argmin(1); cov = int((np.bincount(cells, minlength=8) > 0).sum()); ok &= cov >= 6; print("T5 modes visited after darting from one mode:", cov, "of 8")
# T6 freeze_step: the step is tuned on the first batch only and reused (same value, cheaper later batches); default False tunes on every batch
mf = AdjointVEDartBankMatcher(sweeps=10, n_bank_starts=400, freeze_step=True, sde=sde, buffer=BatchBuffer(2000), grad_term_cost=tc, grad_state_cost=ZeroGradStateCost(), resample_size=256, duplicates=1)
costs = []
for _ in range(2):
    mm.sdeint = lambda sde_, x0_, ts_, only_boundary=True: (x0_, raw)
    try:
        mf.populate_buffer(torch.zeros(256, D), torch.linspace(0, 1, 21), False)
    finally:
        mm.sdeint = real
    costs.append(mf._stats["sbank/cost_dart_batch"])
ok &= mf._hs is not None and costs[1] < costs[0] - 30000 and m._hs is None; print("T6 frozen step", mf._hs, "costs first / second batch", costs, "default tunes every batch:", m._hs is None)
print("ALL PASS" if ok else "SOME FAIL"); sys.exit(0 if ok else 1)
