""" Quench of DW-4 configurations: analytic gradient / Hessian against autograd of the repo energy, convergence to ||∇E|| < 1e-4, energy decrease, invariance of the state.
    python tests/test_dw4_quench.py """
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.energies.double_well_energy import DoubleWellEnergy  # noqa: E402
from adjoint_samplers.utils.dw4_quench import energy_grad_hess, quench  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, state_freq, state_ids  # noqa: E402


def test_derivatives():
    torch.manual_seed(0)
    en = DoubleWellEnergy(dim=8, n_particles=4)
    x = (3 * torch.randn(5, 8)).double()
    E, g, H = energy_grad_hess(x)
    xr = x.clone().requires_grad_(True)
    Er = en.eval(xr)
    gr = torch.autograd.grad(Er.sum(), xr)[0]
    assert torch.allclose(E, Er.detach(), atol=1e-5) and torch.allclose(g, gr, atol=1e-5), (E - Er, g - gr)
    Hr = torch.stack([torch.autograd.functional.hessian(lambda v: en.eval(v[None])[0], x[i]) for i in range(5)])
    assert torch.allclose(H, Hr, atol=1e-4), (H - Hr).abs().max()
    print("ok  analytic energy, gradient and Hessian match autograd of the repo energy")


def test_quench():
    torch.manual_seed(1)
    ref = torch.tensor(np.load(Path(__file__).resolve().parents[1] / "data" / "test_split_DW4.npy", allow_pickle=True)).float()[:1500]
    t0 = time.time()
    q, gn, its = quench(ref)
    E0, E1 = energy_grad_hess(ref.double())[0], energy_grad_hess(q.double())[0]
    assert float(gn.max()) < 1e-3 and (E1 <= E0 + 1e-6).all(), (float(gn.max()), float((E1 - E0).max()))
    conv = float((gn < 1e-4).float().mean())
    # a rigid rotation / translation does not change the quenched state
    th = 0.9
    R = torch.tensor([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]).float()
    q2, _, _ = quench((ref.reshape(-1, 4, 2) @ R.T + 2.0).reshape(-1, 8))
    assert torch.equal(state_ids(q2), state_ids(q))
    from adjoint_samplers.utils.dw4_states import pair_distances
    d = pair_distances(q).flatten()
    fr, fq = state_freq(state_ids(ref)), state_freq(state_ids(q))
    print(f"ok  quench of 1500 reference samples in {time.time() - t0:.1f} s ({its} iterations max): converged ||g|| < 1e-4 for {100 * conv:.1f} %, max ||g|| {float(gn.max()):.1e}, "
          f"energy mean {float(E0.mean()):.1f} -> {float(E1.mean()):.1f}; quenched distances: share in [3.6, 4.4] = {float(((d > 3.6) & (d < 4.4)).float().mean()):.4f}")
    print("    raw vs quenched state frequencies:", {NAMES[k][:16]: (round(float(a), 3), round(float(b), 3)) for k, a, b in zip(KEYS, fr, fq) if a > 0.005 or b > 0.005})


if __name__ == "__main__":
    test_derivatives()
    test_quench()
    print("all passed")
