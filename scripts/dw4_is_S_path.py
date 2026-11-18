""" Path-space IS of the AS controller 94252 in the S metric (report only; local, chunks of a few thousand paths so that one call stays well under a minute).
    Simulation and weights are those of scripts/dw4_path_weights.py (same functions, 99 Euler steps, discrete Girsanov weights); this script only saves (x1, log w) for a given seed and size.
      python scripts/dw4_is_S_path.py --seed 10 --n 3000 --out PATH.pt          then   python scripts/dw4_is_S_report.py """
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dw4_path_weights as pw  # noqa: E402
import dw4_division_step0 as s0  # noqa: E402
from adjoint_samplers.components.model import EGNN_dynamics  # noqa: E402
from adjoint_samplers.components.sde import GraphVESDE  # noqa: E402
from adjoint_samplers.utils.train_utils import get_timesteps  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.manual_seed(a.seed)
ref_sde = GraphVESDE(n_particles=4, spatial_dim=2, sigma_max=2, sigma_min=0.001)
ctrl = EGNN_dynamics(n_particles=4, spatial_dim=2, hidden_nf=128, n_layers=5, act_fn=torch.nn.SiLU(), recurrent=True, tanh=True, attention=True, condition_time=True, agg="sum")
ctrl.load_state_dict(torch.load(pw.ROOT / "results" / "dw4_e1500_94252_controller.pt", map_location="cpu", weights_only=False)["controller"])
ctrl.eval()
ts = get_timesteps(0.0, 1.0, steps=100)
t0 = time.time()
x1, S, V = pw.simulate(lambda t, x: ctrl(t, x), ref_sde.diff, ts, torch.zeros(a.n, 8), ref_sde.randn_like, lambda x: ref_sde.propagate(x, torch.zeros_like(x)))
x1 = x1.double()
lw = -s0.energy(x1) + 0.5 * (x1 ** 2).sum(-1) / V + S
torch.save({"x1": x1, "lw": lw, "seed": a.seed}, a.out)
print(f"seed {a.seed}: {a.n} paths in {time.time() - t0:.0f} s -> {a.out}")
