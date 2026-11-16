""" Task 4 of rules_reference_check.md (no rule): errors of the basin proportions recomputed with p_PT (parallel tempering, results/reference_check_pt.json) as the truth.
    Rows: AS ed8bb67 and AS 94252 (model_samples, binomial SE), endpoint IS (gaussian s = 1 and t5 s = 1, 20000 draws, seed 7, EM components fitted on the AS ed8bb67 samples), path-space IS and path-space IMH (controller 94252, 10000 paths, seed 0, 100 chains x 99 steps).
    Columns: log(p_row / p_PT) with SE_comb = sqrt((SE_row / p_row)^2 + (SE_PT / p_PT)^2), the sum over the 5 basins of |log ratio|, and last the same sum computed against the reference.      python scripts/dw4_reference_rows.py
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dw4_division_b as b  # noqa: E402
import dw4_division_step0 as s0  # noqa: E402
import dw4_division_trunc as tr  # noqa: E402
import dw4_is_inflate as isf  # noqa: E402
import dw4_path_weights as pw  # noqa: E402
from adjoint_samplers.components.model import EGNN_dynamics  # noqa: E402
from adjoint_samplers.components.sde import ControlledSDE, GraphVESDE  # noqa: E402
from adjoint_samplers.utils.dw4_quench import quench  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, state_ids  # noqa: E402
from adjoint_samplers.utils.train_utils import get_timesteps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
t0 = time.time()


def onehot_of(x, types):
    out = []
    for lo in range(0, len(x), 20000):
        q, _, _ = quench(x[lo:lo + 20000].double())
        out.append(state_ids(q.double()))
    lab = torch.cat(out)
    return torch.stack([(lab == k).double() for k in types], 1)


def weighted(oh, w, nb=500, seed=1):
    N = len(w)
    p = (w[:, None] * oh).sum(0) / w.sum()
    g = torch.Generator().manual_seed(seed)
    boots = torch.stack([(lambda i: (w[i][:, None] * oh[i]).sum(0) / w[i].sum())(torch.randint(N, (N,), generator=g)) for _ in range(nb)])
    return p, boots.std(0)


def main():
    ref, q, lab_r, types, mu8 = b.load_reference()
    names = [NAMES[KEYS[k]] for k in types]
    pj = json.load(open(ROOT / "results" / "reference_check_pt.json"))
    p_pt, se_pt, p_ref = (torch.tensor(pj[k], dtype=torch.float64) for k in ("p_pt", "se_pt", "p_ref"))
    rows = {}
    for nm, f in (("AS ed8bb67", "dw4_e1500_ed8bb67_samples.pt"), ("AS 94252", "dw4_e1500_94252_samples.pt")):
        x = torch.load(ROOT / "results" / f, map_location="cpu", weights_only=False)["model_samples"].double()
        p = onehot_of(x, types).mean(0)
        rows[nm] = (p, torch.sqrt(p * (1 - p) / len(x)))
        print(f"  row {nm} done ({time.time() - t0:.0f} s)", flush=True)
    fit = tr.load_as_fit(str(ROOT / "results" / "dw4_e1500_ed8bb67_samples.pt"))["fit"]
    comps = [fit[k] for k in types]
    print(f"  EM components fitted ({time.time() - t0:.0f} s)", flush=True)
    isf.N = 20000
    for fam, nm in (("gauss", "endpoint IS gauss s=1"), ("t5", "endpoint IS t5 s=1")):
        gen = torch.Generator().manual_seed(7)
        x, lr = isf.build(fam, 1.0, comps, gen)
        lw = -s0.energy(x) - lr
        w = torch.exp(lw - lw.max())
        rows[nm] = weighted(onehot_of(x, types), w)
        print(f"  row {nm} done: ESS {float(w.sum() ** 2 / (w ** 2).sum()):.0f} ({time.time() - t0:.0f} s)", flush=True)
    torch.manual_seed(0)
    ref_sde = GraphVESDE(n_particles=4, spatial_dim=2, sigma_max=2, sigma_min=0.001)
    ctrl = EGNN_dynamics(n_particles=4, spatial_dim=2, hidden_nf=128, n_layers=5, act_fn=torch.nn.SiLU(), recurrent=True, tanh=True, attention=True, condition_time=True, agg="sum")
    ctrl.load_state_dict(torch.load(ROOT / "results" / "dw4_e1500_94252_controller.pt", map_location="cpu", weights_only=False)["controller"])
    ctrl.eval()
    ts = get_timesteps(0.0, 1.0, steps=100)
    xs, Ss = [], []
    for lo in range(0, 10000, 2000):
        x1, S, V = pw.simulate(lambda t, x: ctrl(t, x), ref_sde.diff, ts, torch.zeros(2000, 8), ref_sde.randn_like, lambda x: ref_sde.propagate(x, torch.zeros_like(x)))
        xs.append(x1)
        Ss.append(S)
        print(f"  paths {lo + 2000}/10000 ({time.time() - t0:.0f} s)", flush=True)
    x1, S = torch.cat(xs).double(), torch.cat(Ss)
    lw = -s0.energy(x1) + 0.5 * (x1 ** 2).sum(-1) / V + S
    w = torch.exp(lw - lw.max())
    oh = onehot_of(x1, types)
    rows["path-space IS"] = weighted(oh, w)
    lab = oh.argmax(1)
    C, T, BURN = 100, 99, 50
    gen = torch.Generator().manual_seed(2)
    idx = torch.arange(C * (T + 1)).reshape(C, T + 1)
    state, labs = idx[:, 0], []
    for t in range(1, T + 1):
        prop = idx[:, t]
        ok = torch.rand(C, generator=gen, dtype=torch.float64) < torch.exp((lw[prop] - lw[state]).clamp(max=0))
        state = torch.where(ok, prop, state)
        if t >= BURN:
            labs.append(oh[state])
    cm = torch.stack(labs, 0).mean(0)                                                              # (C, 5): chain means of the indicators
    rows["path-space IMH"] = (cm.mean(0), cm.std(0) / math.sqrt(C))
    print(f"  path rows done ({time.time() - t0:.0f} s)", flush=True)

    se_log_pt = se_pt / p_pt
    print("\np (SE):  truth p_PT = " + ", ".join(f"{p:.4f} ({s:.4f})" for p, s in zip(p_pt, se_pt)) + "\n         reference  = " + ", ".join(f"{p:.4f}" for p in p_ref))
    hdr = " | ".join(f"{n[:12]:>15}" for n in names)
    print(f"\nlog(p_row / p_PT) +- SE_comb per basin ({hdr}) | sum |log| vs p_PT | sum |log| vs reference")
    for nm, (p, se) in rows.items():
        lr_ = torch.log(p.clamp(min=1e-12) / p_pt)
        se_c = torch.sqrt((se / p.clamp(min=1e-12)) ** 2 + se_log_pt ** 2)
        s_ref = torch.log(p.clamp(min=1e-12) / p_ref).abs().sum()
        print(f"{nm:24} " + " | ".join(f"{float(l):+6.3f} +-{float(s):5.3f}" for l, s in zip(lr_, se_c)) + f" | {float(lr_.abs().sum()):6.3f} | {float(s_ref):6.3f}")
    print(f"\ntotal {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
