""" DW-4 parallel tempering on exp(-E)/Z (independent estimate of the basin proportions). Protocol and validity rules: rules_reference_check.md (Task 1), committed before the run.

      python scripts/dw4_pt.py [--steps 200000] [--out results/pt_samples_T1.pt]

    64 vectorised chains x 8 temperatures T_k = 5^(k/7), E = DoubleWellEnergy (the training energy), Gaussian random walk in the zero-CoM subspace (deterministic basis s0.B), step per temperature tuned during the burn-in (20 % of the steps)
    to 30 % acceptance then frozen, swaps of neighbouring temperatures every 10 steps, overdispersed initialisation (32 chains from the reference, 32 from the AS samples ed8bb67). T = 1 states are saved every 100 steps after the burn-in,
    labelled by the repo quench + state_ids.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dw4_division_step0 as s0  # noqa: E402
from adjoint_samplers.energies.double_well_energy import DoubleWellEnergy  # noqa: E402
from adjoint_samplers.utils.dw4_quench import quench  # noqa: E402
from adjoint_samplers.utils.dw4_states import state_ids  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C, NT, TARGET, SAVE_EVERY, SWAP_EVERY, ADAPT_EVERY = 64, 8, 0.30, 100, 10, 50


def centre(x):
    z = x.reshape(-1, 4, 2)
    return (z - z.mean(1, keepdim=True)).reshape(-1, 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--out", default=str(ROOT / "results" / "pt_samples_T1.pt"))
    ap.add_argument("--as-samples", default=str(ROOT / "results" / "dw4_e1500_ed8bb67_samples.pt"))
    a = ap.parse_args()
    burn = int(0.2 * a.steps)
    g = torch.Generator().manual_seed(0)
    en = DoubleWellEnergy(dim=8, n_particles=4)
    E = lambda x: en.eval(x.reshape(-1, 8)).reshape(x.shape[:-1])
    B = s0.B
    T = 5.0 ** (torch.arange(NT, dtype=torch.float64) / 7)
    ref = centre(torch.tensor(np.load(ROOT / "data" / "test_split_DW4.npy", allow_pickle=True), dtype=torch.float64))
    asm = centre(torch.load(a.as_samples, map_location="cpu", weights_only=False)["model_samples"].double())
    x0 = torch.cat([ref[torch.randperm(len(ref), generator=g)[:C // 2]], asm[torch.randperm(len(asm), generator=g)[:C // 2]]])      # (64, 8): chains 0-31 from the reference, 32-63 from AS
    x = x0[None].repeat(NT, 1, 1).clone()                                                                                          # (8, 64, 8)
    Ex = E(x)
    logs = torch.log(0.1 * torch.sqrt(T))
    acc_win, acc_tot, n_tot = torch.zeros(NT, dtype=torch.float64), torch.zeros(NT, dtype=torch.float64), 0
    sw_acc, sw_try = torch.zeros(NT - 1, dtype=torch.float64), torch.zeros(NT - 1, dtype=torch.float64)
    saved, savedE = [], []
    t0 = time.time()
    w = 0
    for step in range(1, a.steps + 1):
        s = torch.exp(logs)
        xi = torch.randn(NT, C, 6, generator=g, dtype=torch.float64)
        y = x + s[:, None, None] * (xi @ B.T)
        Ey = E(y)
        ok = torch.log(torch.rand(NT, C, generator=g, dtype=torch.float64)) < -(Ey - Ex) / T[:, None]
        x = torch.where(ok[..., None], y, x)
        Ex = torch.where(ok, Ey, Ex)
        acc_win += ok.double().mean(1)
        if step <= burn and step % ADAPT_EVERY == 0:
            w += 1
            logs = logs + 0.5 * (acc_win / ADAPT_EVERY - TARGET) / (1 + w / 10) ** 0.5
            acc_win.zero_()
        elif step > burn:
            acc_tot += ok.double().mean(1)
            n_tot += 1
        if step % SWAP_EVERY == 0:
            par = (step // SWAP_EVERY) % 2
            for k in range(par, NT - 1, 2):
                dl = (1 / T[k] - 1 / T[k + 1]) * (Ex[k] - Ex[k + 1])
                sw = torch.log(torch.rand(C, generator=g, dtype=torch.float64)) < dl
                xk, xk1 = x[k].clone(), x[k + 1].clone()
                x[k] = torch.where(sw[:, None], xk1, xk)
                x[k + 1] = torch.where(sw[:, None], xk, xk1)
                ek, ek1 = Ex[k].clone(), Ex[k + 1].clone()
                Ex[k] = torch.where(sw, ek1, ek)
                Ex[k + 1] = torch.where(sw, ek, ek1)
                if step > burn:
                    sw_acc[k] += sw.double().mean()
                    sw_try[k] += 1
        if step > burn and step % SAVE_EVERY == 0:
            saved.append(x[0].clone())
            savedE.append(Ex[0].clone())
        if step % 10_000 == 0 or step == a.steps:
            print(f"  step {step}/{a.steps} ({time.time() - t0:.0f} s) {'burn-in' if step <= burn else 'sampling'}; acceptance per T (window): " + " ".join(f"{float(v / ADAPT_EVERY if step <= burn else 0):.2f}" for v in acc_win)
                  + f"; E(T=1) mean {float(Ex[0].mean()):.2f}", flush=True)
    states = torch.stack(saved, 1)                                                                # (64, n_saved, 8)
    Es = torch.stack(savedE, 1)
    print(f"sampling done ({time.time() - t0:.0f} s): {states.shape[1]} saved states per chain; frozen steps per T: " + " ".join(f"{float(v):.3f}" for v in torch.exp(logs))
          + "\n  acceptance after burn-in per T: " + " ".join(f"{float(v / n_tot):.3f}" for v in acc_tot) + "\n  swap acceptance per neighbouring pair: " + " ".join(f"{float(sw_acc[k] / max(sw_try[k], 1)):.3f}" for k in range(NT - 1)), flush=True)
    flat = states.reshape(-1, 8)
    labels, conv = [], []
    for lo in range(0, len(flat), 20_000):
        q, gq, _ = quench(flat[lo:lo + 20_000])
        labels.append(state_ids(q.double()))
        conv.append(gq < 1e-4)
        print(f"  quench {min(lo + 20_000, len(flat))}/{len(flat)} ({time.time() - t0:.0f} s)", flush=True)
    labels = torch.cat(labels).reshape(C, -1)
    conv = torch.cat(conv)
    Path(a.out).parent.mkdir(exist_ok=True)
    torch.save(dict(states=states, E=Es, basin_state_id=labels, group=torch.tensor([0] * 32 + [1] * 32), steps=a.steps, burn=burn, temperatures=T, steps_frozen=torch.exp(logs),
                    quench_converged_frac=float(conv.double().mean()), runtime_s=time.time() - t0), a.out)
    print(f"saved {a.out}; quench converged {100 * float(conv.double().mean()):.2f} %; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
