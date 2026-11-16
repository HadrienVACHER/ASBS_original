""" Report only: do the W2 metrics of the AS evaluation (energy_w2 and eq_w2 of `DW4Evaluator`) see the basin-proportion bias of the DW-4 reference?

      python scripts/dw4_w2_check.py [--n 2000] [--reps 50]

    The metrics are the ones of the official evaluation (`adjoint_samplers.components.evaluator.DW4Evaluator`, called as the training does): `samples` of size n = 2000 (eval_batch_size / num_eval_samples of the configs) are compared with n reference samples drawn at random
    (torch.randperm, the evaluator's own draw) after remove_mean; energy_w2 = emd2_1d(energies)^(1/2), eq_w2 = emd2 of the point-cloud distances (permutation + rotation + reflection alignment)^(1/2).
    Declared deviation: eq_w2 uses the vectorised `dist_point_clouds_fast` (the evaluator's `fast_eq` option; tests/test_eval_vectorized.py: same distances up to ties), because the Python double loop of the default is too slow for ~100 evaluations of 2000 x 2000.
    (a) W2(test split, PT): PT states at T = 1 (results/pt_samples_T1.pt, 2000 drawn at random over chains and times, seed 0) as the `samples` of the evaluator, reference = data/test_split_DW4.npy. (Extra, no rule: the same over `reps` random draws.)
    (b) floor: `reps` = 50 pairs of DISJOINT PT sub-samples taken from different chains (the 64 chains are split at random into two disjoint groups of 32; 2000 states drawn at random from each group): W2(PT_a, PT_b); median, p5, p95.
    (c) for comparison: W2(AS ed8bb67, PT) (the first 2000 `model_samples`, reference = 2000 random PT states) and W2(AS ed8bb67, test split) (the official comparison).
    INTERPRETATION, FIXED BEFORE THE RUN: for a metric, if (a) lies inside [p5, p95] of the floor (b), that metric does NOT see the bias of the reference proportions; if (a) is above p95, it sees it. Nothing else is concluded.
"""
import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.components.evaluator import DW4Evaluator  # noqa: E402
from adjoint_samplers.energies import DoubleWellEnergy  # noqa: E402
from adjoint_samplers.utils.graph_utils import remove_mean  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def w2(ev, samples):
    with contextlib.redirect_stdout(io.StringIO()):
        out = ev(samples)
    return out["energy_w2"], out["eq_w2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()
    t0 = time.time()
    ev = DW4Evaluator("data/test_split_DW4.npy", DoubleWellEnergy(dim=8, n_particles=4), eq_w2_every=0, eq_n=a.n, fast_eq=True)
    ref_test = ev.ref_samples.clone()
    d = torch.load(ROOT / "results" / "pt_samples_T1.pt", map_location="cpu", weights_only=False)
    pt = d["states"].float()                                                                      # (64, 1600, 8)
    C, T, _ = pt.shape
    asm = torch.load(ROOT / "results" / "dw4_e1500_ed8bb67_samples.pt", map_location="cpu", weights_only=False)["model_samples"].float()[:a.n]
    rm = lambda x: remove_mean(x, 4, 2)

    def draw(chains, g):
        c = chains[torch.randint(len(chains), (a.n,), generator=g)]
        t = torch.randint(T, (a.n,), generator=g)
        return pt[c, t]

    # (a)
    g = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    ev.ref_samples = ref_test
    a_val = w2(ev, rm(draw(torch.arange(C), g)))
    print(f"(a) W2(test split, PT): energy_w2 {a_val[0]:.4f}  eq_w2 {a_val[1]:.4f}   ({time.time() - t0:.0f} s)", flush=True)
    extra = []
    for r in range(a.reps):
        g = torch.Generator().manual_seed(1000 + r)
        torch.manual_seed(1000 + r)
        extra.append(w2(ev, rm(draw(torch.arange(C), g))))
    extra = np.array(extra)
    print(f"    extra, no rule: over {a.reps} random draws: energy_w2 median {np.median(extra[:, 0]):.4f} [{np.percentile(extra[:, 0], 5):.4f}, {np.percentile(extra[:, 0], 95):.4f}], eq_w2 median {np.median(extra[:, 1]):.4f} [{np.percentile(extra[:, 1], 5):.4f}, {np.percentile(extra[:, 1], 95):.4f}]   ({time.time() - t0:.0f} s)", flush=True)
    # (b) floor
    fl = []
    for r in range(a.reps):
        g = torch.Generator().manual_seed(2000 + r)
        torch.manual_seed(2000 + r)
        perm = torch.randperm(C, generator=g)
        pa, pb = rm(draw(perm[:C // 2], g)), rm(draw(perm[C // 2:], g))
        ev.ref_samples = pb
        fl.append(w2(ev, pa))
        if (r + 1) % 10 == 0:
            print(f"    floor {r + 1}/{a.reps} ({time.time() - t0:.0f} s)", flush=True)
    fl = np.array(fl)
    q = lambda x: (np.median(x), np.percentile(x, 5), np.percentile(x, 95))
    print(f"(b) floor W2(PT_a, PT_b), {a.reps} disjoint pairs from different chains: energy_w2 median {q(fl[:, 0])[0]:.4f} p5 {q(fl[:, 0])[1]:.4f} p95 {q(fl[:, 0])[2]:.4f}; eq_w2 median {q(fl[:, 1])[0]:.4f} p5 {q(fl[:, 1])[1]:.4f} p95 {q(fl[:, 1])[2]:.4f}")
    # (c)
    g = torch.Generator().manual_seed(3)
    torch.manual_seed(3)
    ev.ref_samples = rm(draw(torch.arange(C), g))
    c_pt = w2(ev, rm(asm))
    ev.ref_samples = ref_test
    torch.manual_seed(4)
    c_test = w2(ev, rm(asm))
    print(f"(c) W2(AS ed8bb67, PT): energy_w2 {c_pt[0]:.4f}  eq_w2 {c_pt[1]:.4f};  W2(AS ed8bb67, test split): energy_w2 {c_test[0]:.4f}  eq_w2 {c_test[1]:.4f}")
    for i, nm in enumerate(("energy_w2", "eq_w2")):
        lo, hi = q(fl[:, i])[1], q(fl[:, i])[2]
        print(f"INTERPRETATION {nm}: (a) {a_val[i]:.4f} vs floor [p5 {lo:.4f}, p95 {hi:.4f}] -> " + ("inside: this metric does NOT see the bias" if lo <= a_val[i] <= hi else "above p95: this metric sees the bias" if a_val[i] > hi else "below p5"))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
