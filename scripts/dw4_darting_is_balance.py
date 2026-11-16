""" DW-4, POST HOC follow-up of the darting test (declared post hoc, report only): is the T1 imbalance of the dominant pair a flaw of the kernel or a property of the REFERENCE proportions?

      python scripts/dw4_darting_is_balance.py --samples results/dw4_e1500_ed8bb67_samples.pt [--n 6000]

    The lumped flux balance w_a K_ab = w_b K_ba of the darting kernel (scripts/dw4_darting_test.py, kappa = 0, same code: the functions are imported) is recomputed on IMPORTANCE-WEIGHTED samples of e^{-E}/Z instead of the reference:
    draws from the encompassing endpoint proposal of scripts/dw4_is_inflate.py (gaussian s = 1: G-averaged EM basin components fitted on the AS samples ed8bb67, mixture weights = the AS frequencies; ESS about 60 % at 20000 draws), weights w = e^{-E}/r.
    Weighted lumped flux F_ij = sum_n w_n [a_n = i] P_n(i -> j) / sum_n w_n, bootstrap over the draws (300), log(F_ij / F_ji) and its SE, for every pair with both fluxes > 1e-3. Nothing of the reference proportions is used.
    DECLARED INTERPRETATION (before the run): if every pair has |log ratio| < 3 SE, the kernel is consistent with pi-invariance on these samples and the T1 imbalance of the reference is a property of the reference (its dominant-pair proportions);
    if the dominant pair (triangle+pendant / path of 2) is out of balance by more than 3 SE with the same sign and size (about +0.3), the kernel is at fault for that pair (or the proposal misses the same region). Inconclusive otherwise.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dw4_darting_test as dt  # noqa: E402
import dw4_division_b as b  # noqa: E402
import dw4_division_step0 as s0  # noqa: E402
import dw4_division_trunc as tr  # noqa: E402
import dw4_is_inflate as isf  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--nboot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ref, q, lab_r, types, mu8 = b.load_reference()
    fit = tr.load_as_fit(a.samples)["fit"]
    comps = [fit[k] for k in types]
    isf.N = a.n
    gen = torch.Generator().manual_seed(7)
    x, lr = isf.build("gauss", 1.0, comps, gen)
    lw = -s0.energy(x) - lr
    w_all = torch.exp(lw - lw.max()).numpy()
    ess = w_all.sum() ** 2 / (w_all ** 2).sum()
    print(f"IS draws {a.n}: ESS {ess:.0f} ({100 * ess / a.n:.1f} %)", flush=True)
    errs = dt.check_energy_against_repo(ref.numpy().reshape(-1, 8))
    X = dt.rc(x.numpy().reshape(-1, 4, 2))
    rng = np.random.default_rng(a.seed)
    # minima and kernel from the reference quench, as in the darting test (3000 random reference samples)
    R = dt.rc(ref.numpy().reshape(-1, 4, 2)[rng.choice(len(ref), 3000, replace=False)])
    QR = dt.quench_precise(R)
    LR = dt.label(QR)
    ER = dt.energy(QR)
    minima = {s: QR[LR == s][np.argmin(ER[LR == s])] for s in dt.ORDER}
    pairs, maxerr = dt.build_model(minima)
    K = len(dt.ORDER)
    res = dt.evaluate(X, pairs, K, [0.0], 4.0, rng)
    keep, aa, P = res["keep"], res["a"], res["probs"][:, :, 0]
    w = w_all[keep]
    n = len(w)
    onehot = np.eye(K)[aa]

    def flux(idx):
        ww = w[idx] / w[idx].sum()
        return np.einsum("n,ni,nj->ij", ww, onehot[idx], P[idx])

    F = flux(np.arange(n))
    boots = np.stack([flux(rng.integers(0, n, n)) for _ in range(a.nboot)])
    pw = (w[:, None] * onehot).sum(0) / w.sum()
    print("weighted basin proportions of the IS draws (kept): " + ", ".join(f"{s} {p:.4f}" for s, p in zip(dt.ORDER, pw)))
    print("[T1 on IS samples] flux balance w_a K_ab vs w_b K_ba (kappa = 0), log-ratio / SE")
    zs = {}
    for i in range(K):
        for j in range(i + 1, K):
            if F[i, j] > 1e-3 and F[j, i] > 1e-3:
                lrt = np.log(F[i, j] / F[j, i])
                se = np.std(np.log(boots[:, i, j] / np.maximum(boots[:, j, i], 1e-12)))
                zs[(i, j)] = lrt / se
                print(f"   {dt.ORDER[i]}-{dt.ORDER[j]}: {F[i, j]:.5f} vs {F[j, i]:.5f}  log-ratio {lrt:+.3f}  SE {se:.3f}  z {lrt / se:+.1f}")
    allok = all(abs(z) < 3 for z in zs.values())
    dom = zs.get((0, 1), float("nan"))
    print(f"\nall pairs |z| < 3: {allok}; dominant pair z = {dom:+.1f}  =>  " + ("kernel consistent with pi-invariance on these samples (reference imbalance = property of the reference)" if allok else "dominant pair out of balance (kernel or proposal at fault)" if abs(dom) >= 3 else "inconclusive"))


if __name__ == "__main__":
    main()
