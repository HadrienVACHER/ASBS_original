""" T1' of the darting test on FRESH samples of e^{-E}/Z: the parallel-tempering states at T = 1 (results/pt_samples_T1.pt). Rule: rules_reference_check.md (Task 3), committed before the run.
    T1 (scripts/dw4_darting_test.py) treated the reference as samples of pi: that was a design error (the reference proportions of the two dominant basins are not those of e^{-E}/Z: Task 1). Its NO-GO stays the written verdict.
    T1' = the same statistic and the same threshold as T1 (lumped flux balance w_a K_ab = w_b K_ba at kappa = 0; for every pair with both lumped fluxes > 1e-3, |log(w_a K_ab / (w_b K_ba))| < 3 bootstrap SE),
    on 8000 PT states drawn at random without replacement (seed 0); kernel built as in the T1 run (minima of the quench of 3000 random reference samples, seed 0). Bootstrap over the states (300, as T1);
    reported besides, without rule: the bootstrap over whole chains.      python scripts/dw4_darting_T1prime.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dw4_darting_test as dt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    t0 = time.time()
    d = torch.load(ROOT / "results" / "pt_samples_T1.pt", map_location="cpu", weights_only=False)
    st = d["states"].numpy()                                                                      # (64, n, 8)
    C, n, _ = st.shape
    rng = np.random.default_rng(0)
    R = dt.load(str(ROOT / "data" / "test_split_DW4.npy"), 3000, rng)                             # same call, same seed, as the T1 run
    QR = dt.quench_precise(R)
    LR = dt.label(QR)
    ER = dt.energy(QR)
    minima = {s: QR[LR == s][np.argmin(ER[LR == s])] for s in dt.ORDER}
    pairs, maxerr = dt.build_model(minima)
    K = len(dt.ORDER)
    print(f"kernel built (T0 max |M_ba M_ab - Id| = {maxerr:.1e}); {time.time() - t0:.0f} s", flush=True)
    rs = np.random.default_rng(0)
    pick = rs.choice(C * n, 8000, replace=False)
    chain = pick // n
    X = dt.rc(st.reshape(-1, 8)[pick].reshape(-1, 4, 2))
    res = dt.evaluate(X, pairs, K, [0.0], 4.0, rng)
    keep, aa, P = res["keep"], res["a"], res["probs"][:, :, 0]
    ch = chain[keep]
    N = len(aa)
    onehot = np.eye(K)[aa]

    def flux(idx):
        return np.einsum("ni,nj->ij", onehot[idx], P[idx]) / len(idx)

    F = flux(np.arange(N))
    boots = np.stack([flux(rng.integers(0, N, N)) for _ in range(300)])
    members = [np.where(ch == c)[0] for c in range(C)]
    bb = []
    for _ in range(300):
        sel = np.concatenate([members[c] for c in rng.integers(0, C, C)])
        bb.append(flux(sel))
    bb = np.stack(bb)
    w = onehot.mean(0)
    print("PT states used: " + ", ".join(f"{s} {p:.4f}" for s, p in zip(dt.ORDER, w)) + f"; valid landing {res['valid']:.3f}; mean Barker weight {res['omega']:.3f}")
    print("[T1'] flux balance w_a K_ab vs w_b K_ba (kappa = 0): log-ratio / SE (states bootstrap = the rule, chain bootstrap = report only)")
    ok = True
    for i in range(K):
        for j in range(i + 1, K):
            if F[i, j] > 1e-3 and F[j, i] > 1e-3:
                lr = np.log(F[i, j] / F[j, i])
                se = np.std(np.log(boots[:, i, j] / np.maximum(boots[:, j, i], 1e-12)))
                seb = np.std(np.log(bb[:, i, j] / np.maximum(bb[:, j, i], 1e-12)))
                ok &= abs(lr) < 3 * se
                print(f"   {dt.ORDER[i]}-{dt.ORDER[j]}: {F[i, j]:.5f} vs {F[j, i]:.5f}  log-ratio {lr:+.3f}  SE {se:.3f} (z {lr / se:+.1f})  chain-bootstrap SE {seb:.3f} (z {lr / seb:+.1f})")
    print(f"\nT1' (every pair |log-ratio| < 3 SE): {ok}   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
