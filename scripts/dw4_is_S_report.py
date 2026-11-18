""" S_R and S_Q (sum_b |p - p_PT|, p_PT of scripts/dw4_reeval_pt.py) of importance-weighted estimators, for comparison with the darting runs (report only; local, seconds).
      python scripts/dw4_is_S_report.py path DIR/path_*.pt          (path-space IS of the AS controller 94252, chunks produced by dw4_is_S_path.py + dw4_is_S_label.py)
      python scripts/dw4_is_S_report.py endpoint FILE.pt ...        (endpoint IS, files with lw, lab_R, lab_Q)
    A single batch of N weighted samples has a sampling error of its own: the bootstrap SE over the samples is printed; S is positive even for a perfect estimator (floor: scripts/dw4_floor_S.py, which concerns 20 x 5000 draws). """
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dw4_reeval_pt as rp  # noqa: E402
import dw4_division_b as b  # noqa: E402

K = rp.K
mode, files = sys.argv[1], sys.argv[2:]
ds = [torch.load(f, weights_only=False) for f in files]
lw = torch.cat([d["lw"] for d in ds]).double()
w = torch.exp(lw - lw.max())
N = len(w)
ess = float(w.sum() ** 2 / (w ** 2).sum())
print(f"{mode}: {N} samples from {len(files)} file(s); ESS {ess:.0f} ({100 * ess / N:.1f} %); k_hat {b.khat(w.numpy()):.3f}")
tr = rp.truth()
rng = np.random.default_rng(0)
wn = w.numpy()
for tag in ("R", "Q"):
    lab = torch.cat([d[f"lab_{tag}"] for d in ds]).numpy()
    truth = tr[tag][0]
    def prop(idx, weights):
        return np.bincount(lab[idx], weights=weights[idx], minlength=K) / weights[idx].sum()
    all_idx = np.arange(N)
    p_is, p_raw = prop(all_idx, wn), prop(all_idx, np.ones(N))
    s_is, s_raw = float(np.abs(p_is - truth).sum()), float(np.abs(p_raw - truth).sum())
    boot = [float(np.abs(prop(i, wn) - truth).sum()) for i in (rng.integers(N, size=N) for _ in range(300))]
    print(f"[{tag}] S IS {s_is:.4f} (bootstrap SD {np.std(boot):.4f})   S unweighted (raw sampler) {s_raw:.4f}")
    print("      p_IS  " + "  ".join(f"{rp.NAME[k]}: {p_is[k]:.4f}" for k in range(K) if truth[k] >= 0.005))
