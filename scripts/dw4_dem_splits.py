""" Report only (no rule): basin proportions (repo quench) of the DEM DW-4 splits (train 100 000, val 10 000, and the test split used as the reference) against p_PT (results/reference_check_pt.json), z per basin.
    SE of a split: binomial (samples assumed independent: the splits are MCMC data, possibly autocorrelated, which would make the SE larger); SE of p_PT: over the 64 chains.
    python -I scripts/dw4_dem_splits.py <dir with train_split_DW4.npy and val_split_DW4.npy>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from adjoint_samplers.utils.dw4_quench import quench  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, state_ids  # noqa: E402
import dw4_division_b as b  # noqa: E402

d = Path(sys.argv[1])
pj = json.load(open(ROOT / "results" / "reference_check_pt.json"))
ref, q, lab_r, types, mu8 = b.load_reference()
names = [NAMES[KEYS[k]] for k in types]
p_pt, se_pt = np.array(pj["p_pt"]), np.array(pj["se_pt"])
splits = {"test (reference)": np.load(ROOT / "data" / "test_split_DW4.npy", allow_pickle=False)}
for s in ("train", "val"):
    splits[s] = np.load(d / f"{s}_split_DW4.npy", allow_pickle=False)
print(f"p_PT: " + ", ".join(f"{n} {p:.4f}" for n, p in zip(names, p_pt)))
for nm, x in splits.items():
    x = torch.tensor(x, dtype=torch.float64)
    lab = []
    conv = 0
    for lo in range(0, len(x), 20000):
        qq, g, _ = quench(x[lo:lo + 20000])
        lab.append(state_ids(qq.double()))
        conv += int((g < 1e-4).sum())
    lab = torch.cat(lab)
    p = np.array([float((lab == k).double().mean()) for k in types])
    se = np.sqrt(p * (1 - p) / len(x))
    z = (p - p_pt) / np.sqrt(se ** 2 + se_pt ** 2)
    print(f"{nm:18} n {len(x):6d} (quench converged {100 * conv / len(x):.1f} %): " + "; ".join(f"{n} {pp:.4f} (z {zz:+.1f}, log {math.log(pp / pt):+.3f})" for n, pp, zz, pt in zip(names, p, z, p_pt)))
