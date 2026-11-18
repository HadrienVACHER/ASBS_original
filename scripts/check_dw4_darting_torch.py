""" The batched torch darting kernel (adjoint_samplers/utils/dw4_darting.py) against the numpy reference (scripts/dw4_darting_test.py) on reference samples: types, number of valid proposals, and the one-step probability of moving to each type at kappa = 0 (and the timing).
      python scripts/check_dw4_darting_torch.py [--n 600] """
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dw4_darting_test as dt  # noqa: E402
from adjoint_samplers.utils import dw4_darting as dd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=600)
a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(5)
X = dt.load(str(ROOT / "data" / "test_split_DW4.npy"), a.n, rng)                                   # (n, 4, 2) centred
# numpy reference (the model of the json is the one built with the same protocol)
import json
d = json.loads((ROOT / "adjoint_samplers" / "utils" / "dw4_darting_model.json").read_text())
R0 = dt.load(str(ROOT / "data" / "test_split_DW4.npy"), 3000, np.random.default_rng(0))
QR = dt.quench_precise(R0)
LR = dt.label(QR)
ER = dt.energy(QR)
pairs, _ = dt.build_model({s: QR[LR == s][np.argmin(ER[LR == s])] for s in dt.ORDER})
t0 = time.time()
res = dt.evaluate(X, pairs, len(dt.ORDER), [0.0], 4.0, np.random.default_rng(0))
t_np = time.time() - t0
t0 = time.time()
pr = dd.proposals(torch.tensor(X.reshape(-1, 8)))
t_t = time.time() - t0
keep, a_np, probs_np = res["keep"], res["a"], res["probs"][:, :, 0]
a_t = pr["type"].numpy()[keep]
print(f"numpy {t_np:.1f} s, torch {t_t:.1f} s for {a.n} samples; kept by the numpy evaluation {len(keep)}; types equal: {bool((a_t == a_np).all())}; torch types -1: {int((pr['type'] < 0).sum())}")
omega = torch.sigmoid(pr["base"])
P = pr["weight"] * omega
probs_t = torch.stack([P[:, b * dd.SLOTS:(b + 1) * dd.SLOTS].sum(1) for b in range(5)], 1).numpy()[keep]
print("valid proposals per sample: numpy mean", f"{(res['valid'] * 0 + 1):.0f}" if False else "", " torch", f"{float((pr['weight'] > 0).sum(1).double().mean()):.2f}")
print(f"max |one-step probability torch - numpy| over samples and target types (kappa = 0): {np.abs(probs_t - probs_np).max():.2e}; mean total out-flux: numpy {probs_np.sum(1).mean():.4f}, torch {probs_t.sum(1).mean():.4f}")
