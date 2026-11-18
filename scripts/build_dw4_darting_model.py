""" Builds the fixed model of the DW-4 darting kernel (adjoint_samplers/utils/dw4_darting_model.json): for every ordered pair of basin types (a, b), the pair-frame copies src_ab / tgt_ab of the two minima, the 8 x 8 map M_ab between
    the vibrational Hessian eigen-directions (in the 8-dim coordinates, so it does not depend on the basis of the zero-CoM subspace), log|det|, and the stabiliser size of tgt. Same construction and the same minima as the offline test
    (scripts/dw4_darting_test.py: minima = lowest-energy quenched representative of each type among 3000 random reference samples, seed 0); nothing depends on the training data after that.
      python scripts/build_dw4_darting_model.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dw4_darting_test as dt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
R = dt.load(str(ROOT / "data" / "test_split_DW4.npy"), 3000, rng)
QR = dt.quench_precise(R)
LR = dt.label(QR)
ER = dt.energy(QR)
minima = {s: QR[LR == s][np.argmin(ER[LR == s])] for s in dt.ORDER}
pairs, maxerr = dt.build_model(minima)
out = dict(order=dt.ORDER, names=["triangle + pendant", "path of 2 (3 particles)", "2 disjoint pairs", "K4 minus an edge", "star (3 edges)"], pairs={})
for (a, b), P in pairs.items():
    M8 = dt.BV @ P["M"] @ dt.BV.T
    out["pairs"][f"{a},{b}"] = dict(src=P["src"].tolist(), tgt=P["tgt"].tolist(), M8=M8.tolist(), logdet=float(P["logdet"]), n_tgt=int(P["n_tgt"]))
(ROOT / "adjoint_samplers" / "utils" / "dw4_darting_model.json").write_text(json.dumps(out))
print("pairs", len(pairs), "T0 max |M_ba M_ab - Id_slice|", maxerr, "stabiliser sizes", {dt.ORDER[a]: pairs[(a, (a + 1) % 5)]["n_tgt"] for a in range(5)}, "(n_tgt of the next type, for information)")
print({f"{a}->{b}": pairs[(a, b)]["n_tgt"] for a in range(5) for b in range(5) if a != b})
