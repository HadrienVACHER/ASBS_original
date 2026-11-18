""" Floor of S: the S of a PERFECT sampler, in the exact metric of the continuation runs (report only; local, seconds).
    A perfect sampler is represented by draws from the PT pool at T = 1 (results/pt_samples_T1.pt: 64 chains x 1600 states, labels R = raw states, Q = quenched basins), with replacement.
    One 'evaluation' = 5000 draws (num_eval_samples of the runs); one S = mean proportions over 20 evaluations compared with p_PT (the pool mean over chains, as in scripts/dw4_reeval_pt.py): S = sum_b |p_mean - p_PT|.
    Repeated 50 times: median, p5, p95.  Variant 2 (adds the uncertainty of p_PT itself): the draws come from a pool of 64 chains resampled with replacement (chain bootstrap) while the truth stays the original p_PT.
      python scripts/dw4_floor_S.py """
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dw4_reeval_pt as rp  # noqa: E402
from adjoint_samplers.utils.dw4_states import state_ids  # noqa: E402

K, NEVAL, NS, REP = rp.K, 20, 5000, 50
d = torch.load(rp.ROOT / "results" / "pt_samples_T1.pt", map_location="cpu", weights_only=False)
st, ids = d["states"], d["basin_state_id"]
raw = torch.stack([state_ids(st[c].double()) for c in range(st.shape[0])])
tr = rp.truth()
rng = np.random.default_rng(0)
for tag, lab in (("R", raw), ("Q", ids)):
    truth = tr[tag][0]
    lab = lab.numpy()                                                      # (64, 1600)
    flat = lab.reshape(-1)
    for variant in ("iid draws from the pool", "chain-bootstrap pool"):
        out = []
        for _ in range(REP):
            pool = flat if variant.startswith("iid") else lab[rng.integers(lab.shape[0], size=lab.shape[0])].reshape(-1)
            props = np.stack([np.bincount(rng.choice(pool, size=NS), minlength=K) / NS for _ in range(NEVAL)])
            out.append(float(np.abs(props.mean(0) - truth).sum()))
        out = np.array(out)
        print(f"[{tag}] floor of S, {variant:26}: median {np.median(out):.4f}  p5 {np.percentile(out, 5):.4f}  p95 {np.percentile(out, 95):.4f}   (50 repetitions x 20 evaluations x {NS} draws)")
