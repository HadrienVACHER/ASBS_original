""" Diagnostic on EXISTING data (no new run): Q1 and TV (basin, Voronoi) of the corrected buffer itself (buffer_x1 of samples_final.pt: the last 10^4 buffer points, i.e. the last ~20 refreshes)
    versus the learned sampler (model_samples of the same file), arm (s) and, as a reference, arm (a). python scripts/selftrain_buffer_diag.py """
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import selftrain_analyse_lib as A
OUT = ROOT / "figures" / "selftrain_dartbank"; RUNS = ROOT / "results" / "selftrain_runs"; res = {}
for c in ("mosw10", "d128", "k8d32"):
    pi, mu, K, pt, M, med = A.target(c); res[c] = {}
    for arm in (("s",) if c == "d128" else ("s", "a")):
        d = torch.load(RUNS / f"sd_{arm}_{c}" / "samples_final.pt", map_location="cpu", weights_only=False)
        for nm in ("buffer_x1", "model_samples"):
            st = A.stats(d[nm].double().numpy(), pi, mu, K, pt, M, med); st.pop("p_vor"); st.pop("p_basin"); res[c][f"{arm}:{nm}"] = st
            print(f"{c} {arm} {nm:13s} n {st['n']} TV basin {st['tv_basin']:.4f} Voronoi {st['tv_vor']:.4f} | cov basin {st['cov_basin']} vor {st['cov_vor']} | Q1 {st['q1_ratio']:.3f}")
json.dump(res, open(OUT / "buffer_diag.json", "w"), indent=1)
