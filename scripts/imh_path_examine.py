""" Post hoc examination (declared AFTER the verdict, no rule) of the I1 'gain' direction in cell C of rules_imh_path.md, on the SAME saved pools.
    python scripts/imh_path_examine.py     per cell and pool: accepted moves, distinct states, IMH ESS per indicator against the ESS of the SAME indicator for self-normalised IS, and 200 replays of the chain with fresh acceptance uniforms (median basin TV) against the IS TV. """
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
src = (ROOT / "scripts/imh_path.py").read_text().split("c = sys.argv[1]")[0]; exec(compile(src, "imh_path_defs", "exec"))
T0 = time.time(); out = {}
for c in "ABCD":
    K, pt, P = pools(c); R = json.load(open(OUT / f"cell_{c}.json")); rows = []
    for i, (name, lw, lab) in enumerate(P):
        lw = np.asarray(lw, float); lab = np.asarray(lab, int); n = len(lw); w = np.exp(lw - logsumexp(lw)); r = R["pools"][i]
        idx, acc = chain(lw, 1000 + i); s = idx[n // 10:]
        # same-functional ratio: IMH ESS_k / IS ESS_k, basins with both defined
        pair = []
        for k, e in r["ess_imh_by_k"].items():
            f = (lab == int(k)).astype(float); ph = float((w * f).sum()); var = float((w ** 2 * (f - ph) ** 2).sum())
            if var > 0 and 0 < ph < 1: pair.append(e / (ph * (1 - ph) / (n * var)))
        tvs = []; accs = []
        for rep in range(200):
            j, a = chain(lw, 5000 + 200 * i + rep); st = lab[j[n // 10:]]; m = st >= 0
            tvs.append(L.tv(np.bincount(st[m], minlength=K) / m.sum(), pt)); accs.append(int(a[n // 10:].sum()))
        rows.append(dict(pool=name, accepted=int(acc[n // 10:].sum()), distinct_states=int(len(np.unique(s))), ess_imh_states=float(r["ess_imh"] * len(s)), ess_ratio_same_functional_median=float(np.median(pair)) if pair else None,
                         tv_replays_median=float(np.median(tvs)), tv_replays_q10=float(np.quantile(tvs, .1)), tv_replays_q90=float(np.quantile(tvs, .9)), tv_is=r["tv_is"], accepted_replays_median=float(np.median(accs))))
    out[c] = rows
    for w_ in rows: print(c, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in w_.items()}, flush=True)
json.dump(out, open(OUT / "examine.json", "w"), indent=1, default=float); print(f"{time.time() - T0:.1f} s")
