""" rules_train_weightfree.md: local CPU analysis of the 9 continuation runs (figures/train_weightfree/runs/wf_<arm>_<cell>/{eval_samples/ep00350.pt, ep00400.pt, ep00450.pt, samples_final.pt, stdout.log}).
    python scripts/wf_analyse.py CELL        CELL in {d8, d32, d128}  (one call per cell, < 1 min)
    python scripts/wf_analyse.py verdict """
import ast, json, re, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import dart_lib as L, basin_lib as B, gmm_dk
T0 = time.time()
OUT = ROOT / "figures" / "train_weightfree"; RUNS = OUT / "runs"
CELLS = {"d8": (8, 8, 8.0), "d32": (32, 8, 6.0), "d128": (128, 8, 6.0)}
EVALS = (350, 400, 450)


def last_stats(log, prefix):
    """ last printed metrics dict of the run containing keys starting with prefix """
    out = None
    for line in open(log):
        if line.startswith("{") and prefix in line:
            try:
                out = ast.literal_eval(line.strip())
            except Exception:
                pass
    return out


c = sys.argv[1]
if c in CELLS:
    D, K, DL = CELLS[c]
    mu = gmm_dk.means(D, K, DL).double().numpy(); wt = gmm_dk.weights(K).double().numpy(); wt /= wt.sum()
    pi = L.Mix("G", mu, wt, 1.0); M = B.polish_minima(pi); pt = gmm_dk.true_proportions(D, K, DL).numpy()
    Xex = B.sample_exact(pi, 10000, np.random.default_rng(2026)); med_ex = float(np.median(B.dist_to_nearest_minimum(Xex, M)))
    res = dict(cell=c, median_dist_exact=med_ex, arms={})
    for arm in ("a", "t", "d"):
        rd = RUNS / f"wf_{arm}_{c}"; per = []
        for e in EVALS:
            x = torch.load(rd / "eval_samples" / f"ep{e:05d}.pt", map_location="cpu", weights_only=False).double().numpy()
            pv = L.props(L.cells(x, mu), K); b, fc, fa = B.basins(pi, x, M); pb = B.basin_props(b, K)
            per.append(dict(epoch=e, n=len(x), tv_vor=L.tv(pv, pt), tv_basin=L.tv(pb, pt), cov_vor=int((pv >= 0.25 * pt).sum()), cov_basin=int((pb >= 0.25 * pt).sum()),
                            q1_ratio=float(np.median(B.dist_to_nearest_minimum(x, M))) / med_ex, frac_converged=fc))
        xf = torch.load(rd / "samples_final.pt", map_location="cpu", weights_only=False)["model_samples"].double().numpy()
        pvf = L.props(L.cells(xf, mu), K); bf, _, _ = B.basins(pi, xf, M)
        agg = {k: dict(mean=float(np.mean([p[k] for p in per])), sd=float(np.std([p[k] for p in per], ddof=1))) for k in ("tv_vor", "tv_basin", "q1_ratio", "cov_vor", "cov_basin")}
        res["arms"][arm] = dict(evals=per, agg=agg, final=dict(tv_vor=L.tv(pvf, pt), tv_basin=L.tv(B.basin_props(bf, K), pt), cov_vor=int((pvf >= 0.25 * pt).sum())),
                                tis_stats=last_stats(rd / "stdout.log", "tis/ess_alpha") if arm == "t" else None,
                                dart_stats=last_stats(rd / "stdout.log", "dart/gamma") if arm == "d" else None)
        print(f"{c} {arm}: TV vor {agg['tv_vor']['mean']:.4f}±{agg['tv_vor']['sd']:.4f} basin {agg['tv_basin']['mean']:.4f}±{agg['tv_basin']['sd']:.4f} cov {agg['cov_vor']['mean']:.1f} "
              f"Q1 {agg['q1_ratio']['mean']:.3f} | final vor {res['arms'][arm]['final']['tv_vor']:.4f}", flush=True)
    res["time_s"] = time.time() - T0; json.dump(res, open(OUT / f"analysis_{c}.json", "w"), indent=1, default=float); print(f"{c}: {res['time_s']:.1f} s")
elif c == "verdict":
    R = {k: json.load(open(OUT / f"analysis_{k}.json")) for k in CELLS}
    tv = lambda k, arm: R[k]["arms"][arm]["agg"]["tv_vor"]["mean"]; q1 = lambda k, arm: R[k]["arms"][arm]["agg"]["q1_ratio"]["mean"]
    F1 = {k: bool(tv(k, "d") <= 0.5 * tv(k, "a")) for k in ("d8", "d32")}
    F2 = dict(vs_a=bool(tv("d128", "d") <= 0.5 * tv("d128", "a")), vs_t=bool(tv("d128", "d") <= 0.5 * tv("d128", "t")))
    F3 = {k: {arm: bool(q1(k, arm) <= 1.1 * q1(k, "a")) for arm in ("d", "t")} for k in CELLS}
    V = dict(F1=dict(per_cell=F1, value=all(F1.values())), F2=dict(**F2, value=F2["vs_a"] and F2["vs_t"]), F3=dict(per_cell=F3, value=all(v for d_ in F3.values() for v in d_.values())),
             tv_vor={k: {arm: tv(k, arm) for arm in ("a", "t", "d")} for k in CELLS}, q1={k: {arm: q1(k, arm) for arm in ("a", "t", "d")} for k in CELLS})
    V["reading"] = ("la correction sans poids marche à l'entraînement là où la correction par poids de trajectoire échoue (contribution à défendre)" if V["F2"]["value"] else "F2 faux")
    json.dump(V, open(OUT / "verdict.json", "w"), indent=1); print(json.dumps(V, indent=1, ensure_ascii=False))
