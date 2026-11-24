""" rules_shape_correction.md: local CPU analysis.
    python scripts/shape_analyse.py calib     calibration floors on the existing oracle runs (o-AS sd_o_d128, o-DSM sd_od_d128), before the rental
    python scripts/shape_analyse.py runs2    rules_shape_correction_2.md: buffer (M') shift 1.0, starts (i) common controller and (ii) re-initialised; arms sd_Mp_<as|dsm>_<i|ii>_d128 ; T, P2'
    python scripts/shape_analyse.py runs      the 4 runs sd_<V|M>_<as|dsm>_d128 : rho_V, rho_M, basin / Voronoi TV, Q1, mean of the evaluations 350, 400, 450 (sd between evaluations) + final; P1, P2, P3 """
import json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import dart_lib as L, basin_lib as B, gmm_dk
OUT = ROOT / "figures" / "shape_correction"; OUT.mkdir(parents=True, exist_ok=True); RUNS = ROOT / "results" / "selftrain_runs"; EVALS = (350, 400, 450); D, K, DL = 128, 8, 6.0
mu = gmm_dk.means(D, K, DL).double().numpy(); wt = gmm_dk.weights(K).double().numpy(); wt /= wt.sum(); pi = L.Mix("G", mu, wt, 1.0); M = B.polish_minima(pi)
pt = gmm_dk.true_proportions(D, K, DL).numpy(); med = float(np.median(B.dist_to_nearest_minimum(B.sample_exact(pi, 10000, np.random.default_rng(2026)), M)))


SHIFT = 0.3


def stats(x):
    pv = L.props(L.cells(x, mu), K); b, fc, _ = B.basins(pi, x, M); pb = B.basin_props(b, K); sm = gmm_dk.shape_measures(x, D, K, DL, shift=SHIFT)
    return dict(tv_vor=L.tv(pv, pt), tv_basin=L.tv(pb, pt), q1_ratio=float(np.median(B.dist_to_nearest_minimum(x, M))) / med, rho_V=sm["rho_V"], rho_M=sm["rho_M"], rho_V_modes=sm["rho_V_modes"], rho_M_modes=sm["rho_M_modes"], counts=sm["counts"])


def run(name):
    per = []
    for e in EVALS:
        x = torch.load(RUNS / name / "eval_samples" / f"ep{e:05d}.pt", map_location="cpu", weights_only=False).double().numpy(); per.append(dict(epoch=e, **stats(x)))
    xf = torch.load(RUNS / name / "samples_final.pt", map_location="cpu", weights_only=False)["model_samples"].double().numpy()
    agg = {k: dict(mean=float(np.mean([p[k] for p in per])), sd=float(np.std([p[k] for p in per], ddof=1))) for k in ("tv_vor", "tv_basin", "q1_ratio", "rho_V", "rho_M")}
    return dict(evals=per, final=stats(xf), agg=agg)


c = sys.argv[1]; T0 = time.time()
if c == "calib":
    R = {a: run(n) for a, n in (("o_as", "sd_o_d128"), ("o_dsm", "sd_od_d128"))}
    for a in R: print(a, {k: (round(v["mean"], 4), round(v["sd"], 4)) for k, v in R[a]["agg"].items()})
    json.dump(R, open(OUT / "calibration.json", "w"), indent=1, default=float)
elif c == "runs":
    R = {}
    for buf in ("V", "M"):
        for a in ("as", "dsm"):
            R[f"{buf}_{a}"] = run(f"sd_{buf}_{a}_d128"); print(buf, a, {k: (round(v["mean"], 4), round(v["sd"], 4)) for k, v in R[f"{buf}_{a}"]["agg"].items()}, "final rho_V %.3f rho_M %.3f" % (R[f"{buf}_{a}"]["final"]["rho_V"], R[f"{buf}_{a}"]["final"]["rho_M"]), flush=True)
    m = lambda k, q: R[k]["agg"][q]["mean"]
    P1 = dict(rho_V_as=m("V_as", "rho_V"), rho_V_dsm=m("V_dsm", "rho_V")); P1["as_ok"] = bool(0.80 <= P1["rho_V_as"] <= 1.10); P1["dsm_ok"] = bool(P1["rho_V_dsm"] >= 1.30); P1["value"] = P1["as_ok"] and P1["dsm_ok"]
    P2 = dict(rho_M_as=m("M_as", "rho_M"), rho_M_dsm=m("M_dsm", "rho_M")); P2["as_ok"] = bool(P2["rho_M_as"] <= 0.5); P2["dsm_ok"] = bool(P2["rho_M_dsm"] >= 0.8); P2["value"] = P2["as_ok"] and P2["dsm_ok"]
    P3 = {k: m(k, "tv_basin") for k in R}; P3["value"] = bool(all(v <= 0.053 for k, v in P3.items()))
    R["verdict"] = dict(P1=P1, P2=P2, P3=P3); json.dump(R, open(OUT / "analysis_runs.json", "w"), indent=1, default=float); print(json.dumps(R["verdict"], indent=1, default=float))
elif c == "runs2":
    SHIFT = 1.0; R = {}
    STARTS = [st for st in ("i", "ii") if (RUNS / f"sd_Mp_as_{st}_d128").exists()]                  # amendment 2026-11-24: start (ii) only
    for st in STARTS:
        for a in ("as", "dsm"):
            R[f"{st}_{a}"] = run(f"sd_Mp_{a}_{st}_d128"); print(st, a, {k: (round(v["mean"], 4), round(v["sd"], 4)) for k, v in R[f"{st}_{a}"]["agg"].items()}, "final rho_V %.3f rho_M %.3f" % (R[f"{st}_{a}"]["final"]["rho_V"], R[f"{st}_{a}"]["final"]["rho_M"]), flush=True)
    m = lambda k, q: R[k]["agg"][q]["mean"]
    T = {st: m(f"{st}_dsm", "rho_M") for st in STARTS}; Tv = {st: bool(v >= 0.8) for st, v in T.items()}
    P2p = {st: dict(rho_M_as=m(f"{st}_as", "rho_M"), rho_M_dsm=T[st], value=bool(m(f"{st}_as", "rho_M") <= 0.3)) for st in STARTS if Tv[st]}
    R["verdict"] = dict(T_values=T, T_per_start=Tv, T=any(Tv.values()), P2prime=P2p, A_m=-0.5442); json.dump(R, open(OUT / "analysis_runs2.json", "w"), indent=1, default=float); print(json.dumps(R["verdict"], indent=1, default=float))
print(f"{c}: {time.time() - T0:.1f} s")
