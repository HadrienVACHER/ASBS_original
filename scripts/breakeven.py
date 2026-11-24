""" Break-even of the corrected training (criterion A) against the post hoc bank + darting on (a), from logged costs (energy and gradient evaluations only; descriptive, no rule).
    python scripts/breakeven.py CELL     CELL in {k8d32, mosw10, d128}   (one call per cell, < 1 min)
    python scripts/breakeven.py report
    N* = [total cost of (s) - cost of (a), bank included] / [cost per sample of the post hoc bank + darting at the smallest S matching the level of (s)].
    The adjoint gradient at the end points is the same count in (a) and (s) (one per point), so the extra cost of (s) = its bank + quench + darting cost (cum, logged). """
import json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import dart_lib as L, basin_lib as B, gmm_dk
T0 = time.time(); OUT = ROOT / "figures" / "selftrain_dartbank"; GMM = {"k8d32": (32, 8, 8.0), "d128": (128, 8, 6.0)}; NU1 = {"k8d32": 2.0 - 1e-6, "mosw10": 23.2 ** 2 - 1e-6, "d128": 1.602001 ** 2 - 1e-6}
A_DIR = {"k8d32": ROOT / "results/selftrain_runs/sd_a_k8d32", "mosw10": ROOT / "results/selftrain_runs/sd_a_mosw10", "d128": ROOT / "results/train_weightfree_runs/wf_a_d128"}
NS, SGRID = 5000, (1, 2, 3, 4, 5, 7, 10, 20, 50)


def target(c):
    if c == "mosw10":
        z = np.load(ROOT / "figures/mosw_posthoc/target_d10.npz"); mu, wt, M = z["mu"], z["wt"], z["M"]
        TR = json.load(open(ROOT / "figures/mog_mos_truth_mosw_d10.json")); med = json.load(open(ROOT / "results/mosw_step3/step3_results/d10/out.json"))["median_dist_exact"]
        return L.Mix("S", mu, wt, 1.0), mu, 10, np.array(TR["p"]), M, float(med)
    D, K, DL = GMM[c]; mu = gmm_dk.means(D, K, DL).double().numpy(); wt = gmm_dk.weights(K).double().numpy(); wt /= wt.sum(); pi = L.Mix("G", mu, wt, 1.0); M = B.polish_minima(pi)
    Xex = B.sample_exact(pi, 10000, np.random.default_rng(2026)); return pi, mu, K, gmm_dk.true_proportions(D, K, DL).numpy(), M, float(np.median(B.dist_to_nearest_minimum(Xex, M)))


def stats(x, pi, mu, K, pt, M, med):
    b, fc, _ = B.basins(pi, x, M); pb = B.basin_props(b, K)
    return dict(tv_basin=L.tv(pb, pt), tv_vor=L.tv(L.props(L.cells(x, mu), K), pt), q1=float(np.median(B.dist_to_nearest_minimum(x, M))) / med)


c = sys.argv[1]
if c in ("k8d32", "mosw10", "d128"):
    pi, mu, K, pt, M, med = target(c); d = pi.d
    an = json.load(open(OUT / f"analysis_{c}.json"))["arms"]["s"]; cost_s = an["log"]["cost"]; lev = dict(tv_basin=an["agg"]["tv_basin"]["mean"], tv_vor=an["agg"]["tv_vor"]["mean"], q1=an["agg"]["q1_ratio"]["mean"])
    if c == "d128": a_ref = json.load(open(ROOT / "figures/train_weightfree/analysis_d128.json"))["arms"]["a"]["agg"]
    else: a_ref = json.load(open(OUT / f"analysis_{c}.json"))["arms"]["a"]["agg"]
    a_lev = dict(tv_basin=a_ref["tv_basin"]["mean"], tv_vor=a_ref["tv_vor"]["mean"], q1=a_ref["q1_ratio"]["mean"])
    x = torch.load(A_DIR[c] / "eval_samples" / "ep00450.pt", map_location="cpu", weights_only=False).double().numpy()[:NS]
    raw = stats(x, pi, mu, K, pt, M, med)
    rng = np.random.default_rng(0); X0 = rng.standard_normal((1000, d)) * np.sqrt(NU1[c]); Y, conv, it = B.quench(pi, X0); bank = []
    for y in Y[conv]:
        if not bank or np.linalg.norm(np.array(bank) - y, axis=1).min() > B.MERGE: bank.append(y)
    bank = np.array(bank); bank_cost = 1000 * (it + 1)
    r = L.darting(pi, x, mu, pt, seed=99, SW=SGRID, minima=bank, keep_final=True)
    rows = []
    for S in SGRID:
        z = r["by_S"][S]; xf = z.pop("X_final"); st = stats(xf, pi, mu, K, pt, M, med); rows.append(dict(S=S, per_sample_cost=z["cost"] / len(x), **st, jump_acc=z["jump_acc"]))
        print(c, "S", S, {k: round(v, 4) for k, v in rows[-1].items()}, flush=True)
    ok = [w for w in rows if w["tv_basin"] <= lev["tv_basin"] and w["q1"] <= lev["q1"]]; ok_tv = [w for w in rows if w["tv_basin"] <= lev["tv_basin"]]
    extra = cost_s["cum"]; res = dict(cell=c, n_bank=len(bank), bank_cost_post=bank_cost, extra_cost_s=extra, cost_s_parts=cost_s["parts"], cost_a_per_epoch_grads=512, level_s=lev, level_a_raw=a_lev, level_a_raw_first5000_ep450=raw, posthoc=rows,
                                      S_star_both=(ok[0]["S"] if ok else None), S_star_tv_only=(ok_tv[0]["S"] if ok_tv else None))
    for key, lst in (("both", ok), ("tv_only", ok_tv)):
        if lst: res[f"N_star_{key}"] = extra / lst[0]["per_sample_cost"]; res[f"N_star_{key}_bank_post_removed"] = (extra - bank_cost) / lst[0]["per_sample_cost"]
        else: res[f"N_star_{key}"] = None
    json.dump(res, open(OUT / f"breakeven_{c}.json", "w"), indent=1, default=float); print(json.dumps({k: v for k, v in res.items() if k not in ("posthoc",)}, indent=1, default=float)); print("%.1f s" % (time.time() - T0))
