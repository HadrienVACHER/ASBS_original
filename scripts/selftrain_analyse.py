""" rules_selftrain_dartbank.md: local CPU analysis of the 7 continuation runs (results/selftrain_runs/sd_<arm>_<cell>/{eval_samples/ep00350.pt, ep00400.pt, ep00450.pt, samples_final.pt, stdout.log}).
    python scripts/selftrain_analyse.py CELL        CELL in {k8d32, mosw10, d128}  (one call per cell, < 1 min)
    python scripts/selftrain_analyse.py verdict
    d128: arm (s) only; (a) and (t) are read from figures/train_weightfree/analysis_d128.json (step B, reused as declared in the rules). """
import ast, json, re, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import dart_lib as L, basin_lib as B, gmm_dk
T0 = time.time()
OUT = ROOT / "figures" / "selftrain_dartbank"; OUT.mkdir(parents=True, exist_ok=True); RUNS = ROOT / "results" / "selftrain_runs"   # eval and final samples (large) stay outside the repository
EVALS = (350, 400, 450)
GMM = {"k8d32": (32, 8, 8.0), "d128": (128, 8, 6.0)}


def target(c):
    """ pi (dart_lib.Mix), Voronoi centres mu, K, true proportions, polished minima, exact median distance to the nearest minimum """
    if c == "mosw10":
        z = np.load(ROOT / "figures" / "mosw_posthoc" / "target_d10.npz"); mu, wt, M = z["mu"], z["wt"], z["M"]
        TR = json.load(open(ROOT / "figures" / "mog_mos_truth_mosw_d10.json")); med = json.load(open(ROOT / "results" / "mosw_step3" / "step3_results" / "d10" / "out.json"))["median_dist_exact"]
        return L.Mix("S", mu, wt, 1.0), mu, 10, np.array(TR["p"]), M, float(med)
    D, K, DL = GMM[c]
    mu = gmm_dk.means(D, K, DL).double().numpy(); wt = gmm_dk.weights(K).double().numpy(); wt /= wt.sum()
    pi = L.Mix("G", mu, wt, 1.0); M = B.polish_minima(pi)
    Xex = B.sample_exact(pi, 10000, np.random.default_rng(2026))
    return pi, mu, K, gmm_dk.true_proportions(D, K, DL).numpy(), M, float(np.median(B.dist_to_nearest_minimum(Xex, M)))


def stats(x, pi, mu, K, pt, M, med):
    pv = L.props(L.cells(x, mu), K); b, fc, _ = B.basins(pi, x, M); pb = B.basin_props(b, K)
    return dict(n=len(x), tv_vor=L.tv(pv, pt), tv_basin=L.tv(pb, pt), cov_vor=int((pv >= 0.25 * pt).sum()), cov_basin=int((pb >= 0.25 * pt).sum()),
                q1_ratio=float(np.median(B.dist_to_nearest_minimum(x, M))) / med, frac_converged=fc, p_vor=pv.tolist(), p_basin=pb.tolist())


def last_dict(log, prefix):
    out = None
    for line in open(log):
        if line.startswith("{") and prefix in line:
            try:
                out = ast.literal_eval(line.strip())
            except Exception:
                pass
    return out


def log_stats(arm, log):
    r = {}
    ts = [float(m) for m in re.findall(r"'time/epoch_s': ([0-9.]+)", open(log).read())]
    if ts:
        r["epoch_s_mean"] = float(np.mean(ts)); r["epoch_s_median"] = float(np.median(ts)); r["n_epochs_logged"] = len(ts)
    if arm == "s":
        last = None
        for line in open(log):
            if line.startswith("[dartbank] pop"):
                last = line
        d = last_dict(log, "sbank/n_bank"); r["sbank_last"] = d
        if last:
            m = re.search(r"pop (\d+) n \d+ bank (\d+).*cum cost ([0-9.e+]+) (\{.*\})", last)
            pops, bank, cum, parts = int(m.group(1)) + 1, int(m.group(2)), float(m.group(3)), ast.literal_eval(m.group(4))
            per_pop = (parts["quench"] + parts["dart"]) / pops
            r["cost"] = dict(pops=pops, bank=bank, cum=cum, parts=parts, per_batch_quench_plus_dart=per_pop, per_epoch=per_pop, per_update=per_pop / 100, a_per_epoch=512, a_per_update=5.12,
                             ratio_vs_a=per_pop / 512)
        banks = [int(m) for m in re.findall(r"\[dartbank\] pop \d+ n \d+ bank (\d+)", open(log).read())]
        r["bank_trajectory"] = dict(first=banks[0] if banks else None, last=banks[-1] if banks else None, max=max(banks) if banks else None)
    if arm == "t":
        es = [(float(a), float(w)) for a, w in re.findall(r"\[tis\] pop \d+ ess_alpha ([0-9.e+-]+) ess_w ([0-9.e+-]+)", open(log).read())]
        if es:
            ea, ew = np.array(es).T
            r["tis_ess"] = dict(n=len(es), ess_alpha=dict(first=float(ea[0]), median=float(np.median(ea)), last=float(ea[-1]), min=float(ea.min())),
                                ess_w=dict(first=float(ew[0]), median=float(np.median(ew)), last=float(ew[-1]), min=float(ew.min())))
    return r


c = sys.argv[1]
if c in ("k8d32", "mosw10", "d128"):
    pi, mu, K, pt, M, med = target(c); res = dict(cell=c, median_dist_exact=med, arms={})
    for arm in (("s",) if c == "d128" else ("a", "t", "s")):
        rd = RUNS / f"sd_{arm}_{c}"; per = []
        for e in EVALS:
            x = torch.load(rd / "eval_samples" / f"ep{e:05d}.pt", map_location="cpu", weights_only=False).double().numpy()
            per.append(dict(epoch=e, **stats(x, pi, mu, K, pt, M, med)))
        xf = torch.load(rd / "samples_final.pt", map_location="cpu", weights_only=False)["model_samples"].double().numpy()
        fin = stats(xf, pi, mu, K, pt, M, med)
        agg = {k: dict(mean=float(np.mean([p[k] for p in per])), sd=float(np.std([p[k] for p in per], ddof=1))) for k in ("tv_vor", "tv_basin", "q1_ratio", "cov_vor", "cov_basin")}
        res["arms"][arm] = dict(evals=per, final=fin, agg=agg, log=log_stats(arm, rd / "stdout.log"))
        print(f"{c} {arm}: TV vor {agg['tv_vor']['mean']:.4f}±{agg['tv_vor']['sd']:.4f} basin {agg['tv_basin']['mean']:.4f}±{agg['tv_basin']['sd']:.4f} | cov vor {[p['cov_vor'] for p in per]}+{fin['cov_vor']} "
              f"basin {[p['cov_basin'] for p in per]}+{fin['cov_basin']} | Q1 {agg['q1_ratio']['mean']:.3f} (final {fin['q1_ratio']:.3f}) | final vor {fin['tv_vor']:.4f}", flush=True)
    res["time_s"] = time.time() - T0; json.dump(res, open(OUT / f"analysis_{c}.json", "w"), indent=1, default=float); print(f"{c}: {res['time_s']:.1f} s")
elif c == "verdict":
    R = {k: json.load(open(OUT / f"analysis_{k}.json")) for k in ("k8d32", "mosw10", "d128")}
    SB = json.load(open(ROOT / "figures" / "train_weightfree" / "analysis_d128.json"))["arms"]                                   # step B, reused (a) and (t) at d = 128
    sets = lambda cell: R[cell]["arms"]["s"]["evals"] + [R[cell]["arms"]["s"]["final"]]
    tv = lambda cell, arm: R[cell]["arms"][arm]["agg"]["tv_vor"]["mean"]; q1 = lambda cell, arm: R[cell]["arms"][arm]["agg"]["q1_ratio"]["mean"]
    S1 = dict(cov8_all4=all(z["cov_vor"] == 8 for z in sets("k8d32")), cov_vor_4=[z["cov_vor"] for z in sets("k8d32")], tv_s=tv("k8d32", "s"), tv_a=tv("k8d32", "a"), tv_t=tv("k8d32", "t"),
              ratio=tv("k8d32", "s") / tv("k8d32", "a"))
    S1["tv_ok"] = bool(S1["tv_s"] <= 0.5 * S1["tv_a"]); S1["value"] = bool(S1["cov8_all4"] and S1["tv_ok"]); S1["cov8_final_only"] = bool(sets("k8d32")[-1]["cov_vor"] == 8)
    S2 = dict(cov_basin_4=[z["cov_basin"] for z in sets("mosw10")], q1_4=[z["q1_ratio"] for z in sets("mosw10")])
    S2["cov_ok_all4"] = all(v >= 9 for v in S2["cov_basin_4"]); S2["q1_ok_all4"] = all(v <= 1.5 for v in S2["q1_4"]); S2["value"] = bool(S2["cov_ok_all4"] and S2["q1_ok_all4"])
    S2["final_only"] = bool(S2["cov_basin_4"][-1] >= 9 and S2["q1_4"][-1] <= 1.5)
    a_d128, t_d128 = SB["a"]["agg"]["tv_vor"]["mean"], SB["t"]["agg"]["tv_vor"]["mean"]
    S3 = dict(tv_s=tv("d128", "s"), tv_a_stepB=a_d128, tv_t_stepB=t_d128, threshold=min(0.5 * a_d128, t_d128), vs_a=bool(tv("d128", "s") <= 0.5 * a_d128), vs_t=bool(tv("d128", "s") <= t_d128))
    S3["value"] = bool(S3["vs_a"] and S3["vs_t"]); S3["cov_vor_4"] = [z["cov_vor"] for z in sets("d128")]
    qa = {"k8d32": q1("k8d32", "a"), "mosw10": q1("mosw10", "a"), "d128": SB["a"]["agg"]["q1_ratio"]["mean"]}
    S4 = dict(q1_s={k: q1(k, "s") for k in qa}, q1_a=qa, per_cell={k: bool(q1(k, "s") <= 1.1 * qa[k]) for k in qa}); S4["value"] = all(S4["per_cell"].values())
    V = dict(S1=S1, S2=S2, S3=S3, S4=S4, reading=("l'auto-entraînement sans poids donne un sampler aux bonnes masses là où les méthodes à poids échouent (contribution à défendre)" if (S1["value"] and S3["value"]) else "S1 et S3 ne sont pas tous deux vrais : lecture non établie"))
    json.dump(V, open(OUT / "verdict.json", "w"), indent=1, default=float); print(json.dumps(V, indent=1, ensure_ascii=False, default=float))
