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


