""" Descriptive diagnostic (no verdict), cell (B) (128, 8, 6) x 0.75: is the path-IS degeneracy due to the TERMINAL shape of the AS samples or to the TRAJECTORIES? Same 3 pools as rules/rules_imh_path.md. CPU, one call per pool.
    python scripts/terminal_is.py POOL     POOL in {d128e1000, d128e1250, d128}
    python scripts/terminal_is.py summary
    1) q_hat = sum_k w_k N(m_k, S_k) fitted per basin on a random half of the AS end points (full, diagonal, isotropic); log w_T = -E(X) - log q_hat(X) on the other half: ESS, Pareto k_hat, decomposition of E[w^2];
    2) analytic 1 + chi^2(pi_k || N(m_k, S_k)) between Gaussians (pi_k = N(mu_k, I)). The path log weights of the SAME evaluation points are used for comparison. """
import json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import dart_lib as L, gmm_dk
T0 = time.time(); OUT = ROOT / "figures" / "imh_path"; R = ROOT / "results" / "darting_large_sigma"; D, K = 128, 8
mu = gmm_dk.means(D, K, 6.0).double().numpy(); wt = gmm_dk.weights(K).double().numpy(); wt /= wt.sum(); pi = L.Mix("G", mu, wt, 1.0)
pt = gmm_dk.true_proportions(D, K, 6.0).numpy()


def kish(lw):
    w = np.exp(lw - logsumexp(lw)); return float(1.0 / (len(w) * (w ** 2).sum()))


def decomp(lw, lab):
    n = len(lw); wn = np.exp(lw - logsumexp(lw)) * n; q = np.array([(lab == k).mean() for k in range(K)])
    mk = np.array([wn[lab == k].mean() for k in range(K)]); m2 = np.array([(wn[lab == k] ** 2).mean() for k in range(K)])
    pih = q * mk; T = float((pih ** 2 / q).sum()); Ew2 = float((q * m2).sum())
    return dict(Ew2=Ew2, T_prop=T, share_mass=math.log(T) / math.log(Ew2), chi2p1=(m2 / mk ** 2).tolist())


def gauss_logpdf(x, m, S, kind):
    z = x - m
    if kind == "full":
        c = np.linalg.cholesky(S); y = np.linalg.solve(c, z.T); return -0.5 * (y ** 2).sum(0) - np.log(np.diag(c)).sum() - 0.5 * D * math.log(2 * math.pi)
    v = S if kind == "diag" else np.full(D, S)
    return -0.5 * (z ** 2 / v).sum(1) - 0.5 * np.log(v).sum() - 0.5 * D * math.log(2 * math.pi)


def chi2_closed(k, m, S, kind):
    """ log of  int pi_k^2 / q_k  for pi_k = N(mu_k, I), q_k = N(m, S); None if infinite """
    Sm = S if kind == "full" else (np.diag(S) if kind == "diag" else S * np.eye(D))
    P2 = np.linalg.inv(Sm); A = 2 * np.eye(D) - P2; ev = np.linalg.eigvalsh(A)
    if ev.min() <= 0: return None
    b = 2 * mu[k] - P2 @ m; cst = 2 * mu[k] @ mu[k] - m @ P2 @ m
    logS2 = np.linalg.slogdet(Sm)[1]
    return float(0.5 * logS2 - 0.5 * np.log(ev).sum() + 0.5 * (b @ np.linalg.solve(A, b) - cst))


c = sys.argv[1]
if c != "summary":
    z = [np.load(R / f"sim_{c}_s{s}_p0of1.npz") for s in (1, 2, 3)]; x = np.concatenate([q["x"] for q in z]).astype(np.float64); lwp = np.concatenate([q["lw"] for q in z])
    lab = np.concatenate([np.load(R / f"raw_{c}_s{s}.npz")["basins"] for s in (1, 2, 3)]); assert (lab >= 0).all()
    perm = np.random.default_rng(4242).permutation(len(x)); fit, ev = perm[:len(x) // 2], perm[len(x) // 2:]
    res = dict(pool=c, n=len(x), n_eval=len(ev), path=dict(ess=kish(lwp[ev]), khat=L.khat(lwp[ev]), **decomp(lwp[ev], lab[ev])))
    nk = np.array([(lab[fit] == k).sum() for k in range(K)]); res["fit_counts"] = nk.tolist(); res["w_hat"] = (nk / len(fit)).tolist()
    # terminal Q1-like per-coordinate variance ratio of AS around each basin mean (descriptive)
    res["var_ratio_mean_per_basin"] = [float(x[fit][lab[fit] == k].var(0, ddof=1).mean()) for k in range(K)]
    xe, le = x[ev], lab[ev]; logp = pi.logp(xe)
    for kind in ("iso", "diag", "full"):
        if kind == "full" and nk.min() < 2 * D: res[kind] = "non calculable : moins de 2d points dans un bassin"; continue
        comps, an, ex = [], [], []
        for k in range(K):
            xf = x[fit][lab[fit] == k]; m = xf.mean(0)
            S = np.cov(xf.T) if kind == "full" else (xf.var(0, ddof=1) if kind == "diag" else float(xf.var(0, ddof=1).mean()))
            comps.append(math.log(nk[k] / len(fit)) + gauss_logpdf(xe, m, S, kind)); lc = chi2_closed(k, m, S, kind); an.append(lc)
        logq = logsumexp(np.array(comps), axis=0); lw = logp - logq
        wh = nk / len(fit); ew2_an = (None if any(v is None for v in an) else float(sum(pt[k] ** 2 / wh[k] * math.exp(an[k]) for k in range(K))))
        res[kind] = dict(ess=kish(lw), khat=L.khat(lw), max_w=float(np.exp(lw - logsumexp(lw)).max()), logw_sd=float(lw.std()), **decomp(lw, le),
                         analytic_log_chi2p1=an, analytic_Ew2=ew2_an, analytic_T_prop=float(sum(pt[k] ** 2 / wh[k] for k in range(K))), analytic_ess=(None if ew2_an is None else 1.0 / ew2_an))
    json.dump(res, open(OUT / f"terminal_{c}.json", "w"), indent=1, default=float)
    print(c, "path ess %.4g khat %.2f" % (res["path"]["ess"], res["path"]["khat"]), "| var ratio", np.round(res["var_ratio_mean_per_basin"], 3), flush=True)
    for kind in ("iso", "diag", "full"):
        r = res[kind]
        if isinstance(r, str): print(kind, r); continue
        print(f"  {kind}: ESS {r['ess']:.4g} (khat {r['khat']:.2f}, max w {r['max_w']:.2e}) E[w2] {r['Ew2']:.3g} T {r['T_prop']:.3g} share_mass {r['share_mass']:.3f} | analytic E[w2] {r['analytic_Ew2']} log chi2+1 mean {np.mean([v for v in r['analytic_log_chi2p1'] if v is not None]):.3f} inf {sum(v is None for v in r['analytic_log_chi2p1'])}", flush=True)
    print("%.1f s" % (time.time() - T0))
else:
    S = {c: json.load(open(OUT / f"terminal_{c}.json")) for c in ("d128e1000", "d128e1250", "d128")}
    for key in ("path", "iso", "diag", "full"):
        for q in ("ess", "khat", "Ew2", "share_mass", "analytic_ess"):
            v = [S[c][key].get(q) for c in S if isinstance(S[c][key], dict) and S[c][key].get(q) is not None]
            if v: print(key, q, "mean %.4g sd %.3g" % (np.mean(v), np.std(v, ddof=1)), [float(f"{z_:.4g}") for z_ in v])
