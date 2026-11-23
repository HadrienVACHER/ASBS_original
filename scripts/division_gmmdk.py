""" rules_division_gmmdk.md (A5): offline division on GMM-dk (32, 8, 6) x 0.5 with unshrunk Gaussian templates. CPU, one call per snapshot (< 1 min).
    python scripts/division_gmmdk.py snap E      E in {350, 400, 450, start}   (positive and negative controls computed in the same call)
    python scripts/division_gmmdk.py verdict """
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import dart_lib as L, basin_lib as B, gmm_dk
from adjoint_samplers.utils.division_cv import division_term, make_params
torch.set_default_dtype(torch.float64)
T0 = time.perf_counter()
OUT = ROOT / "figures" / "division_gmmdk"; OUT.mkdir(parents=True, exist_ok=True)
D, K, DL, SMAX, SMIN, N0, R, NBOOT = 32, 8, 6.0, 1.25, 0.001, 50.0, 4, 200
NU1 = SMAX ** 2 - SMIN ** 2
mu = gmm_dk.means(D, K, DL).double().numpy(); w = gmm_dk.weights(K).double().numpy(); w /= w.sum()
pi = L.Mix("G", mu, w, 1.0); ptrue = gmm_dk.true_proportions(D, K, DL).double().numpy()
Emin = lambda X: -pi.logp(X)                                                          # E = -log pi, normalised (Z = 1)


def load(E):
    if E == "start":
        x = torch.load(ROOT / "results" / "wf_start" / "d32" / "start_samples.pt", map_location="cpu", weights_only=False)
        x = x["model_samples"]
    else:
        x = torch.load(ROOT / "results" / "train_weightfree_runs" / "wf_a_d32" / "eval_samples" / f"ep{int(E):05d}.pt", map_location="cpu", weights_only=False)
    return torch.as_tensor(x).double().numpy()


def logN(X, m, S):
    lam, V = np.linalg.eigh(0.5 * (S + S.T)); y = (X - m) @ V
    return -0.5 * (y ** 2 / lam).sum(1) - 0.5 * np.log(2 * math.pi * lam).sum()


def post_mean(xt, kap, v, wts):
    """ exact posterior mean of x1 | x_t under sum_k wts_k N(mu_k, I), x_t | x1 ~ N(kappa x1, v I): (n, D) """
    s2 = kap ** 2 + v                                                                  # (n, 1)
    d2 = (xt ** 2).sum(1)[:, None] - 2 * kap * xt @ mu.T + kap ** 2 * (mu ** 2).sum(1)[None, :]
    lg = np.log(wts)[None, :] - 0.5 * d2 / s2; r = np.exp(lg - logsumexp(lg, axis=1, keepdims=True))
    m = mu[None, :, :] + (kap / s2)[:, :, None] * (xt[:, None, :] - kap[:, :, None] * mu[None, :, :])
    return (r[:, :, None] * m).sum(1)


def beta_stats(x1, par, wq, seed):
    """ projection coefficient of the division term on the exact correction c* = (M* - M^q~) / nu_{1|t}, R draws of t ~ U(0, 1) per x1; bootstrap SE over x1; by kappa band """
    rg = np.random.default_rng(seed); n = len(x1)
    X1 = np.repeat(x1, R, axis=0); t = rg.uniform(1e-6, 1 - 1e-6, len(X1))
    nut = SMAX ** 2 * (1 - (SMAX / SMIN) ** (-2 * t)); kap = (nut / NU1)[:, None]; v = NU1 * kap * (1 - kap); nu1t = NU1 * (1 - kap)
    xt = kap * X1 + np.sqrt(v) * rg.standard_normal(X1.shape)
    cstar = (post_mean(xt, kap, v, w) - post_mean(xt, kap, v, wq)) / nu1t
    term = np.empty_like(X1)
    for lo in range(0, len(X1), 4000):
        sl = slice(lo, lo + 4000)
        term[sl] = division_term(torch.from_numpy(xt[sl]), torch.from_numpy(X1[sl]), torch.from_numpy(Emin(X1[sl])), torch.from_numpy(kap[sl]), torch.from_numpy(v[sl]),
                                 torch.from_numpy(nu1t[sl]), par).numpy()
    num = (term * cstar).sum(1); den = (cstar ** 2).sum(1)
    out = dict(beta=float(num.sum() / den.sum()), finite=float(np.isfinite(term).all(1).mean()))
    nn, dd = num.reshape(n, R).sum(1), den.reshape(n, R).sum(1); rb = np.random.default_rng(seed + 1)
    bs = [nn[i].sum() / dd[i].sum() for i in (rb.integers(n, size=n) for _ in range(NBOOT))]; out["se"] = float(np.std(bs, ddof=1))
    k1 = kap[:, 0]
    out["by_kappa"] = {f"[{a};{b}]": dict(beta=float(num[(k1 >= a) & (k1 < b)].sum() / max(den[(k1 >= a) & (k1 < b)].sum(), 1e-300)), share_den=float(den[(k1 >= a) & (k1 < b)].sum() / den.sum()),
                                          frac=float(((k1 >= a) & (k1 < b)).mean())) for a, b in ((0, 0.1), (0.1, 0.7), (0.7, 1.0001))}
    return out


cmd = sys.argv[1]
if cmd == "snap":
    E = sys.argv[2]; x = load(E); M = B.polish_minima(pi)
    b, fc, _ = B.basins(pi, x, M); ok = b >= 0
    pas = B.basin_props(b, K); wq = w * pas / ptrue; wq /= wq.sum()
    fit, ev = np.zeros(len(x), bool), np.zeros(len(x), bool)
    for k in range(K):
        idx = np.nonzero(b == k)[0]; fit[idx[0::2]] = True; ev[idx[1::2]] = True
    nk = np.array([int(((b == k) & fit).sum()) for k in range(K)])
    muh = np.array([x[(b == k) & fit].mean(0) for k in range(K)])
    Sk = np.array([np.cov(x[(b == k) & fit].T, ddof=1) for k in range(K)])
    Spool = sum((nk[k] - 1) * Sk[k] for k in range(K)) / (nk.sum() - K)
    Xex = B.sample_exact(pi, 10000, np.random.default_rng(2026)); bex, _, _ = B.basins(pi, Xex, M)
    TEMPL = dict(main=(muh, np.array([(nk[k] * Sk[k] + N0 * Spool) / (nk[k] + N0) for k in range(K)])), pooled=(muh, np.array([Spool] * K)), oracle=(mu, np.array([np.eye(D)] * K)))
    res = dict(E=E, n=len(x), frac_converged=fc, n_fit=nk.tolist(), p_AS=pas.tolist(), p=ptrue.tolist(), tv_AS=L.tv(pas, ptrue), templates={})
    for nm, (mh, Sh) in TEMPL.items():
        r = dict(by_basin=[])
        logZ = np.zeros(K)
        for k in range(K):
            lu_f = logN(x[(b == k) & fit], mh[k], Sh[k]) + Emin(x[(b == k) & fit]); logZ[k] = -(logsumexp(lu_f) - math.log(len(lu_f)))   # Gelfand-Dey: Z_k = 1 / mean u
            lu_e = logN(x[(b == k) & ev], mh[k], Sh[k]) + Emin(x[(b == k) & ev]); lu_x = logN(Xex[bex == k], mh[k], Sh[k]) + Emin(Xex[bex == k])
            lam = np.linalg.eigvalsh(Sh[k])
            r["by_basin"].append(dict(sd_eval=float(lu_e.std(ddof=1)), sd_exact=float(lu_x.std(ddof=1)), khat_eval=L.khat(lu_e), n_eval=len(lu_e),
                                      sd_noise_ref=float(math.sqrt(0.5 * ((1 / lam - 1) ** 2).sum()))))
        pgd = np.exp(logZ - logsumexp(logZ)); dp = ptrue - pas
        r["masses"] = dict(p_GD=pgd.tolist(), sum_Z=float(np.exp(logZ).sum()), tv_GD=L.tv(pgd, ptrue), tv_AS=L.tv(pas, ptrue), proj=float((pgd - pas) @ dp / (dp @ dp)))
        par = make_params(torch.from_numpy(mh), torch.from_numpy(Sh), torch.from_numpy(-logZ), shrink=1.0)
        r["beta"] = beta_stats(x[ev], par, wq, seed=11)
        if nm == "main":                                                               # negative control: same templates, AS masses (no mass information)
            parn = make_params(torch.from_numpy(mh), torch.from_numpy(Sh), torch.from_numpy(-np.log(pas)), shrink=1.0)
            res["negative_control"] = beta_stats(x[ev], parn, wq, seed=11)
        res["templates"][nm] = r
        print(f"snap {E} {nm}: SD log u eval " + " ".join(f"{z['sd_eval']:.2f}" for z in r["by_basin"]) + " | exact " + " ".join(f"{z['sd_exact']:.2f}" for z in r["by_basin"])
              + f" | noise ref " + " ".join(f"{z['sd_noise_ref']:.2f}" for z in r["by_basin"]) + f" | beta {r['beta']['beta']:.3f} ± {r['beta']['se']:.3f} | TV GD {r['masses']['tv_GD']:.4f} vs AS {r['masses']['tv_AS']:.4f} proj {r['masses']['proj']:.2f}", flush=True)
    # positive control: oracle parameters on exact draws of q~
    rg = np.random.default_rng(3); comp = rg.choice(K, size=5000, p=wq); xq = mu[comp] + rg.standard_normal((5000, D))
    paro = make_params(torch.from_numpy(mu), torch.from_numpy(np.array([np.eye(D)] * K)), torch.from_numpy(-np.log(w)), shrink=1.0)
    res["positive_control"] = beta_stats(xq, paro, wq, seed=11)
    res["time_s"] = time.perf_counter() - T0
    json.dump(res, open(OUT / f"snap_{E}.json", "w"), indent=1, default=float)
    print(f"snap {E}: positive control beta {res['positive_control']['beta']:.3f} ± {res['positive_control']['se']:.3f}; negative control {res['negative_control']['beta']:.3f} ± {res['negative_control']['se']:.3f}; "
          f"n_fit {nk.tolist()}; TV AS {res['tv_AS']:.4f} ({res['time_s']:.1f} s)")

elif cmd == "verdict":
    S = [json.load(open(OUT / f"snap_{e}.json")) for e in (350, 400, 450)]; V = {}
    for nm in ("main", "pooled", "oracle"):
        sd = np.array([[z["sd_eval"] for z in s["templates"][nm]["by_basin"]] for s in S]); be = [s["templates"][nm]["beta"]["beta"] for s in S]
        V[nm] = dict(sd_eval_mean=sd.mean(0).tolist(), sd_eval_sd=sd.std(0, ddof=1).tolist(), dispersion_ok=bool((sd.mean(0) <= 1.0).all()), beta=[float(np.mean(be)), float(np.std(be, ddof=1))], beta_ok=bool(np.mean(be) >= 0.5))
        V[nm]["both"] = V[nm]["dispersion_ok"] and V[nm]["beta_ok"]
    pc = [s["positive_control"]["beta"] for s in S]; nc = [s["negative_control"]["beta"] for s in S]
    V["positive_control"] = dict(beta=[float(np.mean(pc)), float(np.std(pc, ddof=1))], ok=bool(0.8 <= np.mean(pc) <= 1.2))
    V["negative_control"] = dict(beta=[float(np.mean(nc)), float(np.std(nc, ddof=1))], ok=bool(abs(np.mean(nc)) <= 0.2))
    V["A5"] = V["main"]["both"]
    V["reading"] = ("l'échec sur DW-4 tenait à la seule contrainte de domination" if V["A5"] else
                    ("faux par la dispersion du gabarit principal, covariance commune vraie : échec attribué à l'estimation des covariances (d/n)" if (not V["main"]["dispersion_ok"] and V["pooled"]["both"]) else "A5 faux"))
    json.dump(V, open(OUT / "verdict.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(V, indent=1, ensure_ascii=False))
