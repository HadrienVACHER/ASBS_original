""" rules_eccentric_gmm.md: Eccentric GMM (d = 50), CPU, every call < 1 min.
    python scripts/toy_eccentric.py disc S0 S1       bank discovery from rho = N(0, I), N_q = 1000 starts, seeds S0..S1 (inclusive)
    python scripts/toy_eccentric.py smc S0 S1        adaptive tempered SMC (N = 1000) from rho, then darting with an independent bank, seeds S0..S1; also the collapsed proxy
    python scripts/toy_eccentric.py verdict
    (seeds 9001+ are for timing tests only, never in the rules) """
import json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import dart_lib as L, basin_lib as B
T0 = time.perf_counter(); OUT = ROOT / "figures" / "eccentric_gmm"; OUT.mkdir(parents=True, exist_ok=True)
D, NP, NQ, S, STD, JP = 50, 1000, 1000, 10, 0.25, 0.5                      # target std 1/4 (variance 1/16)
MU = np.vstack([np.zeros(D), 0.75 * np.ones(D)]); W = np.array([0.5, 0.5]); PI = L.Mix("G", MU, W, STD); PT = np.array([0.5, 0.5])
LOGRHO = lambda X: -0.5 * (X ** 2).sum(1) - 0.5 * D * math.log(2 * math.pi)
vor2 = lambda X: ((X[:, None, :] - MU[None]) ** 2).sum(2).argmin(1)


def bank_from_base(seed):
    """ quench of NQ draws of rho = N(0, I), merge at 0.5; returns (bank, number of starts converged to each true minimum, cost) """
    rng = np.random.default_rng([seed, 1]); X0 = rng.standard_normal((NQ, D)); Y, conv, it = B.quench(PI, X0)
    bank = []
    for y in Y[conv]:
        if not bank or np.linalg.norm(np.array(bank) - y, axis=1).min() > B.MERGE: bank.append(y)
    bank = np.array(bank); near = np.array([np.linalg.norm(MU - m, axis=1).argmin() for m in bank]) if len(bank) else np.array([], int)
    ok = [bool(len(bank) and (np.linalg.norm(bank - MU[k], axis=1) < B.MERGE).any()) for k in range(2)]
    nst = [int((np.linalg.norm(Y[conv] - MU[k], axis=1) < B.MERGE).sum()) for k in range(2)]
    return bank, ok, nst, float(conv.mean()), NQ * (it + 1)


def smc(seed, N=NP, ess_frac=0.5, n_mcmc=5):
    """ adaptive tempered SMC on pi_beta ∝ rho^(1-beta) pi^beta from rho; Delta beta by bisection to a conditional ESS of 0.5 N; systematic resampling at every step; n_mcmc MALA moves with an adapted step """
    rng = np.random.default_rng([seed, 2]); X = rng.standard_normal((N, D)); beta, h, cost = 0.0, 0.01, 0
    lp = np.array([0.0]); lpi, gpi = PI.logp_grad(X); lr = LOGRHO(X); steps = 0
    while beta < 1.0:
        d_ = lpi - lr
        def ess(db):
            lw = db * d_; w = np.exp(lw - lw.max()); return w.sum() ** 2 / (w ** 2).sum()
        if ess(1.0 - beta) >= ess_frac * N: db = 1.0 - beta
        else:
            lo, hi = 0.0, 1.0 - beta
            for _ in range(40):
                mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if ess(mid) >= ess_frac * N else (lo, mid)
            db = max(lo, 1e-12)
        lw = db * d_; w = np.exp(lw - lw.max()); w /= w.sum(); beta = min(1.0, beta + db)
        u = (rng.random() + np.arange(N)) / N; idx = np.minimum(np.searchsorted(np.cumsum(w), u), N - 1)         # systematic resampling
        X, lpi, gpi, lr = X[idx], lpi[idx], gpi[idx], lr[idx]
        lt = lambda lpi_, lr_: (1 - beta) * lr_ + beta * lpi_
        gt = lambda X_, gpi_: (1 - beta) * (-X_) + beta * gpi_
        for _ in range(n_mcmc):
            g = gt(X, gpi); Xp = X + h * g + math.sqrt(2 * h) * rng.standard_normal(X.shape)
            lpip, gpip = PI.logp_grad(Xp); lrp = LOGRHO(Xp); gp = gt(Xp, gpip); cost += N
            a = lt(lpip, lrp) - lt(lpi, lr) - ((X - Xp - h * gp) ** 2).sum(1) / (4 * h) + ((Xp - X - h * g) ** 2).sum(1) / (4 * h)
            acc = np.log(rng.random(N)) < a
            X = np.where(acc[:, None], Xp, X); lpi = np.where(acc, lpip, lpi); lr = np.where(acc, lrp, lr); gpi = np.where(acc[:, None], gpip, gpi)
            h *= 1.15 if acc.mean() > 0.6 else 0.85
        steps += 1
        if steps > 2000: break
    return X, steps, cost


def dart(X, seed, bank):
    if len(bank) < 2: return None
    r = L.darting(PI, X, MU, PT, seed=seed, SW=(S,), minima=bank, keep_final=True, jump_prob=JP)
    if "error" in r: return None
    z = r["by_S"][S]; Xf = z.pop("X_final"); return Xf, dict(jump_acc=z["jump_acc"], mala_acc=z["mala_acc"], cost=z["cost"], map=r.get("map"))


def run_smc(seed):
    out = {"seed": seed}; X, steps, cost = smc(seed); p2 = float((vor2(X) == 1).mean()); out.update(smc_steps=steps, smc_cost=cost, smc_p2=p2, smc_count2=int((vor2(X) == 1).sum()), smc_tv=abs(p2 - 0.5))
    bank, ok, nst, cf, bcost = bank_from_base(10_000 + seed); out.update(bank_n=len(bank), bank_has=ok, bank_cost=bcost)
    d_ = dart(X, 1000 + seed, bank)
    if d_ is not None:
        Xf, info = d_; q2 = float((vor2(Xf) == 1).mean()); out.update(dart_p2=q2, dart_tv=abs(q2 - 0.5), dart=info)
    rng = np.random.default_rng([seed, 3]); Xc = STD * rng.standard_normal((NP, D))                         # collapsed proxy: q = N(0, std^2 I) = the near mode alone
    lq = -0.5 * (Xc ** 2).sum(1) / STD ** 2 - D * math.log(STD) - 0.5 * D * math.log(2 * math.pi); lw = PI.logp(Xc) - lq; w = np.exp(lw - lw.max()); w /= w.sum()
    out.update(proxy_is_p2=float(w[vor2(Xc) == 1].sum()), proxy_raw_p2=float((vor2(Xc) == 1).mean()))
    if d_ is not None:
        Xf, _ = dart(Xc, 2000 + seed, bank); q2 = float((vor2(Xf) == 1).mean()); out.update(proxy_dart_p2=q2, proxy_dart_tv=abs(q2 - 0.5))
    return out


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "disc":
        s0, s1 = int(sys.argv[2]), int(sys.argv[3]); res = []
        for s in range(s0, s1 + 1):
            bank, ok, nst, cf, bcost = bank_from_base(s); res.append(dict(seed=s, bank_n=len(bank), has=ok, starts_per_mode=nst, frac_conv=cf, cost=bcost))
        json.dump(res, open(OUT / f"disc_{s0}_{s1}.json", "w"), indent=1); print(sum(r["has"][0] and r["has"][1] for r in res), "/", len(res), "both found; starts in mode 2:", [r["starts_per_mode"][1] for r in res][:30])
    elif c == "smc":
        s0, s1 = int(sys.argv[2]), int(sys.argv[3]); res = []
        for s in range(s0, s1 + 1):
            t = time.perf_counter(); r = run_smc(s); res.append(r); print(s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("dart", "bank_has")}, "%.1fs" % (time.perf_counter() - t), flush=True)
        json.dump(res, open(OUT / f"smc_{s0}_{s1}.json", "w"), indent=1)
    elif c == "verdict":
        import glob
        dd = [r for f in sorted(glob.glob(str(OUT / "disc_*.json"))) for r in json.load(open(f))]; ss = [r for f in sorted(glob.glob(str(OUT / "smc_*.json"))) for r in json.load(open(f))]
        n1 = [r for r in dd if 1 <= r["seed"] <= 100]; sm = [r for r in ss if 1 <= r["seed"] <= 100]
        E1 = dict(n=len(n1), both=int(sum(r["has"][0] and r["has"][1] for r in n1)), starts_mode2_median=float(np.median([r["starts_per_mode"][1] for r in n1])), starts_mode2_mean=float(np.mean([r["starts_per_mode"][1] for r in n1])))
        E1["value"] = bool(len(n1) == 100 and E1["both"] >= 95)
        c1 = lambda r: int(r["smc_count2"]); col = [r for r in sm if min(c1(r), 1000 - c1(r)) <= 10]; non = [r for r in sm if r not in col]
        med = lambda L_, k: float(np.median([r[k] for r in L_])) if L_ else None; frac = lambda L_, k, t: float(np.mean([r[k] <= t for r in L_ if k in r])) if L_ else None
        E2 = dict(n_runs=len(sm), n_collapse=len(col), frac_collapse=len(col) / max(1, len(sm)), count2_values=sorted(c1(r) for r in sm)[:5] + ["..."] + sorted(c1(r) for r in sm)[-5:])
        dk = [r for r in col if "dart_tv" in r]
        E3 = dict(n_collapse=len(col), n_with_dart=len(dk), med_tv_smc=med(col, "smc_tv"), med_tv_dart=med(dk, "dart_tv"), frac_dart_le_0_10=(float(np.mean([r["dart_tv"] <= 0.10 for r in dk])) if dk else None) if len(dk) == len(col) else float(sum(r["dart_tv"] <= 0.10 for r in dk) / max(1, len(col))))
        E3["value"] = bool(len(col) >= 10 and E3["med_tv_smc"] >= 0.40 and E3["med_tv_dart"] is not None and E3["med_tv_dart"] <= 0.05 and E3["frac_dart_le_0_10"] >= 0.90)
        pd_ = [r for r in sm if "proxy_dart_tv" in r]
        E4 = dict(n=len(sm), med_tv_is=float(np.median([abs(r["proxy_is_p2"] - 0.5) for r in sm])), med_tv_dart=med(pd_, "proxy_dart_tv"), frac_dart_le_0_10=float(sum(r["proxy_dart_tv"] <= 0.10 for r in pd_) / max(1, len(sm))))
        E4["value"] = bool(len(sm) == 100 and E4["med_tv_is"] >= 0.45 and E4["med_tv_dart"] <= 0.05 and E4["frac_dart_le_0_10"] >= 0.90)
        nd = [r for r in non if "dart_tv" in r]; E5 = dict(n_noncollapse=len(non), med_diff=(float(np.median([r["dart_tv"] - r["smc_tv"] for r in nd])) if nd else None)); E5["value"] = (bool(E5["med_diff"] <= 0.03) if len(nd) >= 5 else "non évaluable (moins de 5 runs)")
        E6 = dict(smc_cost_median=med(sm, "smc_cost"), bank_cost_median=med(sm, "bank_cost"), dart_cost_median=float(np.median([r["dart"]["cost"] for r in sm if "dart" in r])), smc_steps_median=med(sm, "smc_steps"), dart_jump_acc_median=float(np.median([r["dart"]["jump_acc"] for r in sm if "dart" in r])))
        V = dict(E1=E1, E2=E2, E3=E3, E4=E4, E5=E5, E6=E6); json.dump(V, open(OUT / "verdict.json", "w"), indent=1, default=float); print(json.dumps(V, indent=1, default=float))
    print("%.1f s" % (time.perf_counter() - T0))
