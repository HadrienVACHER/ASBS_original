""" rules_mosw_posthoc.md (A4): post hoc on the 12 MoS-w step-3 controllers, from the saved endpoints and path log W (no new simulation). CPU, one call per controller and arm (< 1 min each).
    python scripts/mosw_posthoc.py prep D | bank D | raw_is D E | dart D E | mala D E | dartbank D E | mala_bank D E | summary
    D in {10, 50}; E in {250, 500, 750, 1000, 1250, 1500}. """
import json, math, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import dart_lib as L, basin_lib as B
T0 = time.perf_counter()
OUT = ROOT / "figures" / "mosw_posthoc"; OUT.mkdir(parents=True, exist_ok=True)
S3 = ROOT / "results" / "mosw_step3"
K, SW, NQ, EPOCHS = 10, (10, 40), 1000, (250, 500, 750, 1000, 1250, 1500)
MED_EX = {D: json.load(open(S3 / "step3_results" / f"d{D}" / "out.json"))["median_dist_exact"] for D in (10, 50)}   # median distance to the nearest minimum under pi (step 3, same quench)


def jsave(p, r):
    r["time_s"] = time.perf_counter() - T0; json.dump(r, open(p, "w"), indent=1, default=float)


def target(D):
    z = np.load(OUT / f"target_d{D}.npz"); TR = json.load(open(ROOT / f"figures/mog_mos_truth_mosw_d{D}.json"))
    mu, wt = z["mu"], z["wt"]
    return L.Mix("S", mu, wt, 1.0), mu, z["M"], np.array(TR["p"]), float(TR["floor_n"]), float(z["nu1"])


def bstats(pi, X, M, ptrue, w=None, b=None):
    if b is None:
        b, _, _ = B.basins(pi, X, M)
    pb = B.basin_props(b, K, w); pv = L.props(L.cells(X, pi.mu), K, w)
    return dict(tv_basin=L.tv(pb, ptrue), cov_basin=int((pb >= 0.25 * ptrue).sum()), tv_vor=L.tv(pv, ptrue), cov_vor=int((pv >= 0.25 * ptrue).sum()), p_basin=pb.tolist())


def endpoints(D, E):
    x = np.load(S3 / "step3_results" / f"d{D}" / f"endpoints_d{D}_ep{E}.npy").astype(np.float64)
    lw = np.load(S3 / "step3_results" / f"d{D}" / f"is_logw_d{D}_ep{E}.npz")["lw"]
    return x, lw


cmd = sys.argv[1]
if cmd == "prep":                                                                   # target parameters from the run configuration (CPU), polished minima
    D = int(sys.argv[2])
    import hydra, train  # noqa: F401 (resolvers)
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(S3 / "asbs" / "results" / f"mosw3_d{D}" / "config.yaml"); OmegaConf.resolve(cfg)
    en = hydra.utils.instantiate(cfg.energy, device="cpu")
    mu = en.dist.means.cpu().double().numpy(); wt = en.dist.weights.cpu().double().numpy()
    M = B.polish_minima(L.Mix("S", mu, wt, 1.0))
    nu1 = float(cfg.sigma_max) ** 2 - float(cfg.sigma_min) ** 2
    np.savez(OUT / f"target_d{D}.npz", mu=mu, wt=wt, M=M, nu1=nu1)
    print(f"prep d{D}: sigma_max {cfg.sigma_max} sigma_min {cfg.sigma_min} nu1 {nu1:.2f}; weights {np.round(wt, 3).tolist()} ({time.perf_counter() - T0:.1f} s)")

elif cmd == "bank":                                                                 # N_q = 1000 starts from N(0, nu1 I), seed 0, converged quench, merge < B.MERGE
    D = int(sys.argv[2]); pi, mu, Mt, ptrue, floor, nu1 = target(D)
    X0 = np.random.default_rng(0).standard_normal((NQ, D)) * math.sqrt(nu1)
    Y, conv, its = B.quench(pi, X0)
    bank, first = [], []
    for i in range(len(Y)):
        if conv[i] and not (bank and np.linalg.norm(np.array(bank) - Y[i], axis=1).min() < B.MERGE):
            bank.append(Y[i].copy()); first.append(i)
    bank = np.array(bank)
    match = [int(np.linalg.norm(Mt - m, axis=1).argmin()) for m in bank]; dm = [float(np.linalg.norm(Mt - m, axis=1).min()) for m in bank]
    found = sorted(set(j for j, d_ in zip(match, dm) if d_ < B.MERGE))
    n_all, seen = None, set()
    for j, i0, d_ in sorted(zip(match, first, dm), key=lambda z: z[1]):
        if d_ < B.MERGE:
            seen.add(j)
            if len(seen) == K:
                n_all = i0 + 1; break
    lab = [int(np.linalg.norm(Mt - y, axis=1).argmin()) for y, cv in zip(Y, conv) if cv and np.linalg.norm(Mt - y, axis=1).min() < B.MERGE]
    np.save(OUT / f"bank_d{D}.npy", bank)
    r = dict(nu1=nu1, n_starts=NQ, frac_converged=float(conv.mean()), quench_iterations=int(its), n_bank=int(len(bank)), true_minima_found=len(found), missing=[j for j in range(K) if j not in found],
             starts_to_find_all=n_all, start_counts_per_true_minimum=np.bincount(np.array(lab, dtype=int), minlength=K).tolist(), weights=pi.w.tolist(), grad_evals_upper_bound=int(NQ * its))
    jsave(OUT / f"bank_d{D}.json", r)
    print(f"bank d{D}: {r['n_bank']} minima ({r['true_minima_found']}/10 true, missing {r['missing']}), starts to find all: {n_all}, converged {r['frac_converged']:.3f}, iters {its}, per minimum {r['start_counts_per_true_minimum']} ({r['time_s']:.1f} s)")

elif cmd in ("raw_is", "dart", "mala", "dartbank", "mala_bank"):
    D, E = int(sys.argv[2]), int(sys.argv[3]); pi, mu, M, ptrue, floor, nu1 = target(D); x, lw = endpoints(D, E); x1 = x[:10000]
    tag = f"d{D}_ep{E}"
    if cmd == "raw_is":
        r = dict(raw=bstats(pi, x1, M, ptrue))
        b20, _, _ = B.basins(pi, x, M); w = np.exp(lw - lw.max())
        ps, k = L.psis_weights(lw); wtr = np.minimum(w, math.sqrt(len(w)) * w.mean())
        r["IS"] = {nm: dict(**bstats(pi, x, M, ptrue, ww, b20), ess=float(ww.sum() ** 2 / (len(ww) * (ww ** 2).sum()))) for nm, ww in (("simple", w), ("psis", ps), ("trunc", wtr))}
        r["khat"] = k; r["n_IS"] = len(lw)
        jsave(OUT / f"raw_is_{tag}.json", r)
        print(f"raw_is {tag}: raw basin TV {r['raw']['tv_basin']:.3f} cov {r['raw']['cov_basin']} | IS basin TV simple {r['IS']['simple']['tv_basin']:.3f} psis {r['IS']['psis']['tv_basin']:.3f} trunc {r['IS']['trunc']['tv_basin']:.3f} "
              f"cov {r['IS']['simple']['cov_basin']}/{r['IS']['psis']['cov_basin']}/{r['IS']['trunc']['cov_basin']} ESS {r['IS']['simple']['ess']:.2e} khat {k:.2f} ({r['time_s']:.1f} s)")
    elif cmd in ("dart", "dartbank"):
        bank = np.load(OUT / f"bank_d{D}.npy") if cmd == "dartbank" else None
        dr = L.darting(pi, x1, mu, ptrue, SW=SW, seed=99, keep_final=True, minima=bank)
        r = dict(minima=dr.get("minima_found"), error=dr.get("error"), mala_step=dr.get("mala_step"), by_S={})
        if dr.get("by_S"):
            Mm = None
            for S in SW:
                v = dr["by_S"][S]; Xf = v.pop("X_final")
                r["by_S"][str(S)] = dict(**bstats(pi, Xf, M, ptrue), cost=int(v["cost"]), jump_acc=v["jump_acc"], mala_acc=v["mala_acc"],
                                         median_dist_ratio=float(np.median(B.dist_to_nearest_minimum(Xf, M))) / MED_EX[D])
        jsave(OUT / f"{cmd}_{tag}.json", r)
        print(f"{cmd} {tag}: minima {r['minima']} {r['error'] or ''} " + " ".join(f"S{S}: basin TV {v['tv_basin']:.4f} cov {v['cov_basin']} (vor {v['tv_vor']:.4f}) jump acc {v['jump_acc']:.2f} med dist {v['median_dist_ratio']:.1f}"
                                                                         for S, v in r["by_S"].items()) + f" ({r['time_s']:.1f} s)")
    else:                                                                          # MALA alone, same step and equal cost as the original-protocol darting (mala; rule)
        dj = json.load(open(OUT / f"{'dart' if cmd == 'mala' else 'dartbank'}_{tag}.json"))   # mala_bank: same, against the bank darting (complement, not in the rule)
        if not dj["by_S"]:
            print(f"{cmd} {tag}: no darting result ({dj['error']})"); jsave(OUT / f"{cmd}_{tag}.json", dict(error=dj["error"])); sys.exit()
        costs = [dj["by_S"][str(S)]["cost"] for S in SW]
        mo = L.mala_only(pi, x1, mu, ptrue, dj["mala_step"], costs, keep_final=True)
        r = dict(by_S={str(S): dict(**bstats(pi, mo[c].pop("X_final"), M, ptrue), cost=c, acc=mo[c]["acc"]) for S, c in zip(SW, costs)})
        jsave(OUT / f"{cmd}_{tag}.json", r)
        print(f"{cmd} {tag}: " + " ".join(f"S{S}: basin TV {v['tv_basin']:.4f} cov {v['cov_basin']} acc {v['acc']:.2f}" for S, v in r["by_S"].items()) + f" ({r['time_s']:.1f} s)")

elif cmd == "summary":
    S_ = {}
    for D in (10, 50):
        pi, mu, M, ptrue, floor, nu1 = target(D); rows = {}
        for E in EPOCHS:
            tag = f"d{D}_ep{E}"; ld = lambda n: json.load(open(OUT / f"{n}_{tag}.json"))
            ri, dj, mj, bj = ld("raw_is"), ld("dart"), ld("mala"), ld("dartbank")
            raw = ri["raw"]["tv_basin"]
            arms = dict(IS=min(ri["IS"].values(), key=lambda v: v["tv_basin"]))
            for nm, j in (("dart", dj), ("mala", mj), ("dartbank", bj)):
                arms[nm] = j["by_S"]["40"] if j.get("by_S") else None
            st = {}
            for nm, v in arms.items():
                if v is None:
                    st[nm] = dict(status="n/a"); continue
                ok = v["cov_basin"] == K and v["tv_basin"] <= 2 * floor; part = v["tv_basin"] <= 0.5 * raw
                st[nm] = dict(tv_basin=v["tv_basin"], cov_basin=v["cov_basin"], status="rattrapé" if ok else ("partiel" if part else "non"))
            rows[E] = dict(raw_tv_basin=raw, raw_cov_basin=ri["raw"]["cov_basin"], arms=st)
        nb = sum(rows[E]["arms"]["dartbank"].get("status") == "rattrapé" for E in EPOCHS)
        npart = sum(rows[E]["arms"]["dartbank"].get("status") in ("rattrapé", "partiel") for E in EPOCHS)
        S_[f"d{D}"] = dict(floor=floor, rows=rows, n_rescued_bank=nb, n_at_least_partial_bank=npart, answer="oui" if nb >= 4 else ("en partie" if npart >= 4 else "non"))
    json.dump(S_, open(OUT / "summary.json", "w"), indent=1, default=float, ensure_ascii=False)
    for k, v in S_.items():
        print(k, "floor", v["floor"], "answer", v["answer"], "rescued", v["n_rescued_bank"], "partial+", v["n_at_least_partial_bank"])
        for E, r in v["rows"].items():
            print(f"  ep{E}: raw {r['raw_tv_basin']:.3f} ({r['raw_cov_basin']}/10) | " + " | ".join(f"{nm} {a.get('tv_basin', float('nan')):.4f} ({a.get('cov_basin', '-')}) {a['status']}" for nm, a in r["arms"].items()))
