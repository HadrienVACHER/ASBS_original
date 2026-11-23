""" rules_darting_large_sigma.md: post-hoc darting where path IS collapses (GMM-dk (128, 8, 6) and (256, 8, 6) at sigma_max x 0.75, (32, 8, 4) at x 2). CPU, each call < 1 min.
    python scripts/darting_large_sigma.py floor C
    python scripts/darting_large_sigma.py sim C SET [--part p --nparts n]     simulate 10^4 paths (or the part p of n)
    python scripts/darting_large_sigma.py raw C SET                          cells and basins of the raw end points
    python scripts/darting_large_sigma.py dart C SET                         darting S in {1, 5, 10} + basins of the final states
    python scripts/darting_large_sigma.py mala C SET                         MALA alone at the cost of each S + basins
    python scripts/darting_large_sigma.py is C                               path IS per set and pooled (simple, PSIS; Voronoi and basins)
    python scripts/darting_large_sigma.py verdict
    C in {d32, d128, d256}, SET in {1, 2, 3}. """
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import dart_lib as L
import basin_lib as B
import gmm_dk

T0 = time.perf_counter()
CTRL = {"d128": dict(d=128, K=8, delta=6.0, ckpt="figures/gmmdk_step4/controllers/gmmdk4_d128_sx0.75_ep1500.pt", cfg="figures/gmmdk_step4/configs/gmmdk4_d128_sx0.5.config.yaml", qjson="figures/gmmdk_step4/q_gmmdk4_d128_sx0.75.json"),
        "d256": dict(d=256, K=8, delta=6.0, ckpt="figures/gmmdk_step4/controllers/gmmdk4_d256_sx0.75_ep1500.pt", cfg="figures/gmmdk_step4/configs/gmmdk4_d256_sx0.5.config.yaml", qjson="figures/gmmdk_step4/q_gmmdk4_d256_sx0.75.json"),
        "d32": dict(d=32, K=8, delta=4.0, ckpt="results/gmm_dk_window/w1/gmmdk_d32_K8_D4_s0_sx2/controller_final.pt", cfg="results/gmm_dk_window/w1/gmmdk_d32_K8_D4_s0_sx2/config.yaml", qjson=None)}
for _d in (128, 256):                                                                           # rules_darting_large_sigma_snapshots.md: the other saved x0.75 snapshots
    for _e in (750, 1000, 1250):
        CTRL[f"d{_d}e{_e}"] = dict(CTRL[f"d{_d}"], ckpt=f"figures/gmmdk_step4/controllers/gmmdk4_d{_d}_sx0.75_ep{_e}.pt")
for _m, _rd in (("x1", "results/gmm_dk_sweep/gmmdk_d32_K8_D8_s0"), ("x0.5", "results/gmm_dk_window/w1/gmmdk_d32_K8_D8_s0_sx0.5"), ("x0.75", "results/gmm_dk_window/w1/gmmdk_d32_K8_D8_s0_sx0.75"),
                 ("x1.5", "results/gmm_dk_window/w1/gmmdk_d32_K8_D8_s0_sx1.5"), ("x2", "results/gmm_dk_window/w1/gmmdk_d32_K8_D8_s0_sx2")):   # rules_bank_darting_dk.md: (32, 8, 8)
    CTRL[f"k8{_m}"] = dict(d=32, K=8, delta=8.0, ckpt=f"{_rd}/controller_final.pt", cfg=f"{_rd}/config.yaml", qjson=None)
SW = (1, 5, 10); N = 10000
OUTF = ROOT / "figures" / "darting_large_sigma"; OUTF.mkdir(parents=True, exist_ok=True)
OUTR = ROOT / "results" / "darting_large_sigma"; OUTR.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("ctrl", nargs="?"); ap.add_argument("set", nargs="?", type=int)
ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
a = ap.parse_args()


def target(c):
    C = CTRL[c]; D, K, DELTA = C["d"], C["K"], C["delta"]
    mu = gmm_dk.means(D, K, DELTA).double().numpy(); wt = gmm_dk.weights(K).double().numpy(); wt = wt / wt.sum()
    return D, K, DELTA, mu, L.Mix("G", mu, wt, 1.0), gmm_dk.true_proportions(D, K, DELTA).numpy()


def jload(p):
    return json.load(open(p)) if Path(p).exists() else {}


def jsave(p, d):
    json.dump(d, open(p, "w"), indent=1, default=float)


def bprops(b, K, w=None):
    return B.basin_props(b, K, w)


c = a.ctrl
if a.cmd == "floor":
    D, K, DELTA, mu, pi, ptrue = target(c)
    g = torch.Generator().manual_seed(5); _mu = torch.tensor(mu[:, :K])
    fl = float(np.median([L.tv(np.bincount(gmm_dk.voronoi(gmm_dk.sample_exact(N, D, K, DELTA, g)[0].double(), _mu).numpy(), minlength=K) / N, ptrue) for _ in range(100)]))
    jsave(OUTF / f"floor_{c}.json", dict(floor=fl, p=ptrue.tolist(), time_s=time.perf_counter() - T0)); print(f"floor {c}: {fl:.4f} ({time.perf_counter() - T0:.1f} s)")

elif a.cmd == "sim":
    import train  # registers resolvers
    import hydra
    from omegaconf import OmegaConf
    from adjoint_samplers.components.sde import ControlledSDE
    from adjoint_samplers.utils import girsanov, train_utils
    Cc = CTRL[c]; cfg = OmegaConf.load(ROOT / Cc["cfg"])
    if Cc["qjson"]:
        cfg.sigma_max = float(json.load(open(ROOT / Cc["qjson"]))["sigma_max"])                   # declared: the x0.75 config = the x0.5 config with the x0.75 sigma_max
    OmegaConf.resolve(cfg); assert int(cfg.dim) == Cc["d"]
    ts = train_utils.get_timesteps(**cfg.timesteps); energy = hydra.utils.instantiate(cfg.energy, device="cpu")
    ref = hydra.utils.instantiate(cfg.ref_sde); ctrl = hydra.utils.instantiate(cfg.controller)
    ctrl.load_state_dict(torch.load(ROOT / Cc["ckpt"], map_location="cpu", weights_only=False)["controller"]); sde = ControlledSDE(ref, ctrl)
    n = N // a.nparts; torch.manual_seed(1000 * a.set + a.part)
    with torch.no_grad():
        lw, x = girsanov.log_path_weights(sde, energy, torch.zeros(n, Cc["d"]), ts)
    np.savez_compressed(OUTR / f"sim_{c}_s{a.set}_p{a.part}of{a.nparts}.npz", x=x.double().numpy().astype(np.float32), lw=lw.double().numpy(), sigma_max=float(cfg.sigma_max))
    print(f"sim {c} set {a.set} part {a.part}/{a.nparts}: n {n}, sigma_max {float(cfg.sigma_max)}, {time.perf_counter() - T0:.1f} s")

else:
    def endpoints(c, s):
        ps = sorted(OUTR.glob(f"sim_{c}_s{s}_p*of*.npz"))
        z = [np.load(p) for p in ps]; x = np.concatenate([q["x"] for q in z]).astype(np.float64); lw = np.concatenate([q["lw"] for q in z])
        assert len(x) == N, (c, s, len(x)); return x, lw

    if a.cmd == "raw":
        D, K, DELTA, mu, pi, ptrue = target(c); M = B.polish_minima(pi); x, lw = endpoints(c, a.set)
        cl = L.cells(x, mu); b, fc, fa = B.basins(pi, x, M)
        np.savez_compressed(OUTR / f"raw_{c}_s{a.set}.npz", cells=cl, basins=b)
        r = dict(tv_vor=L.tv(L.props(cl, K), ptrue), tv_basin=L.tv(bprops(b, K), ptrue), frac_converged=fc, frac_assigned=fa, time_s=time.perf_counter() - T0)
        jsave(OUTF / f"raw_{c}_s{a.set}.json", r); print(f"raw {c} s{a.set}: TV vor {r['tv_vor']:.4f} basin {r['tv_basin']:.4f} conv {fc:.3f} ({r['time_s']:.1f} s)")

    elif a.cmd == "dart":
        D, K, DELTA, mu, pi, ptrue = target(c); M = B.polish_minima(pi); x, _ = endpoints(c, a.set)
        dr = L.darting(pi, x, mu, ptrue, SW=SW, seed=99 + a.set, keep_final=True)
        r = dict(minima_found=dr.get("minima_found"), error=dr.get("error"), mala_step=dr.get("mala_step"), by_S={})
        for S in SW if dr.get("by_S") else []:
            v = dr["by_S"][S]; Xf = v.pop("X_final"); bb, _, _ = B.basins(pi, Xf, M)
            r["by_S"][str(S)] = dict(tv_vor=v["tv"], tv_basin=L.tv(bprops(bb, K), ptrue), cost=int(v["cost"]), jump_acc=v.get("jump_acc"))
        r["time_s"] = time.perf_counter() - T0; jsave(OUTF / f"dart_{c}_s{a.set}.json", r)
        print(f"dart {c} s{a.set}: minima {r['minima_found']} " + " ".join(f"S{S}: vor {v['tv_vor']:.4f} basin {v['tv_basin']:.4f} cost {v['cost']:.2e}" for S, v in r["by_S"].items()) + f" ({r['time_s']:.1f} s)")

    elif a.cmd == "mala":
        D, K, DELTA, mu, pi, ptrue = target(c); M = B.polish_minima(pi); x, _ = endpoints(c, a.set)
        dj = json.load(open(OUTF / f"dart_{c}_s{a.set}.json")); costs = [dj["by_S"][str(S)]["cost"] for S in SW]
        mo = L.mala_only(pi, x, mu, ptrue, dj["mala_step"], costs, seed=77 + a.set, keep_final=True); r = dict(by_cost={})
        for cst, v in mo.items():
            bb, _, _ = B.basins(pi, v.pop("X_final"), M); r["by_cost"][str(cst)] = dict(tv_vor=v["tv"], tv_basin=L.tv(bprops(bb, K), ptrue))
        r["time_s"] = time.perf_counter() - T0; jsave(OUTF / f"mala_{c}_s{a.set}.json", r)
        print(f"mala {c} s{a.set}: " + " ".join(f"cost {k}: vor {v['tv_vor']:.4f} basin {v['tv_basin']:.4f}" for k, v in r["by_cost"].items()) + f" ({r['time_s']:.1f} s)")

    elif a.cmd == "is":
        D, K, DELTA, mu, pi, ptrue = target(c); res = dict(sets={})
        allw, allc, allb = [], [], []
        for s in (1, 2, 3):
            x, lw = endpoints(c, s); z = np.load(OUTR / f"raw_{c}_s{s}.npz"); cl, bb = z["cells"], z["basins"]
            st = L.is_stats(lw, cl, K, ptrue); w = np.exp(lw - lw.max()); pw, kp = L.psis_weights(lw)
            res["sets"][str(s)] = dict(tv_vor=st["tv"], ess=st["ess"], khat=st["khat"], tv_basin=L.tv(bprops(bb, K, w), ptrue),
                                       psis_tv_vor=L.weighted_stats(pw, cl, K, ptrue)["tv"], psis_tv_basin=L.tv(bprops(bb, K, pw), ptrue))
            allw.append(lw); allc.append(cl); allb.append(bb)
        lw, cl, bb = np.concatenate(allw), np.concatenate(allc), np.concatenate(allb)
        st = L.is_stats(lw, cl, K, ptrue); w = np.exp(lw - lw.max()); pw, kp = L.psis_weights(lw)
        res["pool"] = dict(n=len(lw), tv_vor=st["tv"], ess=st["ess"], khat=st["khat"], tv_basin=L.tv(bprops(bb, K, w), ptrue),
                           psis_tv_vor=L.weighted_stats(pw, cl, K, ptrue)["tv"], psis_tv_basin=L.tv(bprops(bb, K, pw), ptrue))
        res["time_s"] = time.perf_counter() - T0; jsave(OUTF / f"is_{c}.json", res)
        print(f"is {c}: pool n {len(lw)} TV vor {res['pool']['tv_vor']:.4f} basin {res['pool']['tv_basin']:.4f} ESS {res['pool']['ess']:.2e} khat {res['pool']['khat']:.2f} PSIS vor {res['pool']['psis_tv_vor']:.4f} ({res['time_s']:.1f} s)")

    elif a.cmd == "bank":                                                                       # rules_bank_darting_dk.md: minima bank by quench from the base law N(0, nu_1 I), N_q = 1000 starts
        from omegaconf import OmegaConf
        import train  # noqa: F401 (resolvers)
        D, K, DELTA, mu, pi, ptrue = target(c); cfg = OmegaConf.load(ROOT / CTRL[c]["cfg"]); OmegaConf.resolve(cfg)
        nu1 = float(cfg.sigma_max) ** 2 - float(cfg.sigma_min) ** 2
        X0 = np.random.default_rng(0).standard_normal((1000, D)) * np.sqrt(nu1)
        Y, conv, its = B.quench(pi, X0)
        bank, first = [], []
        for i in range(len(Y)):
            if not conv[i]:
                continue
            if bank and np.linalg.norm(np.array(bank) - Y[i], axis=1).min() < B.MERGE:
                continue
            bank.append(Y[i].copy()); first.append(i)
        bank = np.array(bank); Mtrue = B.polish_minima(pi)
        match = [int(np.linalg.norm(Mtrue - m, axis=1).argmin()) for m in bank]; dmatch = [float(np.linalg.norm(Mtrue - m, axis=1).min()) for m in bank]
        found = sorted(set(j for j, dm in zip(match, dmatch) if dm < B.MERGE))
        n_all = None
        if len(found) == K:
            seen = set()
            for j, i0 in sorted(zip(match, first), key=lambda z: z[1]):
                seen.add(j)
                if len(seen) == K:
                    n_all = i0 + 1; break
        np.save(OUTR / f"bank_{c}.npy", bank)
        r = dict(nu1=nu1, n_starts=1000, frac_converged=float(conv.mean()), quench_iterations=int(its), n_bank=int(len(bank)), true_minima_found=len(found), starts_to_find_all=n_all,
                 grad_evals_upper_bound=int(1000 * its), start_counts_per_true_minimum=np.bincount(np.array([int(np.linalg.norm(Mtrue - y, axis=1).argmin()) for y, cv in zip(Y, conv) if cv and np.linalg.norm(Mtrue - y, axis=1).min() < B.MERGE], dtype=int), minlength=K).tolist(),
                 time_s=time.perf_counter() - T0)
        jsave(OUTF / f"bank_{c}.json", r); print(f"bank {c}: {r['n_bank']} minima ({r['true_minima_found']}/8 true), starts to find all 8: {n_all}, converged {r['frac_converged']:.3f}, iters {its} ({r['time_s']:.1f} s)")

    elif a.cmd == "dartbank":                                                                   # darting with the external bank (same protocol) + MALA alone at equal cost
        D, K, DL, mu, pi, ptrue = target(c); x, _ = endpoints(c, a.set); bank = np.load(OUTR / f"bank_{c}.npy")
        dr = L.darting(pi, x, mu, ptrue, SW=SW, seed=99 + a.set, keep_final=True, minima=bank)
        r = dict(minima=dr.get("minima_found"), error=dr.get("error"), mala_step=dr.get("mala_step"), by_S={})
        for S in SW if dr.get("by_S") else []:
            v = dr["by_S"][S]; Xf = v.pop("X_final"); pv = L.props(L.cells(Xf, mu), K)
            r["by_S"][str(S)] = dict(tv_vor=v["tv"], coverage=int((pv >= 0.25 * ptrue).sum()), cost=int(v["cost"]), jump_acc=v.get("jump_acc"))
        if dr.get("by_S"):
            costs = [r["by_S"][str(S)]["cost"] for S in SW]
            mo = L.mala_only(pi, x, mu, ptrue, dr["mala_step"], costs, seed=77 + a.set, keep_final=True)
            r["mala"] = {str(k): dict(tv_vor=v["tv"], coverage=int((L.props(L.cells(v.pop("X_final"), mu), K) >= 0.25 * ptrue).sum())) for k, v in mo.items()}
        pr = L.props(L.cells(x, mu), K); r["raw"] = dict(tv_vor=L.tv(pr, ptrue), coverage=int((pr >= 0.25 * ptrue).sum()))
        r["time_s"] = time.perf_counter() - T0; jsave(OUTF / f"dartbank_{c}_s{a.set}.json", r)
        print(f"dartbank {c} s{a.set}: raw TV {r['raw']['tv_vor']:.4f} cov {r['raw']['coverage']} | " + " ".join(f"S{S}: TV {v['tv_vor']:.4f} cov {v['coverage']} acc {v['jump_acc']:.2f}" for S, v in r["by_S"].items()) + f" ({r['time_s']:.1f} s)")

    elif a.cmd == "verdict_bank":
        fl = jload(OUTF / "floor_k8x1.json")["floor"]; out = {}
        sets = [jload(OUTF / f"dartbank_k8x1_s{s}.json") for s in (1, 2, 3)]
        tv10 = [z["by_S"]["10"]["tv_vor"] for z in sets]; cov10 = [z["by_S"]["10"]["coverage"] for z in sets]
        out["A3"] = dict(tv_S10=[float(np.mean(tv10)), float(np.std(tv10, ddof=1))], coverage_S10=cov10, floor=fl, value=bool(all(c_ == 8 for c_ in cov10) and np.mean(tv10) <= 2 * fl))
        jsave(OUTF / "verdict_bank.json", out); print(json.dumps(out, indent=1))

    elif a.cmd == "verdict_snap":                                                                # rule A1: G (S in {5, 10}) and H_saut_bassin on >= 2 of the 3 snapshots, at each dimension
        V = {}
        for dd in (128, 256):
            snaps = {}
            for e in (750, 1000, 1250, 1500):
                c = f"d{dd}e{e}" if e != 1500 else f"d{dd}"
                fl = jload(OUTF / f"floor_d{dd}.json")["floor"]; IS = jload(OUTF / f"is_{c}.json")
                dj = {s: jload(OUTF / f"dart_{c}_s{s}.json") for s in (1, 2, 3)}; mj = {s: jload(OUTF / f"mala_{c}_s{s}.json") for s in (1, 2, 3)}
                rj = {s: jload(OUTF / f"raw_{c}_s{s}.json") for s in (1, 2, 3)}
                out = dict(raw_vor=float(np.mean([rj[s]["tv_vor"] for s in (1, 2, 3)])), raw_basin=float(np.mean([rj[s]["tv_basin"] for s in (1, 2, 3)])),
                           IS_pool=IS["pool"], IS_set_ess=float(np.mean([IS["sets"][str(s)]["ess"] for s in (1, 2, 3)])), by_S={})
                G = False
                for S in (5, 10):
                    d_v = [dj[s]["by_S"][str(S)]["tv_vor"] for s in (1, 2, 3)]; d_b = [dj[s]["by_S"][str(S)]["tv_basin"] for s in (1, 2, 3)]
                    cost = float(np.mean([dj[s]["by_S"][str(S)]["cost"] for s in (1, 2, 3)])); LB = fl * math.sqrt(N / max(IS["pool"]["ess"] * cost, 1e-300)); TVeq = min(IS["pool"]["tv_vor"], LB)
                    m_v = [mj[s]["by_cost"][str(dj[s]["by_S"][str(S)]["cost"])]["tv_vor"] for s in (1, 2, 3)]; m_b = [mj[s]["by_cost"][str(dj[s]["by_S"][str(S)]["cost"])]["tv_basin"] for s in (1, 2, 3)]
                    g = bool(np.mean(d_v) <= 0.5 * TVeq and np.mean(d_v) <= 2 * fl); G |= g
                    out["by_S"][str(S)] = dict(TV_eq=TVeq, LB=LB, dart_vor=[float(np.mean(d_v)), float(np.std(d_v, ddof=1))], dart_basin=[float(np.mean(d_b)), float(np.std(d_b, ddof=1))],
                                               mala_vor=[float(np.mean(m_v)), float(np.std(m_v, ddof=1))], mala_basin=[float(np.mean(m_b)), float(np.std(m_b, ddof=1))], G=g)
                H = bool(out["by_S"]["10"]["dart_basin"][0] <= 0.5 * out["by_S"]["10"]["mala_basin"][0])
                out.update(G=G, H=H, both=bool(G and H)); snaps[e] = out
                print(f"d{dd} ep{e}: G {G} H {H} | dart S10 vor {out['by_S']['10']['dart_vor'][0]:.4f} basin {out['by_S']['10']['dart_basin'][0]:.4f} | MALA basin {out['by_S']['10']['mala_basin'][0]:.4f} | IS pool ESS {IS['pool']['ess']:.2e} khat {IS['pool']['khat']:.2f}")
            n_ok = sum(snaps[e]["both"] for e in (750, 1000, 1250))
            V[f"d{dd}"] = dict(snapshots=snaps, n_both_750_1250=int(n_ok), A1=bool(n_ok >= 2))
        V["A1"] = bool(all(V[f"d{dd}"]["A1"] for dd in (128, 256)))
        jsave(OUTF / "verdict_snapshots.json", V); print("A1 (>= 2/3 snapshots at each dimension):", V["A1"], {k: V[k]["n_both_750_1250"] for k in ("d128", "d256")})

    elif a.cmd == "lwvar":                                                                       # descriptive: Var(log W) within / between basins, and ESS of W^alpha
        def decomp(lw, lab):
            lw = np.asarray(lw, float); m = lab >= 0; lw, lab = lw[m], lab[m]
            tot = float(lw.var()); ks = np.unique(lab); p = np.array([(lab == k).mean() for k in ks])
            within = float(sum(pk * lw[lab == k].var() for pk, k in zip(p, ks))); between = float(sum(pk * (lw[lab == k].mean() - lw.mean()) ** 2 for pk, k in zip(p, ks)))
            from scipy.special import logsumexp as lse
            ess = {str(al): float(np.exp(2 * lse(al * lw) - lse(2 * al * lw)) / len(lw)) for al in (0.25, 0.5, 1.0)}
            return dict(n=int(len(lw)), var=tot, within=within, between=between, between_frac=between / tot if tot > 0 else float("nan"), ess_alpha=ess)
        R = {}
        for dd in (128, 256):
            for e in (750, 1000, 1250, 1500):                                                    # x0.75: our simulated sets, basins of the quench
                c = f"d{dd}e{e}" if e != 1500 else f"d{dd}"
                lws, labs = [], []
                for s in (1, 2, 3):
                    _, lw = endpoints(c, s); lws.append(lw); labs.append(np.load(OUTR / f"raw_{c}_s{s}.npz")["basins"])
                R[f"x0.75_d{dd}_ep{e}"] = decomp(np.concatenate(lws), np.concatenate(labs))
            for e in (250, 500, 750, 1000, 1250, 1500):                                          # x0.5: step-4 log w, Voronoi cell as the basin proxy (declared)
                z = np.load(ROOT / "figures" / "gmmdk_step4" / "logw" / f"is_logw_gmmdk4_d{dd}_sx0.5_ep{e}.npz")
                R[f"x0.5_d{dd}_ep{e}"] = decomp(z["lw"], z["cells"])
        for k, v in R.items():
            print(f"{k:22} var {v['var']:9.3f} within {v['within']:9.3f} between {v['between']:8.3f} (between {100 * v['between_frac']:5.1f} %)  ESS(W^a) a=0.25 {v['ess_alpha']['0.25']:.3f} a=0.5 {v['ess_alpha']['0.5']:.3e} a=1 {v['ess_alpha']['1.0']:.2e}")
        jsave(OUTF / "lwvar.json", R)

    elif a.cmd == "verdict":
        V = {}
        for c in CTRL:
            if not (OUTF / f"is_{c}.json").exists():
                continue
            fl = jload(OUTF / f"floor_{c}.json")["floor"]; IS = jload(OUTF / f"is_{c}.json")
            dj = {s: jload(OUTF / f"dart_{c}_s{s}.json") for s in (1, 2, 3)}; mj = {s: jload(OUTF / f"mala_{c}_s{s}.json") for s in (1, 2, 3)}; rj = {s: jload(OUTF / f"raw_{c}_s{s}.json") for s in (1, 2, 3)}
            ms = lambda vals: dict(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)), values=[float(v) for v in vals])
            out = dict(floor=fl, raw=dict(tv_vor=ms([rj[s]["tv_vor"] for s in (1, 2, 3)]), tv_basin=ms([rj[s]["tv_basin"] for s in (1, 2, 3)])),
                       IS_set=dict(tv_vor=ms([IS["sets"][str(s)]["tv_vor"] for s in (1, 2, 3)]), tv_basin=ms([IS["sets"][str(s)]["tv_basin"] for s in (1, 2, 3)]),
                                   ess=ms([IS["sets"][str(s)]["ess"] for s in (1, 2, 3)]), khat=ms([IS["sets"][str(s)]["khat"] for s in (1, 2, 3)]),
                                   psis_tv_vor=ms([IS["sets"][str(s)]["psis_tv_vor"] for s in (1, 2, 3)])), IS_pool=IS["pool"], by_S={})
            G_any, G_sets = False, [False, False, False]
            for S in SW:
                d_v = [dj[s]["by_S"][str(S)]["tv_vor"] for s in (1, 2, 3)]; d_b = [dj[s]["by_S"][str(S)]["tv_basin"] for s in (1, 2, 3)]
                cost = float(np.mean([dj[s]["by_S"][str(S)]["cost"] for s in (1, 2, 3)]))
                LB = fl * math.sqrt(N / max(IS["pool"]["ess"] * cost, 1e-300)); TVeq = min(IS["pool"]["tv_vor"], LB)
                m_v = [mj[s]["by_cost"][str(dj[s]["by_S"][str(S)]["cost"])]["tv_vor"] for s in (1, 2, 3)]
                m_b = [mj[s]["by_cost"][str(dj[s]["by_S"][str(S)]["cost"])]["tv_basin"] for s in (1, 2, 3)]
                G = bool(np.mean(d_v) <= 0.5 * TVeq and np.mean(d_v) <= 2 * fl); G_any |= G
                per = [bool(v <= 0.5 * TVeq and v <= 2 * fl) for v in d_v]; G_sets = [x or y for x, y in zip(G_sets, per)]
                out["by_S"][str(S)] = dict(cost=cost, LB=LB, TV_eq=TVeq, dart_vor=ms(d_v), dart_basin=ms(d_b), mala_vor=ms(m_v), mala_basin=ms(m_b), G=G, G_per_set=per)
            s10 = out["by_S"]["10"]
            H = bool(s10["dart_basin"]["mean"] <= 0.5 * s10["mala_basin"]["mean"])
            out.update(G=G_any, G_sets_true=int(sum(G_sets)), H_saut_bassin=H,
                       H_per_set=[bool(a_ <= 0.5 * b_) for a_, b_ in zip(s10["dart_basin"]["values"], s10["mala_basin"]["values"])],
                       reading=("le darting tient là où l'IS s'effondre (bras d'entraînement de la famille 2)" if (G_any and H) else
                                ("la correction vient des pas locaux" if G_any else "le darting ne tient pas non plus")))
            V[c] = out
            print(f"{c}: G {G_any} ({out['G_sets_true']}/3 sets), H_saut_bassin {H} -> {out['reading']}")
        jsave(OUTF / "verdict.json", V)
