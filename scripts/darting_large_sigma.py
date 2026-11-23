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
