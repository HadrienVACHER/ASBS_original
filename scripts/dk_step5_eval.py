""" Step 5 (rules_dk_step5.md): GMM-dk (128, 256) and MoS-dk (32, 128), real AS (generalisation of scripts/gmmdk_step4_eval.py; `--fam gmm|mos`). Subcommands q, cmp (as in step 4, darting S in {1, 5, 10, 40}), truth (basin truth, descriptive).
    Step-4 documentation follows.
    python scripts/gmmdk_step4_eval.py q RUN_DIR OUT_DIR --d D [--epochs 250,...,1500] [--deadline UNIX] [--wait] [--smoke]
        Q (Q1, Q2) on every saved controller of ONE run (20 000 paths: log w, cells and end points saved); the path-IS summary (k-hat, log Z-hat, SE, ESS) is computed ONLY for controllers that pass Q (input of the sigma_max rule).
    python scripts/gmmdk_step4_eval.py cmp OUT_DIR --d D --runs NAME_A NAME_B --rdirs DIR_A DIR_B [--deadline UNIX] [--smoke]
        per dimension: candidate = last saved controller of each run that passes Q; sigma_max rule among the candidates; comparison (path IS, PSIS, truncated IS, darting S in {1, 5, 10}, MALA alone at equal cost,
        Voronoi AND basin proportions, equal-cost path IS) on the chosen controller only; I, G, H_saut. """
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
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["q", "cmp", "truth"])
ap.add_argument("--fam", choices=["gmm", "mos"], default="gmm"); ap.add_argument("--fixed", default=None, help="cmp only: NAME:EPOCH:RDIR, skips the Q files and the sigma_max rule and re-simulates the 20 000 paths of that controller (step-4 GMM-dk controllers, supplement S = 40)")
ap.add_argument("pos", nargs="*")
ap.add_argument("--d", type=int, required=True); ap.add_argument("--K", type=int, default=8); ap.add_argument("--delta", type=float, default=6.0)
ap.add_argument("--epochs", default="250,500,750,1000,1250,1500"); ap.add_argument("--n_paths", type=int, default=20000)
ap.add_argument("--deadline", type=float, default=float("inf")); ap.add_argument("--wait", action="store_true"); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--runs", nargs="*", default=[]); ap.add_argument("--rdirs", nargs="*", default=[])
args = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D, K, DELTA, NFE = args.d, args.K, args.delta, 200
SW = (1, 2) if args.smoke else (1, 5, 10, 40)
NPATH = 1000 if args.smoke else args.n_paths
NDART = 1000 if args.smoke else 10000
import train  # registers resolvers
import hydra
from omegaconf import OmegaConf
from adjoint_samplers.components.sde import ControlledSDE
from adjoint_samplers.utils import girsanov, train_utils

# ---- target (one definition: gmm_dk.py), truth, floor, minima, exact reference
mu = gmm_dk.means(D, K, DELTA).double().numpy(); wt = gmm_dk.weights(K).double().numpy(); wt = wt / wt.sum()
NUT = 3.0
if args.fam == "gmm":
    pi = L.Mix("G", mu, wt, 1.0)
    ptrue = gmm_dk.true_proportions(D, K, DELTA).numpy()
else:                                                                                    # MoS-dk: product of 1D t_3 of squared scale 4/3 (Hessian I at the modes); truth: figures/mog_mos_truth_mosdk_d<D>.json (scripts/mosdk_truth.py)
    pi = L.Mix("S", mu, wt, 1.0, nu=NUT, s2=(NUT + 1) / NUT)
    _tr = json.load(open(ROOT / f"figures/mog_mos_truth_mosdk_d{D}.json")); ptrue = np.array(_tr["p"])
_rngf = np.random.default_rng(5); _muK = mu[:, :K]


def _drawK(n):                                                                           # first K coordinates of exact draws (all that the nearest-mean cell depends on: equal-norm means on the first K axes)
    comp = _rngf.choice(K, size=n, p=wt); z = _rngf.standard_normal((n, K))
    if args.fam == "gmm":
        return _muK[comp] + z
    chi = _rngf.gamma(NUT / 2, 2.0 / NUT, size=(n, K)); return _muK[comp] + math.sqrt((NUT + 1) / NUT) * z / np.sqrt(chi)


floor = float(np.median([L.tv(np.bincount((_drawK(10000) @ _muK.T - 0.5 * (_muK ** 2).sum(1)[None]).argmax(1), minlength=K) / 10000, ptrue) for _ in range(100)]))
M = B.polish_minima(pi)
Xex = B.sample_exact(pi, 10000, np.random.default_rng(2026))
med_ex = float(np.median(B.dist_to_nearest_minimum(Xex, M)))
bex, fc_ex, fa_ex = B.basins(pi, Xex, M)


def bstats(b, w=None):
    pb = B.basin_props(b, K, w)
    return dict(tv=L.tv(pb, ptrue), coverage=int((pb >= 0.25 * ptrue).sum()), p_hat=pb.tolist(), assigned=float((b >= 0).mean()))


def load_run(rdir):
    cfg = OmegaConf.load(Path(rdir) / "config.yaml"); OmegaConf.resolve(cfg)
    assert int(cfg.dim) == D and int(cfg.K) == K
    ts = train_utils.get_timesteps(**cfg.timesteps).to(dev)
    energy = hydra.utils.instantiate(cfg.energy, device=str(dev))
    return cfg, ts, energy


def make_sde(cfg, ckpt):
    ref = hydra.utils.instantiate(cfg.ref_sde).to(dev); ctrl = hydra.utils.instantiate(cfg.controller).to(dev)
    ctrl.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["controller"])
    return ControlledSDE(ref, ctrl).to(dev)


def simulate(sde, energy, ts, n, seed=0, chunk=10000):
    lws, xs = [], []
    for i in range(0, n, chunk):
        torch.manual_seed(seed + i)
        lw, x = girsanov.log_path_weights(sde, energy, torch.zeros(min(chunk, n - i), D, device=dev), ts)
        lws.append(lw.double().cpu().numpy()); xs.append(x.double().cpu().numpy())
    return np.concatenate(xs), np.concatenate(lws)


def is_summary(lw):
    st = L.is_stats(lw, np.zeros(len(lw), int), K, ptrue)
    return dict(khat=st["khat"], logz=st["logz"], logz_se=st["logz_se"], ess=st["ess"])


def dump(out, path):
    out["total_time_s"] = time.perf_counter() - T0
    json.dump(out, open(path, "w"), indent=1, default=float)


# =====================================================================================================  basin truth (descriptive)
if args.mode == "truth":
    odir = Path(args.pos[0]); odir.mkdir(parents=True, exist_ok=True)
    n = 2000 if args.smoke else 200_000; rng = np.random.default_rng(777); X = B.sample_exact(pi, n, rng)
    t = time.perf_counter(); bb, fcc, faa = B.basins(pi, X, M); pb = B.basin_props(bb, K)
    res = dict(fam=args.fam, d=D, n=n, frac_converged=fcc, frac_assigned=faa, p_basin=pb.tolist(), p_voronoi=ptrue.tolist(), tv_basin_vs_voronoi=L.tv(pb, ptrue), floor_n1e4=floor, median_dist_exact=med_ex, time_s=time.perf_counter() - t)
    json.dump(res, open(odir / f"truth_{args.fam}_d{D}.json", "w"), indent=1)
    print(f"truth {args.fam} d={D}: basin proportions from {n} exact draws, converged {fcc:.3f}; TV(basin, Voronoi) = {res['tv_basin_vs_voronoi']:.4f}; floor {floor:.4f}; median distance to the nearest minimum (exact) {med_ex:.2f}", flush=True)
    sys.exit(0)

# =====================================================================================================  Q
if args.mode == "q":
    rdir, odir = Path(args.pos[0]), Path(args.pos[1]); odir.mkdir(parents=True, exist_ok=True); (odir / "data").mkdir(exist_ok=True)
    cfg, ts, energy = load_run(rdir)
    name = rdir.name; qpath = odir / f"q_{name}.json"
    out = dict(run=name, d=D, K=K, delta=DELTA, sigma_max=float(cfg.sigma_max), floor=floor, p=ptrue.tolist(), median_dist_exact=med_ex,
               exact=dict(frac_converged=fc_ex, frac_assigned=fa_ex, **bstats(bex)), epochs={})
    print(f"{name}: exact reference: median distance to nearest minimum {med_ex:.2f} (sqrt d = {math.sqrt(D):.2f}); basin TV vs truth {out['exact']['tv']:.4f} (floor {floor:.4f}); Q1 threshold {1.5 * med_ex:.2f}", flush=True)
    for E in [int(e) for e in args.epochs.split(",")]:
        ck = rdir / f"controller_ep{E}.pt"
        while args.wait and not (ck.exists() and time.time() - ck.stat().st_mtime > 5):
            if time.time() > args.deadline: break
            time.sleep(10)
        if not ck.exists():
            out["epochs"][E] = dict(status="missing"); dump(out, qpath); print(f"ep{E}: controller missing", flush=True); continue
        if time.time() > args.deadline:
            out["epochs"][E] = dict(status="skipped: deadline"); dump(out, qpath); print(f"ep{E}: skipped (deadline)", flush=True); continue
        sde = make_sde(cfg, ck); t = time.perf_counter()
        x, lw = simulate(sde, energy, ts, NPATH); t_sim = time.perf_counter() - t
        c = L.cells(x, mu)
        np.savez_compressed(odir / f"is_logw_{name}_ep{E}.npz", lw=lw, cells=c)
        np.save(odir / "data" / f"endpoints_{name}_ep{E}.npy", x.astype(np.float32))
        xq = x[:10000]; med_as = float(np.median(B.dist_to_nearest_minimum(xq, M)))
        bq, fc, fa = B.basins(pi, xq, M); qs = bstats(bq)
        Q1 = bool(med_as <= 1.5 * med_ex); Q2 = bool(fc >= 0.95 and qs["coverage"] == K)
        pv = L.props(c[:10000], K)
        r = dict(status="evaluated", time_sim_s=t_sim, n_paths=len(lw), Q=dict(median_dist_AS=med_as, median_dist_exact=med_ex, ratio=med_as / med_ex, Q1=Q1, frac_converged=fc, frac_assigned=fa,
                 basins_reached_ge1=int(np.bincount(bq[bq >= 0], minlength=K).astype(bool).sum()), basin_coverage=qs["coverage"], basin_p_hat=qs["p_hat"], Q2=Q2, Q=bool(Q1 and Q2)),
                 raw=dict(tv_voronoi=L.tv(pv, ptrue), coverage_voronoi=int((pv >= 0.25 * ptrue).sum()), basin_tv=qs["tv"]))
        out["epochs"][E] = r; dump(out, qpath)
        print(f"{name} ep{E}: Q1 median dist {med_as:.2f} (ratio {med_as / med_ex:.2f}) -> {Q1}; Q2 converged {fc:.3f}, basin coverage {qs['coverage']}/{K} -> {Q2}; Q = {Q1 and Q2} | raw TV vor {r['raw']['tv_voronoi']:.3f} basin {qs['tv']:.3f}", flush=True)
        if (Q1 and Q2) or args.smoke:                                                       # (--smoke: always) IS summary only for controllers that pass Q (input of the sigma_max rule)
            r["IS_summary"] = is_summary(lw); dump(out, qpath)
            print(f"{name} ep{E}: [passes Q] path IS: khat {r['IS_summary']['khat']:.2f}, log Z {r['IS_summary']['logz']:+.3f} +- {r['IS_summary']['logz_se']:.3f}, ESS {r['IS_summary']['ess']:.4f}", flush=True)
    print(f"total {time.perf_counter() - T0:.1f} s")
    sys.exit(0)

# =====================================================================================================  comparison on the chosen controller
odir = Path(args.pos[0]); odir.mkdir(parents=True, exist_ok=True)
cpath = odir / (f"cmp_{args.fam}_d{D}_supp.json" if args.fixed else f"cmp_{args.fam}_d{D}.json")
out = dict(d=D, K=K, delta=DELTA, floor=floor, p=ptrue.tolist(), runs=args.runs, candidates={}, chosen=None)
cands = {}
if args.fixed:                                                                           # step-4 GMM-dk controller already chosen by the step-4 rule: supplement S = 40, no sigma_max rule
    fname, fep, frdir = args.fixed.split(":"); args.runs, args.rdirs = [], []
    cands[fname] = (int(fep), None, frdir, float(OmegaConf.load(Path(frdir) / "config.yaml").sigma_max)); out["candidates"][fname] = dict(fixed=True, candidate_epoch=int(fep))
for name, rdir in zip(args.runs, args.rdirs):
    q = json.load(open(odir / f"q_{name}.json"))
    passing = [int(e) for e, r in q["epochs"].items() if r.get("status") == "evaluated" and (r["Q"]["Q"] or args.smoke)]
    out["candidates"][name] = dict(sigma_max=q["sigma_max"], epochs_passing_Q=sorted(passing), candidate_epoch=max(passing) if passing else None)
    if passing:
        cands[name] = (max(passing), q["epochs"][str(max(passing))]["IS_summary"], rdir, q["sigma_max"])
print(f"d={D}: candidates (last saved controller passing Q per run): " + (", ".join(f"{n}: ep{v[0]}" for n, v in cands.items()) or "NONE"), flush=True)
if not cands:
    out["verdict"] = "no controller passes Q at this dimension: no comparison (rule)"; dump(out, cpath); print(out["verdict"]); sys.exit(0)
# sigma_max rule (rules_sigma_select.md) among the candidates
ret = {n: v for n, v in cands.items() if v[1] is not None and v[1]["khat"] < 0.7}
if args.fixed:
    chosen = fname; rule = dict(step="fixed: step-4 candidate (supplement)", retained=[])
elif ret:
    best = max(ret, key=lambda n: ret[n][1]["logz"])
    tie = [n for n in ret if abs(ret[n][1]["logz"] - ret[best][1]["logz"]) < 2 * math.sqrt(ret[n][1]["logz_se"] ** 2 + ret[best][1]["logz_se"] ** 2) or n == best]
    chosen = max(tie, key=lambda n: ret[n][1]["ess"]); rule = dict(step="khat<0.7 filter; argmax log Z; ties within 2 SE -> largest ESS", retained=list(ret), argmax=best, ties=tie)
else:
    chosen = max(cands, key=lambda n: cands[n][1]["ess"]); rule = dict(step="no candidate with khat<0.7: largest ESS (flagged)", retained=[])
E, summ, rdir, sig = cands[chosen]
out["chosen"] = dict(run=chosen, epoch=E, sigma_max=sig, rule=rule); dump(out, cpath)
print(f"chosen: {chosen} ep{E} (sigma_max {sig}); rule: {rule['step']}", flush=True)
if time.time() + 200 > args.deadline:
    out["verdict"] = "comparison skipped: deadline"; dump(out, cpath); print(out["verdict"]); sys.exit(0)
cfg, ts, energy = load_run(rdir); sde = make_sde(cfg, Path(rdir) / f"controller_ep{E}.pt")
if args.fixed:                                                                           # re-simulate the 20 000 paths of the controller (seed 0, as in the Q stage)
    x, lw = simulate(sde, energy, ts, NPATH); c = L.cells(x, mu)
else:
    z = np.load(odir / f"is_logw_{chosen}_ep{E}.npz"); lw, c = z["lw"], z["cells"]
    x = np.load(odir / "data" / f"endpoints_{chosen}_ep{E}.npy").astype(np.float64)
w = np.exp(lw - lw.max()); b20, _, _ = B.basins(pi, x, M)
r = dict(n_paths=len(lw), raw=dict(tv_voronoi=L.tv(L.props(c[:10000], K), ptrue), basin=bstats(b20[:10000])))
r["IS_simple"] = dict(**L.weighted_stats(w, c, K, ptrue), khat=L.khat(lw), logz=float(np.log(np.mean(w)) + lw.max()), basin=bstats(b20, w))
ps, k = L.psis_weights(lw); r["IS_psis"] = dict(**L.weighted_stats(ps, c, K, ptrue), khat=k, basin=bstats(b20, ps))
wt_ = np.minimum(w, math.sqrt(len(w)) * w.mean()); r["IS_trunc"] = dict(**L.weighted_stats(wt_, c, K, ptrue), basin=bstats(b20, wt_))
r["I"] = bool(r["IS_simple"]["ess"] < 0.05)
out["compared"] = r; dump(out, cpath)
print(f"IS (n={len(lw)}): simple TV {r['IS_simple']['tv']:.4f} (basin {r['IS_simple']['basin']['tv']:.4f}) ESS {r['IS_simple']['ess']:.4f} khat {r['IS_simple']['khat']:.2f}; PSIS {r['IS_psis']['tv']:.4f}; trunc {r['IS_trunc']['tv']:.4f}; I (ESS < 5 %) = {r['I']}", flush=True)
t = time.perf_counter(); dr = L.darting(pi, x[:NDART], mu, ptrue, SW=SW, keep_final=True); dr["time_cpu_s"] = time.perf_counter() - t
finals = {S: dr["by_S"][S].pop("X_final") for S in dr.get("by_S", {})}
r["darting"] = dr
if dr.get("by_S"):
    for S in SW:
        bb, _, _ = B.basins(pi, finals[S], M); dr["by_S"][S]["basin"] = bstats(bb)
    costs = [int(dr["by_S"][s]["cost"]) for s in SW]
    mo = L.mala_only(pi, x[:NDART], mu, ptrue, dr["mala_step"], costs, keep_final=True)
    for cst, v in mo.items():
        bb, _, _ = B.basins(pi, v.pop("X_final"), M); v["basin"] = bstats(bb)
    r["mala_only"] = {str(cst): v for cst, v in mo.items()}
    r["H_saut"] = bool(dr["by_S"][SW[-1]]["tv"] <= 0.5 * r["mala_only"][str(costs[-1])]["tv"])
    r["H_saut_basin"] = bool(dr["by_S"][SW[-1]]["basin"]["tv"] <= 0.5 * r["mala_only"][str(costs[-1])]["basin"]["tv"])
    # equal-cost path IS: one simulation of max cost paths, sliced at the cost of each S (time-guarded)
    need = {S: int(dr["by_S"][S]["cost"]) for S in SW}; nmax = max(need.values())
    lws, cs, done, t = [], [], 0, time.perf_counter(); trunc = False
    while done < nmax:
        if time.time() + 60 > args.deadline:
            trunc = True; break
        xx, ll = simulate(sde, energy, ts, min(20000, nmax - done), seed=1000 + done); lws.append(ll); cs.append(L.cells(xx, mu)); done += len(ll)
    lw_all, c_all = np.concatenate(lws), np.concatenate(cs)
    r["equal_cost"] = dict(time_s=time.perf_counter() - t, n_done=int(len(lw_all)), truncated=trunc, by_S={})
    for S, nn in need.items():
        m = min(nn, len(lw_all)); ls, cc = lw_all[:m], c_all[:m]; ww = np.exp(ls - ls.max())
        st = L.is_stats(ls, cc, K, ptrue); pw, kk = L.psis_weights(ls)
        tv_p = L.weighted_stats(pw, cc, K, ptrue)["tv"]; tv_t = L.weighted_stats(np.minimum(ww, math.sqrt(m) * ww.mean()), cc, K, ptrue)["tv"]
        best = min(st["tv"], tv_p, tv_t); td = dr["by_S"][S]["tv"]
        avail = nn <= len(lw_all)
        r["equal_cost"]["by_S"][S] = dict(n_IS=nn, n_used=m, available=avail, tv_simple=st["tv"], tv_psis=tv_p, tv_trunc=tv_t, ess=st["ess"], khat=st["khat"], tv_dart=td,
                                          G=bool(avail and td <= 0.5 * st["tv"] and td <= 0.5 * best and td <= 2 * floor), network_passes=nn * NFE)
    r["G"] = bool(any(v["G"] for v in r["equal_cost"]["by_S"].values()))
    best20 = min(r["IS_simple"]["tv"], r["IS_psis"]["tv"], r["IS_trunc"]["tv"])
    r["G_n20000"] = {S: bool(dr["by_S"][S]["tv"] <= 0.5 * best20 and dr["by_S"][S]["tv"] <= 2 * floor) for S in SW}              # reported next to G (comparator = best variant at 20 000 paths)
    print("darting: minima %s; " % dr.get("minima_found") + " ".join(f"S{S}: TV {v['tv']:.4f} (basin {v['basin']['tv']:.4f}) jump acc {v['jump_acc']:.2f}" for S, v in dr["by_S"].items())
          + " | MALA-only " + " ".join(f"{v['tv']:.3f}/{v['basin']['tv']:.3f}" for v in r["mala_only"].values()) + f" | darting CPU {dr['time_cpu_s']:.0f} s", flush=True)
    for S, v in r["equal_cost"]["by_S"].items():
        print(f"  equal cost S={S}: darting TV {v['tv_dart']:.4f} vs IS n={v['n_IS']:.2e} (used {v['n_used']:.2e}{'' if v['available'] else ' TRUNCATED'}): simple {v['tv_simple']:.4f} PSIS {v['tv_psis']:.4f} trunc {v['tv_trunc']:.4f}; G_S {v['G']}", flush=True)
    print(f"I = {r['I']}; G = {r['G']}; H_saut = {r['H_saut']} (basin {r['H_saut_basin']}); floor {floor:.4f}", flush=True)
else:
    print(dr.get("error"), flush=True)
dump(out, cpath); print(f"total {time.perf_counter() - T0:.1f} s")
