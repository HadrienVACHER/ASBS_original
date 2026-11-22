""" Step 3 evaluation (rules_mosw_step3.md): quality criterion Q on every saved controller, comparisons ONLY on the controllers that pass Q.
    python scripts/mosw_step3_eval.py RUN_DIR OUT_JSON --d 50 [--epochs 250,500,...,1500] [--n_paths 20000] [--deadline UNIX_TS] [--wait] [--smoke]
    Q is computed and written to OUT_JSON before any comparison quantity is computed on the controller. """
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import dart_lib as L
import basin_lib as B

T0 = time.perf_counter()
ap = argparse.ArgumentParser()
ap.add_argument("rdir"); ap.add_argument("out"); ap.add_argument("--d", type=int, required=True)
ap.add_argument("--epochs", default="250,500,750,1000,1250,1500"); ap.add_argument("--n_paths", type=int, default=20000)
ap.add_argument("--deadline", type=float, default=float("inf")); ap.add_argument("--wait", action="store_true"); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--q_only", action="store_true")   # rules_mosw_ns.md: Q and basin TV only, no comparison (default off: step-3 behaviour unchanged)
args = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D, K, NFE = args.d, 10, 200
SWS = (2, 4) if args.smoke else (10, 40)
NDART = 1000 if args.smoke else 10000
TR = json.load(open(ROOT / f"figures/mog_mos_truth_mosw_d{D}.json")); ptrue = np.array(TR["p"]); floor = TR["floor_n"]
import train  # registers resolvers
import hydra
from omegaconf import OmegaConf
from adjoint_samplers.components.sde import ControlledSDE
from adjoint_samplers.utils import girsanov, train_utils
cfg = OmegaConf.load(Path(args.rdir) / "config.yaml"); OmegaConf.resolve(cfg)
assert int(cfg.dim) == D
ts = train_utils.get_timesteps(**cfg.timesteps).to(dev)
energy = hydra.utils.instantiate(cfg.energy, device=str(dev))
mu = energy.dist.means.cpu().double().numpy(); wt = energy.dist.weights.cpu().double().numpy()
pi = L.Mix("S", mu, wt, 1.0)
M = B.polish_minima(pi)
out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)


def make_sde(ckpt):
    ref = hydra.utils.instantiate(cfg.ref_sde).to(dev); ctrl = hydra.utils.instantiate(cfg.controller).to(dev)
    ctrl.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["controller"])
    return ControlledSDE(ref, ctrl).to(dev)


def simulate(sde, n, seed=0, chunk=10000):
    lws, xs = [], []
    for i in range(0, n, chunk):
        torch.manual_seed(seed + i)
        lw, x = girsanov.log_path_weights(sde, energy, torch.zeros(min(chunk, n - i), D, device=dev), ts)
        lws.append(lw.double().cpu().numpy()); xs.append(x.double().cpu().numpy())
    return np.concatenate(xs), np.concatenate(lws)


def bstats(b, w=None):
    pb = B.basin_props(b, K, w)
    return dict(tv=L.tv(pb, ptrue), coverage=int((pb >= 0.25 * ptrue).sum()), p_hat=pb.tolist(), assigned=float((b >= 0).mean()))


# ---- exact reference (same quench, same minima): the denominators of Q1 and the basin truth
rng = np.random.default_rng(2026)
Xex = B.sample_exact(pi, 10000, rng)
med_ex = float(np.median(B.dist_to_nearest_minimum(Xex, M)))
bex, fc_ex, fa_ex = B.basins(pi, Xex, M)
out = dict(d=D, floor=floor, p=ptrue.tolist(), weights=wt.tolist(), sigma_max=float(cfg.sigma_max), median_dist_exact=med_ex,
           exact=dict(frac_converged=fc_ex, frac_assigned=fa_ex, **bstats(bex)), epochs={})
print(f"d={D}: exact reference: median distance to nearest minimum {med_ex:.2f}; quench converged {fc_ex:.3f}; basin TV vs truth {out['exact']['tv']:.4f} (floor {floor:.4f}); Q1 threshold {1.5 * med_ex:.2f}", flush=True)


def dump():
    out["total_time_s"] = time.perf_counter() - T0
    json.dump(out, open(out_path, "w"), indent=1, default=float)


last_pass = None
for E in [int(e) for e in args.epochs.split(",")]:
    ck = Path(args.rdir) / f"controller_ep{E}.pt"
    while args.wait and not (ck.exists() and time.time() - ck.stat().st_mtime > 5):
        if time.time() > args.deadline:
            break
        time.sleep(10)
    if not ck.exists():
        out["epochs"][E] = dict(status="missing (deadline)" if args.wait else "missing"); dump(); print(f"ep{E}: controller missing", flush=True); continue
    if time.time() > args.deadline:
        out["epochs"][E] = dict(status="skipped: deadline"); dump(); print(f"ep{E}: skipped (deadline)", flush=True); continue
    sde = make_sde(ck); t = time.perf_counter()
    x, lw = simulate(sde, 1000 if args.smoke else args.n_paths); t_sim = time.perf_counter() - t
    c = L.cells(x, mu); w = np.exp(lw - lw.max())
    np.savez_compressed(out_path.parent / f"is_logw_d{D}_ep{E}.npz", lw=lw, cells=c)
    np.save(out_path.parent / f"endpoints_d{D}_ep{E}.npy", x.astype(np.float32))
    # ---------- Q (before anything else)
    xq = x[:10000]
    dq = B.dist_to_nearest_minimum(xq, M); med_as = float(np.median(dq))
    bq, fc, fa = B.basins(pi, xq, M)
    qs = bstats(bq)
    reached = int(np.bincount(bq[bq >= 0], minlength=K).astype(bool).sum())
    Q1 = bool(med_as <= 1.5 * med_ex); Q2 = bool(fc >= 0.95 and qs["coverage"] == K)
    r = dict(status="evaluated", time_sim_s=t_sim, n_paths=len(lw),
             Q=dict(median_dist_AS=med_as, median_dist_exact=med_ex, ratio=med_as / med_ex, Q1=Q1, frac_converged=fc, frac_assigned=fa, basins_reached_ge1=reached, basin_coverage=qs["coverage"], basin_p_hat=qs["p_hat"], Q2=Q2, Q=bool(Q1 and Q2)))
    r["raw"] = dict(tv_voronoi=L.tv(L.props(c[:10000], K), ptrue), coverage_voronoi=int((L.props(c[:10000], K) >= 0.25 * ptrue).sum()), basin_tv=qs["tv"])
    out["epochs"][E] = r; dump()
    print(f"ep{E}: Q1 median dist AS {med_as:.2f} vs exact {med_ex:.2f} (ratio {med_as / med_ex:.2f}) -> {Q1}; Q2 converged {fc:.3f}, basin coverage {qs['coverage']}/10 (>=1 sample: {reached}) -> {Q2}; Q = {Q1 and Q2} | raw TV vor {r['raw']['tv_voronoi']:.3f} basin {qs['tv']:.3f}", flush=True)
    if args.q_only or (not (Q1 and Q2) and not args.smoke):                                            # --smoke only: comparisons run regardless of Q to test the code path
        continue
    if time.time() + 150 > args.deadline:
        r["comparisons"] = "skipped: deadline"; dump(); print(f"ep{E}: comparisons skipped (deadline)", flush=True); continue
    # ---------- comparisons (only controllers that pass Q)
    b20, _, _ = B.basins(pi, x, M)
    r["IS_simple"] = dict(**L.weighted_stats(w, c, K, ptrue), khat=L.khat(lw), logz=float(np.log(np.mean(w)) + lw.max()), basin=bstats(b20, w))
    ps, k = L.psis_weights(lw); r["IS_psis"] = dict(**L.weighted_stats(ps, c, K, ptrue), khat=k, basin=bstats(b20, ps))
    wt_ = np.minimum(w, math.sqrt(len(w)) * w.mean()); r["IS_trunc"] = dict(**L.weighted_stats(wt_, c, K, ptrue), basin=bstats(b20, wt_))
    t = time.perf_counter(); dr = L.darting(pi, x[:NDART], mu, ptrue, SW=SWS, keep_final=True); dr["time_cpu_s"] = time.perf_counter() - t
    finals = {S: dr["by_S"][S].pop("X_final") for S in dr.get("by_S", {})}
    r["darting"] = dr
    if dr.get("by_S"):
        for S in SWS:
            bb, _, _ = B.basins(pi, finals[S], M); dr["by_S"][S]["basin"] = bstats(bb)
        costs = [int(dr["by_S"][s]["cost"]) for s in SWS]
        mo = L.mala_only(pi, x[:NDART], mu, ptrue, dr["mala_step"], costs, keep_final=True)
        for cst, v in mo.items():
            bb, _, _ = B.basins(pi, v.pop("X_final"), M); v["basin"] = bstats(bb)
        r["mala_only"] = {str(cst): v for cst, v in mo.items()}
        best = min(r["IS_simple"]["tv"], r["IS_psis"]["tv"], r["IS_trunc"]["tv"]); r["best_IS_tv_n20000"] = best
        r["G"] = {S: bool(dr["by_S"][S]["tv"] <= 0.5 * best and dr["by_S"][S]["tv"] <= 2 * floor) for S in SWS}
        tm = r["mala_only"][str(costs[1])]["tv"]; r["H_saut"] = bool(dr["by_S"][SWS[1]]["tv"] <= 0.5 * tm)
        bestb = min(r["IS_simple"]["basin"]["tv"], r["IS_psis"]["basin"]["tv"], r["IS_trunc"]["basin"]["tv"])
        r["G_basin"] = {S: bool(dr["by_S"][S]["basin"]["tv"] <= 0.5 * bestb and dr["by_S"][S]["basin"]["tv"] <= 2 * floor) for S in SWS}
        r["H_saut_basin"] = bool(dr["by_S"][SWS[1]]["basin"]["tv"] <= 0.5 * r["mala_only"][str(costs[1])]["basin"]["tv"])
    last_pass = E; dump()
    print(f"ep{E}: path IS n={len(lw)}: simple TV {r['IS_simple']['tv']:.3f} (basin {r['IS_simple']['basin']['tv']:.3f}) ESS {r['IS_simple']['ess']:.5f} khat {r['IS_simple']['khat']:.2f}; PSIS {r['IS_psis']['tv']:.3f}; trunc {r['IS_trunc']['tv']:.3f} | darting minima {dr.get('minima_found')} "
          + (" ".join(f"S{S}: TV {v['tv']:.4f} (basin {v['basin']['tv']:.4f}) jump acc {v['jump_acc']:.2f}" for S, v in dr.get("by_S", {}).items()) + " | MALA-only " + " ".join(f"{v['tv']:.3f}/{v['basin']['tv']:.3f}" for v in r["mala_only"].values()) + f" | G {r['G']} H_saut {r['H_saut']} | G_basin {r['G_basin']} H_saut_basin {r['H_saut_basin']}" if dr.get("by_S") else dr.get("error", ""))
          + f" | darting CPU {dr['time_cpu_s']:.0f} s", flush=True)

# ---- equal-cost simple path IS for the LAST controller that passed Q (time-guarded)
if last_pass is not None and out["epochs"][last_pass]["darting"].get("by_S") and time.time() + 200 < args.deadline and not args.smoke:
    E = last_pass; sde = make_sde(Path(args.rdir) / f"controller_ep{E}.pt"); lastr = out["epochs"][E]
    need = {S: int(lastr["darting"]["by_S"][S]["cost"]) for S in SWS}; nmax = max(need.values())
    lws, cs, done, t = [], [], 0, time.perf_counter(); trunc = False
    while done < nmax:
        if time.time() + 60 > args.deadline:
            trunc = True; break
        x, lw = simulate(sde, min(20000, nmax - done), seed=1000 + done); lws.append(lw); cs.append(L.cells(x, mu)); done += len(lw)
    lw_all, c_all = np.concatenate(lws), np.concatenate(cs); out["equal_cost"] = dict(epoch=E, time_s=time.perf_counter() - t, n_done=int(len(lw_all)), truncated=trunc, by_S={})
    for S, nn in need.items():
        m = min(nn, len(lw_all)); st = L.is_stats(lw_all[:m], c_all[:m], K, ptrue); td = lastr["darting"]["by_S"][S]["tv"]
        out["equal_cost"]["by_S"][S] = dict(n_IS=nn, n_used=m, trunc=nn > len(lw_all), tv_IS=st["tv"], ess=st["ess"], khat=st["khat"], tv_dart=td, G=bool(td <= 0.5 * st["tv"] and td <= 2 * floor and nn <= len(lw_all)), network_passes=nn * NFE)
        print(f"  equal cost S={S}: darting TV {td:.4f} vs simple path IS n={nn:.2e} (used {m:.2e}{' TRUNCATED' if nn > len(lw_all) else ''}): TV {st['tv']:.4f} ESS {st['ess']:.5f} khat {st['khat']:.2f}; network passes {nn * NFE:.2e}", flush=True)
out["last_pass_epoch"] = last_pass
dump(); print(f"total {out['total_time_s']:.1f} s")
