""" Step 2 evaluation (rules_mos_step1.md, amendment 1): MoS-w d = 50 controllers at epochs 100/200/300. python scripts/mosw_step2_eval.py RUN_DIR OUT_JSON [--epochs 100,200,300] [--n_paths 20000] [--budget_s 600] """
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import dart_lib as L

T0 = time.perf_counter()
ap = argparse.ArgumentParser(); ap.add_argument("rdir"); ap.add_argument("out"); ap.add_argument("--epochs", default="100,200,300"); ap.add_argument("--n_paths", type=int, default=20000); ap.add_argument("--budget_s", type=float, default=600)
args = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TR = json.load(open(ROOT / "figures/mog_mos_truth_mosw_d50.json")); ptrue = np.array(TR["p"]); floor = TR["floor_n"]; K = 10; NFE = 200
import train  # registers resolvers
import hydra
from omegaconf import OmegaConf
from adjoint_samplers.components.sde import ControlledSDE
from adjoint_samplers.utils import girsanov, train_utils
cfg = OmegaConf.load(Path(args.rdir) / "config.yaml"); OmegaConf.resolve(cfg)
ts = train_utils.get_timesteps(**cfg.timesteps).to(dev)
energy = hydra.utils.instantiate(cfg.energy, device=str(dev))
mu = energy.dist.means.cpu().double().numpy(); wt = energy.dist.weights.cpu().double().numpy()
pi = L.Mix("S", mu, wt, 1.0)


def make_sde(ckpt):
    ref = hydra.utils.instantiate(cfg.ref_sde).to(dev); ctrl = hydra.utils.instantiate(cfg.controller).to(dev)
    ctrl.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["controller"])
    return ControlledSDE(ref, ctrl).to(dev)


def simulate(sde, n, seed=0, chunk=10000):
    lws, xs = [], []
    for i in range(0, n, chunk):
        torch.manual_seed(seed + i)
        lw, x = girsanov.log_path_weights(sde, energy, torch.zeros(min(chunk, n - i), 50, device=dev), ts)
        lws.append(lw.double().cpu().numpy()); xs.append(x.double().cpu().numpy())
    return np.concatenate(xs), np.concatenate(lws)


out = dict(floor=floor, p=ptrue.tolist(), weights=wt.tolist(), epochs={})
for E in [int(e) for e in args.epochs.split(",")]:
    ck = Path(args.rdir) / f"controller_ep{E}.pt"
    sde = make_sde(ck); t = time.perf_counter()
    x, lw = simulate(sde, args.n_paths); t_sim = time.perf_counter() - t
    c = L.cells(x, mu); ph = L.props(c, K); w = np.exp(lw - lw.max())
    np.savez_compressed(Path(args.out).parent / f"is_logw_ep{E}.npz", lw=lw, cells=c)
    r = dict(tv_raw=L.tv(ph, ptrue), coverage=int((ph >= 0.25 * ptrue).sum()), ratios=(ph / ptrue).round(3).tolist(), time_sim_s=t_sim, n_paths=args.n_paths)
    r["IS_simple"] = dict(**L.weighted_stats(w, c, K, ptrue), khat=L.khat(lw), logz=float(np.log(np.mean(w)) + lw.max()))
    ps, k = L.psis_weights(lw); r["IS_psis"] = dict(**L.weighted_stats(ps, c, K, ptrue), khat=k)
    r["IS_trunc"] = L.weighted_stats(np.minimum(w, math.sqrt(len(w)) * w.mean()), c, K, ptrue)
    t = time.perf_counter(); dr = L.darting(pi, x[:10000], mu, ptrue, SW=(10, 40)); dr["time_cpu_s"] = time.perf_counter() - t
    r["darting"] = dr
    if dr.get("by_S"):
        costs = [int(dr["by_S"][s]["cost"]) for s in (10, 40)]
        r["mala_only"] = {str(cst): v for cst, v in L.mala_only(pi, x[:10000], mu, ptrue, dr["mala_step"], costs).items()}
        best = min(r["IS_simple"]["tv"], r["IS_psis"]["tv"], r["IS_trunc"]["tv"]); r["best_IS_tv_n20000"] = best
        r["G"] = {S: bool(dr["by_S"][S]["tv"] <= 0.5 * best and dr["by_S"][S]["tv"] <= 2 * floor) for S in (10, 40)}
        tm = r["mala_only"][str(costs[1])]["tv"]; r["H_saut"] = bool(dr["by_S"][40]["tv"] <= 0.5 * tm)
    out["epochs"][E] = r
    print(f"ep{E}: raw TV {r['tv_raw']:.3f} cov {r['coverage']}/10 | path IS n={args.n_paths}: simple TV {r['IS_simple']['tv']:.3f} ESS {r['IS_simple']['ess']:.5f} khat {r['IS_simple']['khat']:.2f} logZ {r['IS_simple']['logz']:+.1f}; PSIS TV {r['IS_psis']['tv']:.3f}; trunc TV {r['IS_trunc']['tv']:.3f} | darting minima {dr.get('minima_found')} " +
          (" ".join(f"S{S}: TV {v['tv']:.4f} jump acc {v['jump_acc']:.2f}" for S, v in dr.get("by_S", {}).items()) + f" | MALA-only " + " ".join(f"{v['tv']:.3f}" for v in r.get("mala_only", {}).values()) + f" | G {r.get('G')} H_saut {r.get('H_saut')}" if dr.get("by_S") else dr.get("error", "")) +
          f" | times: sim {t_sim:.1f} s, darting CPU {dr['time_cpu_s']:.0f} s", flush=True)
# equal-cost simple path IS for the last controller (GPU), time-guarded
last = out["epochs"][int(args.epochs.split(",")[-1])]
if last["darting"].get("by_S") and time.perf_counter() - T0 < args.budget_s:
    E = int(args.epochs.split(",")[-1]); sde = make_sde(Path(args.rdir) / f"controller_ep{E}.pt")
    need = {S: int(last["darting"]["by_S"][S]["cost"]) for S in (10, 40)}; nmax = max(need.values())
    lws, cs, done, t = [], [], 0, time.perf_counter(); trunc = False
    while done < nmax:
        if time.perf_counter() - T0 > args.budget_s: trunc = True; break
        x, lw = simulate(sde, min(20000, nmax - done), seed=1000 + done); lws.append(lw); cs.append(L.cells(x, mu)); done += len(lw)
    lw_all, c_all = np.concatenate(lws), np.concatenate(cs); out["equal_cost"] = dict(epoch=E, time_s=time.perf_counter() - t, n_done=int(len(lw_all)), truncated=trunc, by_S={})
    for S, nn in need.items():
        m = min(nn, len(lw_all)); st = L.is_stats(lw_all[:m], c_all[:m], K, ptrue); td = last["darting"]["by_S"][S]["tv"]
        out["equal_cost"]["by_S"][S] = dict(n_IS=nn, n_used=m, trunc=nn > len(lw_all), tv_IS=st["tv"], ess=st["ess"], khat=st["khat"], tv_dart=td, G=bool(td <= 0.5 * st["tv"] and td <= 2 * floor and nn <= len(lw_all)), network_passes=nn * NFE)
        print(f"  equal cost S={S}: darting TV {td:.4f} vs simple path IS n={nn:.2e} (used {m:.2e}{' TRUNCATED' if nn > len(lw_all) else ''}): TV {st['tv']:.4f} ESS {st['ess']:.5f} khat {st['khat']:.2f}; network passes {nn * NFE:.2e}", flush=True)
out["total_time_s"] = time.perf_counter() - T0
json.dump(out, open(args.out, "w"), indent=1, default=float); print(f"total {out['total_time_s']:.1f} s")
