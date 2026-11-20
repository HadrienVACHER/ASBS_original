""" Evaluation of the MoS pilot (rules_mos_pilot.md). python scripts/mos_pilot_eval.py RESULTS_DIR OUT_JSON [--n_paths 20000] [--budget_s 420]
    RESULTS_DIR holds mospilot_s<sigma>/{config.yaml, controller_final.pt, samples_final.pt}. Steps: coverage (P1) -> path IS + darting post hoc (protocol of the phase 0) on each controller -> retention rule -> equal-cost path IS for the retained one. """
import argparse, glob, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import dart_lib as L

T0 = time.perf_counter()
ap = argparse.ArgumentParser(); ap.add_argument("rdir"); ap.add_argument("out"); ap.add_argument("--n_paths", type=int, default=20000); ap.add_argument("--budget_s", type=float, default=420); ap.add_argument("--test_force", action="store_true", help="local pipeline test only: continue even if P1 is false")
args = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TR = json.load(open(ROOT / "figures/mog_mos_truth_mos_d50.json")); ptrue = np.array(TR["p"]); floor = TR["floor_n"]; K = 10
NFE = 200


def load(rd):
    import train  # registers resolvers
    import hydra
    from omegaconf import OmegaConf
    from adjoint_samplers.components.sde import ControlledSDE
    from adjoint_samplers.utils import train_utils
    cfg = OmegaConf.load(Path(rd) / "config.yaml"); OmegaConf.resolve(cfg)
    ref = hydra.utils.instantiate(cfg.ref_sde).to(dev); ctrl = hydra.utils.instantiate(cfg.controller).to(dev)
    ctrl.load_state_dict(torch.load(Path(rd) / "controller_final.pt", map_location=dev, weights_only=False)["controller"])
    sde = ControlledSDE(ref, ctrl).to(dev)
    energy = hydra.utils.instantiate(cfg.energy, device=str(dev))
    ts = train_utils.get_timesteps(**cfg.timesteps).to(dev)
    return sde, energy, ts, int(cfg.dim), float(cfg.sigma_max)


def simulate(sde, energy, ts, d, n, chunk=10000, seed=0):
    from adjoint_samplers.utils import girsanov
    lws, xs = [], []
    for i in range(0, n, chunk):
        torch.manual_seed(seed + i)
        lw, x = girsanov.log_path_weights(sde, energy, torch.zeros(min(chunk, n - i), d, device=dev), ts)
        lws.append(lw.double().cpu().numpy()); xs.append(x.double().cpu().numpy())
    return np.concatenate(xs), np.concatenate(lws)


runs = sorted(glob.glob(str(Path(args.rdir) / "mospilot_s*")))
out = dict(runs={}, floor=floor, p=ptrue.tolist())
# ---- P1: coverage from the 10^4 final model samples
mu_t = None
for rd in runs:
    xs = torch.load(Path(rd) / "samples_final.pt", weights_only=False)["model_samples"].double().numpy()
    sde, energy, ts, d, sig = load(rd)
    mu = energy.dist.means.cpu().double().numpy(); mu_t = mu
    ph = L.props(L.cells(xs, mu), K); cov = int((ph >= 0.25 * ptrue).sum())
    out["runs"][Path(rd).name] = dict(sigma_max=sig, coverage=cov, tv_raw=L.tv(ph, ptrue), lost=[int(k) for k in np.nonzero(ph < 0.25 * ptrue)[0]], ratios=(ph / ptrue).round(3).tolist())
    print(f"{Path(rd).name}: sigma_max {sig}: coverage {cov}/10, raw TV {L.tv(ph, ptrue):.3f}; ratios {np.round(ph / ptrue, 2).tolist()}", flush=True)
P1 = any(v["coverage"] == 10 for v in out["runs"].values()); out["P1"] = P1
print(f"P1 (at least one sigma_max with 10/10 modes): {P1}")
if not P1 and not args.test_force:
    json.dump(out, open(args.out, "w"), indent=1); print("P1 FALSE: stop (no full batch). time %.1f s" % (time.perf_counter() - T0)); sys.exit(0)
# ---- path IS (20000 trajectories) and darting on each controller
w_eq = np.array(TR["p"])  # placeholder to keep the name; target weights are equal (1/K)
pi = L.Mix("S", mu_t, np.full(K, 1.0 / K), 1.0)
for rd in runs:
    name = Path(rd).name; sde, energy, ts, d, sig = load(rd)
    t = time.perf_counter(); x, lw = simulate(sde, energy, ts, d, args.n_paths); t_sim = time.perf_counter() - t
    c = L.cells(x, mu_t); st = L.is_stats(lw, c, K, ptrue); st["tv_unweighted"] = L.tv(L.props(c, K), ptrue); st["time_sim_s"] = t_sim; st["n_paths"] = args.n_paths
    t = time.perf_counter(); dr = L.darting(pi, x[:10000], mu_t, ptrue); dr["time_cpu_s"] = time.perf_counter() - t
    out["runs"][name].update(IS=st, darting=dr)
    print(f"{name}: path IS (N={args.n_paths}): TV {st['tv']:.4f} ESS {st['ess']:.4f} khat {st['khat']:.2f} logZ {st['logz']:+.3f} ± {st['logz_se']:.3f} ({t_sim:.1f} s GPU/CPU) | darting: minima {dr.get('minima_found')} " +
          (" ".join(f"S{S}: TV {v['tv']:.4f} jump acc {v['jump_acc']:.2f} cov {v['coverage']}" for S, v in dr.get("by_S", {}).items()) if dr.get("by_S") else dr.get("error", "")) + f" ({dr['time_cpu_s']:.1f} s CPU)", flush=True)
# ---- retention rule: 10/10 runs; k_hat < 0.7; argmax log Z_hat, ties within 2 SE -> largest ESS; none with khat < 0.7 -> largest ESS (flagged)
cand = [(n, v) for n, v in out["runs"].items() if v["coverage"] == 10] or ([(n, v) for n, v in out["runs"].items()] if args.test_force else [])
ok = [(n, v) for n, v in cand if v["IS"]["khat"] < 0.7]
if ok:
    top = max(ok, key=lambda nv: nv[1]["IS"]["logz"])
    ties = [nv for nv in ok if top[1]["IS"]["logz"] - nv[1]["IS"]["logz"] <= 2 * math.sqrt(nv[1]["IS"]["logz_se"] ** 2 + top[1]["IS"]["logz_se"] ** 2)]
    ret = max(ties, key=lambda nv: nv[1]["IS"]["ess"]); flag = ""
else:
    ret = max(cand, key=lambda nv: nv[1]["IS"]["ess"]); flag = "no 10/10 run has khat < 0.7: largest ESS (flagged)"
out["retained"] = ret[0]; out["retained_flag"] = flag
print(f"retained: {ret[0]} (sigma_max {ret[1]['sigma_max']}) {flag}; P2: khat at the retained sigma_max = {ret[1]['IS']['khat']:.2f} ({'> 0.7: tails too light, as the synthetic sampler' if ret[1]['IS']['khat'] > 0.7 else '<= 0.7'})", flush=True)
# ---- equal-cost path IS for the retained controller
dr = ret[1]["darting"]
if dr.get("by_S"):
    need = {S: int(v["cost"]) for S, v in dr["by_S"].items()}; nmax = max(need.values())
    sde, energy, ts, d, sig = load(str(Path(args.rdir) / ret[0]))
    lws, cs, done, t = [], [], 0, time.perf_counter(); truncated = False
    while done < nmax:
        if time.perf_counter() - T0 > args.budget_s:
            truncated = True; break
        x, lw = simulate(sde, energy, ts, d, min(20000, nmax - done), seed=1000 + done)
        lws.append(lw); cs.append(L.cells(x, mu_t)); done += len(lw)
    t_eq = time.perf_counter() - t
    lw_all, c_all = np.concatenate(lws), np.concatenate(cs)
    out["equal_cost"] = dict(time_s=t_eq, n_done=int(len(lw_all)), truncated=truncated, by_S={})
    for S, nn in need.items():
        m = min(nn, len(lw_all)); st = L.is_stats(lw_all[:m], c_all[:m], K, ptrue)
        dd = dr["by_S"][S]
        G = bool(dd["tv"] <= 0.5 * st["tv"] and dd["tv"] <= 2 * floor and nn <= len(lw_all))
        out["equal_cost"]["by_S"][S] = dict(n_IS=nn, n_used=m, trunc=nn > len(lw_all), tv_IS=st["tv"], ess_IS=st["ess"], khat_IS=st["khat"], tv_dart=dd["tv"], G=G,
                                            cost_EG=dd["cost"], cost_with_network_passes=dd["cost"] + 0, network_passes_IS=nn * NFE)
        print(f"  S={S}: darting TV {dd['tv']:.4f} (cost {dd['cost']:.2e} E+grad) vs path IS n={nn:.2e} [used {m:.2e}{', TRUNCATED' if nn > len(lw_all) else ''}]: TV {st['tv']:.4f} ESS {st['ess']:.4f} khat {st['khat']:.2f}; G {G}; network passes of the IS {nn * NFE:.2e}; IS time {t_eq:.0f} s total, darting CPU {dr['time_cpu_s']:.0f} s")
out["total_time_s"] = time.perf_counter() - T0
json.dump(out, open(args.out, "w"), indent=1, default=float)
print(f"total {out['total_time_s']:.1f} s")
