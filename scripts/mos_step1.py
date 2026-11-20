""" Step 1 of rules_mos_step1.md (CPU). python scripts/mos_step1.py --part a|b|c|verdict [--n 10000]
    a: darting S in {10, 20, 40} on the retained MoS d = 50 controller's samples; b: MALA-only control at the same cost; c: path IS variants (simple, PSIS, truncated) on 20 000 resimulated trajectories; verdict: G and H_jump. """
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import dart_lib as L

T0 = time.perf_counter()
ap = argparse.ArgumentParser(); ap.add_argument("--part", required=True); ap.add_argument("--n", type=int, default=10000); ap.add_argument("--n_paths", type=int, default=20000)
args = ap.parse_args()
K = 10
TR = json.load(open(ROOT / "figures/mog_mos_truth_mos_d50.json")); ptrue = np.array(TR["p"])
MU = np.load(ROOT / "data/mog_mos_means.npz")["mos_K10_d50"]
pi = L.Mix("S", MU, np.full(K, 1.0 / K), 1.0)
n = args.n
RUN = ROOT / "results/mos_pilot/r/mospilot_s23.2"
fpath = ROOT / f"figures/mos_step1_floor_n{n}.json"


def floor_n():
    if n == 10000: return TR["floor_n"]
    if fpath.exists(): return json.load(open(fpath))["floor"]
    rng = np.random.default_rng(3)
    fl = []
    for _ in range(100):
        comp = rng.choice(K, size=n); X = MU[comp] + rng.standard_t(2.0, size=(n, 50))
        fl.append(L.tv(L.props(L.cells(X, MU), K), ptrue))
    json.dump(dict(floor=float(np.median(fl)), n=n), open(fpath, "w")); return float(np.median(fl))


floor = floor_n()
X = torch.load(RUN / "samples_final.pt", weights_only=False)["model_samples"].double().numpy()[:n]
ra = ROOT / f"figures/mos_step1_a_n{n}.json"

if args.part == "a":
    res = L.darting(pi, X, MU, ptrue, SW=(10, 20, 40)); res["n"] = n; res["floor"] = floor; res["time_s"] = time.perf_counter() - T0
    res["tv_raw"] = L.tv(L.props(L.cells(X, MU), K), ptrue)
    json.dump(res, open(ra, "w"), indent=1, default=float)
    print(f"(a) n={n}: floor {floor:.4f}; raw TV {res['tv_raw']:.4f}; minima {res['minima_found']}; MALA step {res['mala_step']:.3g}; cost before sweeps {res['cost_pre_sweeps']:.2e}")
    for S, v in res["by_S"].items(): print(f"   S={S}: TV {v['tv']:.4f} jump acc {v['jump_acc']:.3f} MALA acc {v['mala_acc']:.3f} cost {v['cost']:.3e} coverage {v['coverage']}")
    print(f"   time {res['time_s']:.1f} s")
elif args.part == "b":
    a = json.load(open(ra)); costs = [a["by_S"][s]["cost"] for s in ("10", "20", "40")]
    out = L.mala_only(pi, X, MU, ptrue, a["mala_step"], costs)
    res = dict(n=n, floor=floor, mala_step=a["mala_step"], by_cost={str(c): v for c, v in out.items()}, time_s=time.perf_counter() - T0)
    json.dump(res, open(ROOT / f"figures/mos_step1_b_n{n}.json", "w"), indent=1, default=float)
    print(f"(b) MALA only (same step {a['mala_step']:.3g}) n={n}:")
    for c, v in out.items(): print(f"   cost {c:.3e}: {v['steps']} MALA steps, TV {v['tv']:.4f}, acc {v['acc']:.3f}, coverage {v['coverage']}")
    print(f"   time {res['time_s']:.1f} s")
elif args.part == "c":
    import train  # registers resolvers
    import hydra
    from omegaconf import OmegaConf
    from adjoint_samplers.components.sde import ControlledSDE
    from adjoint_samplers.utils import girsanov, train_utils
    cfg = OmegaConf.load(RUN / "config.yaml"); OmegaConf.resolve(cfg)
    ref = hydra.utils.instantiate(cfg.ref_sde); ctrl = hydra.utils.instantiate(cfg.controller)
    ctrl.load_state_dict(torch.load(ROOT / "figures/mos_pilot_controllers/mospilot_s23.2.pt", map_location="cpu", weights_only=False)["controller"])
    sde = ControlledSDE(ref, ctrl); energy = hydra.utils.instantiate(cfg.energy, device="cpu"); ts = train_utils.get_timesteps(**cfg.timesteps)
    torch.set_num_threads(10)
    lws, cs, done = [], [], 0
    t = time.perf_counter()
    while done < args.n_paths and time.perf_counter() - T0 < 45:
        torch.manual_seed(500 + done)
        lw, x = girsanov.log_path_weights(sde, energy, torch.zeros(5000, 50), ts)
        lws.append(lw.double().numpy()); cs.append(L.cells(x.double().numpy(), MU)); done += 5000
    lw, c = np.concatenate(lws), np.concatenate(cs); N = len(lw); t_sim = time.perf_counter() - t
    out = dict(n_paths=N, time_sim_s=t_sim)
    w = np.exp(lw - lw.max())
    out["simple"] = dict(**L.weighted_stats(w, c, K, ptrue), khat=L.khat(lw))
    ps, k = L.psis_weights(lw); out["psis"] = dict(**L.weighted_stats(ps, c, K, ptrue), khat=k)
    tw = np.minimum(w, math.sqrt(N) * w.mean()); out["truncated"] = dict(**L.weighted_stats(tw, c, K, ptrue))
    out["unweighted_tv"] = L.tv(L.props(c, K), ptrue); out["logz"] = float(np.log(np.mean(w)) + lw.max())
    for sub in (N // 4, N // 2):
        ws = np.exp(lw[:sub] - lw[:sub].max()); out[f"simple_n{sub}"] = L.weighted_stats(ws, c[:sub], K, ptrue)
    json.dump(out, open(ROOT / "figures/mos_step1_c.json", "w"), indent=1, default=float)
    print(f"(c) {N} paths resimulated on CPU in {t_sim:.1f} s: unweighted TV {out['unweighted_tv']:.4f}; log Z_hat {out['logz']:+.2f}")
    for k_ in ("simple", "psis", "truncated"): print(f"   {k_:>9}: TV {out[k_]['tv']:.4f} ESS {out[k_]['ess']:.5f}" + (f" khat {out[k_]['khat']:.2f}" if "khat" in out[k_] else ""))
    print(f"   time {time.perf_counter() - T0:.1f} s")
