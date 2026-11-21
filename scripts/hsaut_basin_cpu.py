""" CPU control of rules_hsaut_basin.md on the MoS-dk d = 128, sigma_max x 0.75, epoch-1500 controller (step 5). Each execution < 1 min, never in the background.
    python scripts/hsaut_basin_cpu.py dart I [--n N] [--timing]   chain I: n AS end points (CPU simulation, torch seed 1000 + I), darting S in {1, 5, 40} (chain seed 100 + I), basin + Voronoi TV of the snapshots
    python scripts/hsaut_basin_cpu.py mala I                     chain I: MALA alone from the same end points, at the cost of each S (seed 200 + I)
    python scripts/hsaut_basin_cpu.py agg                        mean +- sd over the chains, floors, H_saut_basin """
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import dart_lib as L
import basin_lib as B
import gmm_dk

T0 = time.perf_counter()
mode = sys.argv[1]
OUT = ROOT / "figures/hsaut_cpu"; OUT.mkdir(exist_ok=True)
D, K, DELTA, NUT = 128, 8, 6.0, 3.0
SW = (1, 5, 40)
args = sys.argv[2:]
N = int(args[args.index("--n") + 1]) if "--n" in args else 10000
TIMING = "--timing" in args
torch.set_num_threads(10)
mu = gmm_dk.means(D, K, DELTA).double().numpy(); wt = gmm_dk.weights(K).double().numpy(); wt /= wt.sum()
pi = L.Mix("S", mu, wt, 1.0, nu=NUT, s2=(NUT + 1) / NUT)
p_vor = np.array(json.load(open(ROOT / "figures/mog_mos_truth_mosdk_d128.json"))["p"])
tr = json.load(open(ROOT / "figures/dk_step5/truth_mos_d128.json")); p_bas = np.array(tr["p_basin"])
M = B.polish_minima(pi)


def tvs(X):
    b, fc, fa = B.basins(pi, X, M)
    return dict(tv_basin=L.tv(B.basin_props(b, K), p_bas), tv_vor=L.tv(L.props(L.cells(X, mu), K), p_vor), conv=fc)


if mode in ("dart", "mala"):
    i = 99 if TIMING else int(args[0])
    f = OUT / f"chain{i}_{'dart' if mode == 'dart' else 'mala'}.json"
    xf = OUT / f"chain{i}_endpoints.npy"
    if mode == "dart":
        import train  # registers resolvers
        import hydra
        from omegaconf import OmegaConf
        from adjoint_samplers.components.sde import ControlledSDE
        from adjoint_samplers.utils import train_utils
        rd = ROOT / "results/dk_step5/asbs/results/mosdk5_d128_sx0.75"
        cfg = OmegaConf.load(rd / "config.yaml"); OmegaConf.resolve(cfg)
        ref = hydra.utils.instantiate(cfg.ref_sde); ctrl = hydra.utils.instantiate(cfg.controller)
        ctrl.load_state_dict(torch.load(ROOT / "figures/dk_step5/controllers/mosdk5_d128_sx0.75_ep1500.pt", map_location="cpu", weights_only=False)["controller"])
        sde = ControlledSDE(ref, ctrl); ts = train_utils.get_timesteps(**cfg.timesteps)
        from adjoint_samplers.utils import girsanov
        energy = hydra.utils.instantiate(cfg.energy, device="cpu")
        torch.manual_seed(1000 + i); t = time.perf_counter()
        with torch.no_grad():
            _, xt = girsanov.log_path_weights(sde, energy, torch.zeros(N, D), ts)
        x = xt.double().numpy()
        t_sim = time.perf_counter() - t; np.save(xf, x.astype(np.float32))
        r = dict(chain=i, n=N, time_sim_s=t_sim, raw=tvs(x))
        t = time.perf_counter(); dr = L.darting(pi, x, mu, np.zeros(K) + 1.0 / K, SW=SW, seed=100 + i, keep_final=True); r["time_darting_s"] = time.perf_counter() - t
        r["minima_found"] = dr.get("minima_found"); r["mala_step"] = dr.get("mala_step"); r["by_S"] = {}
        for S in SW:
            v = dr["by_S"][S]; X = v.pop("X_final")
            r["by_S"][S] = dict(cost=v["cost"], jump_acc=v["jump_acc"], mala_acc=v["mala_acc"], **tvs(X))
        r["time_total_s"] = time.perf_counter() - T0
    else:
        dj = json.load(open(OUT / f"chain{i}_dart.json")); x = np.load(xf).astype(np.float64)
        costs = [int(dj["by_S"][str(S)]["cost"]) for S in SW]
        mo = L.mala_only(pi, x, mu, np.zeros(K) + 1.0 / K, dj["mala_step"], costs, seed=200 + i, keep_final=True)
        r = dict(chain=i, n=len(x), by_S={})
        for S, c in zip(SW, costs):
            v = mo[c]; X = v.pop("X_final"); r["by_S"][S] = dict(cost=c, steps=v["steps"], acc=v["acc"], **tvs(X))
        r["time_total_s"] = time.perf_counter() - T0
    json.dump(r, open(f, "w"), indent=1, default=float)
    print(f"{mode} chain {i} n={r['n']}: " + (f"sim {r['time_sim_s']:.1f} s; raw {r['raw']}; " if mode == "dart" else "") + " ".join(f"S{S}: basin {v['tv_basin']:.4f} vor {v['tv_vor']:.4f}" for S, v in r["by_S"].items()) + f" | total {r['time_total_s']:.1f} s", flush=True)
    sys.exit(0)

# ---- aggregate
n = json.load(open(OUT / "chain0_dart.json"))["n"]
chains = range(5)
D_ = [json.load(open(OUT / f"chain{i}_dart.json")) for i in chains]; M_ = [json.load(open(OUT / f"chain{i}_mala.json")) for i in chains]
rng = np.random.default_rng(31)


def floor(p, reps=2000):
    return float(np.median([0.5 * np.abs(np.bincount(rng.choice(K, size=n, p=p), minlength=K) / n - p).sum() for _ in range(reps)]))


res = dict(n=n, floor_basin=floor(p_bas), floor_vor=floor(p_vor), by_S={})
ms = lambda a: (float(np.mean(a)), float(np.std(a, ddof=1)))
for S in SW:
    res["by_S"][S] = dict(
        darting_basin=ms([d["by_S"][str(S)]["tv_basin"] for d in D_]), darting_vor=ms([d["by_S"][str(S)]["tv_vor"] for d in D_]),
        mala_basin=ms([m["by_S"][str(S)]["tv_basin"] for m in M_]), mala_vor=ms([m["by_S"][str(S)]["tv_vor"] for m in M_]),
        cost=float(np.mean([d["by_S"][str(S)]["cost"] for d in D_])), jump_acc=float(np.mean([d["by_S"][str(S)]["jump_acc"] for d in D_])))
res["raw_basin"] = ms([d["raw"]["tv_basin"] for d in D_]); res["raw_vor"] = ms([d["raw"]["tv_vor"] for d in D_])
res["H_saut_basin"] = bool(res["by_S"][40]["darting_basin"][0] <= 0.5 * res["by_S"][40]["mala_basin"][0])
res["H_saut_voronoi_descriptive"] = bool(res["by_S"][40]["darting_vor"][0] <= 0.5 * res["by_S"][40]["mala_vor"][0])
cmp_ = json.load(open(ROOT / "figures/dk_step5/cmp_mos_d128.json"))["compared"]
res["IS_trunc_reminder_n20000_GPU"] = dict(tv_voronoi=cmp_["IS_trunc"]["tv"], basin_vs_voronoi_truth=cmp_["IS_trunc"]["basin"]["tv"], basin_vs_basin_truth=float(0.5 * np.abs(np.array(cmp_["IS_trunc"]["basin"]["p_hat"]) - p_bas).sum()))
json.dump(res, open(OUT / "aggregate.json", "w"), indent=1, default=float)
print(f"n = {n}; floors (multinomial, median of 2000): basin {res['floor_basin']:.4f}, Voronoi {res['floor_vor']:.4f}")
print(f"raw AS: basin {res['raw_basin'][0]:.4f} +- {res['raw_basin'][1]:.4f}; Voronoi {res['raw_vor'][0]:.4f} +- {res['raw_vor'][1]:.4f}")
for S, v in res["by_S"].items():
    print(f"S={S:2d} (cost {v['cost']:.2e}, jump acc {v['jump_acc']:.2f}): darting basin {v['darting_basin'][0]:.4f} +- {v['darting_basin'][1]:.4f}, Voronoi {v['darting_vor'][0]:.4f} +- {v['darting_vor'][1]:.4f} | MALA basin {v['mala_basin'][0]:.4f} +- {v['mala_basin'][1]:.4f}, Voronoi {v['mala_vor'][0]:.4f} +- {v['mala_vor'][1]:.4f}")
print(f"IS tronque (20 000 trajectoires, GPU, rappel): Voronoi {res['IS_trunc_reminder_n20000_GPU']['tv_voronoi']:.4f}; bassins contre verite Voronoi {res['IS_trunc_reminder_n20000_GPU']['basin_vs_voronoi_truth']:.4f}, contre verite bassin {res['IS_trunc_reminder_n20000_GPU']['basin_vs_basin_truth']:.4f}")
print(f"H_saut_basin (mean TV basin darting S=40 <= 0.5 x mean TV basin MALA at equal cost) = {res['H_saut_basin']}   (Voronoi version, descriptive: {res['H_saut_voronoi_descriptive']})")
