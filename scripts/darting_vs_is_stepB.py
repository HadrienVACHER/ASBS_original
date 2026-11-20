""" Step B of rules_darting_vs_is.md: trained AS controllers (seed 0, sigma_max x 0.5, cells with 8/8 coverage), path IS (girsanov.log_path_weights) versus darting post hoc on the SAME 10^4 end points.
    python scripts/darting_vs_is_stepB.py [--n 10000] [--time-only]   CPU only, one budget of < 1 min (if the projected time exceeds it, the script stops and says so). """
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gmm_dk as G
import darting_vs_is_stepA as A

T0 = time.perf_counter()
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 10_000
TIME_ONLY = "--time-only" in sys.argv
CELLS = [("gmmdk_d8_K8_D8_s0_sx0.5", 8, 8.0), ("gmmdk_d32_K8_D4_s0_sx0.5", 32, 4.0), ("gmmdk_d32_K8_D6_s0_sx0.5", 32, 6.0)]
MEM = [("gmmdk_d32_K8_D8_s0_sx0.5", 32, 8.0)]                       # for memory only (coverage after darting)
DIRS = ["results/gmm_dk_sweep", "results/gmm_dk_window/w1"]
torch.set_num_threads(10)


def find(run):
    for b in DIRS:
        if (Path(b) / run / "controller_final.pt").exists():
            return Path(b) / run
    raise FileNotFoundError(run)


def simulate(run, n):
    import train  # registers the resolvers of the configs
    import hydra
    from omegaconf import OmegaConf
    from adjoint_samplers.components.sde import ControlledSDE
    from adjoint_samplers.utils import girsanov, train_utils
    d = find(run)
    cfg = OmegaConf.load(d / "config.yaml"); OmegaConf.resolve(cfg)
    ref_sde = hydra.utils.instantiate(cfg.ref_sde)
    ctrl = hydra.utils.instantiate(cfg.controller)
    ctrl.load_state_dict(torch.load(d / "controller_final.pt", map_location="cpu", weights_only=False)["controller"])
    sde = ControlledSDE(ref_sde, ctrl)
    energy = hydra.utils.instantiate(cfg.energy, device="cpu")
    ts = train_utils.get_timesteps(**cfg.timesteps)
    dim = int(cfg.dim)
    lws, xs = [], []
    for i in range(0, n, 5000):
        torch.manual_seed(100 + i)
        lw, x = girsanov.log_path_weights(sde, energy, torch.zeros(min(5000, n - i), dim), ts)
        lws.append(lw.double().numpy()); xs.append(x.double().numpy())
    return np.concatenate(xs), np.concatenate(lws)


def analyse(run, d, delta, n):
    t = time.perf_counter()
    x, lw = simulate(run, n)
    t_sim = time.perf_counter() - t
    K = A.K
    A.MM = delta / math.sqrt(2)                                      # the step-A machinery with this cell's geometry (all components Gaussian, A1 family)
    ptrue = G.true_proportions(d, K, delta).numpy()
    rng = np.random.default_rng(55)
    floor = float(np.median([A.tv(A.props(A.cells(A.draw_pi_K(rng, n, "A1"))), ptrue) for _ in range(100)]))
    c = A.cells(x[:, :K])
    pre = A.props(c)
    cov_pre = int((pre >= 0.25 * ptrue).sum())
    res = dict(run=run, d=d, delta=delta, n=n, time_sim_s=t_sim, floor_n=floor, tv_as=A.tv(pre, ptrue), coverage_as=cov_pre,
               path_is=A.is_stats(lw, c, ptrue))
    w = np.exp(lw - lw.max()); res["path_is"]["p_hat"] = (A.props(c, w)).tolist(); res["path_is"]["coverage"] = int((A.props(c, w) >= 0.25 * ptrue).sum())
    mins, Hs = A.discover_minima(x[:500], "A1")
    herr = float(np.abs(Hs - np.eye(d)[None]).max()) if len(mins) else float("nan")
    res["minima"] = dict(found=int(len(mins)), max_hessian_dev_from_I=herr)
    sq, Ft, xK = A.stats(x, "A1", [])
    if len(mins) == K and herr < 1e-6:
        res["map"] = "translation (H = I to 1e-6)"
        snaps = A.darting_chain(xK, sq, Ft, "A1", d, rng)
    elif len(mins) >= 2:
        res["map"] = f"general Hessian map among the {len(mins)} discovered minima"
        snaps = A.darting_chain_general(x, "A1", d, mins, Hs, rng)
    else:
        snaps = None
    res["darting"] = None if snaps is None else {S: dict(tv=A.tv(A.props(snaps[S][0]), ptrue), acc=snaps[S][1], coverage=int((A.props(snaps[S][0]) >= 0.25 * ptrue).sum()), cost=n * (1 + S)) for S in (1, 5, 10)}
    res["time_total_s"] = time.perf_counter() - t
    return res


def main():
    out = []
    first = True
    for run, d, delta in CELLS + MEM:
        el = time.perf_counter() - T0
        if not first and el + out[0]["time_sim_s"] * 1.3 > 58:
            print(f"*** time budget: stopping before {run} (elapsed {el:.1f} s, one simulation ~{out[0]['time_sim_s']:.1f} s): a short GPU job would be needed (offer, cost and cap to submit before any rental) ***")
            break
        if TIME_ONLY and not first: break
        r = analyse(run, d, delta, N)
        first = False
        out.append(r)
        pi_ = r["path_is"]; dd = r["darting"]
        print(f"{run} (d={d}, Delta={delta:.0f}), n={N}: simulation {r['time_sim_s']:.1f} s; AS: TV {r['tv_as']:.4f} coverage {r['coverage_as']}/8 | path IS: TV {pi_['tv']:.4f} ESS {pi_['ess']:.3f} khat {pi_['khat']:.2f} coverage {pi_['coverage']}/8"
              + (f" | darting [{r.get('map')}] " + " ".join(f"S{S}: TV {dd[S]['tv']:.4f} acc {dd[S]['acc']:.3f} cov {dd[S]['coverage']}/8" for S in (1, 5, 10)) if dd else f" | darting n/a (minima found {r['minima']['found']}/8)")
              + f" | floor(n) {r['floor_n']:.4f} | total {time.perf_counter() - T0:.1f} s")
    if TIME_ONLY:
        print(f"time-only: elapsed {time.perf_counter() - T0:.1f} s"); return
    cells3 = [r for r in out if (r["run"], r["d"], r["delta"]) in CELLS and r["darting"]]
    r3 = [(r["run"], r["darting"][5]["tv"], 2 * r["path_is"]["tv"], r["darting"][5]["tv"] <= 2 * r["path_is"]["tv"]) for r in cells3]
    R3 = len(cells3) == len(CELLS) and all(x[3] for x in r3)
    total = time.perf_counter() - T0
    print(f"\nR3 (TV_darting(S=5) <= 2 x TV_IS(path), all 3 cells): {'TRUE' if R3 else 'FALSE'}  {[(x[0].replace('gmmdk_', ''), round(x[1], 4), round(x[2], 4), x[3]) for x in r3]}")
    print(f"TOTAL TIME {total:.1f} s (budget 60 s){'  *** OVER BUDGET ***' if total > 60 else ''}")
    json.dump(dict(cells=out, R3=R3, R3_detail=r3, total_time_s=total), open("figures/darting_vs_is_stepB.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
