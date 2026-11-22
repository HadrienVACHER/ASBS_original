""" rules_ns_damping.md: evaluation worker (one target = one Delta; several runs). For every saved controller (epochs in order, all runs of an epoch before the next epoch):
    20 000 paths (seed 0): path-IS summary (k-hat, log Z, SE, ESS; computed for EVERY controller, used by the rules only for runs that pass Q), Q (Q1, Q2, definition of step 4),
    per-mode mean errors (Voronoi cells) for R3. After the last epoch: per-mode mean errors of every eval dump (eval_samples/ep*.pt, 5000 samples) of each run.
    python scripts/ns_damping_eval.py RUN_DIR [RUN_DIR ...] --out OUT --delta DELTA [--final_only NAME ...] [--epochs 250,...,1500] [--n_paths 20000] [--wait] [--deadline UNIX]
    A run listed in --final_only is evaluated at epoch 1500 only (reused AS runs: controller_ep1500.pt = their controller_final.pt).
    Waiting (--wait): a snapshot is awaited until it exists, or until RUN_DIR/TRAIN_DONE exists (training ended: snapshot missing), or until the deadline. """
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
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", required=True); ap.add_argument("--delta", type=float, required=True)
ap.add_argument("--d", type=int, default=32); ap.add_argument("--K", type=int, default=8)
ap.add_argument("--epochs", default="250,500,750,1000,1250,1500"); ap.add_argument("--n_paths", type=int, default=20000)
ap.add_argument("--final_only", nargs="*", default=[]); ap.add_argument("--final_epoch", type=int, default=1500)
ap.add_argument("--deadline", type=float, default=float("inf")); ap.add_argument("--wait", action="store_true")
args = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D, K, DELTA = args.d, args.K, args.delta
import train  # registers resolvers
import hydra
from omegaconf import OmegaConf
from adjoint_samplers.components.sde import ControlledSDE
from adjoint_samplers.utils import girsanov, train_utils

odir = Path(args.out); odir.mkdir(parents=True, exist_ok=True)
# ---- target (one definition: gmm_dk.py), truth, minima, exact reference (same construction as step 4)
mu = gmm_dk.means(D, K, DELTA).double().numpy(); wt = gmm_dk.weights(K).double().numpy(); wt = wt / wt.sum()
pi = L.Mix("G", mu, wt, 1.0)
ptrue = gmm_dk.true_proportions(D, K, DELTA).numpy()
M = B.polish_minima(pi)
Xex = B.sample_exact(pi, 10000, np.random.default_rng(2026))
med_ex = float(np.median(B.dist_to_nearest_minimum(Xex, M)))
bex, fc_ex, fa_ex = B.basins(pi, Xex, M)
print(f"Delta {DELTA}: exact reference median distance {med_ex:.3f}; converged {fc_ex:.3f}; Q1 threshold {1.5 * med_ex:.3f}", flush=True)


def mode_stats(x):
    """ per-mode (Voronoi cell = nearest mean) counts and mean error vectors e_k = mean(x in cell k) - mu_k (None if the cell is empty) """
    c = L.cells(x, mu)
    n, err = [], []
    for k in range(K):
        m = c == k; n.append(int(m.sum()))
        err.append((x[m].mean(0) - mu[k]).tolist() if m.any() else None)
    return dict(n=n, err=err)


def load_run(rdir):
    cfg = OmegaConf.load(Path(rdir) / "config.yaml"); OmegaConf.resolve(cfg)
    assert int(cfg.dim) == D and int(cfg.K) == K and abs(float(cfg.delta) - DELTA) < 1e-9
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


def dump(out, path):
    out["total_time_s"] = time.perf_counter() - T0
    json.dump(out, open(path, "w"), indent=1, default=float)


runs = [Path(r) for r in args.runs]
state = {}
for r in runs:
    t_cfg = time.time()
    while not (r / "config.yaml").exists() and time.time() - t_cfg < 600 and time.time() < args.deadline:   # training processes write it at start-up
        time.sleep(5)
    cfg, ts, energy = load_run(r)
    state[r.name] = dict(cfg=cfg, ts=ts, energy=energy, path=odir / f"q_{r.name}.json",
                         out=dict(run=r.name, d=D, K=K, delta=DELTA, sigma_max=float(cfg.sigma_max), matcher=str(cfg.adjoint_matcher._target_).split(".")[-1],
                                  final_only=r.name in args.final_only, p=ptrue.tolist(), median_dist_exact=med_ex, exact_frac_converged=fc_ex, epochs={}, eval_dumps={}))
EPS = [int(e) for e in args.epochs.split(",")]
for E in EPS:
    for r in runs:
        st = state[r.name]; out = st["out"]
        if out["final_only"] and E != args.final_epoch:
            continue
        ck = r / f"controller_ep{E}.pt"
        while args.wait and not (ck.exists() and time.time() - ck.stat().st_mtime > 5) and not (r / "TRAIN_DONE").exists() and time.time() < args.deadline:
            time.sleep(10)
        if not ck.exists():
            out["epochs"][E] = dict(status="missing"); dump(out, st["path"]); print(f"{r.name} ep{E}: controller missing", flush=True); continue
        if time.time() > args.deadline:
            out["epochs"][E] = dict(status="skipped: deadline"); dump(out, st["path"]); print(f"{r.name} ep{E}: skipped (deadline)", flush=True); continue
        sde = make_sde(st["cfg"], ck); t = time.perf_counter()
        x, lw = simulate(sde, st["energy"], st["ts"], args.n_paths); t_sim = time.perf_counter() - t
        np.savez_compressed(odir / f"is_logw_{r.name}_ep{E}.npz", lw=lw, cells=L.cells(x, mu))
        xq = x[:10000]; med_as = float(np.median(B.dist_to_nearest_minimum(xq, M)))
        bq, fc, fa = B.basins(pi, xq, M); pb = B.basin_props(bq, K)
        cov = int((pb >= 0.25 * ptrue).sum()) if np.isfinite(pb).all() else 0
        Q1 = bool(med_as <= 1.5 * med_ex); Q2 = bool(fc >= 0.95 and cov == K)
        isst = L.is_stats(lw, np.zeros(len(lw), int), K, ptrue)
        pv = L.props(L.cells(xq, mu), K)
        r_ = dict(status="evaluated", time_sim_s=t_sim, n_paths=len(lw),
                  Q=dict(median_dist_AS=med_as, ratio=med_as / med_ex, Q1=Q1, frac_converged=fc, frac_assigned=fa, basin_coverage=cov, basin_p_hat=pb.tolist(), Q2=Q2, Q=bool(Q1 and Q2)),
                  IS_summary=dict(khat=isst["khat"], logz=isst["logz"], logz_se=isst["logz_se"], ess=isst["ess"]),
                  raw=dict(tv_voronoi=L.tv(pv, ptrue), coverage_voronoi=int((pv >= 0.25 * ptrue).sum())),
                  modes=mode_stats(x))
        out["epochs"][E] = r_; dump(out, st["path"])
        print(f"{r.name} ep{E}: Q1 {med_as:.2f} ({med_as / med_ex:.2f}x) {Q1}; Q2 conv {fc:.3f} cov {cov}/{K} {Q2}; Q {Q1 and Q2} | IS ESS {isst['ess']:.4f} khat {isst['khat']:.2f} "
              f"logZ {isst['logz']:+.3f}+-{isst['logz_se']:.3f} | raw TV vor {r_['raw']['tv_voronoi']:.3f} | sim {t_sim:.1f}s", flush=True)
for r in runs:                                                                                  # R3: per-mode mean errors of the eval dumps (5000 samples every 50 epochs)
    st = state[r.name]; out = st["out"]
    for f in sorted((r / "eval_samples").glob("ep*.pt")) if (r / "eval_samples").exists() else []:
        x = torch.load(f, map_location="cpu", weights_only=False).double().numpy()
        if x.shape[1] == D:
            out["eval_dumps"][int(f.stem[2:])] = mode_stats(x)
    dump(out, st["path"])
    print(f"{r.name}: {len(out['eval_dumps'])} eval dumps processed", flush=True)
print(f"total {time.perf_counter() - T0:.1f} s")
