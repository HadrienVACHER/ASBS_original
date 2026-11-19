""" Analysis of the (d, K, Delta) study (predictions_dk.md). Usage:
      python scripts/gmm_dk_analysis.py [--n_is 20000] [--no_is] RUN_DIR [RUN_DIR ...]
    RUN_DIR holds config.yaml, eval_samples/ep*.pt (dump_eval_samples), controller_final.pt. Report only; nothing here is a rule.
    Metrics: coverage (p_hat >= 0.25 p) and lost modes; TV = mean over the last 10 evaluations of the per-evaluation TV, with the exact-sampler floor;
    ratios p_hat/p and beta from p_hat averaged over the last 10 evaluations; path-space IS (adjoint_samplers/utils/girsanov.py): ESS fraction, khat (generalised Pareto fit,
    scipy genpareto, shape c of the largest min(0.2 N, 3 sqrt N) weights), TV after reweighting. """
import argparse, glob, json, math, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gmm_dk as G


def khat(lw):
    w = np.exp((lw - lw.max()).cpu().double().numpy())
    w.sort()
    M = int(min(0.2 * len(w), 3 * math.sqrt(len(w))))
    from scipy.stats import genpareto
    tail = w[-M:] - w[-M - 1]
    if M < 10 or tail.max() <= 0:
        return float("nan")
    return float(genpareto.fit(tail[tail > 0], floc=0)[0])


def is_paths(run, d, K, delta, n_is, device):
    import train  # registers the resolvers of the configs
    import hydra
    from omegaconf import OmegaConf
    from adjoint_samplers.components.sde import ControlledSDE
    from adjoint_samplers.utils import girsanov, train_utils
    cfg = OmegaConf.load(Path(run) / "config.yaml"); OmegaConf.resolve(cfg)
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
    ctrl = hydra.utils.instantiate(cfg.controller).to(device)
    ctrl.load_state_dict(torch.load(Path(run) / "controller_final.pt", map_location=device, weights_only=False)["controller"])
    sde = ControlledSDE(ref_sde, ctrl).to(device)
    energy = hydra.utils.instantiate(cfg.energy, device=str(device))   # DistEnergy moves its distribution to `device`
    timesteps = train_utils.get_timesteps(**cfg.timesteps).to(device)
    lws, xs, done = [], [], 0
    while done < n_is:
        B = min(5000, n_is - done)
        lw, x = girsanov.log_path_weights(sde, energy, torch.zeros(B, d, device=device), timesteps)
        lws.append(lw.cpu()); xs.append(x.cpu()); done += B
    return torch.cat(lws), torch.cat(xs)


def analyse(run, n_is, do_is):
    cfg = {l.split(":")[0].strip(): l.split(":")[1].strip() for l in open(Path(run) / "config.yaml") if l.split(":")[0].strip() in ("dim", "K", "delta") and not l.startswith(" ")}
    d, K, delta = int(cfg["dim"]), int(cfg["K"]), float(cfg["delta"])
    p = G.true_proportions(d, K, delta)
    floor = G.tv_floor(d, K, delta, p, n=5000, reps=100)
    files = sorted(glob.glob(str(Path(run) / "eval_samples" / "ep*.pt")))
    last = files[-10:]
    phs = [G.proportions(torch.load(f, weights_only=False), d, K, delta) for f in last]
    ph = torch.stack(phs).mean(0)
    cov, lost, c = G.coverage(ph, p)
    b, se = G.shrink_beta(ph, p, c)
    out = dict(run=Path(run).name, d=d, K=K, delta=delta, n_evals=len(files), tv=float(np.mean([G.tv(x, p) for x in phs])), tv_sd=float(np.std([G.tv(x, p) for x in phs])),
               floor_median=floor["median"], floor_p5=floor["p5"], floor_p95=floor["p95"], coverage=cov, lost=lost, beta=b, beta_se=se,
               ratios=[round(float(x), 3) for x in (ph / p)], p=[round(float(x), 4) for x in p], p_hat=[round(float(x), 4) for x in ph])
    if do_is:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lw, x = is_paths(run, d, K, delta, n_is, dev)
        w = torch.softmax(lw.double(), 0)
        out.update(is_n=n_is, is_ess_frac=float(1.0 / (w ** 2).sum() / len(w)), is_khat=khat(lw),
                   is_tv_weighted=G.tv(G.proportions(x, d, K, delta, w), p), is_tv_unweighted=G.tv(G.proportions(x, d, K, delta), p))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--n_is", type=int, default=20000); ap.add_argument("--no_is", action="store_true")
    a = ap.parse_args()
    res = [analyse(r, a.n_is, not a.no_is) for r in a.runs]
    for r in res:
        print(f"{r['run']}: {r['n_evals']} evals | coverage {r['coverage']}/{r['K']} lost {r['lost']} | TV {r['tv']:.3f} (sd {r['tv_sd']:.3f}) floor {r['floor_median']:.3f} [{r['floor_p5']:.3f}-{r['floor_p95']:.3f}] | beta {r['beta']:+.2f} ± {r['beta_se']:.2f}")
        print(f"    p      {r['p']}\n    p_hat  {r['p_hat']}\n    ratios {r['ratios']}")
        if "is_ess_frac" in r:
            print(f"    path IS (N={r['is_n']}): ESS frac {r['is_ess_frac']:.4g}, khat {r['is_khat']:.2f}, TV weighted {r['is_tv_weighted']:.3f} (unweighted {r['is_tv_unweighted']:.3f})")
    json.dump(res, open("gmm_dk_analysis.json", "w"), indent=1)
