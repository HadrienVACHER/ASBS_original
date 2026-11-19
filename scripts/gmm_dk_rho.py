""" H_rho (predictions_dk.md, addendum 2): rho = sqrt(d nu1) / |mu|, nu1 = left Riemann sum of g(t_k)^2 dt over linspace(0, 1, 200) (the real scheme), |mu| = Delta / sqrt 2.
    Usage: python scripts/gmm_dk_rho.py OUT_PNG OUT_JSON ANALYSIS_JSON... ; CPU, seconds. Report only. """
import json, math, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train  # noqa: F401  (registers the dk_sigma resolver)
from omegaconf import OmegaConf

def sigma_of(run):
    p = next(Path("results").glob(f"gmm_dk*/{run}/config.yaml"))
    cfg = OmegaConf.load(p); OmegaConf.resolve(cfg)
    return float(cfg.sigma_max), float(cfg.sigma_min)

def nu1(smax, smin, steps=200):
    t = torch.linspace(0, 1, steps).double()
    g2 = smin ** 2 * (smax / smin) ** (2 * (1 - t[:-1])) * 2 * math.log(smax / smin)
    return float((g2 * (t[1:] - t[:-1])).sum())

rows = []
for f in sys.argv[3:]:
    for r in json.load(open(f)):
        smax, smin = sigma_of(r["run"])
        n1 = nu1(smax, smin); mu = r["delta"] / math.sqrt(2)
        rows.append(dict(run=r["run"], d=r["d"], K=r["K"], delta=r["delta"], sigma_max=smax, nu1=n1, rho=math.sqrt(r["d"] * n1) / mu, coverage=r["coverage"], tv=r["tv"],
                         floor=r["floor_median"], beta=r["beta"], beta_se=r["beta_se"], abl=("_sx" in r["run"])))
rows.sort(key=lambda r: (r["d"], r["K"], r["delta"], r["sigma_max"]))
json.dump(rows, open(sys.argv[2], "w"), indent=1)
print(f"{'d':>3} {'K':>2} {'Delta':>5} {'sigma_max':>9} {'nu1':>8} {'rho':>6} {'cov':>5} {'TV/floor':>9} {'beta':>7}")
for r in rows:
    print(f"{r['d']:3d} {r['K']:2d} {r['delta']:5.1f} {r['sigma_max']:9.3f} {r['nu1']:8.3f} {r['rho']:6.2f} {r['coverage']:2d}/{r['K']:<2d} {r['tv']/r['floor']:9.1f} {r['beta']:+7.2f}" + ("   (ablation)" if r["abl"] else ""))
# criterion of the addendum: best common threshold, then per-d and per-Delta groups
lost = np.array([r["coverage"] < r["K"] for r in rows]); rho = np.array([r["rho"] for r in rows])
def best(mask):
    xs = np.sort(rho[mask]); cands = np.concatenate([[xs[0] - 1], (xs[:-1] + xs[1:]) / 2, [xs[-1] + 1]])
    errs = [int(((rho[mask] > c) != lost[mask]).sum()) for c in cands]
    ok = [c for c, e in zip(cands, errs) if e == min(errs)]
    return min(errs), min(ok), max(ok)
e, lo, hi = best(np.ones(len(rows), bool))
print(f"\ncommon threshold over the {len(rows)} runs: min errors {e}  (rho* in [{lo:.2f}, {hi:.2f}])  -> {'SUPPORTED' if e <= 2 else 'not supported'} (criterion: <= 2 errors)")
for key, name in (("d", "d"), ("delta", "Delta")):
    for v in sorted(set(r[key] for r in rows)):
        m = np.array([r[key] == v for r in rows]); e, lo, hi = best(m)
        print(f"  {name} = {v}: {m.sum()} runs, min errors {e}, rho* in [{lo:.2f}, {hi:.2f}]")
