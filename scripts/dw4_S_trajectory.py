""" S_Q and S_R per evaluation (epochs 0..490) for D30 seeds 1-4 and the AS controls C1-C4, and the slope of S over epochs 300-490 (report only; local, seconds).
    S(e) = sum_b |p_b(e) - p_PT,b| with p(e) the proportions of ONE evaluation (5000 samples; its own sampling noise adds to S: floor of one evaluation ~ 0.02, see below).
    Slope: OLS on the 20 evaluations of epochs 300-490, in S units per 100 epochs; SE_ols (iid residuals) and SE_ar1 = SE_ols * sqrt((1 + rho) / (1 - rho)) with rho the lag-1 autocorrelation of the residuals (honest when the residuals are autocorrelated).
      python scripts/dw4_S_trajectory.py """
import ast
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dw4_reeval_pt as rp  # noqa: E402

K, ROOT = rp.K, Path(__file__).resolve().parents[1]
tr = rp.truth()


def from_log(path):
    ev = [ast.literal_eval(l) for l in Path(path).read_text().splitlines() if l.startswith("{'energy_w2'")]
    return {t: np.array([[e[f"{'basin' if t == 'Q' else 'state'}/{k}"] for k in range(K)] for e in ev]) for t in "QR"}


def from_wandb(runs, name):
    r = [x for x in runs if x.name == name and (x.config.get("git_commit") or "")[:7] == rp.COMMIT][0]
    h = r.history(samples=6000, x_axis="_step", keys=["basin_tv"] + [f"basin/{k}" for k in range(K)] + [f"state/{k}" for k in range(K)]).dropna(subset=["basin_tv"]).sort_values("_step")
    return {"Q": h[[f"basin/{k}" for k in range(K)]].values, "R": h[[f"state/{k}" for k in range(K)]].values}


import wandb  # noqa: E402
runs = list(wandb.Api(timeout=120).runs("CCDS_NTU/as", per_page=200, order="-created_at"))
arms = {
    "D30 s1": from_log(ROOT / "results" / "dw4_cont_D30_seed1_fast" / "stdout.log"),
    "D30 s2": from_log(ROOT / "results" / "confirm" / "dw4_cont_D30_s2" / "stdout.log"),
    "D30 s3": from_log(ROOT / "results" / "confirm" / "dw4_cont_D30_s3" / "stdout.log"),
    "D30 s4": from_log(ROOT / "results" / "confirm" / "dw4_cont_D30_s4" / "stdout.log"),
    "C1 (cluster)": from_wandb(runs, "dw4_cont_C1"),
    "C2 (re-run)": from_log(ROOT / "results" / "confirm" / "dw4_cont_C2_s2" / "stdout.log"),
    "C2 (cluster)": from_wandb(runs, "dw4_cont_C2"),
    "C3 (re-run)": from_log(ROOT / "results" / "confirm" / "dw4_cont_C3_s3" / "stdout.log"),
    "C4 (re-run)": from_log(ROOT / "results" / "confirm" / "dw4_cont_C4_s4" / "stdout.log"),
}
ep = np.arange(50) * 10.0
out = {}
for tag in "QR":
    truth = tr[tag][0]
    print(f"\n===== variant {tag}: S per evaluation, mean S by blocks of 100 epochs, slope over epochs 300-490 =====")
    print(f"{'arm':14} {'0-99':>7} {'100-199':>8} {'200-299':>8} {'300-399':>8} {'400-490':>8} | {'slope /100 ep':>14} {'SE ols':>7} {'SE ar1':>7} {'z (ar1)':>8} | {'S 300-490':>9}")
    for n, d in arms.items():
        assert len(d[tag]) == 50, (n, len(d[tag]))
        S = np.abs(d[tag] - truth).sum(1)
        out[(tag, n)] = S
        blocks = [S[a:b].mean() for a, b in ((0, 10), (10, 20), (20, 30), (30, 40), (40, 50))]
        x, y = ep[30:], S[30:]
        xc = x - x.mean()
        slope = float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
        res = y - y.mean() - slope * xc
        se = float(math.sqrt((res ** 2).sum() / (len(x) - 2) / (xc ** 2).sum()))
        rho = float((res[1:] * res[:-1]).sum() / (res ** 2).sum())
        se_ar = se * math.sqrt((1 + max(rho, 0.0)) / (1 - max(rho, 0.0)))
        print(f"{n:14} " + " ".join(f"{b:8.4f}" for b in blocks) + f" | {100 * slope:+14.4f} {100 * se:7.4f} {100 * se_ar:7.4f} {slope / se_ar:+8.2f} | {y.mean():9.4f}   (rho {rho:+.2f})")
    for t in ("D30", "C"):
        sel = [n for n in arms if n.startswith(t) and "cluster" not in n or (t == "C" and n.startswith("C1"))]
        sl = []
        for n in sel:
            y = out[(tag, n)][30:]
            xc = ep[30:] - ep[30:].mean()
            sl.append(100 * float((xc * (y - y.mean())).sum() / (xc ** 2).sum()))
        print(f"   mean slope over {', '.join(sel)}: {np.mean(sl):+.4f} per 100 epochs (SD between arms {np.std(sl, ddof=1):.4f})")
print("\nfloor of ONE evaluation (5000 draws from the PT pool, S vs p_PT): see below")
rng = np.random.default_rng(0)
import torch  # noqa: E402
d = torch.load(ROOT / "results" / "pt_samples_T1.pt", map_location="cpu", weights_only=False)
from adjoint_samplers.utils.dw4_states import state_ids  # noqa: E402
flat = {"Q": d["basin_state_id"].numpy().reshape(-1), "R": torch.stack([state_ids(d["states"][c].double()) for c in range(64)]).numpy().reshape(-1)}
for tag in "QR":
    s1 = [float(np.abs(np.bincount(rng.choice(flat[tag], 5000), minlength=K) / 5000 - tr[tag][0]).sum()) for _ in range(200)]
    print(f"  [{tag}] S of one 5000-sample evaluation of a perfect sampler: median {np.median(s1):.4f} (p5 {np.percentile(s1, 5):.4f}, p95 {np.percentile(s1, 95):.4f})")
