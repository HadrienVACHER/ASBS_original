""" eq_w2 and energy_w2 of the confirmation arms, time-averaged over the evaluations of epochs 300-490 (report only; local, seconds; no rule).
    Evaluations are every 10 epochs (epochs 0..490, 50 evaluations); energy_w2 at every evaluation (20 values in 300-490), eq_w2 only at epochs 0, 100, ..., 400 and 490 (the last evaluation): 3 values in 300-490 (epochs 300, 400, 490).
    SE: batch means over lots of 5 consecutive evaluations (4 lots for energy_w2); eq_w2 has at most one value per lot of 5 evaluations, so its lots are single values: SE = SD / sqrt(3) (3 values only: crude).
    Paired difference D30 - C per seed: SE = sqrt(SE_D^2 + SE_C^2) (the two runs share the seed, not the trajectory: the independence assumption is conservative).
      python scripts/dw4_confirm_w2_timeavg.py """
import ast
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dw4_reeval_pt as rp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def from_log(path):
    ev = [ast.literal_eval(l) for l in Path(path).read_text().splitlines() if l.startswith("{'energy_w2'")]
    idx = range(len(ev) - 20, len(ev))                                                   # epochs 300..490
    en = np.array([ev[i]["energy_w2"] for i in idx])
    eq = np.array([ev[i]["eq_w2"] for i in idx if "eq_w2" in ev[i]])
    return en, eq


def stats(en, eq):
    lots = en.reshape(4, 5).mean(1)
    se_en = float(lots.std(ddof=1) / math.sqrt(4))
    se_eq = float(eq.std(ddof=1) / math.sqrt(len(eq)))
    return float(en.mean()), se_en, float(eq.mean()), se_eq, len(eq)


arms = {}
for n in ("D30_s2", "D30_s3", "D30_s4", "C2_s2", "C3_s3", "C4_s4"):
    arms[n] = stats(*from_log(ROOT / "results" / "confirm" / f"dw4_cont_{n}" / "stdout.log"))
for g in (3, 10, 30):
    arms[f"D{g}_s1(local fast)"] = stats(*from_log(ROOT / "results" / f"dw4_cont_D{g}_seed1_fast" / "stdout.log"))
for g in (3,):
    arms[f"D{g}_s2(local fast, phase 1)"] = stats(*from_log(ROOT / "results" / f"dw4_cont_D{g}_seed2_fast" / "stdout.log"))
import wandb  # noqa: E402
runs = list(wandb.Api(timeout=120).runs("CCDS_NTU/as", per_page=200, order="-created_at"))
for arm in ("C1", "C2"):
    r = [x for x in runs if x.name == rp.ARMS[arm] and (x.config.get("git_commit") or "")[:7] == rp.COMMIT][0]
    he = r.history(samples=6000, x_axis="_step", keys=["energy_w2"]).dropna().sort_values("_step")
    hq = r.history(samples=6000, x_axis="_step", keys=["eq_w2"]).dropna().sort_values("_step")
    en = he["energy_w2"].values.astype(float)[-20:]                                                # epochs 300-490
    eq = hq[hq["_step"] >= 300]["eq_w2"].values.astype(float)                                       # epochs 300, 400, 490
    arms[f"{arm}(cluster)"] = stats(en, eq)
print(f"{'arm':28} {'energy_w2':>10} {'SE':>7} | {'eq_w2':>7} {'SE':>7} (n eq)")
for n, (me, se, mq, sq, k) in arms.items():
    print(f"{n:28} {me:10.3f} {se:7.3f} | {mq:7.3f} {sq:7.3f} ({k})")
print("\npaired D30 - C (same seed), time means over epochs 300-490")
rows = []
for s, c in ((2, "C2_s2"), (3, "C3_s3"), (4, "C4_s4")):
    (de, dse, dq, dsq, _), (ce, cse, cq, csq, _) = arms[f"D30_s{s}"], arms[c]
    rows.append((s, de - ce, math.hypot(dse, cse), dq - cq, math.hypot(dsq, csq)))
(de, dse, dq, dsq, _), (ce, cse, cq, csq, _) = arms["D30_s1(local fast)"], arms["C1(cluster)"]
row1 = (1, de - ce, math.hypot(dse, cse), dq - cq, math.hypot(dsq, csq))
print(f"{'seed':>5} | {'d energy_w2':>11} {'SE':>7} {'z':>6} | {'d eq_w2':>8} {'SE':>7} {'z':>6}")
for s, d1, s1, d2, s2 in rows + [row1]:
    print(f"{s:5d} | {d1:+11.3f} {s1:7.3f} {d1 / s1:+6.2f} | {d2:+8.3f} {s2:7.3f} {d2 / s2:+6.2f}" + ("   (seed 1: C1 from the cluster, seed used to choose gamma: not in the mean)" if s == 1 else ""))
for name, j in (("energy_w2", 1), ("eq_w2", 3)):
    d = np.array([r[j] for r in rows])
    se = np.array([r[j + 1] for r in rows])
    m, sm = float(d.mean()), float(np.sqrt((se ** 2).sum()) / 3)
    same = bool((d > 0).all() or (d < 0).all())
    nz = int((np.abs(d / se) > 2).sum())
    print(f"mean over seeds 2-4 of D30 - C, {name}: {m:+.3f} (SE {sm:.3f}, z {m / sm:+.2f}); same sign on the 3 seeds: {same}; seeds with |z| > 2: {nz}; 'systematically higher' (same positive sign AND |z| > 2 on at least 2 seeds): {bool((d > 0).all() and nz >= 2)}")
