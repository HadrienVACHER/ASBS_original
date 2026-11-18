""" Reading of the ONE darting continuation run (rules_darting_run.md, committed before the run): S against p_PT (variants R and Q), errors per class against the two AS runs C1 and C2, sanity.
      python scripts/dw4_darting_run_analysis.py [results/dw4_cont_D3_s1/stdout.log] """
import ast
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dw4_reeval_pt as rp  # noqa: E402

log = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results" / "dw4_cont_D3_s1" / "stdout.log")
txt = log.read_text().splitlines()
evals = [ast.literal_eval(l) for l in txt if l.startswith("{'energy_w2'")]
stats = [ast.literal_eval(re.sub(r"\bnan\b", "float('nan')", l)) if False else l for l in txt if l.startswith("{'time/epoch_s'")]
losses = [float(m.group(2)) for l in txt for m in [re.match(r"\[adjoint \| ep=(\d+)\] loss=([0-9a-z.+-]+)", l)] if m]
print(f"run: {len(losses)} epochs logged, {len(evals)} evaluations; loss finite: {all(math.isfinite(v) for v in losses)} (min {min(losses):.3f}, max {max(losses):.3f}, last {losses[-1]:.3f})")
K = rp.K
Q = np.array([[e[f"basin/{k}"] for k in range(K)] for e in evals])
R = np.array([[e[f"state/{k}"] for k in range(K)] for e in evals])
last = slice(len(evals) - 20, len(evals))
tr = rp.truth()
import wandb  # noqa: E402
runs = list(wandb.Api(timeout=120).runs("CCDS_NTU/as", per_page=200, order="-created_at"))
ref = {}
for arm in ("C1", "C2"):
    r = [x for x in runs if x.name == rp.ARMS[arm] and x.config.get("git_commit", "")[:7] == rp.COMMIT][0]
    h = r.history(samples=6000, x_axis="_step", keys=["basin_tv"] + [f"basin/{k}" for k in range(K)] + [f"state/{k}" for k in range(K)]).dropna(subset=["basin_tv"]).set_index("_step").iloc[-20:]
    e = r.history(samples=6000, x_axis="_step", keys=["eq_w2"])["eq_w2"].dropna().values[-2:]
    en = r.history(samples=6000, x_axis="_step", keys=["energy_w2"])["energy_w2"].dropna().values[-2:]
    ref[arm] = dict(Q=h[[f"basin/{k}" for k in range(K)]].values, R=h[[f"state/{k}" for k in range(K)]].values, eq=float(e.mean()), en=float(en.mean()))
print("\n=== 1. S against p_PT (mean of the last 20 evaluations, epochs 300-490) ===")
for tag, X in (("R", R), ("Q", Q)):
    tru = tr[tag][0]
    s = lambda M: float(np.abs(M.mean(0) - tru).sum())
    sD, s1, s2 = s(X[last]), s(ref["C1"][tag]), s(ref["C2"][tag])
    pos = "below both" if sD < min(s1, s2) else "between" if sD < max(s1, s2) else "above both"
    print(f"  [{tag}] S_darting {sD:.4f}   S_C1 {s1:.4f}   S_C2 {s2:.4f}   |S_C1 - S_C2| {abs(s1 - s2):.4f}   => {pos}")
print("\n=== 2. error |log(p / p_PT)| per class, darting vs C1 and C2 (classes with p_PT >= 0.005) ===")
better_total = {}
for tag, X in (("Q", Q), ("R", R)):
    tru = tr[tag][0]
    better = 0
    print(f"  [{tag}]  {'class':26} {'p_PT':>7} {'darting':>8} {'C1':>8} {'C2':>8} | err D / C1 / C2")
    for k in range(K):
        if tru[k] < 0.005:
            continue
        pd_, p1, p2 = X[last][:, k].mean(), ref["C1"][tag][:, k].mean(), ref["C2"][tag][:, k].mean()
        ed, e1, e2 = (abs(math.log(max(p, 1e-9) / tru[k])) for p in (pd_, p1, p2))
        small = rp.NAME[k] in ("2 disjoint pairs", "star (3 edges)", "K4 minus an edge")
        if small and ed < min(e1, e2):
            better += 1
        print(f"        {rp.NAME[k]:26} {tru[k]:7.4f} {pd_:8.4f} {p1:8.4f} {p2:8.4f} | {ed:.3f} / {e1:.3f} / {e2:.3f}" + ("   (small basin)" if small else ""))
    better_total[tag] = better
print("  small basins (two disjoint pairs, K4 minus an edge, star) where darting is better than BOTH AS runs: " + ", ".join(f"{t}: {b}/3" for t, b in better_total.items()))
print("\n=== 3. sanity ===")
eq = [e["eq_w2"] for e in evals if "eq_w2" in e][-2:]
en = [e["energy_w2"] for e in evals][-2:]
print(f"  eq_w2 (last two): darting {np.mean(eq):.3f}  C1 {ref['C1']['eq']:.3f}  C2 {ref['C2']['eq']:.3f}   energy_w2 (last two evaluations): darting {np.mean(en):.3f}  C1 {ref['C1']['en']:.3f}  C2 {ref['C2']['en']:.3f}")
for l in txt:
    if l.startswith("{'time/epoch_s'"):
        last_stats = l
vals = {}
for l in txt:
    if l.startswith("{'time/epoch_s'"):
        for k, v in re.findall(r"'(dart/[^']+)': ([0-9.e+-]+|nan)", l):
            vals.setdefault(k, []).append(float(v))
print("  matcher statistics (mean over the run): " + "; ".join(f"{k.replace('dart/', '')} {np.nanmean(v):.4f}" for k, v in sorted(vals.items()) if "outflux" in k or "term_over" in k or "valid" in k or "untyped" in k))
