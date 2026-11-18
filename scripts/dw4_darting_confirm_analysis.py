""" Verdict of rules_darting_confirm.md (written and committed before any run).
      python scripts/dw4_darting_confirm_analysis.py DIR_D30_s2 DIR_D30_s3 DIR_D30_s4 DIR_C2 DIR_C3 DIR_C4        (each DIR contains stdout.log; C2 = the RE-RUN C2 of the same instance, amendment 1)
    The cluster C2 is read from W&B (CCDS_NTU/as, dw4_cont_C2, commit aadf33b) and only REPORTED. """
import ast
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dw4_reeval_pt as rp  # noqa: E402

K, THR = rp.K, 0.05
tr = rp.truth()


def read_log(d):
    txt = (Path(d) / "stdout.log").read_text().splitlines()
    evals = [ast.literal_eval(l) for l in txt if l.startswith("{'energy_w2'")]
    losses = [float(m.group(2)) for l in txt for m in [re.match(r"\[adjoint \| ep=(\d+)\] loss=([0-9a-z.+-]+)", l)] if m]
    vals = {}
    for l in txt:
        if l.startswith("{'time/epoch_s'"):
            for k, v in re.findall(r"'(dart/[^']+)': ([0-9.e+-]+|nan)", l):
                vals.setdefault(k, []).append(float(v))
    last = evals[-20:]
    return dict(Q=np.array([[e[f"basin/{k}"] for k in range(K)] for e in last]), R=np.array([[e[f"state/{k}"] for k in range(K)] for e in last]),
                eq=float(np.mean([e["eq_w2"] for e in evals if "eq_w2" in e][-2:])), en=float(np.mean([e["energy_w2"] for e in evals][-2:])),
                n_eval=len(evals), n_ep=len(losses), finite=bool(np.isfinite(losses).all()), stats=vals)


def S(X, tag):
    return float(np.abs(X.mean(0) - tr[tag][0]).sum())


import wandb  # noqa: E402
runs = list(wandb.Api(timeout=120).runs("CCDS_NTU/as", per_page=200, order="-created_at"))
r = [x for x in runs if x.name == "dw4_cont_C2" and x.config.get("git_commit", "")[:7] == rp.COMMIT][0]
K11 = list(range(K))
h = r.history(samples=6000, x_axis="_step", keys=["basin_tv"] + [f"basin/{k}" for k in K11] + [f"state/{k}" for k in K11]).dropna(subset=["basin_tv"]).set_index("_step").iloc[-20:]
C_CLUSTER = {2: dict(Q=h[[f"basin/{k}" for k in K11]].values, R=h[[f"state/{k}" for k in K11]].values,
             eq=float(r.history(samples=6000, x_axis="_step", keys=["eq_w2"])["eq_w2"].dropna().values[-2:].mean()),
             en=float(r.history(samples=6000, x_axis="_step", keys=["energy_w2"])["energy_w2"].dropna().values[-2:].mean()), finite=True)}
C = {}
D = {}
for seed, d in zip((2, 3, 4), sys.argv[1:4]):
    D[seed] = read_log(d)
for seed, d in zip((2, 3, 4), sys.argv[4:7]):
    C[seed] = read_log(d)
for seed in D:
    print(f"D30 seed {seed}: {D[seed]['n_ep']} epochs, {D[seed]['n_eval']} evaluations, loss finite {D[seed]['finite']}")
for seed in (2, 3, 4):
    print(f"C{seed} (re-run): {C[seed]['n_ep']} epochs, {C[seed]['n_eval']} evaluations, loss finite {C[seed]['finite']}")
print("\n=== S (mean of the last 20 evaluations, epochs 300-490) ; floor of a perfect sampler: Q median 0.0034 (p95 0.0060), R 0.0036 (0.0070) ===")
print(f"{'seed':>5} | {'S_Q D30':>8} {'S_Q C':>8} {'Delta_Q':>8} | {'S_R D30':>8} {'S_R C':>8} {'Delta_R':>8} | {'eq_w2 D30 / C':>16} {'energy_w2 D30 / C':>19}")
(res_i, res_iii) = {}, {}
for s in (2, 3, 4):
    sq, sqc, sr, src = S(D[s]["Q"], "Q"), S(C[s]["Q"], "Q"), S(D[s]["R"], "R"), S(C[s]["R"], "R")
    res_i[s] = sq < sqc - THR
    res_iii[s] = D[s]["eq"] <= 1.10 * max(C[t]["eq"] for t in C) and D[s]["en"] <= 1.10 * max(C[t]["en"] for t in C)
    print(f"{s:5d} | {sq:8.4f} {sqc:8.4f} {sq - sqc:+8.4f} | {sr:8.4f} {src:8.4f} {sr - src:+8.4f} | {D[s]['eq']:7.3f} /{C[s]['eq']:7.3f} {D[s]['en']:9.3f} /{C[s]['en']:7.3f}")
sD = [S(D[s]["Q"], "Q") for s in (2, 3, 4)]
sC = [S(C[s]["Q"], "Q") for s in (2, 3, 4)]
res_ii = float(np.median(sD)) < min(sC)
finite = all(D[s]["finite"] for s in D)
print(f"\n(i)   S_Q(D30_s) < S_Q(C_s) - {THR} for each seed: " + ", ".join(f"seed {s}: {res_i[s]}" for s in res_i) + f"  => {all(res_i.values())}")
print(f"(ii)  median S_Q(D30) {np.median(sD):.4f} < min S_Q(C2, C3, C4) {min(sC):.4f}  => {res_ii}")
print(f"(iii) eq_w2 and energy_w2 of D30_s <= 1.1 x max over C2, C3, C4 (eq_w2 max {max(C[t]['eq'] for t in C):.3f}, energy_w2 max {max(C[t]['en'] for t in C):.3f}) for each seed: " + ", ".join(f"seed {s}: {res_iii[s]}" for s in res_iii) + f"  => {all(res_iii.values())}")
print(f"finite losses in all D30 arms: {finite}")
print("VERDICT: " + ("effet confirme" if all(res_i.values()) and res_ii and all(res_iii.values()) and finite else "non confirme"))
print(f"\nreported only, cluster C2: S_Q {S(C_CLUSTER[2]['Q'], 'Q'):.4f}, S_R {S(C_CLUSTER[2]['R'], 'R'):.4f}, eq_w2 {C_CLUSTER[2]['eq']:.3f}, energy_w2 {C_CLUSTER[2]['en']:.3f}")
print("\n=== per basin: p_PT, p-bar and |log(p-bar / p_PT)| of D30_s and of C_s (variante Q) ===")
tq = tr["Q"][0]
for s in (2, 3, 4):
    print(f"  seed {s}:  {'basin':26} {'p_PT':>7} {'D30':>8} {'C':>8} | err D30 / C")
    for k in range(K):
        if tq[k] < 0.005:
            continue
        pd_, pc = D[s]["Q"].mean(0)[k], C[s]["Q"].mean(0)[k]
        print(f"            {rp.NAME[k]:26} {tq[k]:7.4f} {pd_:8.4f} {pc:8.4f} | {abs(math.log(max(pd_, 1e-9) / tq[k])):.3f} / {abs(math.log(max(pc, 1e-9) / tq[k])):.3f}")
print("\n=== dose (matcher statistics, mean over the run) ===")
for s in (2, 3, 4):
    print(f"  D30 seed {s}: " + "; ".join(f"{k.replace('dart/', '')} {np.nanmean(v):.3f}" for k, v in sorted(D[s]["stats"].items()) if "term_over" in k or "valid" in k))
