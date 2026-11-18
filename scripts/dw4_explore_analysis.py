""" Descriptive reading of the seed-1 exploration (D100, D30 + focus, AS + focus, D100 + focus) next to D3, D10, D30 and C1 of seed 1 (RESULTS.md, no verdict, no effect rule).
      python scripts/dw4_explore_analysis.py
    S (variants Q and R) = sum_b |p_bar - p_PT| over the last 20 evaluations (epochs 300-490); error |log(p_bar / p_PT)| per basin (Q); time means of eq_w2 and energy_w2 over epochs 300-490 with SE by batch means (lots of 5 consecutive evaluations;
    eq_w2 has 20 values in the exploration arms, 3 in the older arms: SD / sqrt(3) there); slope of the per-evaluation S over epochs 300-490 (OLS, SE corrected for AR(1)); losses; matcher statistics. """
import ast
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dw4_reeval_pt as rp  # noqa: E402

K, ROOT = rp.K, Path(__file__).resolve().parents[1]
tr = rp.truth()
ARMS = {
    "D100": "explore/dw4_cont_D100_s1", "D30F": "explore/dw4_cont_D30F_s1", "CF (AS+focus)": "explore/dw4_cont_CF_s1", "D100F": "explore/dw4_cont_D100F_s1",
    "D3 (s1)": "dw4_cont_D3_seed1_fast", "D10 (s1)": "dw4_cont_D10_seed1_fast", "D30 (s1)": "dw4_cont_D30_seed1_fast",
}


def read(rel):
    txt = (ROOT / "results" / rel / "stdout.log").read_text().splitlines()
    ev = [ast.literal_eval(l) for l in txt if l.startswith("{'energy_w2'")]
    losses = [float(m.group(2)) for l in txt for m in [re.match(r"\[adjoint \| ep=(\d+)\] loss=([0-9a-z.+-]+)", l)] if m]
    st = {}
    for l in txt:
        if l.startswith("{'time/epoch_s'"):
            for k, v in re.findall(r"'(dart/[^']+)': ([0-9.e+-]+|nan)", l):
                st.setdefault(k, []).append(float(v))
    return dict(Q=np.array([[e[f"basin/{k}"] for k in range(K)] for e in ev]), R=np.array([[e[f"state/{k}"] for k in range(K)] for e in ev]),
                en=np.array([e["energy_w2"] for e in ev]), eq=[(i, e["eq_w2"]) for i, e in enumerate(ev) if "eq_w2" in e], losses=np.array(losses), stats=st)


import wandb  # noqa: E402
runs = list(wandb.Api(timeout=120).runs("CCDS_NTU/as", per_page=200, order="-created_at"))
r = [x for x in runs if x.name == "dw4_cont_C1" and (x.config.get("git_commit") or "")[:7] == rp.COMMIT][0]
hb = r.history(samples=6000, x_axis="_step", keys=["basin_tv"] + [f"basin/{k}" for k in range(K)] + [f"state/{k}" for k in range(K)]).dropna(subset=["basin_tv"]).sort_values("_step")
he = r.history(samples=6000, x_axis="_step", keys=["energy_w2"]).dropna().sort_values("_step")
hq = r.history(samples=6000, x_axis="_step", keys=["eq_w2"]).dropna().sort_values("_step")
D = {n: read(p) for n, p in ARMS.items()}
D["C1 (cluster)"] = dict(Q=hb[[f"basin/{k}" for k in range(K)]].values, R=hb[[f"state/{k}" for k in range(K)]].values, en=he["energy_w2"].values.astype(float),
                         eq=[(40 + (int(s) - 400) // 10 if s >= 400 else int(s) // 10, v) for s, v in zip(hq["_step"], hq["eq_w2"])], losses=np.array([np.nan]), stats={})
order = ["C1 (cluster)", "D3 (s1)", "D10 (s1)", "D30 (s1)", "D100", "D30F", "CF (AS+focus)", "D100F"]
tq = tr["Q"][0]
print("=== 1. S (mean of the last 20 evaluations, epochs 300-490); floor of a perfect sampler (20 x 5000): Q 0.0034, R 0.0036 ===")
print(f"{'arm':16} {'S_Q':>8} {'S_R':>8} | slope of per-evaluation S_Q / 100 ep (SE ar1, z) | S_Q by blocks of 100 ep")
for n in order:
    d = D[n]
    sq, sr = (float(np.abs(d[t][-20:].mean(0) - tr[t][0]).sum()) for t in "QR")
    S = np.abs(d["Q"] - tq).sum(1)
    ep = np.arange(50) * 10.0
    x, y = ep[30:], S[30:]
    xc = x - x.mean()
    slope = float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
    res = y - y.mean() - slope * xc
    se = math.sqrt((res ** 2).sum() / (len(x) - 2) / (xc ** 2).sum())
    rho = max(float((res[1:] * res[:-1]).sum() / (res ** 2).sum()), 0.0)
    sea = se * math.sqrt((1 + rho) / (1 - rho))
    blocks = " ".join(f"{S[a:b].mean():.3f}" for a, b in ((0, 10), (10, 20), (20, 30), (30, 40), (40, 50)))
    print(f"{n:16} {sq:8.4f} {sr:8.4f} | {100 * slope:+8.4f} ({100 * sea:.4f}, {slope / sea:+.2f})                    | {blocks}")
print("\n=== 2. error |log(p_bar / p_PT)| per basin, variant Q (p_bar: last 20 evaluations) ===")
names = [rp.NAME[k] for k in range(K) if tq[k] >= 0.005]
print(f"{'arm':16} " + " ".join(f"{n[:18]:>19}" for n in names))
print(f"{'p_PT':16} " + " ".join(f"{tq[k]:19.4f}" for k in range(K) if tq[k] >= 0.005))
for n in order:
    p = D[n]["Q"][-20:].mean(0)
    print(f"{n:16} " + " ".join(f"{abs(math.log(max(p[k], 1e-9) / tq[k])):9.3f} ({p[k]:.4f})" for k in range(K) if tq[k] >= 0.005))
print("\n=== 3. time means over epochs 300-490 (SE: batch means of lots of 5 evaluations; eq_w2 with 3 values: SD / sqrt(3)) ===")
print(f"{'arm':16} {'energy_w2':>10} {'SE':>7} | {'eq_w2':>7} {'SE':>7} (n)")
for n in order:
    d = D[n]
    en = d["en"][-20:]
    eqv = np.array([v for i, v in d["eq"] if i >= 30])
    lots = en.reshape(4, 5).mean(1)
    se_en = lots.std(ddof=1) / 2
    if len(eqv) == 20:
        se_eq = eqv.reshape(4, 5).mean(1).std(ddof=1) / 2
    else:
        se_eq = eqv.std(ddof=1) / math.sqrt(len(eqv))
    print(f"{n:16} {en.mean():10.3f} {se_en:7.3f} | {eqv.mean():7.3f} {se_eq:7.3f} ({len(eqv)})")
print("\n=== 4. stability and matcher statistics ===")
for n in order:
    d = D[n]
    l = d["losses"]
    msg = f"{n:16} epochs {len(l)}, loss finite {bool(np.isfinite(l).all())}, min {np.nanmin(l):.2f} max {np.nanmax(l):.2f} last {l[-1]:.2f}" if len(l) > 1 else f"{n:16} (cluster run: not read)"
    st = d["stats"]
    if st:
        msg += " | term/grad [0,.1] [.1,.7] [.7,1]: " + " ".join(f"{np.nanmean(st[k]):.3f}" for k in sorted(st) if "term_over" in k) + " | frac_t: " + " ".join(f"{np.nanmean(st[k]):.3f}" for k in sorted(st) if "frac_t" in k) + f" | valid/sample {np.nanmean(st['dart/valid_proposals_per_sample']):.2f}"
    print(msg)
