""" Decomposition of the cost of arm (s) per refresh (rules_selftrain_dartbank.md; dart_lib cost convention: 1 energy = 1, 1 gradient = 1). Components are computed from the code of dart_lib._darting_from_minima
    with the logged jump acceptance, and CHECKED against the logged total per batch. python scripts/selftrain_cost_decomp.py """
import json, re
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; RUNS = ROOT / "results" / "selftrain_runs"
CELLS = {"k8d32": (32, 8), "mosw10": (10, 10), "d128": (128, 8)}
S = 10; out = {}
for c, (d, M) in CELLS.items():
    log = open(RUNS / f"sd_s_{c}" / "stdout.log").read()
    rows = [(int(a), int(n), float(j), float(m), float(co)) for a, n, j, m, co in re.findall(r"\[dartbank\] pop (\d+) n (\d+) bank \d+ \(\+\d+\) jump acc ([0-9.]+) mala acc ([0-9.]+) cost (\d+)", log)]
    rows = [r for r in rows if r[0] >= 20]                                                      # refreshes of the training epochs (n = 512)
    n = np.mean([r[1] for r in rows]); acc = np.mean([r[2] for r in rows]); logged = np.mean([r[4] for r in rows])
    comp = dict(
        mala_sweeps=S * 5 * 2 * n,                                   # 5 MALA steps per sweep, 1 energy + 1 gradient each, n points
        jumps=S * (n + acc * n),                                     # 1 energy per proposal + 1 gradient per accepted jump
        step_tuning=500 * 2 + 8 * 5 * 2 * 500,                       # pilot state (E+G on 500 points) + 8 bisection rounds x 5 MALA steps x (E+G) on 500 points
        initial_state=2 * n,                                         # E and gradient at the batch before the sweeps
        minima_and_hessians=M * 50 + M * 2 * d)                      # 50 gradient ascent steps per bank minimum + central differences 2d gradients per minimum
    tot = sum(comp.values())
    q = [float(x) for x in re.findall(r"'quench': (\d+)", log)]; qb = (q[-1] - q[0]) / (len(q) - 1)  # quench cost per refresh (cumulative counter)
    out[c] = dict(n=n, jump_acc=acc, logged_dart_per_batch=logged, computed_dart_per_batch=tot, check_ratio=tot / logged, components=comp, shares={k: v / tot for k, v in comp.items()},
                  quench_per_batch=qb, total_per_batch=logged + qb, per_update=(logged + qb) / 100, ratio_vs_a=(logged + qb) / 512)
    print(f"{c}: logged {logged:.0f} computed {tot:.0f} (x{tot / logged:.3f}) | " + " ".join(f"{k} {v / tot * 100:.1f}%" for k, v in comp.items()) + f" | quench {qb:.0f}/batch ({qb / (logged + qb) * 100:.1f}% of total) | total {logged + qb:.0f} = {(logged + qb) / 512:.0f} x AS per epoch")
json.dump(out, open(ROOT / "figures" / "selftrain_dartbank" / "cost_decomp.json", "w"), indent=1)
