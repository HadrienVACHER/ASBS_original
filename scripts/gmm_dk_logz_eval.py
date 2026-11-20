""" Evaluation of the sigma_max selection criterion of predictions_dk.md, addendum 4 (rule committed in d7170f6, not modified).
    python scripts/gmm_dk_logz_eval.py OUT_JSON TABLE_JSON IS_JSON...   (TABLE_JSON: figures/gmm_dk_window_table.json; IS_JSON: outputs of gmm_dk_analysis.py --is_only with log Z_hat)
    Rule: per cell (d, K, Delta), chosen sigma_max = argmax log Z_hat; every run within 2 SE of the maximum (SE of the difference = sqrt(SE_i^2 + SE_max^2)) is a tie and the tie is broken by the LARGEST ESS
    (TV does not enter). Success: in each cell where a K/K coverage exists, the chosen run has the maximal coverage of the cell. Verdict 'validated' iff all such cells succeed.
    Extra, NO rule: log Z_hat versus log of the target mass of the covered modes (sum of p_k over the modes with p_hat >= 0.25 p; log Z = 0 exactly for this normalised target). """
import json, math, sys
import numpy as np

tab = {t["run"]: t for t in json.load(open(sys.argv[2])) if t["seed"] == 0}
isr = {r["run"]: r for f in sys.argv[3:] for r in json.load(open(f))}
CELLS = [(32, 8, 4), (32, 8, 6), (32, 8, 8), (8, 8, 8)]
rows = []
for run, r in isr.items():
    t = tab[run]
    lost = {int(i) for i, _ in (t["lost"] or [])}
    p = np.array(t["p"], float)
    cov_mass = float(sum(p[k] for k in range(len(p)) if k not in lost))
    rows.append(dict(run=run, cell=(t["d"], t["K"], int(t["delta"])), mult=t["mult"], sigma=t["sigma_max"], coverage=t["coverage"], K=t["K"], logz=r["is_logz"], se=r["is_logz_se"], ess=r["is_ess_frac"], khat=r["is_khat"],
                     tv_as=t["tv"], tv_is=r["is_tv_weighted"], p=p.tolist(), p_is=r["is_p_w"], ratios_as=t["ratios"], cov_mass=cov_mass, log_cov_mass=math.log(cov_mass)))
rows.sort(key=lambda r: (r["cell"], r["mult"]))
print(f"{'cell':>11} {'m':>5} {'sigma':>6} {'cov':>4} {'logZ_hat ± SE':>16} {'ESS':>6} {'khat':>5} {'TV_AS':>6} {'TV_IS':>6} | {'log(covered mass)':>17} {'logZ - log M':>12}")
for r in rows:
    print(f"{str(r['cell']):>11} {r['mult']:5.2f} {r['sigma']:6.3f} {r['coverage']:2d}/{r['K']} {r['logz']:+9.3f} ±{r['se']:5.3f} {r['ess']:6.3f} {r['khat']:5.2f} {r['tv_as']:6.3f} {r['tv_is']:6.3f} | {r['log_cov_mass']:+17.3f} {r['logz'] - r['log_cov_mass']:+12.3f}")
print("\n=== rule (addendum 4) ===")
out = []
for cell in CELLS:
    L = [r for r in rows if r["cell"] == cell]
    top = max(L, key=lambda r: r["logz"])
    ties = [r for r in L if top["logz"] - r["logz"] <= 2 * math.sqrt(r["se"] ** 2 + top["se"] ** 2)]
    chosen = max(ties, key=lambda r: r["ess"])
    best = sorted(L, key=lambda r: (-r["coverage"], r["tv_as"]))[0]
    maxcov = max(r["coverage"] for r in L)
    applies = maxcov == cell[1]
    ok = chosen["coverage"] == maxcov
    print(f"{cell}: argmax logZ = x{top['mult']} ({top['logz']:+.3f} ± {top['se']:.3f}); ties within 2 SE: {[r['mult'] for r in ties]}; chosen (largest ESS) x{chosen['mult']} (sigma {chosen['sigma']:.3f}, coverage {chosen['coverage']}/{cell[1]}, ESS {chosen['ess']:.3f}); "
          f"real best x{best['mult']} (sigma {best['sigma']:.3f}, coverage {best['coverage']}/{cell[1]}, TV {best['tv_as']:.3f}); rule applies (K/K exists): {applies}; success: {ok}")
    out.append(dict(cell=cell, chosen=chosen["run"], best=best["run"], applies=applies, success=ok))
    pw = np.array(best["p_is"]); p = np.array(best["p"])
    print(f"      after path IS at the real best sigma (x{best['mult']}): p_hat_IS/p = {np.round(pw / p, 3).tolist()}  (AS ratios: {best['ratios_as']})")
app = [o for o in out if o["applies"]]
verdict = "critere valide" if all(o["success"] for o in app) else "critere NON valide"
print(f"\nVERDICT: {verdict}  (cells where the rule applies: {len(app)}, successes: {sum(o['success'] for o in app)})")
json.dump(dict(rows=rows, rule=out, verdict=verdict), open(sys.argv[1], "w"), indent=1)
