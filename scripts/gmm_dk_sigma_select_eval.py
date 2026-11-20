""" Evaluation of rules_sigma_select.md (log Z_hat criterion with a khat < 0.7 filter, validation on seed 1). Report only, rule not modified.
    python scripts/gmm_dk_sigma_select_eval.py OUT_JSON SEED ANALYSIS_JSON...   (outputs of gmm_dk_analysis.py with IS: coverage, tv, is_logz, is_logz_se, is_ess_frac, is_khat, is_tv_weighted, is_p_w) """
import json, math, re, sys
import numpy as np

seed = int(sys.argv[2])
rows = {}
for f in sys.argv[3:]:
    for r in json.load(open(f)):
        rows.setdefault(r["run"], {}).update(r)
R = []
for run, r in rows.items():
    m = re.match(r"gmmdk_d(\d+)_K(\d+)_D([\d.]+)_s(\d+)(?:_sx([\d.]+))?$", run)
    if int(m[4]) != seed or "is_logz" not in r or "coverage" not in r: continue
    d, K, D = int(m[1]), int(m[2]), int(float(m[3])); mult = float(m[5]) if m[5] else 1.0
    p = np.array(r["p"], float); lost = {int(i) for i, _ in (r["lost"] or [])}
    cm = float(sum(p[k] for k in range(K) if k not in lost))
    R.append(dict(run=run, cell=(d, K, D), mult=mult, sigma=mult * 2 * math.sqrt(1 + D ** 2 / (2 * d)), coverage=r["coverage"], K=K, logz=r["is_logz"], se=r["is_logz_se"], ess=r["is_ess_frac"], khat=r["is_khat"],
                  tv_as=r["tv"], tv_is=r["is_tv_weighted"], floor=r["floor_median"], p=p.tolist(), p_is=r["is_p_w"], ratios_as=r["ratios"], log_cov_mass=math.log(cm)))
R.sort(key=lambda r: (r["cell"], r["mult"]))
print(f"seed {seed}: {len(R)} runs")
print(f"{'cell':>11} {'m':>5} {'sigma':>6} {'cov':>4} {'logZ ± SE':>15} {'ESS':>6} {'khat':>5} {'TV_AS':>6} {'TV_IS':>6} | {'logZ-logM':>9}")
for r in R:
    print(f"{str(r['cell']):>11} {r['mult']:5.2f} {r['sigma']:6.3f} {r['coverage']:2d}/{r['K']} {r['logz']:+8.3f} ±{r['se']:5.3f} {r['ess']:6.3f} {r['khat']:5.2f} {r['tv_as']:6.3f} {r['tv_is']:6.3f} | {r['logz'] - r['log_cov_mass']:+9.3f}")

def pick(L, use_khat):
    S = [r for r in L if r["khat"] < 0.7] if use_khat else L
    if not S:
        return max(L, key=lambda r: r["ess"]), "no khat < 0.7: max ESS (flag)"
    top = max(S, key=lambda r: r["logz"])
    ties = [r for r in S if top["logz"] - r["logz"] <= 2 * math.sqrt(r["se"] ** 2 + top["se"] ** 2)]
    return max(ties, key=lambda r: r["ess"]), f"{len(S)} runs with khat<0.7, ties {[t['mult'] for t in ties]}"
print("\n=== rule (rules_sigma_select.md) ===")
out = []
for cell in sorted({r["cell"] for r in R}):
    L = [r for r in R if r["cell"] == cell]; K = cell[1]
    ch, note = pick(L, True); orig, _ = pick(L, False); mxe = max(L, key=lambda r: r["ess"])
    maxcov = max(r["coverage"] for r in L); applies = maxcov == K
    ok = ch["coverage"] == K and ch["tv_is"] < 2 * ch["floor"] if applies else None
    print(f"{cell}: chosen x{ch['mult']} (sigma {ch['sigma']:.3f}, cov {ch['coverage']}/{K}, TV_IS {ch['tv_is']:.3f} vs 2*floor {2 * ch['floor']:.3f}, ESS {ch['ess']:.3f}, khat {ch['khat']:.2f}) [{note}]; "
          f"rule applies (8/8 exists): {applies}; success: {ok}")
    print(f"      original rule (d7170f6, no khat filter) would choose x{orig['mult']} (cov {orig['coverage']}/{K}); max ESS alone would choose x{mxe['mult']} (cov {mxe['coverage']}/{K})")
    print(f"      ratios p_hat/p after IS at the chosen sigma: {np.round(np.array(ch['p_is']) / np.array(ch['p']), 3).tolist()}  (AS ratios {ch['ratios_as']})")
    out.append(dict(cell=cell, chosen=ch["run"], original=orig["run"], max_ess=mxe["run"], applies=applies, success=ok))
app = [o for o in out if o["applies"]]
verdict = "no verdict (no cell with an 8/8 run)" if not app else ("critere valide" if all(o["success"] for o in app) else "critere NON valide")
print(f"\nVERDICT: {verdict}  (cells where the rule applies: {len(app)}, successes: {sum(bool(o['success']) for o in app)})")
json.dump(dict(rows=R, rule=out, verdict=verdict), open(sys.argv[1], "w"), indent=1)
