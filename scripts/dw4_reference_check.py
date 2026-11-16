""" Analysis of the parallel-tempering run (scripts/dw4_pt.py): validity (i)-(iii), p_PT with SE, verdict on the reference, and Task 2a (energy above the minimum per basin).
    Rules: rules_reference_check.md (Tasks 1 and 2), committed before the run.   python scripts/dw4_reference_check.py [--pt results/pt_samples_T1.pt]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dw4_division_b as b  # noqa: E402
import dw4_division_step0 as s0  # noqa: E402
from adjoint_samplers.energies.double_well_energy import DoubleWellEnergy  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P_IS = {"triangle + pendant": (0.4118, 0.0081), "path of 2 (3 particles)": (0.4683, 0.0087)}      # path-space IS of the controller 94252 (RESULTS.md)


def rhat(x):
    """ split Gelman-Rubin of x (m chains, n draws): chains cut in two halves; nan if the within variance is 0 """
    m, n = x.shape
    h = n // 2
    y = np.concatenate([x[:, :h], x[:, h:2 * h]], 0)
    W = y.var(1, ddof=1).mean()
    Bv = h * y.mean(1).var(ddof=1)
    if W == 0:
        return float("nan")
    return float(np.sqrt(((h - 1) / h * W + Bv / h) / W))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=str(ROOT / "results" / "pt_samples_T1.pt"))
    a = ap.parse_args()
    ref, q, lab_r, types, mu8 = b.load_reference()
    names = [NAMES[KEYS[k]] for k in types]
    n_ref = len(ref)
    p_ref = np.array([float((lab_r == k).double().mean()) for k in types])
    d = torch.load(a.pt, map_location="cpu", weights_only=False)
    ids = d["basin_state_id"].numpy()                                                              # (64, n)
    Es = d["E"].numpy()
    grp = d["group"].numpy()
    C, n = ids.shape
    ind = np.stack([(ids == k) for k in types], 0).astype(float)                                   # (5, 64, n)
    print(f"PT: {C} chains x {n} saved states at T = 1 ({d['steps']} steps, burn-in {d['burn']}, {d['runtime_s']:.0f} s); quench converged {100 * d['quench_converged_frac']:.2f} %")
    print(f"reference ({n_ref}): p_ref = " + ", ".join(f"{nm} {p:.4f}" for nm, p in zip(names, p_ref)))

    # (i) R-hat
    print("\n(i) R-hat of the basin indicators: (a) 64 chains  (b) group A (32, reference start)  group B (32, AS start)  |  group means (A, B), diff / SE_diff")
    ok_i = True
    for i, nm in enumerate(names):
        ra, rA, rB = rhat(ind[i]), rhat(ind[i][grp == 0]), rhat(ind[i][grp == 1])
        mA, mB = ind[i][grp == 0].mean(1), ind[i][grp == 1].mean(1)
        se = math.sqrt(mA.var(ddof=1) / 32 + mB.var(ddof=1) / 32)
        zd = (mA.mean() - mB.mean()) / se if se > 0 else (0.0 if mA.mean() == mB.mean() else float("inf"))
        good = all(np.isfinite(v) and v < 1.05 for v in (ra, rA, rB)) and abs(zd) < 3
        ok_i &= good
        print(f"   {nm:26} (a) {ra:6.3f}  (b) {rA:6.3f} {rB:6.3f}  | {mA.mean():.4f} {mB.mean():.4f}  z_diff {zd:+.2f}   {'ok' if good else 'FAIL'}")
    # (ii) transitions
    t, p = types[0], types[1]
    lab_t = ids
    trans = int((((lab_t[:, :-1] == t) & (lab_t[:, 1:] == p)) | ((lab_t[:, :-1] == p) & (lab_t[:, 1:] == t))).sum())
    ok_ii = trans >= 500
    # (iii) KS
    ks = ks_2samp(Es[grp == 0].ravel(), Es[grp == 1].ravel())
    ok_iii = ks.pvalue > 0.01
    print(f"(ii) transitions triangle+pendant <-> path of 2 between consecutive saved T=1 states: {trans} (>= 500: {ok_ii})")
    print(f"(iii) KS on the T=1 energies between the two initial groups: statistic {ks.statistic:.4f}, p = {ks.pvalue:.3g} (> 0.01: {ok_iii})")
    valid = ok_i and ok_ii and ok_iii
    print(f"VALIDITY: (i) {ok_i}  (ii) {ok_ii}  (iii) {ok_iii}  =>  {'VALID' if valid else 'NOT VALID (reported without concluding)'}")

    # estimation
    cm = ind.mean(2)                                                                                # (5, 64) chain means
    p_pt = cm.mean(1)
    se_pt = cm.std(1, ddof=1) / math.sqrt(C)
    print("\np_PT (chain batch means, SE over 64 chains) vs p_ref:")
    for i, nm in enumerate(names):
        print(f"   {nm:26} p_PT {p_pt[i]:.4f} +- {se_pt[i]:.4f}   p_ref {p_ref[i]:.4f}   log(p_PT/p_ref) {math.log(p_pt[i] / p_ref[i]):+.3f}")
    json.dump(dict(names=names, p_pt=p_pt.tolist(), se_pt=se_pt.tolist(), p_ref=p_ref.tolist(), n_ref=n_ref, valid=bool(valid)), open(ROOT / "results" / "reference_check_pt.json", "w"), indent=1)
    if not valid:
        print("\nthe run is not valid: no verdict, no Task 2a / 3 / 4 (propose a longer setting)")
        return

    # verdict
    print("\nVERDICT on the two dominant basins (|log| in units of SE_comb):")
    flags_b, flags_c = [], []
    for i in (0, 1):
        nm = names[i]
        l_pt, s_pt = math.log(p_pt[i]), se_pt[i] / p_pt[i]
        s_ref = math.sqrt((1 - p_ref[i]) / (p_ref[i] * n_ref))
        p_is, se_is = P_IS[nm]
        z_ref = (l_pt - math.log(p_ref[i])) / math.hypot(s_pt, s_ref)
        z_is = (l_pt - math.log(p_is)) / math.hypot(s_pt, se_is / p_is)
        flags_b.append(abs(z_ref) > 3 and abs(z_is) < 3)
        flags_c.append(abs(z_ref) < 3 and abs(z_is) > 3)
        print(f"   {nm:26} z(PT vs ref) {z_ref:+.1f}   z(PT vs path IS) {z_is:+.1f}   (p_IS {p_is:.4f} +- {se_is:.4f})")
    verdict = "Référence biaisée" if all(flags_b) else "Référence correcte" if all(flags_c) else "non conclusif"
    print(f"   => {verdict}")

    # Task 2a
    en = DoubleWellEnergy(dim=8, n_particles=4)
    Eref = en.eval(ref).numpy()
    Eq = en.eval(q.double()).numpy()
    print("\nTask 2a: mean of E - E_min(basin) on the reference and on PT (T = 1)")
    zs, diffs = [], []
    for i, k in enumerate(types):
        selr = (lab_r == k).numpy()
        Emin = Eq[selr].min()
        dr = Eref[selr] - Emin
        mr, sr = dr.mean(), dr.std(ddof=1) / math.sqrt(len(dr))
        chain_means = np.array([(Es[c][ids[c] == k] - Emin).mean() for c in range(C) if (ids[c] == k).sum() > 0])
        mp, sp = chain_means.mean(), chain_means.std(ddof=1) / math.sqrt(len(chain_means))
        z = (mr - mp) / math.hypot(sr, sp)
        zs.append(z)
        diffs.append(mr - mp)
        print(f"   {names[i]:26} E_min {Emin:8.3f}  ref {mr:6.3f} +- {sr:.3f} (n {len(dr)})  PT {mp:6.3f} +- {sp:.3f} ({len(chain_means)} chains)  diff {mr - mp:+.3f}  z {z:+.1f}  ratio ref/PT {mr / mp:.3f}")
    same_sign = max(sum(1 for z in zs if z > 3), sum(1 for z in zs if z < -3))
    rule = "énergie ou température différente" if same_sign >= 4 else "énergies cohérentes, défaut de mélange de la référence" if all(abs(z) < 3 for z in zs) else "non conclusif"
    print(f"   same-sign |z| > 3 in {same_sign}/5 basins; all |z| < 3: {all(abs(z) < 3 for z in zs)}  =>  {rule}")


if __name__ == "__main__":
    main()
