""" Run de la variable de contrôle par DIVISION sur le GMM isotrope d = K = 36, graine 0 : CRITÈRES PRÉ-ENREGISTRÉS (engagés dans git avant la soumission).

      PYTHONPATH=. python scripts/analyse_division_gmm.py --commit <hash du run>

    Métrique primaire = celle des runs Barker finaux : mean_j |log R̄_j|, R̄_j = rapport de masse du mode j moyenné sur les 30 DERNIÈRES évaluations (plancher 1e-6), 2000 epochs, évaluation toutes
    les 50 epochs, 10 000 échantillons. Références de la même graine (valeurs vérifiées sur W&B le 2026-10-27) :
        AS big (gmm36d_as_big_cap, 203f59d)                 : 0.38949   (quart léger 0.4049, energy_w2 0.552)
        Barker gamma = 3 big (gmm36d_as_barker_g3_thr_big, 3f672a7) : 0.11660   (quart léger 0.7746, energy_w2 0.720)
    SUCCÈS     <=>  métrique <= 0.5 x Barker (= 0.05830) ET en dessous de AS ;
    « ÉGALE BARKER »  <=>  métrique dans ]0.5, 1] x Barker (0.05830 < m <= 0.11660) ;
    ÉCHEC      sinon (m > 0.11660 ou non finie).
    Rapportés (hors critère) : quart léger, min R̄, energy_w2, et par version des paramètres : nombre de clusters, softmax(-L) contre les poids vrais, couverture ; par tranche de kappa
    ([0, 0.1), [0.1, 0.7), [0.7, 1]) : ||gamma terme|| / ||grad g||, ESS des poids des échantillons et fraction du poids maximal (surveillance de la variance).
    Attente (commit) : succès probable, car L_k est exact sur ce GMM ; le run valide l'implémentation de bout en bout, pas la généralité de la méthode.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils.dist_utils import GMM  # noqa: E402

D = 36
AS_REF, BARKER_REF = 0.38949, 0.11660
SUCCESS_MAX = 0.5 * BARKER_REF
SHOW = ["K_clusters", "K_matched_distinct", "w_relerr_max_ge1pct", "w_relerr_median", "xi_mean", "xi_std", "xi_in_0.5_2", "softmax_L_sum", "centre_err_mean", "version", "warmup_frac"]


def cell(runs, prefix, commit=None):
    for r in runs:
        if r.name.startswith(prefix) and r.state == "finished" and (commit is None or r.config.get("git_commit", "")[:7] == commit[:7]):
            return r
    return None


def stats(run, light):
    keys = ["energy_w2"] + [f"mode_ratio/{j}" for j in range(D)]
    h = run.history(keys=keys, samples=6000, pandas=True, x_axis="_step").dropna(subset=["energy_w2"])
    Rm = h[keys[1:]].values[-30:].mean(0)
    return dict(m=float(np.mean(np.abs(np.log(np.maximum(Rm, 1e-6))))), light=float(Rm[light].mean()), minR=float(Rm.min()), ew2=float(h["energy_w2"].values[-30:].mean()), n=len(h), last=int(h["_step"].values[-1]), R=Rm)


def main():
    import wandb
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--project", default="CCDS_NTU/div", help="W&B project of the division run")
    ap.add_argument("--ref-project", default="CCDS_NTU/adjoint_sampler_cv", help="W&B project of the AS / Barker reference cells")
    a = ap.parse_args()
    api = wandb.Api(timeout=120)
    runs = list(api.runs(a.project, per_page=300, order="-created_at"))
    ref_runs = list(api.runs(a.ref_project, per_page=300, order="-created_at"))
    g = GMM(dim=D, K=D, seed=0, layout="spread", weight_law="loguniform", weight_ratio=20.0, radius=4.25)
    w = g.weights.numpy()
    light = np.argsort(w)[: D // 4]
    r = cell(runs, "gmm36d_as_division_K36_s0_spread", a.commit)
    if r is None:
        print("run division introuvable ou non terminé")
        return
    s = stats(r, light)
    print(f"DIVISION (gamma = 1 après échauffement), graine 0, d = K = 36 : mean|log R̄| = {s['m']:.5f}   quart léger {s['light']:.4f}   min R̄ {s['minR']:.4f}   energy_w2 {s['ew2']:.3f}   ({s['n']} évaluations, dernière epoch {s['last']})")
    for nm, pref in (("AS big", "gmm36d_as_big_cap_K36_s0_spread"), ("Barker gamma = 3 big", "gmm36d_as_barker_g3_thr_big_K36_s0_spread")):
        c = cell(ref_runs, pref)
        if c is not None:
            t = stats(c, light)
            print(f"  {nm:22} mean|log R̄| = {t['m']:.5f}   quart léger {t['light']:.4f}   min R̄ {t['minR']:.4f}   energy_w2 {t['ew2']:.3f}")
    m = s["m"]
    if not np.isfinite(m):
        verdict = "ÉCHEC (métrique non finie)"
    elif m <= SUCCESS_MAX and m < AS_REF:
        verdict = "SUCCÈS (<= 0.5 x Barker et sous AS)"
    elif SUCCESS_MAX < m <= BARKER_REF:
        verdict = "ÉGALE BARKER (dans ]0.5, 1] x Barker)"
    else:
        verdict = "ÉCHEC (> Barker)"
    print(f"\nseuils : succès <= {SUCCESS_MAX:.5f} ; égale Barker <= {BARKER_REF:.5f} ; AS {AS_REF:.5f}  =>  VERDICT : {verdict}")

    h = r.history(samples=6000, x_axis="_step")
    cols = [c for c in h.columns if c.startswith("div/")]
    if cols:
        hd = h[["_step"] + cols].dropna(how="all", subset=cols).set_index("_step")
        print("\nlogs par version / par tranche de kappa (premier, milieu, dernier point loggé) :")
        for c in sorted(cols):
            v = hd[c].dropna()
            if len(v):
                print(f"  {c:42} " + "  ".join(f"{float(v.iloc[i]):.4g}" for i in (0, len(v) // 2, len(v) - 1)))
        print("  (clés : div/<stat> par version ; div/kappa[lo,hi]/{frac_t, term_over_gradg, ess_frac, max_weight_frac} par tranche)")


if __name__ == "__main__":
    main()
