""" Règle PRÉ-ENREGISTRÉE pour ManyWell-32 (FAB), appliquée à AS SEUL (réseau 256x6, graines 0-2) :

      python scripts/manywell_rule.py

    La cible est informative si
        rapport moyen du puits léger  < 0.6      (moyenne sur les 16 coordonnées doubles-puits, ratios moyennés sur les 30 dernières
                                                  évaluations, moyenne sur les 3 graines)
        ET chaque coordonnée visite les deux puits (R_i >= 0.1 et part du puits droit >= 0.1 x sa part exacte, pour les 16 coordonnées
                                                  et pour CHAQUE graine)
    Sinon : on passe aux benchmarks du papier AS (DW-4, LJ-13), sans autre recherche de cible. Lit uniquement les runs `manywell_as`.
"""
import re

import numpy as np

P_LEFT = 0.15569  # masse exacte du puits léger (FAB)
N_WELLS = 16
LIGHT_MAX, VISIT_MIN = 0.6, 0.1


def main():
    import wandb
    rows = []
    for r in wandb.Api(timeout=120).runs("CCDS_NTU/adjoint_sampler_cv", per_page=200, order="-created_at"):
        if r.name != "manywell_as" or r.state != "finished":
            continue
        keys = ["energy_w2", "well_logratio_mean", "wells_visited_frac"] + [f"well_ratio/{i}" for i in range(N_WELLS)]
        h = r.history(keys=keys, samples=6000, pandas=True, x_axis="_step").dropna(subset=["energy_w2"])
        Rm = h[keys[3:]].values[-30:].mean(0)
        right = (1 - Rm * P_LEFT) / (1 - P_LEFT)
        rows.append(dict(seed=r.config.get("seed"), R=Rm, mean=float(Rm.mean()), logr=float(np.mean(np.abs(np.log(np.maximum(Rm, 1e-3))))),
                         both=bool((Rm >= VISIT_MIN).all() and (right >= VISIT_MIN).all()), n_vis=int(((Rm >= VISIT_MIN) & (right >= VISIT_MIN)).sum()),
                         ew2=float(h["energy_w2"].values[-30:].mean()), minR=float(Rm.min())))
    rows.sort(key=lambda d: d["seed"])
    if len(rows) < 3:
        print(f"runs AS manquants ou non terminés ({len(rows)}/3)")
        return
    print(f"{'graine':>6}{'R moyen':>9}{'mean|logR|':>12}{'min R_i':>9}{'coord. visitant 2 puits':>25}{'energy_w2':>11}")
    for d in rows:
        print(f"{d['seed']:6d}{d['mean']:9.3f}{d['logr']:12.3f}{d['minR']:9.3f}{d['n_vis']:>20d}/16{d['ew2']:11.2f}")
    m = float(np.mean([d["mean"] for d in rows]))
    all_vis = all(d["both"] for d in rows)
    print(f"\nrapport moyen du puits léger (3 graines) = {m:.3f}  (seuil < {LIGHT_MAX}) ; toutes les coordonnées visitent les 2 puits à chaque graine : {all_vis}")
    informative = m < LIGHT_MAX and all_vis
    print("=> cible INFORMATIVE : lancer AS contre Barker (orbite de sous-groupe, s = 3, gamma = 3)" if informative else
          "=> cible NON informative : passer aux benchmarks du papier AS (DW-4, LJ-13), sans autre recherche de cible")
    if not informative:
        why = []
        if m >= LIGHT_MAX:
            why.append(f"rapport moyen {m:.3f} >= {LIGHT_MAX} (pas de déficit net)")
        if not all_vis:
            why.append("au moins une coordonnée ne visite pas les deux puits (AS ne découvre pas tous les puits)")
        print("   raison:", "; ".join(why))


if __name__ == "__main__":
    main()
