""" Règle PRÉ-ENREGISTRÉE pour ManyWell-32 (FAB), appliquée à AS SEUL (réseau 256x6, graines 0-2) :

      python scripts/manywell_rule.py

    La cible est informative si
        rapport moyen du puits léger  < 0.6      (moyenne sur les 16 coordonnées doubles-puits, ratios moyennés sur les 30 dernières
                                                  évaluations, moyenne sur les 3 graines)
        ET chaque coordonnée visite les deux puits (R_i >= 0.1 et part du puits droit >= 0.1 x sa part exacte, pour les 16 coordonnées
                                                  et pour CHAQUE graine)
    ET (condition de convergence) le rapport moyen du puits léger est sur un plateau sur les 500 dernières epochs : écart entre la moyenne
        de la 1re et de la 2e moitié de la fenêtre <= 0.05, pour CHAQUE graine. Sinon la règle n'est PAS appliquée : on prolonge l'entraînement
        (jamais "informative" parce qu'AS est sous-entraîné).
    Sinon : on passe aux benchmarks du papier AS (DW-4, LJ-13), sans autre recherche de cible. Lit uniquement les runs `manywell_as` du commit
    donné (--commit, hash court du tag W&B git:<hash>) ; les runs a67ed56 (divergés) sont ignorés.

      python scripts/manywell_rule.py --commit <hash>            # règle complète, 3 graines
      python scripts/manywell_rule.py --commit <hash> --check    # run de vérification (graine 0) : stabilité de la loss et plateau, sans verdict
"""
import argparse

import numpy as np

P_LEFT = 0.15569  # masse exacte du puits léger (FAB)
N_WELLS = 16
LIGHT_MAX, VISIT_MIN = 0.6, 0.1
PLATEAU_WINDOW, PLATEAU_TOL = 500, 0.05


def main():
    import wandb
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    assert not args.commit.startswith("a67ed56"), "runs a67ed56 diverged"
    rows = []
    for r in wandb.Api(timeout=120).runs("CCDS_NTU/adjoint_sampler_cv", per_page=200, order="-created_at"):
        if r.name != "manywell_as" or r.config.get("git_commit", "")[:7] != args.commit[:7] or (r.state != "finished" and not args.check):
            continue
        loss = r.history(keys=["adjoint_loss"], samples=6000, pandas=True, x_axis="_step")["adjoint_loss"].dropna().values
        if len(loss) and not np.isfinite(loss).all():
            print(f"graine {r.config.get('seed')}: adjoint_loss NaN/inf : DIVERGÉ")
            continue
        keys = ["energy_w2", "well_logratio_mean", "wells_visited_frac"] + [f"well_ratio/{i}" for i in range(N_WELLS)]
        h = r.history(keys=keys, samples=6000, pandas=True, x_axis="_step").dropna(subset=["energy_w2"])
        steps = h["_step"].values
        curve = h[keys[3:]].values.mean(1)                              # rapport moyen des 16 coordonnées à chaque évaluation
        win = curve[steps > steps[-1] - PLATEAU_WINDOW]
        half = len(win) // 2
        drift = float(win[half:].mean() - win[:half].mean()) if half >= 2 else float("nan")
        Rm = h[keys[3:]].values[-30:].mean(0)
        right = (1 - Rm * P_LEFT) / (1 - P_LEFT)
        rows.append(dict(seed=r.config.get("seed"), R=Rm, mean=float(Rm.mean()), logr=float(np.mean(np.abs(np.log(np.maximum(Rm, 1e-3))))),
                         both=bool((Rm >= VISIT_MIN).all() and (right >= VISIT_MIN).all()), n_vis=int(((Rm >= VISIT_MIN) & (right >= VISIT_MIN)).sum()),
                         ew2=float(h["energy_w2"].values[-30:].mean()), minR=float(Rm.min()), drift=drift, loss_max=float(np.max(loss)) if len(loss) else float('nan'),
                         loss_end=float(np.mean(loss[-50:])), epochs=int(steps[-1])))
    rows.sort(key=lambda d: d["seed"])
    if args.check:
        for d in rows:
            print(f"graine {d['seed']}: epoch {d['epochs']}, adjoint_loss max {d['loss_max']:.3g}, moyenne des 50 dernières {d['loss_end']:.3g}, "
                  f"R moyen {d['mean']:.3f}, dérive sur {PLATEAU_WINDOW} epochs {d['drift']:+.3f} (plateau si |dérive| <= {PLATEAU_TOL})")
        return
    if len(rows) < 3:
        print(f"runs AS manquants ou non terminés ({len(rows)}/3)")
        return
    print(f"{'graine':>6}{'R moyen':>9}{'mean|logR|':>12}{'min R_i':>9}{'coord. visitant 2 puits':>25}{'energy_w2':>11}")
    for d in rows:
        print(f"{d['seed']:6d}{d['mean']:9.3f}{d['logr']:12.3f}{d['minR']:9.3f}{d['n_vis']:>20d}/16{d['ew2']:11.2f}")
    bad = [d["seed"] for d in rows if not abs(d["drift"]) <= PLATEAU_TOL]
    if bad:
        print(f"\nPAS DE PLATEAU (graines {bad} : dérive sur {PLATEAU_WINDOW} epochs {[round(d['drift'], 3) for d in rows]}) : "
              "règle NON appliquée, prolonger l'entraînement.")
        return
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
