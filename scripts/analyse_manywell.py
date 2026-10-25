""" ManyWell-32 : AS contre Barker, graine 0, métrique FIGÉE avant tout run Barker (écart documenté à la règle initiale) :

      python scripts/analyse_manywell.py --as-commit c4345ea --barker-commit <hash>

    Les coordonnées sont indépendantes ; l'ensemble F des coordonnées où le run AS graine 0 visite les deux puits est gelé ici (lu sur le run AS
    c4345ea : R_i moyen des 30 dernières évaluations > 0, i.e. 1, 2, 3, 4, 5, 7, 9, 15). Les 8 autres (découverte du puits léger, hors de portée du
    variate de contrôle) ne sont que comptées.
    Métrique primaire : mean |log R_i| sur F (R_i moyenné sur les 30 dernières évaluations, plancher 1e-3).
    Rapporté aussi : nombre de coordonnées visitant les deux puits (R_i >= 0.1 et part droite >= 0.1 x exacte), sur les 16 et hors de F, energy_w2.
"""
import argparse

import numpy as np

FROZEN = [1, 2, 3, 4, 5, 7, 9, 15]
P_LEFT, N_WELLS = 0.15569, 16


def run_stats(commit, name):
    import wandb
    for r in wandb.Api(timeout=120).runs("CCDS_NTU/adjoint_sampler_cv", per_page=200, order="-created_at"):
        if r.name != name or r.config.get("seed") != 0 or r.config.get("git_commit", "")[:7] != commit[:7] or r.state != "finished":
            continue
        keys = ["energy_w2"] + [f"well_ratio/{i}" for i in range(N_WELLS)]
        h = r.history(keys=keys, samples=6000, pandas=True, x_axis="_step").dropna(subset=["energy_w2"])
        R = h[keys[1:]].values[-30:].mean(0)
        right = (1 - R * P_LEFT) / (1 - P_LEFT)
        vis = (R >= 0.1) & (right >= 0.1)
        hist = r.history(keys=["adjoint_loss"], samples=6000, pandas=True, x_axis="_step")["adjoint_loss"].dropna().values
        return dict(R=R, vis=vis, ew2=float(h["energy_w2"].values[-30:].mean()), loss_ok=bool(np.isfinite(hist).all()), epochs=int(h["_step"].values[-1]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-commit", required=True)
    ap.add_argument("--barker-commit", required=True)
    a = ap.parse_args()
    runs = {"AS": run_stats(a.as_commit, "manywell_as"), "Barker": run_stats(a.barker_commit, "manywell_as_barker")}
    for k, v in runs.items():
        if v is None:
            print(f"run {k} (graine 0) introuvable ou non terminé")
            return
    print(f"{'':8}{'mean|logR| sur F':>18}{'visitent (16)':>15}{'visitent hors F':>17}{'energy_w2':>11}{'loss finie':>12}")
    out = {}
    for k, v in runs.items():
        out[k] = float(np.mean(np.abs(np.log(np.maximum(v["R"][FROZEN], 1e-3)))))
        rest = [i for i in range(N_WELLS) if i not in FROZEN]
        print(f"{k:8}{out[k]:18.3f}{int(v['vis'].sum()):13d}/16{int(v['vis'][rest].sum()):15d}/8{v['ew2']:11.2f}{str(v['loss_ok']):>12}")
    print(f"\nvariation de mean|logR| sur F : {100 * (out['Barker'] / out['AS'] - 1):+.0f} %")
    for k, v in runs.items():
        print(f"R_i ({k}), coordonnées F :", np.round(v["R"][FROZEN], 2))


if __name__ == "__main__":
    main()
