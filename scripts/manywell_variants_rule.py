""" Règle PRÉ-ENREGISTRÉE, variantes ManyWell-32 V1 (-x^4 + 6 x^2 + 0.3 x, puits léger 26.6 %) et V2 (-x^4 + 5 x^2 + 0.5 x, 17.9 %), AS SEUL, graine 0 :

      python scripts/manywell_variants_rule.py --commit <hash>

    PASS d'une variante si, sur les 30 dernières évaluations :
        wells_visited_frac = 1.0 (les deux puits visités sur les 16 coordonnées, à CHAQUE évaluation)
        ET mean |log R_i| > 0.3 sur les 16 coordonnées (R_i moyenné sur les 30 dernières évaluations, plancher 1e-3 : erreur de proportion nette)
        ET plateau : dérive du rapport moyen sur les 500 dernières epochs <= 0.05 (écart entre les moyennes des deux moitiés de la fenêtre)
    Deux PASS -> garder V1. Aucun PASS -> ManyWell est fermé pour de bon, aucune autre variante.
    La masse exacte du puits léger de chaque variante vient de la quadrature (ManyWell.light_mass), pas d'une constante.
    Rapporté : wells_visited_frac, R_i par coordonnée, mean |log R_i|, dérive, stabilité de adjoint_loss, R_i à l'epoch 50 (transitoire).
"""
import argparse

import numpy as np

VARIANTS = {"V1": ("manywell_v1_as", 6.0, 0.3), "V2": ("manywell_v2_as", 5.0, 0.5)}
LOGR_MIN, DRIFT_MAX, WINDOW, N = 0.3, 0.05, 500, 16


def evaluate(runs, label, name, qa, lb, commit):
    """ prints the report of one run and returns (PASS, R at the end, R at epoch 50); None if the run is missing or unfinished """
    from adjoint_samplers.utils.dist_utils import ManyWell
    cols = [f"well_ratio/{i}" for i in range(N)]
    p_left = ManyWell(dim=2 * N, a=-lb, b=-qa).light_mass
    sel = [r for r in runs if r.name == name and r.config.get("seed") == 0 and r.config.get("git_commit", "")[:7] == commit[:7]]
    if not sel or sel[0].state != "finished":
        print(f"{label}: run introuvable ou non terminé")
        return None
    r = sel[0]
    h = r.history(keys=cols + ["energy_w2", "wells_visited_frac"], samples=6000, pandas=True, x_axis="_step").dropna(subset=["energy_w2"]).set_index("_step")
    loss = r.history(keys=["adjoint_loss"], samples=6000, pandas=True, x_axis="_step")["adjoint_loss"].dropna()
    R = h[cols].values
    Rm = R[-30:].mean(0)
    steps = h.index.values
    win = R.mean(1)[steps > steps[-1] - WINDOW]
    half = len(win) // 2
    drift = float(win[half:].mean() - win[:half].mean())
    logr = float(np.mean(np.abs(np.log(np.maximum(Rm, 1e-3)))))
    vis = h["wells_visited_frac"].values[-30:]
    finite = bool(np.isfinite(loss.values).all())
    late = loss[loss.index >= 200] if len(loss) else loss
    ok = bool(vis.min() == 1.0 and logr > LOGR_MIN and abs(drift) <= DRIFT_MAX and finite)
    print(f"\n== {label} ({name}, light well {100 * p_left:.1f} % par la quadrature) : {'PASS' if ok else 'FAIL'}")
    print(f"  wells_visited_frac (30 dernières) : min {vis.min():.3f}, moyenne {vis.mean():.3f}   (exigé 1.0)")
    print(f"  mean |log R_i| (16 coordonnées)   : {logr:.3f}   (exigé > {LOGR_MIN})")
    print(f"  dérive du rapport moyen / {WINDOW} epochs : {drift:+.3f}   (exigé |.| <= {DRIFT_MAX})")
    print(f"  adjoint_loss : finie {finite}, max {loss.max():.3g}, moyenne des 50 dernières {loss.values[-50:].mean():.3g}, max après l'epoch 200 {late.max() if len(late) else float('nan'):.3g}")
    print("  R_i (30 dernières évaluations)    :", np.round(Rm, 2))
    r50 = h.loc[50, cols].values.astype(float) if 50 in h.index else None
    if r50 is not None:
        print("  R_i à l'epoch 50 (transitoire)    :", np.round(r50, 2))
    print(f"  energy_w2 (30 dernières) : {h['energy_w2'].values[-30:].mean():.2f}")
    return ok, Rm, r50


def main():
    import wandb
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    args = ap.parse_args()
    runs = list(wandb.Api(timeout=120).runs("CCDS_NTU/adjoint_sampler_cv", per_page=200, order="-created_at"))
    passed = {}
    for v, (name, qa, lb) in VARIANTS.items():
        out = evaluate(runs, v, name, qa, lb, args.commit)
        if out is None:
            return
        passed[v] = out[0]
    print()
    if all(passed.values()):
        print("=> V1 et V2 PASS : garder V1 (barrière la plus proche de la cible standard)")
    elif any(passed.values()):
        print("=> garder", [v for v, ok in passed.items() if ok][0])
    else:
        print("=> aucune variante ne passe : ManyWell est fermé pour de bon, aucune autre variante")


if __name__ == "__main__":
    main()
