""" Règle de choix de la géométrie de la cible hétérogène, FIXÉE D'AVANCE et appliquée à AS SEUL (anti-surajustement).

      python scripts/hetero_geometry_rule.py

    Lit uniquement les runs W&B `hetero24d_geom{G}_as_K24_s{seed}` (AS, réseau 256x6, 3 graines). Aucun run Barker n'est ouvert.
    Géométries dans l'ordre G1, G2, G3, G4 (scripts/hetero_geometry_as.pbs) puis E1, E2, E3, E4 (hetero_geometry_as_ext.pbs), puis F1 (hetero_geometry_as_last.pbs). Une géométrie est retenue (la première de
    l'ordre) si, sur les 3 graines (ratios moyennés sur les 30 dernières évaluations) :
        quart léger moyen < 0.6     (AS a un vrai déficit sur les modes légers)
        modes_covered moyen >= 0.9  (AS trouve les modes : sinon c'est un effondrement, pas un déficit de poids)
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils.dist_utils import HeteroGMM  # noqa: E402

D = 24
COMMON = dict(scale_min=0.6, scale_max=1.4, min_sep=2.5)
GEOMS = {  # identique à GEOM dans hetero_geometry_as.pbs
    "G1": dict(radius_min=4.5, radius_max=6.5, weight_ratio=20, **COMMON),
    "G2": dict(radius_min=4.5, radius_max=6.5, weight_ratio=100, **COMMON),
    "G3": dict(radius_min=5.5, radius_max=8.0, weight_ratio=20, **COMMON),
    "G4": dict(radius_min=5.5, radius_max=8.0, weight_ratio=100, **COMMON),
}
# EXTENSION (décidée après l'échec de G1-G4 : AS ne trouve que 8-18 % des modes, séparation normalisée médiane 3.9 à 6.6 contre 3.05 dans
# le GMM isotrope où AS trouve tous les modes). Géométries plus serrées, séparation normalisée médiane 3.45 (rayons 3.2-4.2) et 3.75
# (rayons 3.8-5.0), minimum 3.0 ; construites sans aucun résultat Barker. Parcourues APRÈS G1-G4, dans l'ordre E1, E2, E3, E4.
COMMON_E = dict(scale_min=0.6, scale_max=1.1, min_sep=3.0)
GEOMS.update({  # identique à GEOM dans hetero_geometry_as_ext.pbs
    "E1": dict(radius_min=3.2, radius_max=4.2, weight_ratio=20, **COMMON_E),
    "E2": dict(radius_min=3.2, radius_max=4.2, weight_ratio=100, **COMMON_E),
    "E3": dict(radius_min=3.8, radius_max=5.0, weight_ratio=20, **COMMON_E),
    "E4": dict(radius_min=3.8, radius_max=5.0, weight_ratio=100, **COMMON_E),
})
# DERNIER CANDIDAT, fixé d'avance (règle d'arrêt : si E1-E4 échouent, F1 est le seul essai restant ; s'il échoue aussi, la cible hétérogène est
# abandonnée pour le Many-Well asymétrique). Positions ET poids du GMM isotrope (simplexe de rayon 4.25, où AS trouve tous les modes), seules les
# covariances sont hétérogènes (échelles 1.0-1.4, orientations aléatoires) : séparation normalisée <= 3.05.
GEOMS["F1"] = dict(layout="simplex", simplex_radius=4.25, radius_min=4.25, radius_max=4.25, scale_min=1.0, scale_max=1.4, weight_ratio=20, min_sep=0.0)
LIGHT_MAX, COVER_MIN = 0.6, 0.9


def main():
    import wandb
    runs = list(wandb.Api(timeout=120).runs("CCDS_NTU/adjoint_sampler_cv", per_page=300, order="-created_at"))
    chosen = None
    print(f"{'géométrie':10s}{'n':>3}{'quart léger':>13}{'modes_covered':>15}{'|log R| (>0)':>14}{'min R':>8}   verdict")
    for name, kw in GEOMS.items():
        g = HeteroGMM(dim=D, K=D, seed=0, **kw)
        w = g.weights.numpy().astype(float); w /= w.sum()
        light = np.argsort(w)[:D // 4]
        rows = {}
        for r in runs:
            m = re.match(rf"^hetero24d_geom{name}_as_K{D}_s(\d)(?:_random|_simplex)?$", r.name)   # AS seul (le Writer ajoute le layout au nom)
            if m and r.state == "finished" and int(m.group(1)) not in rows:
                keys = ["modes_covered"] + [f"mode_ratio/{j}" for j in range(D)]
                h = r.history(keys=keys, samples=6000, pandas=True, x_axis="_step").dropna(subset=["modes_covered"])
                Rm = h[keys[1:]].values[-30:].mean(0)
                rows[int(m.group(1))] = (float(Rm[light].mean()), float(h["modes_covered"].values[-30:].mean()),
                                         float(np.mean(np.abs(np.log(np.maximum(Rm, 1e-3))))), float(Rm.min()))
        if len(rows) < 3:
            print(f"{name:10s}{len(rows):3d}   (runs AS manquants ou non terminés)")
            continue
        v = np.array(list(rows.values()))
        lq, cov = v[:, 0].mean(), v[:, 1].mean()
        ok = lq < LIGHT_MAX and cov >= COVER_MIN
        verdict = "RETENUE" if ok and chosen is None else ("ok mais une géométrie plus facile est déjà retenue" if ok else
                  ("trop difficile : AS ne trouve pas les modes" if cov < COVER_MIN else "trop facile : AS n'a pas de déficit net"))
        if ok and chosen is None:
            chosen = name
        print(f"{name:10s}{len(rows):3d}{lq:13.3f}{cov:15.3f}{v[:, 2].mean():14.3f}{v[:, 3].mean():8.3f}   {verdict}")
    print("\n=> géométrie retenue:", chosen if chosen else "aucune (étendre la liste, toujours sur AS seul)")
    if chosen:
        print("   paramètres:", {k: v for k, v in GEOMS[chosen].items()})


if __name__ == "__main__":
    main()
