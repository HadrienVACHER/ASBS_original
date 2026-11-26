# Étape A : robustesse du darting post hoc dans le régime où l'IS s'effondre (autres instantanés) et variance de log W par bassin

Écrit et commité AVANT tout calcul. Aucune règle n'est modifiée après coup. CPU uniquement, aucune location ; chaque exécution < 1 min (chronométrée).

## 1. Darting post hoc sur les autres instantanés (règle A1)
- **Contrôleurs** : GMM-dk (128, 8, 6) et (256, 8, 6), σ_max × 0,75, **époques 750, 1000 et 1250** (`figures/gmmdk_step4/controllers/gmmdk4_d{128,256}_sx0.75_ep{750,1000,1250}.pt`). Configuration : celle du run × 0,5 avec le σ_max × 0,75, comme dans `rules_darting_large_sigma.md`.
- **Protocole identique à `rules_darting_large_sigma.md`** (`scripts/darting_large_sigma.py`) : 3 jeux indépendants de 10⁴ points terminaux par instantané (graines 11, 12, 13) ; darting post hoc (S ∈ {1, 5, 10} calculés) ; MALA seul à coût égal ; IS de chemin par jeu et réuni ; TV par Voronoï et par bassin ; plancher ; comparateur de coût égal TV_eq(S) = min(TV de l'IS réuni, LB(S)).
- **G** (pour cet instantané) : pour au moins un S ∈ {5, 10}, TV_darting(S) ≤ 0,5 × TV_eq(S) ET ≤ 2 × plancher (Voronoï, moyennes sur les 3 jeux).
- **H_saut_bassin** : TV_darting(S = 10) ≤ 0,5 × TV du MALA seul de même coût, par bassin (moyennes sur les 3 jeux).
- **Règle A1 : G et H_saut_bassin vrais sur au moins 2 instantanés sur 3 (750, 1000, 1250), à chaque dimension.** Sinon, le résultat de l'époque 1500 n'est pas robuste. Le résultat de l'époque 1500 (255f24d) est rapporté à côté.
- Variabilité (règle permanente) : moyenne ± écart-type sur les 3 jeux, et les 3 instantanés donnent la variabilité entre évaluations du modèle.

## 2. Variance de log W conditionnellement au bassin (descriptif, sans règle)
Pour la baseline IS tempéré (objection de Jensen), on décompose Var(log W) = E_b[Var(log W | b)] + Var_b(E[log W | b]) : variance intra-bassin et inter-bassins, pondérées par la fréquence des échantillons.
- Runs × 0,75 (d = 128, 256 ; époques 750 à 1500) : log w et bassins (quench) des jeux simulés ci-dessus.
- Runs × 0,5 (d = 128, 256 ; époques 250 à 1500) : log w sauvegardés de l'étape 4 (`figures/gmmdk_step4/logw/`, 20 000 trajectoires). **Le bassin y est remplacé par la cellule de Voronoï** (déclaré) : les points terminaux de l'étape 4 n'ont pas été rapatriés, et à × 0,5 bassins et cellules coïncident presque (TV brut Voronoï / bassin 0,408 / 0,413 à d = 128).
- On rapporte aussi la part inter-bassins et l'ESS normalisé des poids W^α pour α ∈ {0,25 ; 0,5 ; 1}.

## Exécution
`scripts/darting_large_sigma.py` (contrôleurs ajoutés : `d128e750`, `d128e1000`, `d128e1250`, `d256e750`, …) ; `verdict_snap` pour A1 ; `lwvar` pour la décomposition. Résultats dans `figures/darting_large_sigma/`. Section dans RESULTS.md.
