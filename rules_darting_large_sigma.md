# Darting post hoc dans le régime où l'IS de chemin s'effondre (réouverture n° 2 de l'audit, élargie)

Écrit et commité AVANT tout calcul. Aucune règle n'est modifiée après coup. CPU uniquement, aucune location ; chaque exécution < 1 min (chronométrée), découpée par contrôleur, par jeu et par étape.

## Contrôleurs (déjà sauvegardés, aucun entraînement)
| nom | cible | σ_max | fichier | configuration |
|---|---|---|---|---|
| d128 | GMM-dk (128, 8, 6) | × 0,75 = 1,602001 | `figures/gmmdk_step4/controllers/gmmdk4_d128_sx0.75_ep1500.pt` | celle du run × 0,5 (`figures/gmmdk_step4/configs/gmmdk4_d128_sx0.5.config.yaml`) avec σ_max remplacé par celui du JSON de Q × 0,75 (seule différence entre les deux runs ; déclaré : la configuration × 0,75 n'a pas été rapatriée) |
| d256 | GMM-dk (256, 8, 6) | × 0,75 = 1,551839 | `.../gmmdk4_d256_sx0.75_ep1500.pt` | idem, depuis `gmmdk4_d256_sx0.5.config.yaml` |
| d32 | GMM-dk (32, 8, 4) | × 2 = 4,472136 | `results/gmm_dk_window/w1/gmmdk_d32_K8_D4_s0_sx2/controller_final.pt` (bras AS du test Newton-Stein) | sa `config.yaml` |
Rappel GPU, sans règle : ESS de l'IS de chemin à 20 000 trajectoires 5,8·10⁻⁴ (k̂ 0,97), 7·10⁻⁵ (k̂ 3,3) et 4,7·10⁻³ (k̂ 0,70) ; Q vrai pour les trois.

## Protocole (par contrôleur ; 3 jeux indépendants de 10⁴ points terminaux, graines 11, 12, 13)
- **Simulation sur CPU** : 10⁴ trajectoires par jeu (`girsanov.log_path_weights`, schéma du run, nfe 200) → points terminaux et log w.
- **IS de chemin sur chaque jeu** (n = 10⁴) : simple et PSIS ; TV (Voronoï, et par bassin pondéré), ESS, k̂. **Sur les 3 jeux réunis** (3 × 10⁴) : les mêmes grandeurs.
- **Darting post hoc** (`dart_lib.darting`, protocole inchangé : découverte des minima sur 500 points, quench, Hessiennes, balayages = 1 saut + 5 pas MALA, pas réglé une fois) sur les 10⁴ points de chaque jeu ; S ∈ {1, 5, 10} ; coût mesuré en évaluations d'énergie et de gradient.
- **MALA seul** à coût égal (`dart_lib.mala_only`, même pas, mêmes points) aux coûts de S = 1, 5 et 10.
- **TV par Voronoï** contre les proportions exactes (`gmm_dk.true_proportions`). **TV par bassin** (quench convergée de `basin_lib`, échantillons assignés) contre la même vérité de Voronoï, comme aux étapes 4 et 5 : bassins ≈ cellules pour le GMM-dk, TV(bassin, Voronoï) = 0,004 sous π.
- **Plancher** : TV d'un sampler exact à n = 10⁴, médiane de 100 répétitions (comme à l'étape 4), le même pour Voronoï et bassins.
- **Variabilité** (règle permanente) : pour chaque grandeur, moyenne ± écart-type sur les 3 jeux. Les verdicts portent sur les moyennes, et le nombre de jeux qui satisfont la règle seuls est rapporté.

## Comparateur de coût égal (déviation déclarée, contrainte CPU)
L'IS de coût égal demande n_IS(S) = coût(S) trajectoires (environ 1,8·10⁵ à 1,2·10⁶), hors de portée du CPU en exécutions de moins d'une minute. Il est remplacé par **TV_eq(S) = min(TV_IS,pool, LB(S))**, avec :
- **TV_IS,pool** : TV de l'IS simple sur les 3 × 10⁴ trajectoires réunies. Au coût réel de S, l'IS aurait plus de trajectoires et un TV plus petit, mais d'ordre ≥ LB.
- **LB(S) = plancher(10⁴) × √(10⁴ / (ESS_pool × coût(S)))** : le plancher multinomial au nombre d'échantillons effectifs ESS_pool × coût(S) de l'IS à coût égal (borne d'ordre de grandeur), avec ESS_pool l'ESS normalisé des 3 × 10⁴ trajectoires réunies. Pour des poids à queue lourde, l'ESS mesuré tend à être surestimé, donc LB sous-estimé, ce qui est défavorable au darting.
Prendre le minimum des deux est conservateur pour le darting.

## Règles (inchangées par rapport aux étapes 4 et 5, sauf le comparateur ci-dessus)
- **G** : pour au moins un S ∈ {1, 5, 10}, TV_darting(S) ≤ 0,5 × TV_eq(S) ET TV_darting(S) ≤ 2 × plancher (TV par Voronoï). La version par bassin est rapportée.
- **H_saut_bassin** : TV_darting(S = 10) ≤ 0,5 × TV du MALA seul de même coût, **par bassin**.
- **Lecture fixée d'avance**, par contrôleur :
  - G et H_saut vrais : le darting tient là où l'IS s'effondre (on le garde comme bras d'entraînement dans la famille 2) ;
  - G vrai mais H_saut faux : la correction vient des pas locaux ;
  - G faux : le darting ne tient pas non plus.

## Exécution
`scripts/darting_large_sigma.py` avec les étapes `floor`, `sim C SET`, `dart C SET`, `mala C SET`, `is C`, `verdict` (C ∈ {d32, d128, d256}, SET ∈ {1, 2, 3}). Une simulation trop longue est découpée en morceaux (déclaré au rapport). Points terminaux et log w dans `results/darting_large_sigma/` (hors dépôt), JSON dans `figures/darting_large_sigma/`. Section dans RESULTS.md, avec tableau contrôleur × méthode, verdicts et temps.
