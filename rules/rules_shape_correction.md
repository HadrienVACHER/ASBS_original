# Test « AS corrige les formes d'un buffer aux bonnes masses » à (128, 8, 6) × 0,75

Écrit pour : Hadrien. Écrit et commité **avant** la location et avant tout résultat de ce lot ; aucune règle ne sera modifiée après coup (amendements datés seulement). **Métrique de masse principale : TV par bassin** ; Voronoï en secondaire.

## Question
Dans le régime |A_m| < 1 (σ_tar² / ν₁ = 0,390, A_m = −0,54 ; RESULTS.md), la mise à jour d'AS corrige-t-elle les erreurs de **forme** (largeur, position moyenne) d'un buffer dont les **masses** sont exactes, comme le prédit la mise à jour de population, alors que le DSM les recopie ?

## Montage
- Cible GMM-dk (128, 8, 6), σ_max = 1,602001 (× 0,75), cellule, architecture (FourierMLP 4 × 128), départ (`results/wf_start/d128/controller_start.pt`), optimiseur, 500 époques, nfe 200, graine 0, évaluation toutes les 50 époques sur 10⁴ échantillons du sampler appris : **identiques à l'oracle** (`rules_oracle_buffer.md`).
- **Buffers synthétiques** : 10⁴ tirages i.i.d. au départ (le buffer de départ du fichier est ignoré), puis **512 tirages neufs i.i.d. à chaque époque**, **masses exactes** (composante tirée avec les vrais poids), composantes à covariance unité sauf :
  - **(V)** covariance **1,44 × I** (la vraie fois 1,44), moyennes exactes ;
  - **(M)** covariance I, moyenne de chaque mode **décalée de δ = 0,3 σ_tar (σ_tar = 1) dans une direction unitaire aléatoire u_k fixée par mode** (`gmm_dk.shape_directions`, graine 2027, la même pour le tirage du buffer et la mesure).
- **Deux apprenants par buffer** (4 runs) : **AS** (`AdjointVEShapeASMatcher` : régression d'AS, adjoint = gradient de la VRAIE énergie aux points du buffer) et **DSM** (`AdjointVEShapeDSMMatcher` : même DSM que `rules_oracle_buffer.md`, sans gradient d'énergie). `tests/test_shape_matcher.py` réussi (variance 1,44 et décalage 1,0 aux mesures sur les buffers, 1 et 0 sur tirages exacts ; adjoint = gradient de la vraie énergie ; tirages neufs à chaque appel).

## Mesures (moyenne des évaluations des époques 350, 400, 450 ; écart-type entre évaluations rapporté ; fin de run à côté)
Chaque échantillon est affecté au mode de moyenne vraie la plus proche (erreur < 0,2 %).
- **ρ_V** = variance (autour de la moyenne par mode des échantillons) par mode et par coordonnée, moyennée sur les 8 modes (non pondérée) et les coordonnées, divisée par σ_tar² = 1.
- **ρ_M** = moyenne sur les 8 modes de la projection du décalage de la moyenne des échantillons du mode (moyenne des échantillons − moyenne vraie) sur u_k, divisée par δ = 0,3.
- **TV par bassin** (principale) et TV de Voronoï, Q1, couverture.

## Calibration sur données existantes (CPU, avant la location ; `scripts/shape_analyse.py calib`, `figures/shape_correction/calibration.json`)
Planchers de l'apprenant sur buffer **exact** (oracle, mêmes évaluations) :

| apprenant | ρ_V | ρ_M (décalage résiduel / 0,3) | TV par bassin |
|---|---|---|---|
| o-AS | 1,078 (0,003) | −0,165 (0,068) | 0,047 |
| o-DSM | 1,162 (0,005) | −0,131 (0,036) | 0,043 |

- Le plancher de ρ_V est au-dessus de 1 pour les deux apprenants (+0,08 et +0,16, biais du sampler à 200 pas) ; celui de ρ_M est légèrement négatif (bruit de mesure de l'ordre de 0,1 par évaluation).
- **Position des planchers par rapport aux intervalles** : le plancher d'AS en ρ_V (1,078) est **dans** [0,80 ; 1,10], à 0,022 du bord haut ; celui d'o-DSM (1,162) est hors de ρ_V ≥ 1,30 ; les planchers de ρ_M (−0,17 et −0,13) sont dans ρ_M ≤ 0,5 et hors de ρ_M ≥ 0,8. **Aucun plancher ne sort d'un intervalle qu'il devrait respecter, aucun recalage n'est nécessaire**, seuils inchangés. Deux avertissements : (i) le plancher d'AS sur P1 et P2 satisfait déjà la partie AS (un apprenant qui **ignorerait** le buffer passerait aussi) : ces parties AS distinguent « AS ne recopie pas le buffer » de « AS le recopie » (1,44 ; 1), pas « corrige » de « ignore » ; (ii) le biais d'AS en ρ_V (+0,078) laisse peu de marge au bord 1,10 si la prédiction (0,90) est décalée du même biais (0,98) : à lire en connaissance de cause. P3 : les planchers en TV par bassin (0,047 et 0,043) sont à 0,006 et 0,010 du seuil 0,053.

## Règles (de l'utilisateur)
- **P1 (buffer V)** : ρ_V(AS) ∈ [0,80 ; 1,10] **ET** ρ_V(DSM) ≥ 1,30. Prédictions exactes (1D, application d'AS à un buffer fixe) : ρ_V(AS) ≈ 0,90, ρ_V(DSM) ≈ 1,44.
- **P2 (buffer M)** : ρ_M(AS) ≤ 0,5 **ET** ρ_M(DSM) ≥ 0,8. Prédictions : ρ_M(AS) ≈ −0,54 (= A_m), ρ_M(DSM) ≈ 1.
- **P3 (masses)** : TV par bassin des **deux** apprenants ≤ 0,053 sur **les deux** buffers (4 valeurs).
- **Lecture fixée d'avance (de l'utilisateur)** : P1 et P2 vraies = AS corrige les erreurs de forme d'un buffer dont il hérite des masses, le DSM non (contribution propre à AS dans le régime |A_m| < 1) ; P1 vraie, P2 fausse = seule la largeur est corrigée ; P1 et P2 fausses = la prédiction de population ne survit pas à l'apprentissage. Les autres combinaisons (P1 fausse, P2 vraie ; P1 ou P2 fausse par la seule partie DSM, etc.) sont rapportées telles quelles, avec la valeur de chaque partie, sans lecture ajoutée.
- **Limites annoncées** : une graine ; 500 époques depuis un départ entraîné par AS (le DSM aussi) ; deux déformations seulement ; masses exactes (le test ne dit rien des masses erronées) ; ρ_V et ρ_M moyennés sur les 8 modes sans pondération (les modes de faible poids ont environ 700 échantillons) ; les prédictions sont celles de la population, non du réseau.
- **Mes prédictions (sans effet)** : P1 vraie 40 % (le biais d'AS de +0,08 et une population à 0,90 laissent peu de marge, et le DSM peut dépasser 1,30 : 85 %) ; P2 vraie 45 % (AS ≤ 0,5 : 80 % ; DSM ≥ 0,8 : 65 %, le DSM à l'oracle suit le buffer avec un biais) ; P3 vraie 55 % (planchers à 0,006 et 0,010 du seuil).

## Coût, plafond et arrêts (accord explicite de l'utilisateur : RTX 3070 ≈ 0,134 $/h, plafond dur **0,10 $** pour les 4 bras)
- Offre RTX 3070 ≤ 0,135 $/h, fiabilité > 0,99, ≥ 8 cœurs CPU ; sinon pas de location.
- **4 runs en parallèle** (3 threads chacun) : 0,7 à 0,8 s par époque à 2 en parallèle (oracle) ; **1,0 à 1,6 s à 4**, soit **8 à 14 min** pour 500 époques. Démarrage 1 à 10 min (de 1 min à 9,8 min observés), installation et envoi 3 min, rapatriement 2 min.
- **Central ≈ 20 min ≈ 0,04 $ ; pessimiste ≈ 40 min ≈ 0,075 $** (à 0,112 à 0,134 $/h), sous le plafond de 0,10 $. Si l'estimation pessimiste de l'offre trouvée dépasse 0,10 $, pas de location.
- **Arrêt d'urgence à la minute 45 depuis la création** : tout est arrêté, les résultats terminés et les contrôleurs sont rapatriés, l'instance est détruite, l'utilisateur est prévenu ; création → destruction ≤ 55 min (≤ 0,123 $ à 0,134 $/h : si le tarif dépasse 0,109 $/h, l'arrêt est avancé pour rester ≤ 0,10 $). Jamais de question pendant que l'instance tourne ; destruction dès les résultats rapatriés.

## Exécution
`scripts/vast_shape_remote.sh` (instance : `sd_V_as_d128`, `sd_V_dsm_d128`, `sd_M_as_d128`, `sd_M_dsm_d128`) ; analyse locale `scripts/shape_analyse.py runs` (CPU). Résultats dans `figures/shape_correction/`, grands fichiers dans `results/selftrain_runs/`. Section dans RESULTS.md.
