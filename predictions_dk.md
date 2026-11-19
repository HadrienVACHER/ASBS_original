# Étude AS en fonction de (d, K, Δ) : définition, règle sigma_max et prédictions (écrit et commité AVANT le pilote)

Descriptif, sans règle d'effet. Aucune prédiction n'est modifiée après coup ; les écarts seront rapportés tels quels.

## Cible (une seule définition : `gmm_dk.py`, utilisée par `GMMDK` dans `dist_utils.py` et par l'analyse)
K gaussiennes isotropes de variance 1 en dimension d (K ≤ d), moyennes μ_k = (Δ/√2) e_k, toutes les paires à distance Δ, séparation normalisée Δ/2. Poids w_k ∝ ρ^(k−1), ρ = 10^(−1/(K−1)) (max/min = 10 pour tout K). Vérité : p_k = masse de la cellule de Voronoï k (plus proche moyenne), 2·10^6 tirages exacts (seules les K premières coordonnées sont tirées : les autres sont du bruit N(0,1) identique pour toutes les moyennes ; le test local le vérifie contre un tirage en dimension d complète). Tests locaux : `tests/test_gmm_dk.py` (1 s).

## Tenu fixe
Config `gmm_dk_as` (copie de `gmm_as`) : FourierMLP 4 couches × 128, nfe 200, sigma_min 0,001, 1500 époques, lr 1e-4, buffer 10 000, duplicates 10, resample 1000, graine 0, options rapides (`fast_sampling fast_train buffer_on_device`), évaluation toutes les 50 époques sur 5000 échantillons, `dump_eval_samples=true`.
**Règle sigma_max (identique pour toutes les cellules) :** sigma_max = 2·√(1 + max‖μ_k‖²/d) = 2·√(1 + Δ²/(2d)). Justification : c'est la convention du dépôt pour tous les runs GMM précédents (« 2 × l'écart-type par coordonnée de la cible », `gmm_std:dim,2,radius`), ici avec radius = max‖μ_k‖ = Δ/√2, de sorte que la base N(0, ν₁I) couvre la cible comme dans les études précédentes. Valeurs du pilote : 4,47 (d=8, Δ=8), 2,83 (d=32, Δ=8), 2,24 (d=32, Δ=4). Réserve déclarée : sigma_max dépend de d et de Δ par construction, donc l'effet de d ou de Δ est celui du couple (géométrie, sigma_max) ; les runs précédents avaient la même confusion.

## Métriques (par run)
Couverture (p̂_k ≥ 0,25 p_k, modes perdus avec leur poids) ; TV = moyenne des TV des 10 dernières évaluations (comparable au plancher à 5000 échantillons), plancher TV exact (médiane, p5–p95) par (K, Δ) ; p̂ moyenné sur les 10 dernières évaluations pour la couverture, les ratios p̂_k/p_k et β ; β = pente OLS de log(p̂/p) sur log p sur les modes couverts (SE nulle si < 3 modes couverts : K=2 donne β sans SE) ; IS de chemin sur le contrôleur final (`adjoint_samplers/utils/girsanov.py`, 20 000 trajectoires) : ESS (fraction), k̂ (ajustement Pareto généralisé sur la queue des poids, 20 % des plus grands, `scipy.stats.genpareto`), TV après repondération ; sample_w2 (évaluateur). Une graine.

## Prédictions de l'utilisateur
P1. β se rapproche de −1 quand Δ augmente (masse des barrières ~ exp(−Δ²/8)). P2. À Δ fixé, l'ESS de l'IS de chemin baisse quand d augmente. P3. À Δ grand, la couverture baisse quand K augmente, et les modes légers sont perdus en premier.

## Prédictions chiffrées de l'assistant (pilote)
Base : 2D (r = 2, 4, 6, K = 3, β calculé sur les deux modes extrêmes : ≈ 0 à Δ = 3,5, ≈ −0,4 à Δ = 6,9, ≈ −0,9 à Δ = 10,4) et l'ancien balayage en d = K (transition de couverture vers Δ ≈ 6–7 en d ≥ 16, aucune perte à d = 8 pour Δ = 8).
| cellule (d, K, Δ) | couverture | TV (plancher) | β | ESS de l'IS de chemin |
|---|---|---|---|---|
| 8, 2, 8 | 2/2 | 0,06 (0,02–0,15) (0,002) | −0,5 (−0,9 à −0,2), sans SE | 0,1–0,5 |
| 8, 8, 8 | 8/8 (7–8) | 0,15 (0,08–0,30) (0,014) | −0,5 (−0,9 à −0,2) | 0,01–0,2 |
| 32, 2, 8 | 2/2 | 0,10 (0,03–0,25) (0,002) | −0,7, sans SE | < 0,05 |
| 32, 8, 8 | 5/8 (2–8), modes perdus = les plus légers | 0,35 (0,15–0,60) (0,014) | −0,8 sur les modes couverts | < 10⁻³, k̂ > 0,7 |
| 32, 8, 4 | 8/8 | 0,04 (0,02–0,08) (0,014) | −0,1 (−0,3 à 0,1) | 0,05 (0,005–0,3) |
Temps : 1,0–1,5 s par époque pour chacun des 5 runs en parallèle (soit 25–40 min), coût ≈ 0,10 $ (plafond 0,30 $). Calibrage visé : niveau « proche » = Δ = 4 si la cellule (32, 8, 4) est près du plancher ; niveau « éloigné » = Δ = 8 si tous les modes y sont encore trouvés (couverture complète à (8, 8, 8) et (32, 8, 8)), sinon un Δ intermédiaire (6) à proposer pour la grille.
