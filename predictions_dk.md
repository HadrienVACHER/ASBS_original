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

---
# Addendum 1 (écrit et commité AVANT le lot « balayage en Δ + ablation de sigma_max »)

## Règle sigma_max actuelle (inchangée) : sigma_max = 2·√(1 + Δ²/(2d)) (`gmm_dk.sigma_max_rule`)
Valeurs exactes du balayage : d = 8 : Δ = 4, 5, 6, 7, 8 → 2,828 ; 3,202 ; 3,606 ; 4,031 ; 4,472. d = 32 : Δ = 4, 5, 6, 7, 8 → 2,236 ; 2,358 ; 2,500 ; 2,658 ; 2,828.
Ablation (4 runs), sigma_max × 0,5 et × 2 par rapport à la règle : (d, K, Δ) = (8, 8, 8) : règle 4,472 → 2,236 et 8,944 ; (32, 8, 6) : règle 2,500 → 1,250 et 5,000.

## Plan du lot (une graine, 0, même protocole que le pilote)
Balayage Δ ∈ {4, 5, 6, 7, 8} pour (d, K) ∈ {8, 32} × {2, 8} : 20 cellules, dont les 5 du pilote sont réutilisées ((8,2,8), (8,8,8), (32,2,8), (32,8,8), (32,8,4)) ; 15 runs nouveaux. Ablation : 4 runs, noms avec suffixe `_sx0.5` / `_sx2`. Plafond 0,15 $ facturé pour ce lot. Rapport : même tableau que le pilote, plus la figure couverture et β en fonction de Δ (une courbe par (d, K)) et ESS de l'IS en fonction de Δ.

## Prédictions de l'assistant
1. Δ de la première perte de mode (première valeur du balayage avec couverture < K, seuil p̂ ≥ 0,25 p) : (8, 2) : Δ = 7 (fourchette 6–8, déjà perdu à 8) ; (8, 8) : Δ = 7 (6–8, déjà 6/8 à 8) ; (32, 2) : Δ = 7 (5–8, à 8 le ratio du mode léger est 0,18, juste sous le seuil) ; (32, 8) : Δ = 6 (5–7, 3/8 à 8). Appui : ancien balayage en d = K, transition de couverture vers Δ ≈ 6–7 en d ≥ 16.
2. Zone « tous les modes trouvés et β < −0,3 » à d = 32 : prédite ABSENTE (probabilité que je lui donne : 75 %). Là où tous les modes sont trouvés à d = 32, je prédis β > −0,3 (poids presque justes, β ≈ −0,1 comme à Δ = 4), puis une bascule vers β > 0 en même temps que les premières pertes. Pour K = 2, β est calculé sur 2 points (sans SE) : un mode léger sous-représenté donne β > 0, un aplatissement donne β < 0. À d = 8, une zone d'aplatissement faible (β ≈ −0,2 à −0,5 avec tous les modes trouvés) est possible pour (8, 8) à Δ = 6–7 (probabilité 40 %).
3. Ablation de sigma_max : sigma_max × 2 élargit la base (N(0, ν₁I)) : je prédis plus de modes couverts qu'avec la règle (à (8, 8, 8) : 7–8 modes sur 8 contre 6 avec la règle ; à (32, 8, 6) : +1 à +2 modes par rapport au run de la règle du balayage), et un TV plus faible si la couverture augmente, au prix d'un entraînement plus bruité. sigma_max × 0,5 : moins de modes couverts (à (8, 8, 8) : 4 sur 8 (3–6) ; à (32, 8, 6) : −1 à −2 modes) et β plus positif (concentration sur les modes lourds). Si l'effet de sigma_max est nul ou de signe contraire, l'explication « base trop étroite » est contredite.
4. Temps et coût : 19 runs en parallèle ; je prédis 0,5–0,9 s par époque (contre 0,41 pour 5 runs), soit 13–23 min, plus 5–8 min d'IS de chemin pour 24 runs ; coût 0,05–0,10 $ pour le lot.

---
# Addendum 2 : hypothèse H_ρ (écrite et commitée AVANT de calculer ρ pour les 24 runs et AVANT de regarder la figure couverture/β en fonction de ρ)

## Définitions
ν₁ = Σ_k g(t_k)² (t_{k+1} − t_k) avec g(t)² = σ_min² (σ_max/σ_min)^(2(1−t)) · 2 ln(σ_max/σ_min) (`VESDE.diff`) et t_k = `linspace(0, 1, 200)` (schéma réel des runs, somme de Riemann à gauche, celle de `girsanov.log_path_weights`). ‖μ‖ = Δ/√2. ρ = √(d ν₁) / ‖μ‖ = (rayon typique de la base N(0, ν₁ I_d)) / (rayon des modes). « Mode perdu » = couverture < K (critère p̂_k ≥ 0,25 p_k). Population : les 24 runs (5 pilote, 15 balayage, 4 ablation, une graine).

## H_ρ
La couverture dépend surtout de ρ : les modes sont perdus quand ρ dépasse un seuil ρ* commun à tous les (d, K, Δ).
Critère de lecture (fixé ici, sans seuil ajusté après coup) : H_ρ est « soutenue avec un seuil commun » s'il existe ρ* tel qu'au moins 22 des 24 runs sont bien classés (couverture = K si ρ < ρ*, couverture < K si ρ > ρ*). Sinon je rapporte, pour chaque d et pour chaque Δ, l'intervalle des ρ* qui classent bien tous les runs du groupe (au plus 1 erreur) ; « le seuil dépend de d (ou de Δ) » si ces intervalles ne se recouvrent pas entre groupes.

## Prédiction de l'assistant (avant la figure)
H_ρ avec un seuil commun ne sera PAS soutenue (probabilité que je lui donne : 75 %). Raison, calculée à la main avant d'écrire ceci : ∫g² dt = σ_max² − σ_min², donc ν₁ ≈ σ_max² et ρ ≈ √2 · σ_max · √d / Δ. Avec la règle σ_max = 2√(1 + Δ²/(2d)) : ρ ≈ 2√2 √(d/Δ² + 1/2), qui DÉCROÎT quand Δ augmente à d fixé (≈ 2,8 → 2,2 de Δ = 4 à 8 pour d = 8 ; ≈ 4,5 → 2,8 pour d = 32), alors que les modes sont perdus aux Δ ÉLEVÉS. Les runs de la règle suggèrent donc un seuil qui dépend de Δ (ou de d), pas de ρ seul. L'ablation (σ_max × 2 → ρ × 2 → perte ; × 0,5 → ρ / 2 → pas de perte) va dans le sens de H_ρ à d et Δ fixés : je m'attends à ce que ρ explique l'effet de σ_max à (d, Δ) fixés mais pas la comparaison entre (d, Δ) différents. Si le seuil dépend de Δ, la grandeur pertinente serait plutôt le rapport σ_max/Δ à d fixé (ou ρ/√d, ν₁/Δ²) : à tester, non pré-enregistré ici.
