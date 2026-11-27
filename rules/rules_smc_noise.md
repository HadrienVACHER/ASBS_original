# SMC dans l'espace des bruits avec tempering du poids de chemin

Écrit pour : Hadrien. Écrit et commité **avant tout calcul** (étape 0 comprise) ; aucune règle ne sera modifiée après coup (amendements datés seulement). Tous les réglages sont fixés d'avance et **identiques pour toutes les cellules**. Métrique de masse principale : TV par bassin.

## Méthode
- ξ ∈ R^{200×d} : innovations d'une trajectoire du schéma discret du dépôt, contrôleur AS figé ; X = Φ_θ(ξ). Log-poids ℓ(ξ) = −E(X_N) − log r_N(X_N) + log dP⁰/dQ_θ (rapport exact des transitions discrètes, `girsanov.log_path_weights`, validé : écart ≤ 5·10⁻¹¹).
- Cibles π_β(ξ) ∝ N(ξ ; 0, I) exp(β ℓ(ξ)), β de 0 (AS) à 1 (P*, marginale terminale π). Les cibles π sont normalisées : **Z = 1, log Z = 0** exactement (GMM-dk et MoS-w).
- Initialisation : M trajectoires, β = 0, log Ẑ = 0. À chaque niveau : (i) Δβ par bissection pour que l'ESS relatif des poids incrémentaux exp(Δβ ℓ) vaille **τ = 0,5**, plafonné par 1 − β ; (ii) log Ẑ += logsumexp(Δβ ℓ) − log M ; (iii) rééchantillonnage systématique ; (iv) **K = 5 pas de pCN** invariants pour π_β : ξ' = ρ ξ + √(1 − ρ²) η, acceptation min(1, exp(β (ℓ' − ℓ))) ; ρ initial 0,9, adapté après chaque niveau par log(1 − ρ) ← log(1 − ρ) + 2 (acceptation moyenne − 0,25), 1 − ρ borné à [10⁻⁶, 1]. Au plus 500 niveaux (sinon run déclaré échoué).
- **Calendrier** : passe 1 adaptative ; passe 2 indépendante avec **le calendrier de β et de ρ figé** de la passe 1 (estimateur strictement sans biais de Z). Les deux sont rapportées.
- Sauvegarde par particule et par niveau : ℓ, bassin (trempe convergée), parent du rééchantillonnage ; X_N des particules finales.

## Étape 0 (CPU, chaque exécution < 1 min, avant toute location)
- **Test unitaire** : contrôle nul (u = 0, la proposition est la référence), cible gaussienne en d = 2, N((1 ; 0), 0,7² I), Z = 1, VE σ_max = 2 ; 20 passes adaptatives et 20 passes figées (M = 512). **Réussi si |moyenne(Ẑ) − 1| ≤ 3 erreurs-types** dans les deux cas ; sinon arrêt.
- **Trempe en torch** (utilisée sur le GPU pour les bassins) contre `basin_lib` sur 2000 points : accord ≥ 99,9 % ; sinon arrêt.
- **Essai du script** sur la plus petite cellule disponible avec un contrôleur, (8, 8, 8) × 0,5 (fenêtre, graine 0), M = 256 ; temps par niveau mesuré.
- Le temps GPU total est estimé après l'étape 0 (M ≥ 2048, fixé par amendement daté avant l'étape 1) ; **si ≤ 30 min de GPU : location, sinon job cluster** que l'utilisateur soumet.

## Étape 1
- **Cellules** (σ_max, contrôleurs figés) : (A) témoin de sûreté (128, 8, 6) × 0,5, instantanés 1000, 1250, 1500 ; (B) (128, 8, 6) × 0,75, instantanés 1000, 1250, 1500 ; (C) MoS-w d = 10, instantanés **250, 500, 750** (l'instantané 250 de la demande plus deux, pour la règle permanente des 3 instantanés ; déclaré). 3 répétitions indépendantes par instantané, chacune avec passes 1 et 2.
- **Témoins à coût égal** (même nombre total de trajectoires simulées par run, M_IS = M × (1 + J × K), J = nombre de niveaux du run) : IS de chemin auto-normalisé (log Ẑ et bassins pondérés) ; « IS de l'AS basique » = même estimateur sur les M trajectoires initiales (rapporté). Pour les échantillons seulement : banque (1000 trempes depuis N(0, ν₁ I)) + darting + MALA du protocole existant (S = 2 en (B), S = 50 en (C)), sur les M trajectoires initiales, coût compté en évaluations d'énergie et de gradient (CPU local après coup).
- **Mesures** : erreur log Ẑ − log Z (médiane et quantiles 10–90 % sur les 9 runs par cellule) ; TV par bassin des particules finales ; Q1 ; erreur de position des modes ‖moyenne (ou médiane par coordonnée pour MoS-w) des particules du bassin − minimum‖ ; masses par bassin à chaque niveau β ; J, Δβ, acceptation et ρ par niveau, ancêtres distincts ; coûts séparés (passes du réseau = 200 par trajectoire, évaluations d'énergie, gradients : 0 pour le SMC et l'IS ; la trempe de mesure n'est pas comptée).

## Règles (sur la **passe 2** ; les 9 runs de chaque cellule ; la passe 1 est rapportée à côté)
- **S1 (Z)** : en (B) **et** (C), médiane de |log Ẑ_SMC| ≤ 0,1 **ET** ≤ 0,5 × la médiane de |log Ẑ_IS| à coût égal.
- **S2 (masses)** : en (B) **et** (C), médiane de la TV par bassin du SMC ≤ 0,5 × celle de l'IS à coût égal **ET** ≤ 2 × plancher (médiane de la TV d'un échantillonneur exact de taille M, 200 répétitions ; vérité : Voronoï exacte du GMM-dk, `mog_mos_truth_mosw_d10.json` pour MoS-w).
- **S3 (forme)** : médiane de Q1 ≤ 1,02 en (B), ≤ 1,2 en (C).
- **S4 (sûreté)** : dans chaque cellule et chaque run de la passe 2, aucune masse par bassin ne descend, à aucun niveau β, sous 0,25 × min(q_k, π_k) (q_k = fraction du bassin à β = 0, π_k = vraie ; si q_k = 0, le seuil est nul) ; en (A), médiane de |log Ẑ| et de TV du SMC ≤ 1,5 × celles de l'IS à coût égal.
- **Lecture fixée d'avance** : S1 et S2 vraies avec S4 = le SMC dans l'espace des bruits corrige les masses et estime Z là où l'IS de chemin s'effondre, à coût égal ; S1 vraie seule = gain sur Z seulement ; S1 et S2 fausses = pas de gain à coût égal, en précisant si c'est le mélange du pCN (ρ, acceptation) ou un goulot (masses le long de β).
- Descriptif sans règle : comparaison des échantillons avec banque + darting + MALA (qualité et coût).

## Mes prédictions (sans effet)
Test unitaire réussi 95 %. S1 vraie 35 % ((B) 55 %, (C) 20 %). S2 vraie 25 % ((B) 40 %, (C) 15 % : le pCN dans R^{200×d} ne déplace pas les masses entre modes, ρ proche de 1). S3 : (B) 55 %, (C) 35 %. S4 : 60 % (rééchantillonnage). (A) sûre 70 %. Lecture « corrige » (S1, S2, S4) : 15 %.

## Coût, plafond et arrêts (accord explicite de l'utilisateur si estimation ≤ 30 min de GPU)
RTX 3070, fiabilité > 0,99 de préférence (≥ 0,975 accepté), plafond dur **0,15 $** pour l'ensemble (pCN seul compris s'il est fait ici) ; sauvegarde, détruis, préviens ; destruction dès les résultats rapatriés. Sinon : job cluster.

## Exécution
`scripts/smc_noise.py` (`unit`, `quenchtest`, `trial`, `run`, `darting`, `verdict`). Résultats `figures/smc_noise/` (résumés) et `results/smc_noise/` (particules). Section dans RESULTS.md, ligne 17 de la grille.
