# Étape 5 : dernier lot de la piste darting, GMM-dk en grande dimension et MoS-dk (queues lourdes), vrais AS

Écrit et commité AVANT la location et avant tout résultat de l'étape 5. Aucune règle n'est modifiée après coup ; tout écart serait un amendement daté.

## 0. Questions
(a) GMM-dk (128, 8, 6) et (256, 8, 6) : l'IS de chemin s'effondre-t-il avec la dimension ? (b) MoS-dk (32, 8, 6) et (128, 8, 6) : les queues lourdes font-elles échouer l'IS alors qu'AS apprend ?

## 1. Cibles (graine 0, recette d'origine des GMM-dk : FourierMLP 4 × 128, lr 1e-4, buffer 10 000, 100 itérations par époque, nfe 200, 1500 époques, options rapides, évaluation tous les 50 époques)
- **GMM-dk (d, 8, 6)** : `gmm_dk.py` inchangé (moyennes (Δ/√2) e_k sur axes orthogonaux, poids géométriques de rapport 10, variance unité).
- **MoS-dk (d, 8, 6)** (`MoSDK` de `dist_utils.py`, `configs/problem/mosdk.yaml`) : mêmes moyennes et mêmes poids, mais chaque composante est un **produit de Student 1D avec ν = 3 et s² = 4/3** (densité d'une coordonnée ∝ (1 + u²/(ν s²))^(−(ν+1)/2) = (1 + u²/4)^(−2)) : la Hessienne de −log d'une composante en son mode vaut (ν+1)/(ν s²) = 1 = I. Toutes les composantes ont la même forme (variance par coordonnée 4). Les minima de l'énergie du mélange sont décalés des moyennes de 0,002 à 0,03 (recouvrement des queues, mesuré avant l'écriture) : les Hessiennes aux minima ne sont pas rigoureusement égales, le darting utilise la carte à Hessiennes générales quand l'écart dépasse 10⁻⁶.
- **Vérité** : proportions par cellules de Voronoï (plus proche moyenne) par **10⁶ tirages exacts** (GMM : `gmm_dk.true_proportions` ; MoS-dk : `scripts/mosdk_truth.py`, `figures/mog_mos_truth_mosdk_d{32,128}.json`, qui ne tire que les K premières coordonnées, seules à compter pour la cellule) ; plancher TV (n = 10⁴, sampler exact) = médiane de 100 répétitions (≈ 0,010). **Proportions par bassin de la vérité : 2·10⁵ tirages exacts** (et non 10⁶ : la quench de 10⁶ points coûterait plusieurs minutes de CPU de l'instance par cible), rapportées à côté (descriptif) avec l'écart TV(bassin, Voronoi) ; les TV des proportions par bassin sont calculés contre les proportions de Voronoï à 10⁶ tirages, comme aux étapes 3 et 4.
- **GMM-dk : pas de réentraînement.** Les 4 runs de GMM-dk de cette étape (les deux cellules × σ_max {0,5 ; 0,75}, recette d'origine, graine 0) sont ceux de l'étape 4, déjà faits et commités (`rules_gmmdk_step4.md`, 2527419 ; résultats 325e338, 8ce62d2). Les refaire à l'identique (même graine, même code) ne ferait que dépenser (≈ 0,05 $) ; leurs Q, le choix de σ_max (× 0,5, époque 1500, aux deux dimensions), l'IS (ESS 0,193 et 0,162, k̂ 0,22 et 0,34) et le darting S ∈ {1, 5, 10} sont ceux de l'étape 4, **connus avant l'écriture de cette règle**. Ce qui est nouveau pour le GMM-dk : le darting à **S = 40** (supplément ci-dessous), les règles G et H_saut évaluées sur S ∈ {1, 5, 10, 40}.

## 2. Entraînement des MoS-dk
σ_max = m × 2√(1 + Δ²/(2d)), **m ∈ {0,5 ; 0,75}** (formule de base des GMM-dk) : d = 32 : 1,25 et 1,875 ; d = 128 : 1,068 et 1,602. **4 runs** (8 runs au total avec les 4 GMM-dk de l'étape 4). Contrôleur sauvegardé aux époques 250, 500, 750, 1000, 1250, 1500, et pour chacun les log w, cellules et points terminaux de 20 000 trajectoires de l'IS de chemin (même si Q échoue).

## 3. Filtre Q (évalué sur chaque contrôleur sauvegardé avant toute comparaison sur ce contrôleur ; définition de l'étape 4)
Sur les 10⁴ premiers points terminaux, avec les K minima de l'énergie (moyennes polies par la quench) et 10⁴ échantillons exacts de π (graine 2026) comme référence.
- **Quench convergée** (même algorithme pour AS et π) : descente monotone de −log π, direction P ∇log π avec P = diag((ν s² + u²)/(ν + 1)) pour les Student (u = x − μ de la composante la plus responsable ; P = I pour les gaussiennes), pas 0,7 par échantillon, un pas n'étant accepté que si log π augmente (sinon pas divisé par 2) ; **convergence : ‖∇ log π‖₂ < 10⁻³**, 3000 itérations au plus. Bassin = minimum à moins de 0,5 du point convergé ; un échantillon non convergé n'est pas assigné.
- **Q1 :** médiane de la distance au minimum le plus proche des échantillons d'AS **≤ 1,5 ×** la même médiane sous π.
- **Q2 :** **≥ 95 % des échantillons convergent** et **8/8 bassins avec p̂ ≥ 0,25 p**. Non évaluable = non passé.
- **Q = Q1 ET Q2**, indépendant de toute comparaison. Le résumé de l'IS de chemin (k̂, log Ẑ ± SE, ESS) n'est calculé, pour le choix de σ_max, que sur les contrôleurs qui passent Q.

## 4. Contrôleur comparé et choix de σ_max
Par cible : **candidat** d'un run = son **dernier contrôleur sauvegardé qui passe Q**. Parmi les candidats des deux runs : règle habituelle (`rules_sigma_select.md`) : retenir k̂ < 0,7 ; argmax de log Ẑ (20 000 trajectoires) ; égalité à 2 erreurs-types (SE de la différence = √(SE_i² + SE_max²)) départagée par le plus grand ESS ; **si aucun candidat n'a k̂ < 0,7 (attendu sur le MoS-dk) : plus grand ESS, signalé**. **Si aucun contrôleur d'une cible ne passe Q : rapport, pas de comparaison pour cette cible.** La comparaison porte sur le seul candidat choisi. (Les runs non choisis ne sont pas comparés, même s'ils passent Q.)

## 5. Comparaison (candidat choisi ; protocole de l'étape 2 du MoS et de l'étape 4)
- IS de chemin simple, PSIS, tronqué (min(w, √n · moyenne)) sur 20 000 trajectoires : TV, ESS, k̂, log Ẑ.
- Darting post hoc (`dart_lib.darting`, protocole inchangé) sur les 10⁴ premiers points terminaux : quench de 500 échantillons, minima, Hessiennes par différences finies (translation si l'écart de Hessiennes < 10⁻⁶, sinon carte générale avec jacobien), balayages = 1 saut + 5 pas MALA locaux, pas MALA réglé une fois (acceptation 0,6), **S ∈ {1, 5, 10, 40}**, coût en évaluations d'énergie + gradient.
- **MALA seul** à coût égal ; **IS de chemin à coût égal** : une simulation de n = coût(S = 40) trajectoires (≈ 4,8·10⁶), tranchée au coût de chaque S (IS simple, PSIS, tronqué), protégée par la limite de temps (si tronquée : G faux pour le S concerné, signalé).
- Proportions par cellules de Voronoï et par bassins (quench convergée, assignés seulement, pondérées pour l'IS).
- **Supplément GMM-dk** : les contrôleurs de l'étape 4 retenus (d = 128 et 256, × 0,5, époque 1500) sont réévalués sur GPU (20 000 trajectoires resimulées avec la même graine) avec S ∈ {1, 5, 10, 40} ; ses chiffres d'IS peuvent différer des chiffres de l'étape 4 au bruit de la simulation GPU près (les deux sont rapportés).

## 6. Règles pré-enregistrées (sur le candidat choisi, par cible)
- **I (question clé, descriptive) :** ESS de l'IS de chemin simple (20 000 trajectoires) **< 5 %**.
- **G (inchangé, évalué contre l'IS simple et contre la meilleure variante) :** pour au moins un S ∈ {1, 5, 10, 40} : TV_darting(S) ≤ 0,5 × TV de l'IS simple de coût égal ET ≤ 0,5 × TV de la meilleure variante (simple, PSIS, tronqué) de coût égal ET TV_darting(S) ≤ 2 × plancher (TV par cellules de Voronoï). Le même test contre la meilleure variante à 20 000 trajectoires, et la version par bassins, sont rapportés à côté sans changer le verdict.
- **H_saut (inchangé) :** TV_darting(S = 40) ≤ 0,5 × TV du MALA seul de même coût (S = 10 rapporté à côté).

## 7. Décision écrite à l'avance
**La piste darting n'est rouverte que si, pour au moins une des quatre cibles, on a à la fois Q vrai, I vrai et G vrai. Sinon, elle est fermée.** Tout autre résultat (Q faux, I faux, G faux, ou comparaison non réalisée) ne la rouvre pas ; les chiffres sont rapportés sans autre conclusion.

## 8. Coût, plafond et arrêts (accord explicite : RTX 3070 ≈ 0,134 $/h, estimation ≤ 0,25 $, plafond dur 0,25 $)
- Estimation (temps mesurés : MoS d = 50, K = 10, 0,92 s par époque seul et 1,0 s à 3 en parallèle ; GMM-dk d = 128/256, 4 en parallèle, 0,7 s) : préparation 3 min ; entraînement des 4 MoS-dk en parallèle 1500 époques à 1,0 à 1,3 s (25 à 33 min ; **pessimiste 2,0 s : 50 min**) ; supplément GMM-dk et vérités par bassin en arrière-plan pendant l'entraînement ; comparaison finale des deux MoS-dk (darting S = 40 sur CPU quelques dizaines de secondes à 1 à 2 min, IS à coût égal jusqu'à 4,8·10⁶ trajectoires sur GPU 2 à 6 min) ≈ 8 min ; rapatriement et destruction 2 min. **Total ≈ 40 min ≈ 0,09 $ (central) ; ≈ 63 min ≈ 0,14 $ (pessimiste)**, sous 0,25 $ : location autorisée.
- **Aucune nouvelle évaluation après la minute 80** depuis la création de l'instance ; **arrêt d'urgence à la minute 95** (0,212 $) : tout est arrêté, les résultats terminés sont rapatriés, l'instance est détruite, puis l'utilisateur est prévenu ; durée totale création → destruction ≤ 100 min (0,223 $). Jamais de question à l'utilisateur pendant que l'instance tourne ; destruction dès les résultats rapatriés.
- Si l'entraînement n'est pas fini à la minute 80, les contrôleurs déjà sauvegardés sont évalués ; le rapport dit ce qui manque.

## 9. Rapport
Par cible et par run : Q1 (médianes, rapport), Q2 (fraction convergée, couverture des bassins, p̂/p) pour chaque contrôleur sauvegardé ; candidats, choix de σ_max (avec signalement du cas « aucun k̂ < 0,7 ») ; sur le candidat choisi : IS (TV, ESS, k̂, log Ẑ), PSIS, tronqué, darting S ∈ {1, 5, 10, 40}, MALA seul, coût égal, I, G, H_saut, proportions par bassin et par Voronoï ; décision de la section 7 ; coût réel ; réserves (une graine, deux valeurs de σ_max, GMM-dk non réentraînés).
