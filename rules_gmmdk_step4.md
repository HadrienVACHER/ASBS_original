# Étape 4 : GMM-dk en grande dimension avec un vrai AS (version réelle d'A1) ; dernier test de la piste darting

Écrit et commité AVANT la location et avant tout résultat de l'étape 4. Aucune règle n'est modifiée après coup ; tout écart serait un amendement daté.

## 0. Question
Existe-t-il un cas avec un vrai AS où AS apprend (bonnes formes, 8/8 bassins) et où l'IS de chemin échoue (ESS < 5 %) alors que le darting corrige ? A1 (synthétique, d = 512) disait oui ; ici le sampler est un vrai AS entraîné.

## 1. Cellules et entraînement (une graine, 0)
- (d, K, Δ) = **(128, 8, 6)** et **(256, 8, 6)** ; construction de `gmm_dk.py` inchangée : K gaussiennes de variance 1 sur axes orthogonaux, moyennes (Δ/√2) e_k (distance mutuelle Δ), poids géométriques de rapport 10.
- Recette d'origine des GMM-dk (`configs/experiment/gmm_dk_as.yaml`) : FourierMLP 4 × 128, lr 1e-4, buffer 10 000, 100 itérations par époque, nfe 200, **1500 époques**, options rapides, évaluation tous les 50 époques.
- σ_max = m × 2√(1 + Δ²/(2d)), **m ∈ {0,5 ; 0,75}** (formule de base des GMM-dk) : d = 128 : 1,068 et 1,602 ; d = 256 : 1,034559 et 1,551839. **4 runs**, pas d'autre valeur.
- Contrôleur sauvegardé aux époques 250, 500, 750, 1000, 1250, 1500, et pour chacun les log w, cellules et points terminaux de 20 000 trajectoires de l'IS de chemin (sauvegardés même si Q échoue).

## 2. Filtre Q (évalué sur chaque contrôleur sauvegardé avant toute comparaison sur ce contrôleur)
Sur les 10⁴ premiers points terminaux, avec les K minima de l'énergie (moyennes polies par la quench) et 10⁴ échantillons exacts de π (graine 2026) comme référence.
- **Quench convergée** (même algorithme pour AS et pour π) : descente monotone de −log π, pas 0,7 par échantillon (direction = gradient, Hessienne ≈ I), un pas n'est accepté que si log π augmente (sinon pas divisé par 2) ; **convergence : ‖∇ log π‖₂ < 10⁻³**, 3000 itérations au plus. Bassin = minimum à moins de 0,5 du point convergé ; un échantillon non convergé n'est pas assigné.
- **Q1 :** médiane de la distance au minimum le plus proche des échantillons d'AS **≤ 1,5 ×** la même médiane sous π (π : ≈ √d, soit ≈ 11,3 à d = 128 et ≈ 16,0 à d = 256).
- **Q2 :** **≥ 95 % des échantillons convergent** et **8/8 bassins avec p̂ ≥ 0,25 p** (p : proportions exactes de Voronoï de la cible, 2·10⁶ tirages). Non évaluable = non passé.
- **Q = Q1 ET Q2.** Q ne dépend d'aucune comparaison. Le résumé de l'IS de chemin (k̂, log Ẑ ± SE, ESS) n'est calculé, pour la sélection de σ_max, que sur les contrôleurs qui passent Q.

## 3. Contrôleur comparé et choix de σ_max
Par dimension : le **candidat** d'un run est son **dernier contrôleur sauvegardé qui passe Q**. Parmi les candidats des deux runs (m = 0,5 et 0,75), règle habituelle de `rules_sigma_select.md` : retenir k̂ < 0,7 ; argmax de log Ẑ (20 000 trajectoires, SE par méthode delta) ; égalité à 2 erreurs-types (SE de la différence = √(SE_i² + SE_max²)) départagée par le plus grand ESS ; si aucun candidat n'a k̂ < 0,7 : plus grand ESS, signalé. **Si aucun contrôleur d'une dimension ne passe Q : rapport, pas de comparaison à cette dimension.** La comparaison porte sur le seul candidat choisi.

## 4. Comparaison (protocole de l'étape 2 du MoS ; uniquement sur le candidat choisi)
- IS de chemin simple, PSIS, tronqué (min(w, √n · moyenne)) sur 20 000 trajectoires : TV, ESS, k̂, log Ẑ.
- Darting post hoc (`dart_lib.darting`, protocole inchangé) sur les 10⁴ premiers points terminaux : quench de 500 échantillons, minima, Hessiennes par différences finies (carte de translation), balayages = 1 saut + 5 pas MALA locaux, pas MALA réglé une fois (acceptation 0,6), **S ∈ {1, 5, 10}**, coût compté en évaluations d'énergie + gradient.
- **MALA seul** à coût égal (même pas, mêmes échantillons).
- **IS de chemin à coût égal** : une simulation de n = coût(S = 10) trajectoires, tranchée au coût de chaque S (IS simple, PSIS, tronqué), protégée par la limite de temps (si tronquée : G faux pour le S concerné et signalé).
- Proportions mesurées de deux façons : cellules de Voronoï (plus proche moyenne) et bassins (quench convergée ci-dessus, assignés seulement, pondérées pour l'IS).
- Plancher TV (n = 10⁴, sampler exact) : médiane sur 100 répétitions, calculée par le script (≈ 0,010).

## 5. Règles pré-enregistrées (sur le candidat choisi, par dimension)
- **I (question clé, descriptive) :** ESS de l'IS de chemin simple (20 000 trajectoires) **< 5 %**.
- **G (inchangé) :** pour au moins un S ∈ {1, 5, 10} : TV_darting(S) ≤ 0,5 × TV de l'IS simple de coût égal ET ≤ 0,5 × TV de la meilleure variante (simple, PSIS, tronqué) de coût égal ET TV_darting(S) ≤ 2 × plancher (TV par cellules de Voronoï). Le même test contre la meilleure variante à 20 000 trajectoires, et la version par bassins, sont rapportés à côté sans changer le verdict.
- **H_saut (inchangé) :** TV_darting(S = 10) ≤ 0,5 × TV du MALA seul de même coût.

## 6. Décision écrite à l'avance
- Si **Q passe et ESS ≥ 5 % à d = 256** : l'IS tient ; **on ferme la piste darting.**
- Si **Q échoue aux deux dimensions** : AS n'apprend pas ; **on ferme la piste.**
- **Seul « Q vrai, I vrai, G vrai » (à l'une des deux dimensions) la rouvre.**
- Tout autre résultat (par exemple Q vrai à d = 128 seulement avec ESS ≥ 5 %, ou Q vrai, I vrai, G faux) ne la rouvre pas : elle reste fermée ; les chiffres sont rapportés sans autre conclusion.

## 7. Coût, plafond et arrêts (accord explicite : RTX 3070 ≈ 0,134 $/h, estimation ≤ 0,15 $, plafond dur 0,15 $)
- Estimation, à partir des temps mesurés (GMM-dk, d = 32, 7 runs en parallèle : 0,40 s par époque ; le coût par époque dépend peu de d car le contrôleur est un MLP de largeur 128 et l'énergie du mélange est bon marché ; 4 runs en parallèle ici) : préparation 3 min ; entraînement 1500 époques à 0,5 s (12,5 min), **pessimiste 1,0 s (25 min)** ; Q du dernier contrôleur ≈ 1 min ; comparaison (darting CPU quelques dizaines de secondes à d = 256, IS à coût égal sur GPU 1 à 3 min) ≈ 4 min ; rapatriement et destruction 2 min. **Total ≈ 22 min ≈ 0,05 $ (central) ; ≈ 35 min ≈ 0,08 $ (pessimiste)**, sous 0,15 $ : location autorisée.
- **Aucune nouvelle évaluation après la minute 50** depuis la création de l'instance ; **arrêt d'urgence à la minute 58** (0,130 $) : tout est arrêté, les résultats terminés sont rapatriés, l'instance est détruite, puis l'utilisateur est prévenu ; durée totale création → destruction ≤ 62 min (0,138 $). Jamais de question à l'utilisateur pendant que l'instance tourne ; destruction dès les résultats rapatriés.
- Si l'entraînement n'est pas fini à la minute 50, les contrôleurs déjà sauvegardés (multiples de 250 époques) sont évalués ; le rapport dit ce qui manque.

## 8. Rapport
Par dimension et par run : Q1 (médianes, rapport), Q2 (fraction convergée, couverture des bassins, p̂/p) pour chaque contrôleur sauvegardé ; candidats, choix de σ_max ; sur le candidat choisi : IS (TV, ESS, k̂, log Ẑ), PSIS, tronqué, darting S ∈ {1, 5, 10}, MALA seul, coût égal, I, G, H_saut, proportions par bassin et par Voronoï ; décision de la section 6 ; coût réel ; réserves (une graine, deux valeurs de σ_max).
