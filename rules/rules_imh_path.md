# Metropolis–Hastings indépendant (IMH) dans l'espace des trajectoires, avec AS comme proposition — Phase 1 (CPU, hors ligne)

Écrit pour : Hadrien. Écrit et commité **avant tout calcul** ; aucune règle ne sera modifiée après coup (amendements datés seulement). CPU, chaque exécution < 1 min, aucune location. **Métrique de masse principale : TV par bassin.**

## Idée (note « MH dans l'espace des trajectoires », 26/11, consignée ici)
- Cible sur les trajectoires : dP*/dP⁰ ∝ exp(−E(X_T)) / r_T(X_T), r_T densité terminale de la **référence** (VE brownien depuis 0) ; sa marginale terminale est exactement π. Pour la version discrète, r_T est la marginale exacte de la référence discrétisée, N(0, V I), V = Σ_k v_k, v_k = σ(t_k)² Δt_k.
- Poids : log w̃ = −E(X_T) − log r_T(X_T) + log dP⁰/dQ_θ(X), avec le rapport exact des densités de transition discrètes du schéma d'Euler du dépôt (référence contre contrôlé, même covariance). C'est le log-poids de l'IS de chemin (`utils/girsanov.log_path_weights`), vérifié numériquement avant tout calcul (commande `check`).
- IMH : proposition Y ~ Q_θ indépendante, acceptation min(1, w̃(Y)/w̃(X)) ; chaîne stationnaire P*, extrémités ~ π.
- **Rejeu exact sur un pool** : chaque trajectoire du pool (iid sous Q_θ) est utilisée une seule fois comme proposition, dans l'ordre du fichier ; il suffit de (log w̃, étiquette de bassin). L'état initial est la première trajectoire ; chauffe 10 %.

## Cellules et pools (existants ; aucune simulation nouvelle hors `check`)
Trois **pools** par cellule (la règle permanente de variabilité : écart-type entre les 3 pools). Les bassins sont les K bassins de π ; « Voronoï » = cellule de la moyenne la plus proche.
- **(A) IS qui marche : GMM-dk (128, 8, 6) × 0,5.** 3 instantanés (époques 1000, 1250, 1500), 20 000 trajectoires chacun : `figures/gmmdk_step4/logw/is_logw_gmmdk4_d128_sx0.5_ep*.npz`. Les points terminaux n'ont pas été rapatriés : l'étiquette est la **cellule de Voronoï** (déclaré ; à × 0,5 les formes sont justes, Q1 1,03 à 1,04).
- **(B) IS effondré : (128, 8, 6) × 0,75.** 3 instantanés (époques 1000, 1250, 1500), chacun = les 3 jeux de 10⁴ de `results/darting_large_sigma/sim_d128e*` / `sim_d128_*` mis bout à bout (30 000 ; ordre s1, s2, s3), **bassins par trempe** déjà calculés (`raw_*.npz`).
- **(C) MoS-w d = 10, instantané de l'époque 250.** Pool de 20 000 de `results/mosw_step3/step3_results/d10` (log w, points terminaux), découpé en **3 blocs consécutifs disjoints** (6667, 6667, 6666) : un seul contrôleur, ces 3 « pools » sont 3 évaluations indépendantes du même instantané (déclaré). Bassins par trempe convergée (`basin_lib`), vérité `figures/mog_mos_truth_mosw_d10.json`.
- **(D) modes absents : (32, 8, 8) × 0,5.** Même contrôleur (vérifié égal) : `results/darting_large_sigma/sim_k8x0.5_s1` (10 000) et `results/wf_start/k8d32/start_samples.pt` (10 000, graine 7), mis bout à bout puis découpés en **3 blocs consécutifs** (6667, 6667, 6666). Bassins par trempe.

## Mesures (par pool)
- Taux d'acceptation (après chauffe) ; plus longue séquence de rejets.
- **ESS de la chaîne pour les indicatrices de bassin, par proposition** : autocorrélation intégrée τ_k de 1[bassin = k] (états après chauffe), estimateur de Geyer à suite initiale monotone positive ; ESS_k/n = 1/τ_k. Les indicatrices constantes dans la chaîne sont exclues. **ESS_IMH = médiane sur les bassins** des ESS_k/n. Rapporté aussi : le minimum, et τ estimé par moyennes de 30 lots.
- **ESS de l'IS par proposition** : (Σw)² / (n Σw²) (Kish) sur le pool entier ; rapporté en plus, sans règle, l'ESS par indicatrice de l'IS auto-normalisé (même fonctionnelle que l'IMH).
- **TV par bassin** de la sortie de chaîne (états après chauffe, parmi les états assignés) et de l'IS auto-normalisé sur le pool entier, contre les proportions vraies. Voronoï en secondaire pour A (c'est la seule disponible).
- Poids maximal normalisé.

## Règles (moyennes des 3 pools ; écart-type entre pools rapporté ; « marginal » signalé si moyenne ± écart-type franchit le seuil)
- **I1 (Liu 1996)** : dans **les 4 cellules**, moyenne des rapports par pool ESS_IMH / ESS_IS (par proposition, Kish) **≤ 2**.
- **I2** : en **(B) et (C)**, TV par bassin moyenne de l'IMH **≥ 0,5 ×** TV par bassin moyenne de l'IS (pas de gain là où l'IS échoue).
- **I3** : en **(D)**, les modes absents (aucune proposition dans le bassin sur le pool) restent à 0 dans la chaîne. **Précision déclarée** : dans un rejeu, les états de la chaîne sont des points du pool, donc cette règle est vraie par construction ; elle est rapportée pour le Th. C et les bassins à très peu de propositions sont listés.
- **I4** : en **(A)**, TV (Voronoï) moyenne de l'IMH **≤ 2 × plancher**, plancher = 0,0100 (convention du projet, n = 10⁴ ; la chaîne a 18 000 états après chauffe mais un ESS bien plus petit ; le plancher à n = 18 000 est rapporté à côté).
- **Lecture fixée d'avance** : I1 à I4 vraies = l'IMH sur les trajectoires n'apporte rien par rapport à l'IS de chemin ; la piste est fermée en **R** dans la grille, références Liu (1996) et Mengersen & Tweedie (1996). Une règle fausse **dans le sens d'un gain** = résultat surprenant à examiner avant toute conclusion (en particulier un ESS_k estimé sur une chaîne bloquée est peu fiable). Une règle fausse sans gain est rapportée telle quelle.

## Descriptif : décomposition de l'identité (15) de la note
E_q[w²] = Σ_k π_k² / q_k · (1 + χ²_k), avec w normalisés (moyenne 1 sur le pool), q_k = fraction du pool dans le bassin k, π̂_k = q_k E_q[w | k] (identité exacte avec ces estimations), 1 + χ²_k = E_q[w² | k] / E_q[w | k]². Rapportés : le terme de proportions T = Σ π̂_k²/q_k, E_q[w²], la part log T / log E_q[w²] (masses) et le reste (forme intra-bassin), les χ²_k ; T avec les π_k exacts (infini si q_k = 0, signalé). Ces estimations héritent de la dégénérescence des poids.

## Contrôle préalable (`check`)
Sur 500 trajectoires simulées sur CPU en float64 pour (C) et (D) : log w̃ recalculé depuis les transitions discrètes enregistrées (−E(X_T) − log N(X_T ; 0, V I) + Σ_k [log N(x_{k+1} ; x_k, v_k) − log N(x_{k+1} ; x_k + m_k, v_k)]) égal à `girsanov.log_path_weights` (même graine) à **10⁻⁶ près** ; sinon, arrêt et rapport.

## Mes prédictions (sans effet)
I1 vraie 80 % (la seule raison de fausseté serait un ESS mal estimé sur des chaînes bloquées) ; I2 vraie 85 % ; I3 vraie 100 % (par construction) ; I4 vraie 55 % (le seuil 0,020 est proche du bruit attendu en (A), ESS de l'ordre de 0,2 sur 18 000 états). Rapport attendu : ESS_IMH / ESS_IS entre 0,3 et 1 ; en (B), part des masses faible (< 5 %) dans E_q[w²], dégénérescence intra-bassin (étape A) ; en (C) et (D), part des masses plus grande.

## Exécution
`scripts/imh_path.py check|A|B|C|D|verdict`. Résultats `figures/imh_path/`, section dans RESULTS.md, ligne dans `grille_pistes.md`. **Phase 2 (pCN dans l'espace des bruits) : préparée seulement, pas lancée** (`rules/rules_pcn_path.md`).
