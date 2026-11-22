# Test GPU : Newton-Stein contre amortissement contre AS (GMM-dk, d = 32)

Écrit et commité AVANT la location et avant tout résultat. Aucune règle n'est modifiée après coup ; tout écart serait un amendement daté. Contexte : `memo_cv_failure_modes.md` (§4 et §7), test modifié par l'utilisateur le 2026-11-22 : un bras « amortissement » est ajouté, pour savoir si Newton-Stein apporte plus qu'un amortissement quelconque.

## 1. Cellules, bras, recette
- **Cellules** (graine 0, 1500 époques, recette d'origine `gmm_dk_as` : FourierMLP 4 × 128, lr 1e-4, buffer 10 000, 100 itérations par époque, nfe 200, options rapides, évaluation tous les 50 époques sur 5000 échantillons) :
  - (32, 8, 4), σ_max × {1 ; 2} = 2,236068 et 4,472136. Question : stabilité et IS.
  - (32, 8, 8), σ_max × {1 ; 2} = 2,828427 et 5,656854. Question : un grand σ_max devient-il utilisable, et trouve-t-il les 8 modes ?
- **Bras** (seul le matcher change ; `configs/experiment/gmm_dk_as_{damped,newton_stein}.yaml` héritent de `gmm_dk_as`) :
  - (a) **AS seul** (`AdjointVEMatcher`).
  - (b) **Amortissement** η = 0,5 (éq. 176 de *Failure Modes*, Blessing et al. 2026) : cible (1 − η) u_{k−1}(X_t, t) + η Y, avec u_{k−1} = le contrôleur **gelé au début de chaque époque** (une époque = une itération externe d'AS : nouveaux points terminaux du contrôleur courant, puis les pas de régression). `AdjointVEDampedMatcher`.
  - (c) **Newton-Stein**, Ĥ = I : Ŷ = Λ_t Y + (I − Λ_t) Y_BM, Λ_t = 1/(ν_{1|t} + κ_t) scalaire, ne dépendant que de t, donc exactement de moyenne nulle sous π̃ ; écrit sans singularité : Ŷ = Y/(ν_{1|t} + κ_t) + (1 − 1/ν₁)(X₁ − X_t)/(ν_{1|t} + κ_t). `AdjointVENewtonSteinMatcher`.
- **Tests (CPU, 1,4 s, avant cette règle)** : `tests/test_newton_stein_matcher.py` : T1 Ĥ = 1/ν₁ redonne exactement la cible d'AS (écart relatif 7e−8) ; T2 identité algébrique (4e−7) ; T3 moyenne nulle sous π exacte, (32, 8, 4), 2·10⁵ tirages, 192 statistiques E[(Ŷ − Y)_j f(t, X_t)] : max |z| = 2,66 (seuil 4,5) ; T4 contrôle positif (loi décalée de 0,3) : max |z| = 51,8 ; T5 cible amortie et copie gelée correctes. Essais d'entraînement de 3 époques des deux bras sur CPU : sans erreur.

## 2. Réutilisation des runs AS (bras (a))
Des runs AS identiques (même recette, même graine, mêmes σ_max, 1500 époques) existent : ils sont **réutilisés, pas relancés** :
- (32, 8, 4) × 1 : `results/gmm_dk_sweep/gmmdk_d32_K8_D4_s0` (balayage, commit 788a0e8) ; × 2 : `results/gmm_dk_window/w1/gmmdk_d32_K8_D4_s0_sx2` (fenêtre, code f581ea6).
- (32, 8, 8) × 1 : `results/gmm_dk_sweep/gmmdk_d32_K8_D8_s0` ; × 2 : `results/gmm_dk_window/w1/gmmdk_d32_K8_D8_s0_sx2`.
- La voie AS du code n'a pas changé depuis ces runs (diff f581ea6..HEAD sur `adjoint_samplers/` et `train.py` purement additif : nouvelles classes, sauvegardes optionnelles). Un nouveau run ne serait de toute façon pas identique bit à bit (non-déterminisme GPU).
- **Limites** : ces runs n'ont que le contrôleur final (époque 1500) ; leurs sauvegardes intermédiaires (250 à 1250) n'existent pas. Les dumps d'évaluation (tous les 50 époques) ne sont disponibles localement que pour les deux runs × 1, aux époques 1000 à 1450. R1 et R2 portent donc sur l'époque 1500 pour les trois bras ; R3 n'est évaluable pour (a) qu'à × 1.
- **8 runs nouveaux** : (b) et (c) dans les 4 réglages. Contrôleurs sauvegardés aux époques 250, 500, 750, 1000, 1250, 1500 ; dumps d'évaluation tous les 50 époques.

## 3. Évaluation (`scripts/ns_damping_eval.py`, identique pour les trois bras, sur GPU)
Pour chaque contrôleur sauvegardé (les 4 contrôleurs AS réutilisés : époque 1500 seulement) : 20 000 trajectoires (graine 0) → log w (sauvegardés), ESS normalisé de l'IS de chemin, k̂, log Ẑ ± SE ; **filtre Q de l'étape 4** sur les 10 000 premiers points terminaux : quench convergée (‖∇ log π‖ < 10⁻³, pas 0,7 monotone, 3000 itérations au plus, bassin = minimum à moins de 0,5), **Q1** : médiane de la distance au minimum le plus proche ≤ 1,5 × la même médiane sous π (10⁴ tirages exacts, graine 2026) ; **Q2** : ≥ 95 % des échantillons convergent et 8/8 bassins avec p̂ ≥ 0,25 p (p : proportions exactes de Voronoï). **Q = Q1 ET Q2.** Erreurs de moyenne par mode (cellules de Voronoï) sur chaque contrôleur et sur chaque dump d'évaluation.

## 4. Règles pré-enregistrées (sur les contrôleurs de l'époque 1500 ; `scripts/ns_damping_rules.py`)
- **R1 (32, 8, 4)**, parmi les runs qui passent Q : ESS (c) ≥ 2 × ESS (a) ET ESS (c) ≥ 1,5 × ESS (b), **à × 1 ET à × 2**. Précisions : (c) doit passer Q aux deux σ_max (sinon R1 est faux) ; si (a) ou (b) ne passe pas Q alors que (c) le passe, la condition contre ce bras est tenue pour vraie ; un run manquant ou non évalué rend R1 non évaluable, compté comme non vrai pour la décision.
- **R2 (32, 8, 8)** : (c) passe Q à au moins un σ_max ∈ {× 1 ; × 2}, alors que (a) et (b) ne le passent à aucun (non évaluable si un des 6 runs manque, compté comme non vrai).
- **R3 (descriptif)** : les moyennes par mode oscillent-elles d'une sauvegarde à l'autre pour (a), et pas pour (b) ou (c) ? Mesure : sur les dumps d'évaluation des époques 1000 à 1450 (5000 échantillons, tous les 50 époques ; seule série disponible pour (a), à × 1) et, pour (b) et (c), aussi sur les sauvegardes 250 à 1500 (20 000 trajectoires). Pour chaque mode avec n_k ≥ 50 : erreur e_k = moyenne de la cellule − µ_k, signal z_k = n_k ‖e_k‖²/d (≈ 1 si l'erreur n'est que du bruit Monte-Carlo) ; paires consécutives où z ≥ 4 aux deux dates ; O = fraction de ces paires où cos(e_k(E), e_k(E')) < 0. Étiquette : « signal trop faible » si moins de 4 paires ; « oscille » si O ≥ 0,5 ; « n'oscille pas » si O < 0,25 ; sinon « intermédiaire ». Médiane de z rapportée. Sans effet sur la décision.
- **Décision écrite à l'avance** : Newton-Stein ne vaut la peine d'être poursuivi que si **R1 ou R2 est vrai**. Si seul l'avantage sur AS (a) est observé, il n'apporte rien de plus que l'amortissement : **on arrête**.

## 5. Mes prédictions (écrites avant ; sans effet sur la décision)
- Q passé à l'époque 1500 dans (32, 8, 4) par les trois bras aux deux σ_max : 75 %.
- ESS (32, 8, 4) : (a) connu (0,25 à × 1, 0,011 à × 2 ; réévalué ici avec le même script) ; (b) 0,2–0,5 à × 1 et 0,02–0,15 à × 2 ; (c) 0,2–0,6 à × 1 et 0,03–0,3 à × 2.
- R1 vrai : 20 % (il faut battre l'amortissement d'un facteur 1,5 aux deux σ_max). R2 vrai : 15 % (Théorème C : Newton-Stein n'aide pas la découverte ; seule une perte de modes causée par l'instabilité serait corrigée). Décision « on arrête » : 70 %.
- R3 : pour (a) à × 1, « signal trop faible » le plus probable (erreurs par mode au niveau du bruit à l'époque 3 de l'essai local avec ce contrôleur : z médian 1,7).

## 6. Coût, plafond et arrêts (accord explicite : RTX 3070 ≈ 0,134 $/h, si l'estimation est ≤ 0,15 $ ; plafond dur 0,15 $)
- Offre : RTX 3070, ≤ 0,14 $/h, fiabilité > 0,99, ≥ 8 cœurs CPU effectifs. Sans telle offre : pas de location, rapport.
- Estimation (fenêtre σ_max : 7 runs en parallèle à 0,42–0,50 s par époque ; ici 8 runs + 2 workers d'évaluation) : démarrage, envoi et installation 6 min ; entraînement 1500 époques à 0,6 s (15 min ; **pessimiste 1,0 s : 25 min**) ; évaluations pendant l'entraînement, les dernières ≈ 3 min après ; règles, rapatriement et destruction 3 min. **Total ≈ 27 min ≈ 0,06 $ (central) ; ≈ 40 min ≈ 0,09 $ (pessimiste)**, sous 0,15 $ : location autorisée, aucune cellule retirée.
- **Aucune nouvelle évaluation après la minute 48** depuis la création de l'instance ; **arrêt d'urgence à la minute 56** : tout est arrêté, les résultats terminés sont rapatriés, l'instance est détruite, puis l'utilisateur est prévenu ; durée totale création → destruction ≤ 60 min (≤ 0,14 $). Jamais de question à l'utilisateur pendant que l'instance tourne ; destruction dès les résultats rapatriés.
- Si l'entraînement n'est pas fini à la minute 48, les contrôleurs déjà sauvegardés sont évalués ; R1 et R2 sont alors non évaluables pour les runs sans époque 1500, et le rapport dit ce qui manque.

## 7. Rapport
Par cellule, σ_max et bras : Q (Q1, Q2, couverture), ESS, k̂, log Ẑ, TV brut à l'époque 1500 ; les mêmes grandeurs aux sauvegardes intermédiaires de (b) et (c) (descriptif) ; R1, R2, R3 ; décision ; coût réel ; réserves (une graine, AS réutilisé sans sauvegardes intermédiaires, ESS estimé sur 20 000 trajectoires).
