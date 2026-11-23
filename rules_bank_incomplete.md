# Banque incomplète : un mode lointain, de forme différente, absent de la banque initiale (jouet CPU)

Version **validée avec modifications par l'utilisateur** (2026-11-23) de la proposition commitée en 28f15a4. Écrit et commité AVANT toute exécution ; aucune règle ne sera modifiée après coup. Aucun GPU, aucune location, aucun entraînement : un sampler synthétique fixe, des rafraîchissements simulés. Chaque exécution < 1 min (chronométrée) ; le calcul est découpé en appels, et toute réduction sera dite.

**Statut déclaré d'avance** : c'est une **vérification de plomberie** (enrichissement, banque, saut entre formes différentes) et du darting entre formes différentes. **Ce n'est pas une preuve d'enrichissement** : le sampler est fixe, donc il n'y a pas de boucle d'apprentissage (la correction n'influence pas les rafraîchissements suivants).
**Métrique de masse principale : TV par bassin** (convention de l'utilisateur) ; Voronoï en secondaire.

## Question
Si un mode est trop lointain pour la loi de base, l'enrichissement de la banque par trempe des nouveaux points du sampler le trouve-t-il, à partir de quelle masse ε du sampler, et le saut entre formes différentes (translation ou affine avec jacobien) transfère-t-il la bonne masse ?

## Jouet
- **Cible π** : 9 gaussiennes en d = 8.
  - 8 modes proches : ceux de GMM-dk (8, 8, 8) (covariance I, axes, distance mutuelle 8), poids géométriques de GMM-dk mis à l'échelle de 1 − w_far.
  - **Mode lointain, de forme différente** : moyenne à la distance **R = 40** de l'origine, direction (1, …, 1)/√8 ; **covariance diagonale** de valeurs propres **4** sur les axes 0 à 3 et **0,5** sur les axes 4 à 7 ; poids **w_far = 0,05**.
- **Loi de base** : N(0, ν₁ I), ν₁ = 8.
- **Banque initiale, par graine s ∈ {1, …, 20}** : N_q = 1000 départs de la base (graine s), trempe convergée de `basin_lib.quench`, fusion à 0,5.
  - **Avant tout**, on rapporte **la composition de la banque initiale de chaque graine** : nombre de minima, lesquels des 8 proches, présence du mode lointain, minima spurieux (à plus de 0,5 d'un vrai minimum).
  - **Toute graine dont la trempe depuis la base trouve le mode lointain est exclue et comptée.** Les règles s'appliquent aux graines restantes (n_s).
- **Sampler synthétique fixe** : à chaque rafraîchissement r, 512 points tirés du mélange à 9 composantes de mêmes formes que π (composante par tirage multinomial), avec des masses fausses : modes proches à poids égaux (1 − ε)/8, mode lointain à ε. **ε ∈ {0 ; 10⁻⁴ ; 3·10⁻⁴ ; 10⁻³ ; 3·10⁻³ ; 10⁻²}.** Tirages déterministes par (s, ε, r).
- **Rafraîchissement r = 1 à 40** : trempe des 512 nouveaux points bruts, ajout des minima distincts nouveaux (fusion à 0,5, au plus 64) ; puis darting de ces 512 points (S = 10 balayages, protocole de `dart_lib.darting` avec `minima=` la banque courante, graine 1000 + r).
- **Deux bras de darting**, avec la même banque et les mêmes points :
  - **translation** : x ↦ x − m_i + m_j (option `force_translation` de `dart_lib`, jacobien 1) ;
  - **affine** : x ↦ m_j + A_ij (x − m_i), avec A_ij = H_j^{−1/2} H_i^{1/2} (Hessiennes de l'énergie aux minima de la banque, racines carrées symétriques), acceptation avec le jacobien |A_ij| = (det H_i / det H_j)^{1/2}. C'est l'implémentation existante de `dart_lib` (cas « general Hessian map »). **Différence avec la formule de l'utilisateur** (m_j + L_j L_i⁻¹ (x − m_i), L = Cholesky de la covariance) : la racine symétrique remplace Cholesky ; les deux sont affines, de même jacobien, et valides.
- **Graines d'échantillonnage** : les mêmes 20 graines (une graine s fixe la banque initiale et les tirages).

## Mesures
- **r_found** : premier r où la banque contient le minimum lointain (à moins de 0,5) ; sinon « jamais ». Indépendant du bras de darting (le sampler est fixe).
- **p** : probabilité que la trempe d'un point du mode lointain converge à son minimum (5000 tirages exacts de la composante lointaine, mesuré une fois).
- **Banque** : taille à chaque r et minima spurieux.
- **Masse par bassin du mode lointain**, **TV par bassin** (principale, 9 bassins) et TV par Voronoï, après darting, sur des points regroupés :
  - **fenêtre post-découverte** : les rafraîchissements r_found à r_found + 9 (si r_found ≤ 31) ;
  - **fenêtre finale** : les rafraîchissements 31 à 40 ;
  chacune de 5120 points, **par bras** ; la banque utilisée pour chaque lot est celle du rafraîchissement du lot.
- **Plancher** de TV par bassin à n = 5120 (médiane de 100 tirages exacts).

## Règles
- **B1 (ε = 10⁻², par bras séparément)** : le mode lointain est découvert à **r ≤ 5 dans au moins 95 % des graines retenues (19/20 si aucune n'est exclue)** ET la **médiane, sur les graines retenues, de la masse par bassin du mode lointain dans la fenêtre post-découverte est dans [0,04 ; 0,06]** (vérité 0,05).
  - Précisions déclarées : la médiane remplace « chaque graine » (au nom de la règle de variabilité) ; la fraction de graines dans [0,04 ; 0,06] est rapportée.
- **B2 (ε = 0, limite ; vrai si vrai pour les deux bras)** : le mode lointain n'entre dans la banque dans **aucune graine** en 40 rafraîchissements, ET la **TV par bassin médiane** (fenêtre finale) est ≥ 0,8 × w_far = **0,04**.
- **B3 (ε ∈ {10⁻⁴, 3·10⁻⁴, 10⁻³, 3·10⁻³, 10⁻²}, descriptif)** : courbe de découverte (fraction des graines retenues avec r_found ≤ r, pour r = 1 à 40) contre **1 − exp(−512 r ε p)**, p mesuré ; écart maximal ; médiane de r_found ; TV par bassin de la fenêtre finale, par bras. Aucun seuil.
- **B4 (sûreté, tous ε, par bras)** :
  - (i) aucun minimum spurieux dans la banque, et taille ≤ 9 ;
  - (ii) pour ε = 10⁻², la TV par bassin restreinte aux 8 modes proches (masses renormalisées sur les 8) de la fenêtre finale, avec la banque courante, n'est pas pire de plus de 0,01 que celle des **mêmes points** darting avec la banque à 8 minima proches seule. (Précision déclarée : « banque complète contre banque à 8 minima » est comparée sur la fenêtre finale, et non à r = 1, un lot de 512 étant trop bruité.)
- **Variabilité** : médianes et quantiles sur les graines retenues (entre graines de sampler).

## Lecture fixée d'avance
- B1 et B2 vrais pour un bras : l'enrichissement trouve un mode lointain tant que le sampler en porte une masse non nulle (à la vitesse prévue par B3), et le saut de ce bras transfère la bonne masse entre formes différentes ; sans masse, rien n'est trouvé. Si B1 est vrai pour l'affine seul et faux pour la translation, le jacobien est nécessaire entre formes différentes.
- B1 faux pour les deux bras : soit la découverte, soit le transfert échoue ; on dit lequel d'après r_found et la masse mesurée.
- B2 faux (mode trouvé avec ε = 0) : le jouet fuit ou le support n'est pas nul ; test invalide, on le dit.
- **Dans tous les cas** : plomberie seulement, pas preuve d'enrichissement dans l'entraînement.

## Mes prédictions (sans effet)
- Découverte (partie de B1) : 90 %. Masse dans [0,04 ; 0,06] : affine 65 %, translation 25 % (la translation entre une forme unité et une covariance diag(4, 0,5) a une acceptation de saut basse dans un sens). B1 vrai : affine 60 %, translation 25 %.
- B2 : 95 %. B4 : 80 %.
- B3 : p ≈ 1 ; la courbe suit la prédiction à moins de 0,15 d'écart maximal : 80 %.

## Exécution
`scripts/toy_bank_incomplete.py` (commandes `init`, `disc EPS`, `dart EPS ARM [WINDOW]`, `verdict`). Résultats dans `figures/bank_incomplete/`. Section dans RESULTS.md.
