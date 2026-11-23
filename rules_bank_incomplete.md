# Banque incomplète : un mode lointain absent de la banque initiale (jouet CPU)

**PROPOSITION, non exécutée.** Écrit pour validation par l'utilisateur ; commité avant toute exécution. Aucun GPU, aucune location, aucun entraînement : un sampler synthétique fixe, des rafraîchissements simulés. Chaque exécution < 1 min (chronométrée). Aucune règle ne sera modifiée après coup ; toute correction demandée avant l'exécution est un amendement daté de ce fichier.

**Convention de métrique** (consigne de l'utilisateur du 2026-11-23) : la métrique de masse principale est la **TV par bassin** ; la TV de Voronoï est secondaire.

## Question
Dans l'auto-entraînement (s), la banque de départ était complète (trempe depuis la loi de base). Si un mode est trop lointain pour la base, **l'enrichissement par trempe des nouveaux points du sampler suffit-il à le trouver, et à partir de quelle masse du sampler ?** L'enrichissement n'a jamais ajouté un minimum dans les runs déjà faits.

## Jouet
- **Cible π** : 9 gaussiennes unité en d = 8. Les 8 premières sont les modes de GMM-dk (8, 8, 8) (axes, distance mutuelle 8) avec les poids géométriques de GMM-dk, mis à l'échelle de 1 − w_far. Le **mode lointain** est à la distance **R = 40** de l'origine, sur un axe qui n'est pas celui des 8 modes (direction (1, …, 1)/√8 ; distance ≥ 38 aux autres modes), de poids **w_far = 0,05**.
- **Loi de base** : N(0, ν₁ I) avec ν₁ = 8 (σ_max 2,83, comme (32, 8, 8) × 1). Elle n'atteint pas le mode lointain : la probabilité d'une trempe vers son bassin est inférieure à 10⁻¹⁰.
- **Banque initiale** : N_q = 1000 départs de la loi de base, trempe convergée de `basin_lib.quench`, fusion à 0,5. Attendu : 8 minima.
- **Sampler synthétique fixe** (il remplace AS) : à chaque rafraîchissement, 512 points tirés du mélange à 9 composantes de mêmes formes que π, avec des masses fausses : les 8 modes proches à poids égaux (1 − ε)/8 et le mode lointain à **ε**. Quatre valeurs : **ε ∈ {0 ; 10⁻⁴ ; 10⁻³ ; 10⁻²}**.
- **Rafraîchissement r** (r = 1 à 40, comme les 500 époques de l'entraînement réel à environ 12 lots près) : trempe des 512 nouveaux points, ajout des minima nouveaux distincts (fusion à 0,5, au plus 64), puis darting de ces 512 points (S = 10, protocole de `dart_lib.darting` avec `minima=` la banque courante, graine = 1000 + r).
- **20 graines** de sampler par valeur de ε (graines 1 à 20).

## Mesures
- **r_found** : le premier rafraîchissement où la banque contient le minimum lointain (à moins de 0,5), par graine ; sinon « jamais » dans les 40.
- **Taille de la banque** à chaque r (spurieux : tout minimum à plus de 0,5 d'un vrai minimum).
- **TV par bassin** (principale) et par Voronoï (secondaire) contre la vérité, des **points corrigés des 10 derniers rafraîchissements regroupés** (n = 5120), par graine ; plancher à n = 5120 (médiane de 100 tirages exacts) ; couverture (p̂ ≥ 0,25 p).
- **Masse du mode lointain** avant et après darting, sur ces mêmes points.

## Règles
- **B1 (ε = 10⁻²)** : le mode lointain est dans la banque à r ≤ 5 dans **au moins 19 graines sur 20**, ET, à r = 40, la TV par bassin médiane (sur les graines) des points corrigés est ≤ 2 × plancher, ET la couverture du mode lointain est vraie dans toutes ces graines.
- **B2 (ε = 0, limite)** : le mode lointain n'entre dans la banque dans **aucune graine** en 40 rafraîchissements, ET la TV par bassin médiane est ≥ 0,8 × w_far = 0,04 (la masse du mode absent ne peut pas être transférée). C'est la limite attendue du support (théorème C ; ligne 2 de la grille).
- **B3 (ε = 10⁻³, 10⁻⁴, descriptif)** : fraction des graines où le mode est trouvé en 40 rafraîchissements et médiane de r_found, contre la prédiction 1 − exp(−512 × r × ε) (la trempe d'un point du mode lointain converge dans son bassin, pas de perte), et TV par bassin à r = 40. Aucun seuil.
- **B4 (sûreté, tous ε)** : la banque ne contient jamais de minimum spurieux (taille ≤ 9, tous à moins de 0,5 d'un vrai minimum), ET le darting avec banque complète ne dégrade pas la TV par bassin des 8 modes proches par rapport à la banque à 8 minima seule à r = 1 (rapporté : sinon, le diagnostic est dit).
- **Variabilité** : toutes les valeurs sont des médianes et des quantiles sur 20 graines de sampler (entre évaluations, règle permanente) ; le mélange de 10 rafraîchissements est déclaré.

## Lecture fixée d'avance
- **B1 et B2 vrais** : l'enrichissement trouve un mode lointain tant que le sampler en porte une masse non nulle, avec la vitesse prévue par ε, et ne peut rien quand la masse est nulle. La banque incomplète est alors un problème de support du sampler, pas de la méthode de darting.
- **B1 faux** : l'enrichissement ne suffit pas même avec 1 % de masse ; la banque externe (trempe depuis une autre loi) devient nécessaire.
- **B2 faux** (le mode est trouvé avec ε = 0) : le jouet fuit (trempe depuis les points proches qui atteint le mode lointain) ; le test est invalide, on le dit.

## Mes prédictions (sans effet)
- B1 vrai : 80 % (r_found médian 1 : environ 5 points du mode lointain par lot). B2 vrai : 95 %. B4 vrai : 85 %.
- B3 : ε = 10⁻³, trouvé dans 100 % des graines, r médian 2 ; ε = 10⁻⁴, trouvé dans environ 87 % des graines, r médian de l'ordre de 14.

## Limites annoncées
Jouet isotrope et à formes connues ; sampler fixe (aucune boucle d'apprentissage, donc aucun effet de la correction sur les rafraîchissements suivants ; c'est justement le cas où l'enrichissement a le moins de chances d'aider) ; une seule distance R et un seul w_far ; la valeur de S (10) ; 20 graines.

## Exécution (après validation)
`scripts/toy_bank_incomplete.py` (commandes `run EPS` pour ε ∈ {0 ; 1e-4 ; 1e-3 ; 1e-2}, découpé par graine si > 1 min, puis `verdict`) ; résultats dans `figures/bank_incomplete/` ; section dans RESULTS.md.
