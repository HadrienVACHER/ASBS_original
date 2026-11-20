# Règle pré-enregistrée : test décisif de la forme 1 (AS entraîné AVEC la variable de contrôle darting, sans IS)

Écrite et commitée AVANT toute location. Un seul réglage, pas de balayage.

## Protocole (de l'énoncé)
Départ : contrôleurs de la graine 0 déjà entraînés, sigma_max × 0,5 de la règle, cellules (8,8,8) (sigma_max 2,236) et (32,8,6) (sigma_max 1,25) (formes justes, mauvais poids). Deux continuations par cellule, mêmes hyperparamètres que l'entraînement d'origine (config `gmm_dk_as` : FourierMLP 4 × 128, lr 1e-4, buffer 10 000, 100 itérations par époque, batch 512, nfe 200, options rapides, Adam repart de zéro), même nombre d'époques, même graine (0) :
(a) AS seul (contrôle) ; (b) AS + darting : cible −∇g(X₁) + γ c(X₁, X_t), c = terme de darting (Barker, moyenne sur les M−1 partenaires, divisé par ν_{1|t}), γ = 1 (valeur prédite par la quadrature 1D), à tous les t. 4 runs. Buffer de départ : 10 000 points terminaux du contrôleur de départ (identiques pour (a) et (b)).

## Déclarations d'implémentation (alignement avec les runs darting de DW-4)
- Implémentation : `AdjointVEGMMDartingMatcher` (matcher.py), `adjoint_samplers/utils/gmm_darting.py`, config `gmm_dk_as_darting`. Normalisation alignée sur `AdjointVEDW4DartingMatcher` : cible = −∇g(X₁) + γ h/ν_{1|t}, h = Σ_j ω_j (Y_j − X₁)/(M−1), ω Barker, sans le facteur σ_t que l'énoncé met devant les deux termes (il les multiplie tous les deux, le rapport est inchangé). Différences avec DW-4 : (i) les minima ne sont pas analytiques : ils sont DÉCOUVERTS sur le buffer (quench de 500 échantillons par descente de gradient, fusion à 0,5, minima atteints par ≥ 5 échantillons, Hessiennes par autograd), cartes complètes A_jk = H_j^{-1/2}H_k^{1/2} avec jacobien ; (ii) **les runs darting de DW-4 n'ont aucun rafraîchissement** (noyau analytique, propositions calculées à l'insertion) : l'énoncé demande « le même intervalle que DW-4 », ce qui n'a pas de sens ; je fixe un rafraîchissement tous les 25 époques (appels de `populate_buffer`), avec les minima de départ découverts sur le buffer initial ; (iii) les propositions sont calculées au moment de la régression (d = 32, float64 ; κ, v, ν_{1|t} en float64 par la forme fermée du schéma VE : en float32, ν_t dépasse ν₁ pour t > 0,5 et κ ≥ 1, ce qui donnait une perte NaN au premier essai local, corrigé avant tout run).
- La moyenne nulle de c ne dépend que du fait que les minima/cartes sont fixes entre deux rafraîchissements ; l'auto-dépendance des minima (500 échantillons du buffer) est négligée.

## Test pré-location T1′ (CPU, `tests/test_gmm_darting.py`, minima découverts sur 2000 points d'un buffer simulé du contrôleur (32,8,6) × 0,5, 8 minima sur 8)
Moyenne nulle de c sous π(·|x_t), x₁ ~ π exact : max |z| = 2,7 / 3,2 / 2,5 / 2,7 à κ = 0,1 / 0,3 / 0,5 / 0,7 (n = 2·10⁴) et 2,9 / 2,7 / 2,7 / 2,5 (n = 10⁵), toujours < 4 : RÉUSSI. Contrôle positif (v × 1,2) : max |z| = 15,7 / 7,6 / 5,3 / 2,7 à n = 10⁵ (6,6 / 5,4 / 3,8 / 2,6 à n = 2·10⁴) : détecte à κ ≤ 0,5, pas à κ = 0,7, où le terme est numériquement négligeable (‖terme‖/‖∇g‖ = 2e-4, mean ω = 1,5e-4 ; 8 % à κ = 0,1, 3 % à κ = 0,3, 0,8 % à κ = 0,5, 1e-24 à κ = 0,9) : le test n'a aucune puissance là où le terme est nul.
**Dose à γ = 1 (à connaître avant de louer) :** ‖terme‖/‖cible‖ ≈ 27 % pour κ < 0,1, 6 % pour κ ∈ [0,1 ; 0,7], ≈ 0 au-delà. À titre de comparaison, DW-4 à γ = 3 avait réduit S d'environ moitié en 500 époques. Je m'attends donc à un effet modeste à γ = 1 (probabilité de succès de R_train que je lui donne : 30 %), mais le réglage unique de l'énoncé est conservé.

## Mesures (tous les 10 % de l'entraînement et à la fin, sur 10⁴ échantillons)
Sampler brut, sans poids : TV des proportions (cellules de Voronoï = plus proche moyenne, vérité par échantillonnage exact), couverture (p̂_k ≥ 0,25 p_k), max |log(p̂/p)| sur les modes couverts, plancher TV à n = 10⁴ (médiane d'un sampler exact : 0,0095 environ) ; évaluations aux époques 0, 50, …, 450 (500 époques) et échantillons finaux (10⁴, modèle final). IS de chemin sur le contrôleur final (20 000 trajectoires) : TV après IS, ESS (fraction), k̂. Contrôleurs finaux sauvegardés.

## Règles (de l'énoncé ; quantifications ajoutées)
- **R_train (succès)** : dans les DEUX cellules, à la fin (échantillons finaux du modèle, 10⁴), TV_brut(b) ≤ 0,5 × TV_brut(a) et la couverture de (b) reste 8/8.
- **R_fort (descriptif)** : TV_brut(b) ≤ 3 × plancher (≈ 0,0285), par cellule.
- **R_sûreté** : TV après IS de (b) ≤ 2 × plancher ET ESS(b) ≥ 0,5 × ESS(a), par cellule (IS de chemin, 20 000 trajectoires) ; verdict « vrai » si les deux cellules le vérifient.
- **Vitesse (descriptif)** : TV_brut en fonction de l'époque pour (a) et (b).
- Tout ajout est exploratoire.

## Lot GPU (proposition soumise AVANT toute location ; rien ne se loue sans accord explicite)
500 époques par run, 4 runs en parallèle, évaluation tous les 50 époques ; IS de chemin des 4 contrôleurs finaux sur GPU à la fin ; instance détruite dès les résultats rapatriés. L'offre, la durée, le coût estimé et le plafond (dur) sont indiqués dans la demande à l'utilisateur et reportés ici après son accord.


---
# Amendement 1 (accord explicite de l'utilisateur du 2026-11-20, écrit AVANT la location)

- **Offre 57456544** (RTX 3070, ≈ 0,134 $/h facturé), **plafond DUR 0,10 $** (≈ 44 min d'instance, temps compté depuis l'appel de création, échecs et attentes SSH compris). À l'approche : sauvegarder les résultats terminés, détruire l'instance, puis prévenir ; ne jamais dépasser sans accord explicite. Détruire l'instance dès les résultats rapatriés, sans attendre de réponse. T1′ : lecture acceptée par l'utilisateur (κ = 0,7 n'est pas un échec).
- **Lot : 6 runs de 500 époques** aux cellules (8,8,8) et (32,8,6) : (a) AS seul ; (b30) AS + darting γ = 30 (valeur confirmée sur DW-4, graines 2 à 4) : **bras principal** ; (b1) AS + darting γ = 1 (valeur théorique de la quadrature 1D) : bras secondaire (descriptif : écart théorie / pratique). Mêmes hyperparamètres, même graine 0, mêmes options que l'entraînement d'origine, même buffer de départ pour les 6 runs. **Aucune autre valeur de γ, quel que soit le résultat.**
- R_train, R_fort et R_sûreté (définis plus haut) sont évalués séparément pour b30 et b1 ; **le verdict principal est celui de b30** (R_train : dans les DEUX cellules, TV_brut(b30) ≤ 0,5 × TV_brut(a) et couverture 8/8). b1 est descriptif.
- Descriptif : TV_brut en fonction de l'époque pour (a), (b30), (b1), pour voir si b30 oscille ou plafonne au-dessus du plancher (comme S_Q sur DW-4).
- Tous les t, sans fenêtre (`t_focus = 0` : t uniforme ; γ appliqué à tous les t).
- Dose ‖terme‖/‖cible‖ moyenne par tranche de κ ([0 ; 0,1), [0,1 ; 0,7), [0,7 ; 1]) rapportée à la fin pour b30 et b1 (moyenne des valeurs imprimées aux évaluations des époques 350, 400 et 450).
- **Vérification demandée avant le lancement (faite dans le code) :** à chaque époque, `train_one_epoch` appelle `populate_buffer` M = resample_size // resample_batch_size = 1000 // 512 = 1 fois : 512 NOUVEAUX chemins sont simulés avec le contrôleur COURANT (`adjoint_matcher.sde = ControlledSDE(ref_sde, controller)`, EMA désactivée) et ajoutés au buffer, qui conserve les 10 000 plus récents (renouvellement complet en ≈ 20 époques). C'est la boucle de l'entraînement d'origine ; le matcher darting ne change que la cible de régression (`prepare_target`), pas l'échantillonnage du buffer.
- Estimation : 6 runs en parallèle, ≈ 0,5–0,9 s par époque, 500 époques ≈ 8–12 min d'entraînement, plus installation, essai de compatibilité GPU de 3 époques, IS de chemin des 6 contrôleurs et rapatriement : ≈ 25–30 min, ≈ 0,06–0,07 $ ; arrêt d'urgence (sauvegarde + destruction) déclenché à T + 38 min au plus tard.
