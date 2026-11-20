# Règle pré-enregistrée : choix de sigma_max sans vérité (log Ẑ de l'IS de chemin), validation sur la graine 1

Écrite et commitée AVANT toute location.

## Origine de la règle
Ce critère a été FORMULÉ À PARTIR DES RÉSULTATS DE LA GRAINE 0 : la règle d'origine (addendum 4 de `predictions_dk.md`, commit d7170f6) a réussi 3 cellules sur 3 mais grâce au départage par l'ESS (log Ẑ seul aurait échoué dans 2 cellules sur 3, à cause de runs effondrés à 1/8 dont log Ẑ est sans biais mais de variance énorme). Le filtre sur k̂ et la condition sur le TV après IS ci-dessous sont ajoutés à partir de ces résultats. Le test sur la graine 1 (données neuves : aucun run de graine 1 de ces cellules n'a été vu) en est la VALIDATION. Pour mémoire, appliqué après coup à la graine 0 (descriptif, non indépendant), le critère reformulé choisit × 0,5 dans les 4 cellules, comme la règle d'origine ; « max ESS seul » choisit × 0,5 dans 3 cellules et × 0,25 (1/8) dans (32, 8, 8).

## Critère
Pour chaque cellule (d, K, Δ) et ses 6 multiplicateurs de sigma_max (× 0,25 ; 0,5 ; 0,75 ; 1 ; 1,5 ; 2), graine 1 :
1. Retenir les runs dont l'IS de chemin a k̂ < 0,7.
2. Parmi eux, argmax de log Ẑ = logsumexp(log w) − log N (N = 20 000 trajectoires), SE par la méthode delta ; tous les runs retenus à moins de 2 SE du maximum (SE de la différence = √(SE_i² + SE_max²)) sont à égalité ; départage par le plus grand ESS.
3. Si aucun run n'a k̂ < 0,7 : choisir le plus grand ESS et le signaler.

## Succès (validation)
Dans chaque cellule où au moins un sigma_max couvre 8/8 en graine 1 (couverture p̂_k ≥ 0,25 p_k sur la moyenne des 10 dernières évaluations, comme avant) : le sigma_max choisi couvre 8/8 ET son TV après IS de chemin est inférieur à 2 × le plancher (plancher = médiane du TV d'un échantillonneur exact à 5000 échantillons, `gmm_dk.tv_floor`, ≈ 0,014 pour K = 8 : seuil ≈ 0,028). Verdict « critère validé » si toutes ces cellules réussissent, « non validé » sinon (pas de relâchement) ; si aucune cellule n'a de run à 8/8, pas de verdict.

## Rapporté sans règle
Pour chaque cellule : ce que la règle d'origine (d7170f6, sans filtre k̂) aurait choisi ; ce que « max ESS seul » aurait choisi ; les ratios p̂/p après IS de chemin au sigma_max choisi ; le tableau complet (couverture, log Ẑ ± SE, ESS, k̂, TV avant et après IS) pour les 24 runs ; log Ẑ comparé au log de la masse cible couverte.

## Lot
Les 24 configurations (4 cellules : (32, 8, 4), (32, 8, 6), (32, 8, 8), (8, 8, 8) × 6 multiplicateurs de la règle σ = 2√(1 + Δ²/(2d))) en graine 1, protocole identique à la graine 0 (1500 époques, nfe 200, options rapides, évaluation toutes les 50 époques sur 5000 échantillons) ; puis IS de chemin (20 000 trajectoires) sur chaque contrôleur final, avec sauvegarde des log w et des proportions pondérées par mode. Au plus 7 runs en parallèle. Offre ≤ 0,14 $/h facturé ; si l'instance n'ouvre pas SSH en 5 minutes : la détruire et en prendre une autre. Plafond 0,10 $ (arrêt et demande au-delà).
