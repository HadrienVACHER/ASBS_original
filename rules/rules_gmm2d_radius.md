# Règle pré-enregistrée : AS seul sur GMM 2D à 3 modes, balayage du rayon r

Écrite et commitée AVANT tout lancement. Exploratoire et DESCRIPTIF : aucun test d'hypothèse, aucun seuil.

## Cible
`configs/experiment/gmm_as.yaml` inchangé, `dim=2 K=3 layout=spread` (triangle équilatéral de rayon r), gaussiennes de variance 1, poids log-uniformes (rapport 20), `target_seed=0`. Rayons r ∈ {2, 4, 6} (distance entre modes ≈ 1,7 r = 3,5 / 6,9 / 10,4). `sigma_max` = valeur du résolveur `gmm_std` (≈ 2 × l'écart-type de la cible, règle de gmm_as).

## Protocole
AS seul, sans variable de contrôle. Graine 0, 1000 époques, évaluation toutes les 50 époques, `num_eval_samples=10000` (défaut de gmm), `dump_final=true`. Aucun réglage fait d'après les résultats. Projet W&B `gmm2d_radius` (nouveau). Vast.ai, une instance, 3 runs en parallèle, instance GARDÉE entre les tests (décision de l'utilisateur) ; plafond de série 2 USD, au-delà redemander.

## Lecture (rapport seul)
Par rayon : `mode_ratio/<k>` (poids mesuré / poids cible) moyenné sur les 10 dernières évaluations, `mode_tv`, `modes_covered`, `mass_covered`, `energy_w2`, valeurs `_floor` à côté ; histogramme 2D des échantillons finaux. Pas de seuil d'effet ; une seule graine, donc aucune différence entre rayons n'est « démontrée ».

## Amendement 1 (écrit avant lancement) : poids très inégaux, test de l'aplatissement vers 1/3
Constat du premier lot (poids π = 0,479 / 0,309 / 0,212, rapport 20) : les proportions d'AS se rapprochent de 1/3 quand r augmente (r = 2 : 0,481 / 0,308 / 0,211 ; r = 4 : 0,404 / 0,343 / 0,251 ; r = 6 : 0,352 / 0,371 / 0,275). Hypothèse NON testée : AS aplatit les poids vers l'égalité.
Test : mêmes moyennes (même graine de cible, mêmes directions, vérifié : écart 0,0), mêmes rayons 2, 4, 6, mais poids FIXÉS à 0,80 / 0,15 / 0,05 (`weight_law=fixed weights=[0.8,0.15,0.05]`, option ajoutée à `GMM`, sans effet sur les runs existants). Protocole identique (AS seul, graine 0, 1000 époques, évaluation toutes les 50, 10 000 échantillons), noms `gmm2dw_r<R>`, même projet W&B `gmm2d_radius`, même instance.
Lecture (rapport seul, aucun seuil) : ŵ = ratio × π, moyenne des 10 dernières évaluations, par rayon. Si les ŵ se rapprochent de 1/3 à mesure que r croît, l'hypothèse d'aplatissement est cohérente ; si le mode léger (0,05) disparaît ou si les ŵ restent près de π, elle ne l'est pas. Une seule graine : rien n'est « démontré ».
