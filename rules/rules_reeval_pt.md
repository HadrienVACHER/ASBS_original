# Règles pré-enregistrées : réévaluation des règles DW-4 (Barker, continuation) contre p_PT

Écrites et commitées sur la branche `div` AVANT toute lecture des historiques W&B de cette réévaluation (seules les CLÉS du résumé d'un run ont été listées avant : `state/0..10`, `basin_tv`, etc., aucune valeur). Les anciens verdicts de RESULTS.md restent écrits tels quels ; une section « Réévaluation contre p_PT » est ajoutée à la fin de RESULTS.md. Pas de GPU, pas d'entraînement, pas de darting en entraînement.

## Les règles appliquées littéralement
`scripts/dw4_continuation_rule.py` et `scripts/dw4_barker_rule.py` : seuils, fenêtres, SE, graines et nombres de rééchantillonnages inchangés ; la seule chose remplacée est la VÉRITÉ (la référence `test_split_DW4.npy`) par p_PT, la distribution à T = 1 du parallel tempering (`results/pt_samples_T1.pt`, 64 chaînes x 1600 états, 102 400 états ; valide, rapport du commit b4e35e5).

## Précision à déclarer : ESPACE de la vérité (deux variantes, toutes deux pré-enregistrées)
Les règles écrites calculent leurs statistiques sur des proportions de BASSINS (états après quench, 11 classes) : la règle de continuation lit les colonnes W&B `basin/<k>` et compare à la référence quenchée ; la règle Barker quenche les échantillons finaux. Les historiques contiennent aussi les proportions des états BRUTS (`state/<k>`, graphe des paires courtes sur les coordonnées non quenchées, 11 classes). Pour ne comparer que des quantités du même espace :
- **Variante R (espace brut, ajustement demandé : primaire).** Vérité p_PT^raw = fréquences des 11 classes obtenues en appliquant `dw4_states.state_ids` aux états bruts de PT à T = 1 (sans quench). Continuation : mêmes formules avec les colonnes `state/<k>` à la place de `basin/<k>`. Barker primaire : TV entre les fréquences brutes `state_ids(x)` des 5000 échantillons finaux (sans quench) et p_PT^raw.
- **Variante Q (espace des bassins, la lettre des règles).** Vérité p_PT^basin = fréquences des 11 classes des états de PT à T = 1 après le quench du dépôt (ids sauvegardés dans `pt_samples_T1.pt`). Continuation : colonnes `basin/<k>` ; Barker primaire : échantillons finaux quenchés.
- Les deux variantes sont rapportées côte à côte ; aucune règle ne les combine. En cas de désaccord, les deux sont rapportées. La variante R est celle qui sert au verdict principal demandé ; Q est le contrôle de cohérence avec la lettre des règles.

## Priorité
En cas de désaccord entre les deux règles, **la continuation (pairée, moyennée dans le temps) fait foi** : la règle Barker comparait des instantanés (erreur de conception déjà déclarée).

## Continuation (règle littérale)
Runs `dw4_cont_C1`, `C2`, `G3`, `G10`, `G30` du projet `CCDS_NTU/as`, commit `aadf33b`, évaluations toutes les 10 epochs, 20 dernières évaluations (epochs 300 à 490). S = somme sur les 11 classes de |pbar_b - vérité_b|, pbar_b = moyenne des 20 dernières évaluations ; SE = écart-type par bootstrap des 20 évaluations (1000 rééchantillonnages, graine `np.random.default_rng(0)` comme le script, un même flux de nombres aléatoires parcourant les bras dans l'ordre C1, C2, G3, G10, G30), on prend le plus grand SE des cinq bras ; bruit = |S_C1 - S_C2| + 2 SE_max ; effet à gamma <=> S_gamma < min(S_C1, S_C2) - bruit ; méthode vivante <=> (effet à gamma = 10 OU 30) ET (S_3 + bruit >= S_10 ET S_10 + bruit >= S_30) ; G30 instable (loss non finie ou eq_w2 > 1.5 x contrôles) => rapporté au lieu de S.

## Barker (règle littérale)
Échantillons finaux (`model_samples`, 5000 premiers) du run Barker e1500 (commit 6a955dd, `dw4_as_barker_e1500/seed0/samples_final.pt`) et des deux runs AS (`dw4_e1500_ed8bb67_samples.pt`, `dw4_e1500_94252_samples.pt`) ; PRIMAIRE : TV_CV < min(TV_AS) - 2 SE_comb avec SE par bootstrap des échantillons (200, graine 0), la vérité remplacée par p_PT ; MÉCANISTE : n'utilise aucune vérité (statistique RB sur les échantillons), inchangé, donc son verdict antérieur n'est pas recalculé ici ; DIRECTION : (a) littérale, avec les listes sur-pondérés / sous-pondérés en dur du script (elles viennent de l'ancienne référence : le script ne les calcule pas avec la vérité), inchangée ; (b) variante dérivée de la vérité, déclarée ici : parmi les classes de vérité >= 0.005, « sur-pondéré » = proportion moyenne des deux runs AS > vérité, « sous-pondéré » = <, et le CV doit faire diminuer les premières et augmenter les secondes par rapport à la moyenne des runs AS ; SANITÉ : eq_w2 / energy_w2, inchangée (pas de vérité).

## Rapport seulement (sans règle)
- Différence pairée Barker - AS par état (continuation : G3, G10, G30 contre C1, même graine), moyenne sur les 20 dernières évaluations, SE par bootstrap sur les évaluations (1000, indices pairés), pour les classes de vérité >= 0.005, dans les deux variantes ; pour chaque état, si le signe va dans le sens de p_PT - p_C1.
- Les proportions de bassins des splits train et val DW-4 de `jarridrb/DEM/data` (s'ils existent), quenchés avec le quench du dépôt, comparées à p_PT : z par bassin (SE binomial des échantillons supposés indépendants, SE de p_PT par chaînes), rapport seulement.

## Écarts à rapporter
Tout écart au protocole ci-dessus sera dit dans le rapport et dans la section de RESULTS.md.
