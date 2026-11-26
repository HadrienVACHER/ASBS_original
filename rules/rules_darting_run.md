# Règle pré-enregistrée : UN run de continuation avec le noyau de darting (DW-4), comparé aux deux runs AS existants

Écrite et commitée AVANT le lancement. Pas de GPU : run local sur CPU (≈ 9 s par époque, ≈ 75 min). C'est un run **exploratoire et descriptif** : un seul run ne permet aucun verdict de méthode (le bruit entre les deux graines AS existantes est |S_C1 − S_C2| = 0,157 ; la règle de continuation de `scripts/dw4_continuation_rule.py` appliquée à ce run serait de toute façon sans pouvoir : seuil d'effet négatif). Rien n'est conclu au-delà de ce qui est écrit ici.

## Le run
Même protocole que les bras de continuation existants (`scripts/dw4_continuation.pbs`, commit aadf33b, runs `dw4_cont_C1` et `dw4_cont_C2` dans le projet W&B `CCDS_NTU/as`) : point de départ commun = contrôleur et buffer (`buffer_x1`) du run AS 94252, `sigma_max=2`, `nfe=100`, 500 époques, évaluation toutes les 10 époques à 5000 échantillons, `evaluator.eq_w2_every=100`, t uniforme, Adam repart de zéro, `seed=1` (comme C1, G3, G10, G30). Seul le matcher change : `experiment=dw4_as_darting_states` (`AdjointVEDW4DartingMatcher`, γ = 3, valeur choisie d'avance, la même que Barker gelé ; non réglée sur DW-4). Écarts à déclarer : exécution sur CPU (pas de `fast_sampling` / `fast_train`, qui sont réservés à CUDA), W&B désactivé (les évaluations sont lues dans la sortie standard du run).
Le noyau est celui de `adjoint_samplers/utils/dw4_darting.py` : validé hors ligne (T0, T2, T3, T1′ dans RESULTS.md) et par `tests/test_dw4_darting.py` et `tests/test_dw4_darting_matcher.py` (espérance nulle exacte sous des états exacts de π : max |z| 1,62 ; contrôle positif 13,4).

## Statistiques (mêmes que la réévaluation contre p_PT, `rules_reeval_pt.md`)
Vérité : p_PT (parallel tempering), variantes R (états bruts, colonnes `state/<k>`) et Q (bassins quenchés, colonnes `basin/<k>`), 11 classes, comme dans `scripts/dw4_reeval_pt.py`. S = Σ_b |p̄_b − p_PT,b| avec p̄_b moyenne des 20 dernières évaluations (époques 300 à 490).
Valeurs des deux bras AS existants (déjà écrites dans RESULTS.md) : variante R : S_C1 = 0,3672, S_C2 = 0,2099 ; variante Q : S_C1 = 0,3632, S_C2 = 0,2029.

## Lecture DESCRIPTIVE fixée d'avance (aucun test d'hypothèse)
1. S_darting par rapport aux deux valeurs AS : « sous les deux » (S_D < min(S_C1, S_C2)), « entre » ou « au-dessus des deux », pour chaque variante.
2. Pour chacun des trois petits bassins (deux paires disjointes, K4 moins une arête, étoile) et pour les deux grands (triangle + pendant, chemin de 2) : l'erreur |log(p̄ / p_PT)| du run darting contre celles de C1 et C2 (mêmes évaluations) ; nombre de bassins où le darting est meilleur que les deux AS.
3. Sanité (rapportée) : perte finie, energy_w2 et eq_w2 des deux dernières évaluations à moins de 15 % de ceux de C1/C2 (valeurs lues dans W&B), `dart/kappa[...]/term_over_gradg` et `outflux` (statistiques du matcher), nombre de propositions valides par échantillon.
4. La taille du bruit est rappelée à côté : |S_C1 − S_C2| (0,157 en R, 0,160 en Q) : une différence de S entre le run darting et un run AS plus petite que cela n'est pas interprétable.

## Ce qui ne sera PAS fait
Pas de règle d'effet, pas de comparaison à un seuil, pas de graines supplémentaires, pas de réglage de γ après coup. Une éventuelle suite (4 + 4 graines, règle pré-enregistrée) est une décision séparée.
