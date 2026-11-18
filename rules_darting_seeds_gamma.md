# Règle pré-enregistrée : darting γ = 3 sur les graines 1 et 2 (appariées à C1 et C2), puis dose-réponse en γ

Écrite et commitée AVANT tout lancement. Exploratoire et DESCRIPTIF : aucun test d'hypothèse, aucun seuil d'effet. Toute suite (graines supplémentaires, γ(κ), instants focalisés) est une décision séparée.

## Protocole commun (identique à C1 / C2, commit aadf33b)
Point de départ : contrôleur et buffer (`buffer_x1`) du run AS 94252 ; `sigma_max=2`, `nfe=100`, 500 époques, évaluation toutes les 10 époques à 5000 échantillons, `evaluator.eq_w2_every=100`, t uniforme, Adam repart de zéro. Matcher `experiment=dw4_as_darting_states` (noyau inchangé, commit de ce fichier). **Cette fois AVEC `fast_sampling=true fast_train=true buffer_on_device=true`**, comme C1 / C2 (le premier run darting ne les avait pas). Exécution sur GPU Vast.ai, W&B activé (projet `as`). Aucun réglage fait d'après les résultats.

## Phase 1 : γ = 3, graines 1 et 2 (2 runs)
Noms : `dw4_cont_D3` (graine 1, graine 2). Comparaison APPARIÉE par graine : D3 graine 1 contre C1, D3 graine 2 contre C2.
Statistiques (celles de `rules_darting_run.md`) : vérité p_PT, variantes R (`state/<k>`) et Q (`basin/<k>`), 11 classes, S = Σ|p̄ − p_PT| sur les 20 dernières évaluations (époques 300 à 490). Valeurs AS : R : S_C1 = 0,3672, S_C2 = 0,2099 ; Q : S_C1 = 0,3632, S_C2 = 0,2029.
Lecture, pour chaque variante et chaque graine :
1. Δ_s = S_D3(s) − S_C(s) (négatif = darting plus proche de p_PT) ; position de S_D3(s) par rapport à S_C(s) seul (pas aux deux).
2. Erreurs |log(p̄ / p_PT)| par bassin ; nombre de petits bassins (deux paires disjointes, K4 moins une arête, étoile) où D3(s) bat C(s).
3. Réplication : sur la graine 1, comparaison avec l'ancien run darting sans options rapides (S_R 0,1374, S_Q 0,1222). La différence est rapportée comme mesure du bruit d'exécution du darting à graine égale.
4. Bruit rappelé à côté : |S_C1 − S_C2| = 0,157 (R) / 0,160 (Q).
5. Sanité : perte finie ; energy_w2 et eq_w2 des deux dernières évaluations contre C(s) ; statistiques du matcher (`dart/kappa[...]/term_over_gradg`, `outflux`, propositions valides par échantillon).
Mots autorisés dans le rapport : « Δ négatif / positif sur 2 graines sur 2 », « Δ dans / hors du bruit ». Pas de « démontré ».

## Phase 2 : dose-réponse, γ = 10 et γ = 30, graines 1 et 2 (4 runs)
Noms : `dw4_cont_D10`, `dw4_cont_D30` (`adjoint_matcher.gamma=10.0` / `30.0`), mêmes graines. Lancée seulement sur ordre de l'utilisateur et avec confirmation du coût ; la lecture ne dépend pas des résultats de la phase 1 (aucun γ n'est choisi d'après la phase 1).
Lecture : pour chaque graine, S (R et Q) en fonction de γ ∈ {3, 10, 30} avec la valeur AS de la graine comme γ = 0 ; l'énoncé « S décroît avec γ » est donné par graine (vrai/faux, sans test). Sanité de stabilité : perte finie, eq_w2 ≤ 1,5 × celui de C(s), `term_over_gradg` par plage de κ (le terme à κ > 0,7 doit rester petit devant le gradient ; s'il dépasse 0,5, c'est rapporté comme instabilité probable). La mesure de la dose est `term_over_gradg` : si elle n'augmente pas avec γ, l'absence d'effet ne dit rien sur la méthode.

## Ce qui ne sera PAS fait
Pas de seuil d'effet, pas de réglage de γ après coup, pas d'AS supplémentaire (C1 et C2 servent de témoins), pas de graines autres que 1 et 2.
