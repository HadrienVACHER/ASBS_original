# Règle pré-enregistrée : test de CONFIRMATION du darting γ = 30 sur des graines neuves

## AMENDEMENTS (demande de l'utilisateur, écrits et commités AVANT la location et avant tout lancement ; ils remplacent les passages correspondants plus bas)
1. **6e run : C2 relancé** (`dw4_cont_C2`, `dw4_as_states`, graine 2, répertoire `dw4_cont_C2_s2`) sur la même instance, avec les mêmes options rapides que les autres bras. Dans (i) et (ii), le témoin de la graine 2 est CE C2 relancé ; le C2 du cluster (S_Q 0,2029, eq_w2 0,589, energy_w2 0,329) est seulement rapporté. Raison : comparer tous les bras sur une même plateforme, en cas de repli non compilé de `torch.compile`.
2. **(iii) remplacé** par : pour chaque graine s ∈ {2, 3, 4}, eq_w2(D30_s) ≤ 1,1 × max(eq_w2(C2), eq_w2(C3), eq_w2(C4)) ET energy_w2(D30_s) ≤ 1,1 × max(energy_w2(C2), energy_w2(C3), energy_w2(C4)), avec les C du même protocole (C2 relancé inclus). Raison : la version bras par bras à ±10 % pouvait échouer par bruit seul.
Le reste est inchangé. Plafond de dépense relevé à 0,35 $ pour couvrir le 6e run.

Écrite et commitée AVANT tout lancement (commit de ce fichier). La graine 1 est EXCLUE du test : elle a servi à choisir γ = 30 (dose-réponse, `rules_darting_seeds_gamma.md`, phase 2). γ = 100 n'est pas lancé.

## Bras et protocole
Continuation de 500 époques depuis le contrôleur et le buffer (`buffer_x1`) du run AS 94252 ; `sigma_max=2`, `nfe=100`, évaluation toutes les 10 époques à 5000 échantillons, `evaluator.eq_w2_every=100`, t uniforme, Adam repart de zéro ; options rapides `fast_sampling=true fast_train=true buffer_on_device=true` pour TOUS les bras (si `torch.compile` échoue faute de compilateur, le repli non compilé s'applique à tous les bras de la même instance : écart déclaré).
- Darting : `dw4_as_darting_states`, γ = 30, graines 2, 3, 4 (runs `dw4_cont_D30`, répertoires `dw4_cont_D30_s{2,3,4}`).
- Témoins AS : `dw4_as_states`, graines 3 et 4 (`dw4_cont_C3`, `dw4_cont_C4`), lancés sur la MÊME instance que les bras darting (même protocole et mêmes options). Le témoin de la graine 2 est C2 RELANCÉ sur la même instance (amendement 1) ; le C2 existant (W&B `CCDS_NTU/as`, `dw4_cont_C2`, commit aadf33b, cluster) est seulement rapporté.
Aucun réglage d'après les résultats. Un run qui plante avant la 500e époque est relancé une fois à l'identique (même graine) ; le relancement est déclaré, il ne compte pas comme un nouveau test.

## Statistiques
Vérité p_PT (`scripts/dw4_reeval_pt.py`, `truth()`), S = Σ_b |p̄_b − p_PT,b| avec p̄ la moyenne des 20 dernières évaluations (époques 300 à 490). Variante Q (bassins quenchés, `basin/<k>`) PRINCIPALE ; variante R (états bruts, `state/<k>`) rapportée.
Garde-fous : eq_w2 = moyenne des deux dernières valeurs de `eq_w2` ; energy_w2 = moyenne des deux dernières évaluations (les mêmes définitions que `scripts/dw4_darting_run_analysis.py`).
Le tout est lu par `scripts/dw4_darting_confirm_analysis.py` (commité avec cette règle).

## Verdict (écrit d'avance, sans relâchement)
Pour les graines s ∈ {2, 3, 4} (C_s = le témoin AS de la graine s) :
- (i) pour CHACUNE : S_Q(D30_s) < S_Q(C_s) − 0,05 (0,05 = bruit d'exécution du darting mesuré : deux runs de la graine 1 à γ = 3 diffèrent de 0,05 à 0,06 en S_Q) ;
- (ii) la MÉDIANE sur les graines de S_Q(D30_s) < min(S_Q(C2), S_Q(C3), S_Q(C4)) ;
- (iii) [remplacé par l'amendement 2] garde-fou, pour CHAQUE graine : eq_w2(D30_s) ≤ 1,1 × max(C2, C3, C4) ET energy_w2(D30_s) ≤ 1,1 × max(C2, C3, C4).
« effet confirmé » si (i), (ii) et (iii) sont tous vrais ; « non confirmé » sinon. Une perte non finie ou une valeur NaN dans un bras D30 rend le verdict « non confirmé ».

## Rapporté (sans règle)
S_Q et S_R de chaque bras ; plancher de S (`scripts/dw4_floor_S.py` : S d'un sampler parfait, 20 évaluations de 5000 tirages du pool PT, médiane 0,0034 en Q et 0,0036 en R, p95 0,0060 / 0,0070, 0,0058 / 0,0069 avec l'incertitude de p_PT) ; repères IS en S_Q (IS d'endpoint gaussien s = 1 : 0,011 ; t5 s = 1 : 0,014 ; IS de chemin : 0,011 ; un seul lot chacun) ; tableau par bassin D30 contre C pour chaque graine ; statistiques du matcher (dose ‖terme‖/‖grad‖ par plage de κ) ; coûts ; tout écart au protocole.

## Ce qui ne sera PAS fait
Pas de relâchement du seuil ou des règles après coup, pas de graine supplémentaire, pas de γ = 100, pas de test sur la graine 1.
