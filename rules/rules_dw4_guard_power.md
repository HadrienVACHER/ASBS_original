# Phase 0, tâche 1/4 : garde-fou (iii) de la confirmation du darting DW-4, refait avec assez de puissance (réanalyse déclarée)

Écrit et commité AVANT tout calcul de cette réanalyse. Aucune règle n'est modifiée après coup. CPU uniquement, aucune location, aucun entraînement ; chaque exécution < 1 min (mesurée et rapportée).
Contexte : `rules_darting_confirm.md` (07b422f, amendée 8ec415b). (i) et (ii) sont vrais sur les graines 2, 3, 4 ; (iii) est faux sur la graine 2 (0,655 > 0,631), calculé sur deux évaluations isolées sans marge calibrée sur le bruit (erreur de spécification déclarée, `audit_fermetures.md` n° 7). **Le verdict d'origine « non confirmé » reste visible et n'est pas réécrit** ; ceci est une réanalyse déclarée qui remplace (iii) par G3′ pour elle seule.

## Données
- Bras : `model_samples` (10⁴ échantillons frais du modèle final) des `samples_final.pt` rapatriés : D30 graines 2, 3, 4 (`results/confirm/dw4_cont_D30_s{2,3,4}`) et AS de même graine C2 (relancé), C3, C4 (`results/confirm/dw4_cont_C{2,3,4}_s{2,3,4}`).
- Référence : le pool PT à T = 1 (`results/pt_samples_T1.pt`, 64 chaînes × 1600 états ; b4e35e5), pas la référence DEM.
- Métriques : celles de l'évaluation officielle (`DW4Evaluator`) : energy_w2 = emd2_1d des énergies^(1/2) ; eq_w2 = emd2 des distances entre nuages de points (permutation, rotation et réflexion ; `dist_point_clouds_fast`, comme en b4e35e5)^(1/2) ; `remove_mean` sur les deux ensembles.
- **Taille n = 2000 (sous-échantillonnage déclaré).** n = 10⁴ n'est pas tenté. Une évaluation d'eq_w2 à n = 2000 a pris environ 2,6 s (≈ 100 évaluations en 270 s dans `dw4_w2_check.py`). Le coût des distances croît en n² (× 25 à n = 10⁴, soit plus d'une minute) et le transport optimal exact n × n davantage encore. n = 2000 est aussi la taille de l'évaluation officielle.

## Mesures
- **Plancher et bruit** : 20 répétitions. Pour chacune, les 64 chaînes sont coupées au hasard en deux groupes disjoints de 32 ; n états sont tirés au hasard dans chaque groupe, l'un comme échantillon, l'autre comme référence (comme b4e35e5). On rapporte la médiane, le p95 et l'écart-type de eq_w2 et energy_w2.
- **Chaque bras** : 20 rééchantillonnages bootstrap. Le rééchantillonnage r tire n des 10⁴ `model_samples` avec remise (graine fixe) et une référence de n états tirés au hasard dans tout le pool PT. La référence du rééchantillonnage r est la même pour tous les bras (nombres aléatoires communs, déclaré). Valeur du bras = moyenne des 20 ; écart-type du bras sd = écart-type des 20.
- **Δ_s** = eq_w2(D30_s) − eq_w2(C_s) ; sd(Δ_s) = √(sd_D30,s² + sd_C,s²). Moyenne Δ̄ sur s ∈ {2, 3, 4} ; erreur-type SE(Δ̄) = √(Σ_s sd(Δ_s)²)/3. La combinaison suppose l'indépendance et ignore la référence commune, ce qui est conservateur.

## Garde G3′ (remplace (iii) pour cette réanalyse)
**G3′ vraie** si Δ̄ ≤ 2 × SE(Δ̄) ET aucune graine n'a Δ_s > 3 × sd(Δ_s). Le même calcul est fait pour energy_w2, en descriptif seulement.

## Contrôle positif (fixé avant)
Version dégradée de D30_s : on ajoute un bruit gaussien N(0, σ_s² I) aux 8 coordonnées, puis `remove_mean`. L'amplitude σ_s est choisie, avant de calculer G3′ sur la version dégradée, pour **doubler l'écart au plancher d'eq_w2**.
- Écart du bras : g_s = eq_w2(D30_s) − médiane du plancher, ramené à un minimum égal à l'écart-type du plancher.
- σ_s = la plus petite valeur de {0,05 ; 0,1 ; 0,2 ; 0,3 ; 0,5} telle que la moyenne de 2 évaluations de eq_w2(D30_s dégradé) − médiane du plancher soit ≥ 2 g_s. Si aucune ne suffit, σ_s = 0,5, signalé.
- Puis G3′ est appliquée à (D30 dégradé, C), avec 20 rééchantillonnages bootstrap du bras dégradé (bruit tiré à neuf à chaque rééchantillonnage, graine fixe). **G3′ doit être FAUSSE sur la version dégradée**, sinon la garde n'a pas de puissance.

## Décision écrite à l'avance
- **« confirmé »** si (i), (ii) (verdicts d'origine, vrais) et G3′ sont vraies, et que le contrôle positif est détecté (G3′ fausse sur la version dégradée).
- **« non confirmé »** si G3′ est fausse.
- **« inconclusif »** si G3′ est vraie mais que le contrôle positif n'est pas détecté.

## Exécution
`scripts/dw4_guard_power.py`, en sous-commandes de moins d'une minute chacune, chronométrées : `floor`, `arm`, `calib`, `ctrl`, `verdict`. Résultats dans `figures/dw4_guard_power/` (JSON), puis section « Phase 0 » de RESULTS.md.
