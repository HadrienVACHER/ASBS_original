# A5 : division hors ligne sur les modes gaussiens du GMM-dk (32, 8, 6) × 0,5, gabarits gaussiens non rétrécis

Écrit et commité AVANT tout calcul (point A5 du message de l'utilisateur du 2026-11-23 ; ne conditionne pas l'étape B). CPU uniquement, aucune location ; chaque exécution < 1 min (chronométrée). Aucune règle n'est modifiée après coup.

Question : sur DW-4, la division a échoué parce que la domination imposait un rétrécissement (`rules_dw4_template_variance.md` : SD de log u de 0,3 à 1,2 nat sans rétrécissement, 5 à 8 nats avec). Sur des modes gaussiens, où aucun rétrécissement n'est nécessaire, la division marche-t-elle ?

## Cible et échantillons
- GMM-dk (d, K, Δ) = (32, 8, 6) : composantes N(μ_k, I), poids 0,302 à 0,030, distance minimale entre moyennes 6. σ_max × 0,5 = 1,25, σ_min = 0,001, ν₁ = σ_max² − σ_min².
- **Échantillons AS** : bras (a) de l'étape B (`results/train_weightfree_runs/wf_a_d32/eval_samples/ep00{350,400,450}.pt`, 10⁴ chacun). Ce sont **3 instantanés**, qui donnent la variabilité entre évaluations (règle permanente). Rapporté à côté : le départ de l'étape B (`results/wf_start/d32/start_samples.pt`).
- Bassins : trempe et minima de `basin_lib` (à Δ = 6, bassins ≈ cellules).
- Dans chaque bassin, les indices pairs forment la **moitié d'ajustement**, les impairs la **moitié d'évaluation**.
- E = −log π normalisé (Z = 1).

## Gabarits (non rétrécis : η = 0, s = 1)
- **Principal (estimateur de DW-4)** : par bassin k, μ̂_k = moyenne de la moitié d'ajustement ; Σ̂_k = covariance du bassin rétrécie vers la covariance intra-bassin poolée avec n₀ = 50 (même estimateur que `rules_dw4_template_variance.md`, sans plancher de tangente).
- **Descriptifs** :
  - covariance commune : Σ̂_k = la covariance intra-bassin poolée (n₀ = ∞) ;
  - oracle : N(μ_k, I).
- **Masses (Gelfand–Dey sur la moitié d'ajustement)** : Ẑ_k = 1 / moyenne des u_k = N_k(x) e^{E(x)} sur les points du bassin k. L_k = −log Ẑ_k.

## Mesures
1. **Dispersion** : écart-type de log u_k = log N_k(x) + E(x) sur les points du bassin k :
   - de la moitié d'évaluation AS (règle) ;
   - de 10⁴ tirages exacts de π (rapporté).
   Rapportés aussi : k̂ de u_k dans le bassin ; et, comme repère, la dispersion attendue du seul bruit d'estimation d'une covariance pleine, √(½ tr((Σ̂_k⁻¹ − I)²)).
2. **Correction de la moyenne** (le terme de division de `adjoint_samplers/utils/division_cv.py`, `division_term`, s = 1, paramètres ajustés ci-dessus) :
   - Pour chaque point x₁ de la moitié d'évaluation, R = 4 tirages de t ~ U(0, 1) (la loi des instants de régression d'AS), κ = ν_t/ν₁ avec ν_t = σ_max²(1 − (σ_max/σ_min)^{−2t}), et x_t = κ x₁ + √v ε, v = ν₁ κ(1 − κ), ν_{1|t} = ν₁(1 − κ). Graine fixée.
   - **Correction exacte** c*(x_t) = (M*(x_t) − M^q̃(x_t)) / ν_{1|t}, en forme fermée (mélange gaussien) :
     - M* : la moyenne a posteriori de x₁ sachant x_t sous π ;
     - M^q̃ : la même sous q̃ = Σ_k w̃_k N(μ_k, I), avec w̃_k ∝ w_k p̂_k^AS / p_k. Ce sont les formes exactes avec les masses par bassin de l'instantané AS ; c'est la correction que la méthode doit donner si AS a les bonnes formes et les mauvaises masses.
   - **Coefficient de projection** β = Σ ⟨term, c*⟩ / Σ |c*|². Par la tour des espérances, c'est le coefficient de E_q[term | x_t] sur c*. Erreur-type par bootstrap sur les x₁ (200 tirages). Rapporté par plage de κ ([0 ; 0,1], [0,1 ; 0,7], [0,7 ; 1]).
3. **Masses** (rapporté, analogue de l'étape 0 de DW-4) : p̂^GD = Ẑ / ΣẐ contre p et contre p̂^AS ; TV, et coefficient de projection ⟨p̂^GD − p̂^AS, p − p̂^AS⟩ / |p − p̂^AS|².

## Témoins
- **Positif** : paramètres oracle (μ_k, I, L_k = −log w_k), sur 5000 tirages exacts de q̃. Il faut β ∈ [0,8 ; 1,2] ; sinon, la mesure est signalée comme non fiable.
- **Négatif** : gabarits principaux avec L_k = −log p̂_k^AS (les masses d'AS : aucune information de masse). Il faut |β| ≤ 0,2 ; sinon, la mesure ne discrimine pas, ce qui est signalé.

## Règle A5 (gabarit principal, moyennes sur les 3 instantanés)
**Écart-type de log u_k ≤ 1 nat dans chacun des 8 bassins** (moitié d'évaluation AS) **ET β ≥ 0,5.**

## Lecture fixée d'avance
- **De l'utilisateur** : si A5 est vraie, l'échec sur DW-4 tenait bien à la seule contrainte de domination.
- **Précision déclarée** (mienne, avant calcul) : à d = 32, le plus petit bassin n'a qu'environ 150 points d'ajustement. Le bruit d'estimation d'une covariance pleine y donne à lui seul une dispersion de l'ordre de 1 à 1,5 nat (repère de la mesure 1). Si la règle est fausse par la dispersion du gabarit principal, mais que la covariance commune passe les deux critères, l'échec est attribué à l'estimation des covariances (rapport d/n), et non à la forme ni à la domination. Le verdict de la règle reste celui du gabarit principal.

## Mes prédictions (sans effet)
- Dispersion ≤ 1 nat dans les 8 bassins (principal) : 30 %. Elle sera dépassée dans les petits bassins (bruit d/n).
- β ≥ 0,5 (principal) : 60 %. Covariance commune : dispersion ≤ 1 nat partout à 85 %, β ≥ 0,5 à 70 %.
- A5 vraie : 25 %.
- Témoins : positif dans [0,8 ; 1,2] à 85 % ; négatif |β| ≤ 0,2 à 80 %.

## Exécution
`scripts/division_gmmdk.py` (commandes `snap E`, `control`, `verdict`). Résultats dans `figures/division_gmmdk/`. Section dans RESULTS.md.

## Amendement daté (2026-11-23, après le verdict ; demandé par l'utilisateur ; écrit et commité avant la reprise)
**Verdict d'origine, conservé** : A5 FAUSSE telle qu'écrite (dispersion > 1 nat dans les 5 plus petits bassins, β = 0,13). Mesure de β signalée non fiable : témoin positif 0,69 ± 0,27, témoin négatif 0,46 ± 0,34 (RESULTS.md, cd24a1f). **Cause : erreur de spécification** (R = 4 tirages de t par point, trop peu).
- **Seul changement : R = 100 tirages par point** (erreur-type attendue ≈ 0,05). Tout le reste est inchangé : échantillons, moitiés, gabarits, masses de Gelfand–Dey, loi de t, graines, témoins et leurs seuils, règle A5, lecture.
- Les dispersions et les masses ne dépendent pas de R : elles sont recalculées à l'identique.
- Calcul découpé par gabarit (`snapR E partie`, chaque appel < 1 min). Résultats dans `figures/division_gmmdk/*_R100*.json`, verdict dans `verdict_R100.json`, à côté des fichiers d'origine, qui sont conservés.
