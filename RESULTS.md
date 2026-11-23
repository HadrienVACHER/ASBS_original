# Results log — Barker control variate for Adjoint Sampling (branch `barker`)

One section per target: setup, pre-registered rule, verdict, key numbers, checks. W&B project `CCDS_NTU/adjoint_sampler_cv` (GMM, ManyWell) and `CCDS_NTU/as` (DW-4); runs carry their commit as tag `git:<hash>`.
Single seed (0) unless stated. Pre-registered thresholds were never relaxed after seeing a result.

## Synthèse de la ligne darting (2026-11-21 ; écrite à partir des résultats déjà commités, aucun calcul nouveau)

**Question.** Le « darting » (sauts minimum à minimum avec poids de Barker) peut-il corriger les poids des modes d'Adjoint Sampling (AS), à l'entraînement ou après coup, mieux que l'IS de chemin ? **Réponse sur les cas testés (une graine chacun, sauf mention) : non, avec un vrai AS.** Pour chaque point : règle pré-enregistrée (commit d'écriture, avant les résultats) → verdict tel qu'écrit → commit des résultats.

1. **Le noyau est exact.** Moyenne nulle du terme sous π, GMM : T1′ (`rules_train_darting.md`, b3a8c18) RÉUSSI, max |z| 2,5–3,2 < 4, contrôle positif détecté à κ ≤ 0,5 ; DW-4 : tests d'exactitude sur états de π exacts (aller-retour 2e-15, |z| ≤ 2,66 contre 13–15 pour le contrôle positif ; e082759, f186193). Équilibre détaillé sur échantillons PT : T1′ (`rules_reference_check.md`, 3d5b062) PASS, tout |log-rapport| < 3 SE (3927fd6, b4e35e5) ; le test T1 antérieur sur la référence DW-4 était NO-GO parce que cette référence est biaisée (erreur de conception déclarée, b4e35e5). Stationnarité MoS (descriptif) : depuis 10⁴ échantillons exacts, TV 0,0147 à S = 40 pour un plancher de 0,0123 (24a81f8).
2. **Quadrature 1D** (`rules_quadrature_1d.md`, c204938 → bed2cf3). AS : la correction relative ρ_AS décroît comme e^{−0,8/(2ε²)} (ln ρ affine en 1/(2ε²), pente −0,797, R² = 0,992) ; avec le darting ρ_CV = 0,93–1,42, de l'ordre de 1 pour tout ε, et atteint |ŵ − w| < 0,01 à la 2ᵉ itération. **H_CV soutenue ; H_AS NON soutenue** telle qu'écrite (la pente est dans l'intervalle, mais ρ_AS < 0,05 pour ε ≤ 0,35 est faux : 0,19 et 0,054) ; plancher de la règle gonflé par le recouvrement des composantes (déclaré, verdicts inchangés).
3. **Darting à l'entraînement.**
   - *DW-4, γ = 30* (`rules_darting_confirm.md`, 07b422f puis 8ec415b → 9b4680c) : S_Q 0,06 / 0,09 / 0,05 contre 0,20 / 0,26 / 0,22 pour AS (graines 2, 3, 4) ; (i) et (ii) vrais, (iii) (garde-fou eq_w2) faux sur la graine 2 (0,655 > 0,631) : **verdict « non confirmé »** tel qu'écrit, la règle (iii) étant bruitée (déclaré). Les D30 restent à 15–27 × le plancher de S. Contexte non testé : une graine, γ de 3 à 30, S décroissant avec γ (6594b75, sans verdict).
   - *GMM-dk, γ = 30* (`rules_train_darting.md`, b3a8c18, d96d4c4 → 81f8243) : **R_train FAUX** dans les deux cellules (TV brut 0,105 contre 0,169 en (8,8,8), 0,140 contre 0,047 en (32,8,6) : pire qu'AS) ; R_sûreté vrai. **γ = 1 (descriptif) : effet partiel** (TV brut divisé par 2 à 3 en (8,8,8), nul à la limite du bruit en (32,8,6)).
4. **Darting post hoc contre l'IS, samplers synthétiques et vrais contrôleurs GMM** (`rules_darting_vs_is.md`, f31f199 + amendements 77d0151, 1b6fe90, 8436d13 → bba6570). Synthétique A1 (gaussien) : **R1 vraie** (d = 512 : IS à 10⁵ en échec, TV 0,106 ; darting 0,010 ; une seule dimension franchit le seuil). A2 (Student) : **R2 (a) non évaluable, (b) vraie** (le darting perd, TV 0,12–0,14 contre 0,003–0,005 pour l'IS ; les minima ne se découvrent plus à d ≥ 128). Vrais contrôleurs GMM-dk à 8/8 modes : **R3 vraie** (darting au plancher, mais l'IS de chemin aussi : ESS 0,55–0,88, k̂ < 0,33) ; à 1/8 mode, le darting ne crée pas les modes absents (2/8 contre 1/8 pour AS ; l'IS de chemin atteint 5/8).
5. **MoG / MoS (Beyond ELBOs), MoS-w.**
   - *Sampler synthétique* (`rules_mog_mos_phase0.md`, cb52a93 → 6318c81 ; `rules_mos_pilot.md`, 8771d10 → 052def7) : **G vraie** pour le MoS (d = 2 à S = 10 ; d = 50 et 200 pour S ≥ 5) et le MoS-w (d = 50 et 200, S = 10 seulement), fausse pour le MoG et pour le cas du mode rare.
   - *Vrai AS, MoS d = 50* (pilote 8771d10 → 5d96ff7 ; `rules_mos_step1.md`, 257767a → e54d048) : P1 vraie seulement à σ_max = 23,2 (10/10 modes), P2 : k̂ = 7,43 (queues d'AS trop légères pour la cible Student) ; **G FAUSSE** (darting 0,052–0,074, au-dessus de 2 × plancher 0,0246) ; H_saut vraie (le MALA seul donne 0,261).
   - *Vrai AS, MoS-w d = 50* (amendement 1 de `rules_mos_step1.md`, 08e1b43 → 24a81f8) : **G FAUSSE** aux époques 100, 200, 300 (TV 0,089–0,200 ; IS 0,47–0,85) ; H_saut fausse à 100, vraie à 200 et 300.
   - **Note sur les samplers ratés (a793a82, 24a81f8).** À σ_max = 23,2 et 300 époques, AS n'a pas appris le MoS : distance médiane au minimum 109 contre 15 sous π ; 13 % des sauts sont invalides, 40 % refusés (log-rapport médian −42). Les comparaisons des étapes 1 et 2 portent donc sur un sampler raté et ne testent pas le régime « bonnes formes, mauvais poids ».
6. **Étape 3 : AS à la recette d'origine, critère de qualité Q** (`rules_mosw_step3.md`, a793a82 → c11e387 ; 0,048 $). **Q FAUX pour les 12 contrôleurs** (MoS-w d = 50 et 10, 1500 époques, 6 sauvegardes chacun) : à d = 50 la distance médiane reste 8 à 10 × celle de π et la masse se concentre sur un seul bassin (p̂/p = 6,3) ; à d = 10 les formes sont apprises (Q1 vrai à 1250 et 1500, rapport 1,49) mais 4 ou 5 bassins sur 10 seulement sont couverts (Q2 faux). **Aucune comparaison n'a été calculée** (règle : arrêt si aucun ne passe Q).

7. **Étape 4 : GMM-dk en grande dimension avec un vrai AS** (`rules_gmmdk_step4.md`, 2527419 → ce fichier ; 0,047 $ ; (d, K, Δ) = (128, 8, 6) et (256, 8, 6), σ_max × {0,5 ; 0,75}, 1500 époques). **Q VRAI** aux deux dimensions pour le candidat choisi par la règle de σ_max (× 0,5, époque 1500 : formes 1,03–1,04 × π, 8/8 bassins), **I FAUX** (ESS de l'IS de chemin 0,19 et 0,16 ≥ 5 %, k̂ 0,22 et 0,34), **G FAUX** (l'IS de coût égal bat le darting : TV 0,0012 contre 0,0098 à d = 128 ; 0,0046 contre 0,0080 à d = 256), H_saut vraie. **Décision écrite à l'avance : l'IS tient, la piste darting est fermée.** À côté (non comparé) : les runs × 0,75 passent aussi Q mais ont un ESS de 10⁻⁴–10⁻³ (écartés par k̂ ≥ 0,7).

8. **Étape 5 : GMM-dk (128, 256) et MoS-dk (32, 128) avec un vrai AS** (`rules_dk_step5.md`, c47fe51 → section « Étape 5 » ; 0,063 $). GMM-dk : Q vrai, I faux (ESS 0,16–0,19), **G faux** à S ∈ {1, 5, 10, 40} (l'IS de coût égal est meilleur partout). MoS-dk d = 32 : Q vrai, I faux (ESS 0,164), G faux, H_saut faux. **MoS-dk d = 128 : Q vrai (époque 1500 seulement), I vrai (ESS 8·10⁻⁴, k̂ 1,19), G VRAI (à S = 1 et 5, de justesse : 0,0119 contre 0,0121 et 0,0059 contre 0,0062 ; faux à S = 10 et 40), H_saut faux. Décision écrite à l'avance : « Q vrai, I vrai, G vrai » → la règle rouvre la piste darting.** Réserves : marges au niveau du bruit (plancher 0,0096), non monotone en S, un seul contrôleur, une graine, MALA seul presque aussi bon.

9. **Contrôle CPU de H_saut par bassins** (`rules_hsaut_basin.md`, 9d16c5a ; section « Contrôle CPU de H_saut par bassins »). Sur le contrôleur qui rouvrait la piste (MoS-dk d = 128 × 0,75, époque 1500), 5 chaînes indépendantes : à S = 40, darting 0,0116 ± 0,0036 et MALA seul 0,0116 ± 0,0020 par bassin (plancher 0,0093) : **H_saut_bassin FAUX** (rapport 1,00). **Décision écrite : la réouverture tient aux pas locaux, pas aux sauts ; la piste darting est fermée, définitivement.** (Aux petits S, le darting corrige plus vite que le MALA seul : 0,55 × à S = 1 et 5 ; descriptif.)

10. **Newton-Stein** (`memo_cv_failure_modes.md`, `rules_ns_damping.md` 656b270, `rules_mosw_ns.md` e5c8b5d + amendement ; sections « Test GPU : Newton-Stein… » et « Newton-Stein dans le régime d'instabilité extrême »). **Exact pour une cible gaussienne (Théorème A, vérifié en quadrature 1D), mais sans effet pratique** : GMM-dk d = 32, R1 et R2 faux (ESS 5,6 × AS à (32, 8, 4) × 2, mais seulement 1,2 × à × 1 ; aucun mode retrouvé à Δ = 8) ; MoS-w σ_max = 23,2, S1 faux, S2 vrai seulement tel qu'écrit (artefact : le sampler est resté à la loi de base), faux après l'amendement daté « S2 exige Q1 ». **Dans les mélanges éloignés, Ĥ = courbure du mode sous-relaxe les petits t (Λ_t ≈ 1/800 : gel des premiers instants)**, et l'étape 0 montre que A = 1 − F(ρ) ne prédit pas les échecs d'AS. Piste fermée (0,040 $ + 0,062 $).

**Conclusion.** Il n'existe, dans ce que nous avons testé, **aucun cas avec un vrai AS où AS trouve tous les modes avec la bonne forme et où l'IS de chemin échoue, parmi les contrôleurs que la procédure pré-enregistrée a comparés** (exception non comparée : les runs × 0,75 de l'étape 4, Q vrai avec un ESS ≈ 10⁻⁴, écartés par la règle de σ_max) : sur les GMM-dk à 8/8 modes (et sur DW-4, où l'IS de chemin retrouve les deux bassins dominants du PT à moins de 3 SE) l'IS corrige déjà, sans darting (GMM-dk : TV après IS 0,006–0,009, au niveau du plancher). **Partout où l'IS échoue, AS a déjà échoué en amont** : modes manquants (GMM-dk (32,8,8), MoS-w d = 10 : le darting ne crée pas les modes absents) ou sampler diffus loin des modes (MoS et MoS-w à d = 50 : k̂ 6–8, G fausse). *Mise à jour finale :* le contrôle CPU de H_saut par bassins (point 9) est FAUX : la réouverture de l'étape 5 tenait aux pas locaux, la piste darting est fermée définitivement. *Mise à jour (étape 5) :* la décision écrite de l'étape 5 est réalisée par un cas, MoS-dk d = 128 (Q vrai, I vrai, G vrai), avec les réserves du point 8 : cela contredit la formule « aucun cas où l'IS échoue alors qu'AS apprend » pour ce contrôleur, aux marges près. *Mise à jour (étape 4) :* à (d, K, Δ) = (128, 8, 6) et (256, 8, 6) avec un vrai AS à 8/8 bassins, l'IS de chemin tient (ESS 16–19 %) et bat le darting à coût égal ; la décision écrite ferme la piste. Réserve : les runs × 0,75 passent Q avec un ESS ≈ 10⁻⁴ ; la règle de sélection de σ_max les a écartés et le darting n'y a pas été calculé, donc le cas « Q vrai, I vrai » existe dans les données sans avoir été testé pour G. Le darting ne gagne que contre des samplers synthétiques (A1 d = 512, MoS et MoS-w synthétiques), régime que les vrais AS essayés n'atteignent pas ; à l'entraînement, il n'améliore ni de façon confirmée (DW-4, « non confirmé ») ni de façon fiable (GMM-dk, R_train faux).
*Réserves.* Une graine par cas ; σ_max = 23,2 est la seule valeur à donner 10/10 modes à d = 50 et il n'a pas été exploré au-delà (ni pour d = 10) ; le darting n'a pas été calculé sur les contrôleurs d = 10 aux époques 1250 et 1500 (décision de l'utilisateur : la réponse est connue, aucune décision n'en dépend) ; les cas synthétiques sont mes opérationnalisations, déclarées dans leurs sections.

---

## 1. GMM, isotropic (regular simplex, d = K)

- **Setup.** Spread (regular-simplex) layout, radius 4.25, log-uniform weights; big controller 256x6 for AS and Barker; seeds 0-2; d = 24 and 36; metrics time-averaged over the last 30 evaluations (per-mode ratios averaged first). Frozen method (commit `ec12405`, configs `gmm_as_final`, `gmm_as_barker_final`): Barker orbit control variate, gamma = 3, threshold centres (Barzilai-Borwein basin minima of the buffer, radius = 1 x median mode width, capacity 128, 12 denoising steps).
- **Rule.** gamma = the single global value minimising mean |log R̄| over d = 24 and 36 (3 seeds). Centres from the buffer only (zero mean stays exact).
- **Verdict.** Works.
- **Key numbers** (VERIFIED against W&B on 2026-10-27; mean |log R̄| over the 30 last evaluations, per seed 0 / 1 / 2, floor 1e-6):
  d = 24: AS big 0.268 / 0.255 / 0.183 (mean 0.235), Barker gamma = 3 big 0.032 / 0.037 / 0.034 (mean 0.034): -86 %, better at every seed.
  d = 36: AS big 0.389 / 0.345 / 0.317 (mean 0.351), Barker gamma = 3 big 0.117 / 0.145 / 0.075 (mean 0.112): -68 %, better at every seed.
  energy_w2 (3 seeds): d = 24 0.33-0.36 (AS) vs 0.41-0.43 (Barker); d = 36 0.53-0.59 vs 0.72-0.78.
  Run names in W&B: `gmm{24,36}d_as_big_cap_K*_s*_spread` and `gmm{24,36}d_as_barker_g3_thr_big_K*_s*_spread` (256x6, 2000 epochs; commits `3f672a7` / `203f59d`). The cells named `*_as_final` /
  `*_as_barker_final` (configs of the freeze commit `ec12405`) were never run under those names: the frozen-method numbers are those of the capacity-ablation cells above, which use the same settings.
  Capacity ablation (d = 36, gamma = 2, from my notes, not re-checked): -47 % with the small network, -55 % with the big one.
- **Costs / caveats.** ~140-200 energy calls per gradient update (d = 24 / 36) against ~5 for AS. energy_w2 is worse with Barker (small net effect, dominated by a common 3-5 % mode-width excess that shrinks with capacity; +0.19 at d = 36 reported as is). gamma annealing was not adopted.
- **Checks.** Exact zero mean by orbit enumeration and under the exact posterior (tests/test_barker_matcher.py, test_mh_matcher.py, test_threshold_centres.py); oracle-centre intervention showed that k-means centres limited Barker, which motivated the threshold centres; the weight deficit of the simplex is monotone in the weight (no mode-specific effect).

## 2. GMM, heterogeneous

- **Setup.** `HeteroGMM` (unequal radii / covariances, affine orbit with the Jacobian in the weights, commit `3f672a7`).
- **Rule (AS alone, Barker never read before the choice).** A geometry is informative iff AS-big has light-quarter ratio < 0.6 (3-seed mean) AND modes_covered >= 0.9. Candidates G1-G4, extension E1-E4 (`a58e478`), last candidate F1 (`70a375f`), fixed stopping rule.
- **Verdict.** No informative regime found; hetero GMM dropped.
- **Key numbers** (VERIFIED against W&B with `scripts/hetero_geometry_rule.py` on 2026-10-27; AS big, 3 seeds, mean of the last 30 evaluations; light quarter / modes_covered): G1 0.125 / 0.182, G2 0.018 / 0.124, G3 0.193 / 0.108, G4 0.014 / 0.077 (AS finds 8-18 % of the modes); E1 0.277 / 0.838, E2 0.074 / 0.740, E3 0.134 / 0.570, E4 0.026 / 0.443 (E1 narrowly misses modes_covered >= 0.9; not relaxed); F1 (isotropic simplex positions and weights, heterogeneous covariances) 0.785 / 1.000, too easy. Hetero v1 (radii 7-11): AS found 2-3 of 24 modes (results void, unread).
- **Reading.** On the tested geometries "AS discovers all modes" and "AS has a clear weight deficit" do not overlap. The normalised separation (distance / (sigma_i + sigma_j) along the line) explains the collapse: 3.05 in the isotropic benchmark (AS finds all modes), collapse above ~3.45.

## 3. ManyWell-32 (FAB; 16 double wells + 16 Gaussian coordinates; light well 0.1557)

- **Setup.** AS 256x6, lr 1e-4, force clip 100, 2000 epochs, sigma_max matched to the target (1.41). The first runs (`a67ed56`: no clip, sigma_max 2.81) diverged to NaN; the stable AS seed-0 run is `c4345ea`.
- **Rule (AS alone).** Informative iff the mean light-well ratio < 0.6 and every coordinate visits both wells, plus a plateau condition. For the variants: wells_visited_frac = 1.0, mean |log R_i| > 0.3, plateau drift <= 0.05.
- **Verdict.** Closed.
- **Key numbers.** Standard target: 8 of 16 coordinates never visit the light well (R = 0); on the other 8 both wells are visited with wrong weights (R from 0.45 to 2.8). Variants V1 (light 26.6 %), V2 (17.9 %), V3 (33.7 %), V1 with lr warm-up: wells_visited_frac (last 30) = 0.67 / 0.50 / 0.65 / 0.43, all failing the rule.
- **Barker** (seed 0, product-subgroup orbit s = 3, metric on the 8 coordinates AS visits, mean |log R_i|: AS 0.571): gamma = 3 (`d6aac91`) 3.18; gamma = 1 (`ae74379`) 2.88; persistent well bank + 1D Newton denoising (`b3147ef`) 3.57. Coordinates that were over-filled in AS were emptied (R = 0 or 6.42) and stayed absorbing. One positive side effect: the light well appeared on 4-6 of the 8 coordinates AS never visits.
- **Pre-trained-sampler correction experiment (stopped).** Pre-training on the symmetric target with a fixed buffer failed the gate twice: v1 (`0e30dc8`): 54 % of samples with a double-well coordinate at |x| > 3 (exact: 0 %), energy_w2 300-540; v2 (70 % of regression times in [0, 0.3], `979996a`): energy_w2 2146. Phase 2 and the post-hoc baselines were never run.
- **Checks.** Product-orbit zero mean exact by enumeration and under the exact posterior with random subsets (max |z| <= 2.5 over the 32 coordinates, tests/test_product_orbit.py); the flux of the term has the right sign (positive for R > 1, ~0 for R = 1); 1D Newton denoising raised the converged fraction from 0.13 to 1.0.

## 4. DW-4 (4 particles in 2D, EGNN, official AS config)

- **Setup.** sigma_max 2, lr 1e-4, clip 100, nfe 100, 1500 epochs, seed 0; reference = DEM benchmark MCMC (10 000 samples). Short-pair graph state (d < 4) up to isomorphism (11 classes), also read on quenched basins (verified against an independent adjacency / L-BFGS classification).
- **AS sanity.** Runs `ed8bb67` and `94252` (`c166b5d`): eq_w2 0.49-0.61, energy_w2 0.53-0.56 (paper: 0.62 / 0.55).
- **Pre-registered AS rule.** Informative iff (i) every state of reference frequency >= 1 % has model frequency >= 0.1 x reference, (ii) state TV > 3 x p95 of the sample-size floor, (iii) plateau (TV drift <= 0.01 over the last 20 %). Result for `ed8bb67`: (i) true, (ii) 0.090 > 0.067 true, (iii) drift +0.019 false (literal verdict: not informative; (iii) is noise-limited, per-evaluation SD ~0.02 with 6 points at 1500 epochs).
- **AS weight errors are real at the basin level** (quenched, TV 0.0845 / 0.0795 for the two AS runs, floor p95 0.022): two disjoint pairs x2.0, star x2.8, K4 minus an edge x1.3, triangle + pendant x0.87-0.90, 2-edge path x0.92-0.94. Only ~6 % of the TV comes from thresholding raw distances. The two dominant states fluctuate by 0.05-0.07 between evaluations.
- **Barker kernel** (12 line + 12 point reflections, R = 24, gamma = 3 frozen, closed-form weights in `prepare_target`). Pre-registered criteria (`dw4_barker_rule.py`): basin TV < min(AS) - 2 SE; mechanistic; direction; sanity. Full run `6a955dd`: all four fail (basin TV 0.1042 ± 0.0070 vs 0.0851 / 0.0801, threshold 0.0621; mechanistic 3 of 5; direction 3 of 5; energy_w2 17.6 % below the AS, outside the +-15 % band). Time-averaged over the last 10 evaluations, the three runs are indistinguishable (TV of the averaged proportions 0.110 / 0.095 / 0.103).
- **Paired continuation test** (`aadf33b`, from the AS `94252` controller and buffer, 500 epochs, eval every 10, last 20 evaluations; rule `dw4_continuation_rule.py`): S = sum_b |p̄_b - ref_b|: AS seed 1 0.2246, AS seed 2 0.1675, gamma 3 / 10 / 30: 0.2171 / 0.2567 / 0.1870; noise 0.102, effect threshold 0.065: no effect at any gamma, no dose-response. Only hint: "two disjoint pairs" 0.104 -> 0.102 / 0.095 / 0.089 (gamma 3 / 10 / 30; -14 % at gamma 30), same order as the seed noise. Verdict: **no demonstrated effect**; no more seeds.
- **Checks.** End-to-end exact zero-mean test under the training law of x_t (tests/test_dw4_end_to_end.py): max |z| = 1.24 over 20 kappa x 3 test functions; positive control (v_t x 1.2 in the Barker factor only) 17.3; the diagnostic's own generator 1.99. State-level flux Delta_b(kappa) has the right sign for every class at every kappa <= 0.7 (z up to 12, `dw4_flux_kappa.py`), and is compatible with 0 on the reference. Involution, |det J| = 1 on the zero-CoM subspace, equivariance, detailed balance (tests/test_dw4_reflections.py). The pool-based vector diagnostic (alignment of E[h | x_t] with the exact correction) was dropped: its estimator has a non-vanishing residual that is not in kappa / v / the x_t generator.
- **Structural explanation (conjecture, not tested).** The weight information sits at intermediate-to-large kappa, where a reflection move costs dE ~ 4 and has ~1 % flux; at small kappa the term is a vector field that vanishes by symmetry.

## 5. Division (Gelfand-Dey) control variate, isotropic GMM (branch `div`; GMM d = 36 closed: first run numerical failure, clean relaunch matches Barker; DW-4 closed: no-go)

- **Method.** term = sum_k exp(-L_k + log Q_k - log C_k + log N_k(x1) + E(x1)/tau) (M_hat - m_k) / nu_{1|t}, with N_k the shrunk and q_k the unshrunk Gaussian of mode k, C_k and Q_k their smoothed marginals,
  m_k the posterior mean of x1 under q_k, pi_hat = softmax(-L + log Q), M_hat = sum_k pi_hat_k m_k (`adjoint_samplers/utils/division_cv.py`, `AdjointVEDivisionMatcher`). Zero mean under pi(. | x_t) for ANY frozen
  parameters; for disjoint modes with exact shapes and exp(-L_k) = Z w_k, E_q[term | x_t] equals the exact Tweedie correction (M* - M^q) / nu_{1|t} at every kappa (no kappa-dependent scale). Parameters (threshold centres,
  Laplace Sigma_k, L_k by log-space regression) are fitted at buffer refresh v, stored as version v and used only for samples inserted after v; gamma = 0 before the first version (warm-up).
- **Step 1 (unit tests, `tests/test_division_cv.py`, `tests/test_division_matcher.py`, thresholds fixed in the file header before the runs).**
  (a) zero mean: quadrature ~1e-27 for random parameters; Monte Carlo max |z| 2.1, positive control 144. (b) oracle, wrong-weight buffer, exact posterior sampling: separation 6 sd: cos 0.995-0.997, scale 0.987-1.006
  at kappa 0.05-0.7 (4 sd: scale 0.91-0.94; 3 sd: 0.74-0.79, reported only). (c) d = 24, 36 (isotropic frozen geometry): LITERAL FAIL at 1500 posterior draws per x_t (cos 0.88-0.97 at kappa <= 0.3), PASS at 24000 draws
  (cos 0.990-0.998, scale 0.975-1.0). (d) estimated parameters (buffer of 10 000 samples with weights ∝ w^0.5): K found = K, centres, weights and coverage exact on this GMM (the mixture is exactly representable),
  literal FAIL at d = 36 with 1500 draws, PASS with 24000 (cos 0.990-0.999, scale 0.98-0.99). Matcher end to end through its own bridge sampler and prepare_target: max |z| 2.1 (exact endpoints), 5.5 with wrong-weight endpoints.
  The threshold specification (mine) lacked a number of draws: the 1500-draw measurement was underpowered (the Monte Carlo noise of the estimate of E_q[term | x_t] biases the cosine downward; the cosine rises monotonically with the
  number of draws, the scale stays ~1). Declared amendments, accepted: that draw count; the positive control of (a) (variance x 2.5, parameters around the truth); the GMM built as in train.py (float32).
- **Step 2 (prepared, NOT submitted).** One run, d = K = 36, seed 0, frozen GMM setup (256x6, 2000 epochs, eval every 50, 10 000 samples), gamma = 1 after warm-up (`configs/experiment/gmm_as_division.yaml`,
  `scripts/gmm_as_division.pbs`). Pre-registered (`scripts/analyse_division_gmm.py`): metric = mean |log R̄| over the last 30 evaluations; seed-0 references AS big 0.38949, Barker gamma = 3 big 0.11660;
  success iff metric <= 0.5 x Barker (0.05830) and below AS; "matches Barker" iff in ]0.5, 1] x Barker; fail otherwise. Expectation: likely success, because L_k is exact on this GMM; the run validates the implementation end to end,
  not the method's generality.

- **Step 2 result (commit a8e0c53, seed 0): NUMERICAL FAILURE, as pre-registered; no rule amendment.** The run reached epoch 1999 with no NaN loss, but `div/nonfinite_frac` was 1.95e-5 (one sample of 51 200, zeroed by the guard)
  at epochs 86 and 1479, so the validity rule (== 0 throughout) fails; mean `div/clipped_frac` = 0. The verdict is not a method result. The cut removed terms of max norm 18.4 in [0.99, 0.999) and 4.2 in [0.999, 1]
  (`div/cut_diag`), the max ||term|| in kappa in [0.7, 1] was 79.6 and the max component 31: no sign of a 1/nu_{1|t} blow-up. The source of the two non-finite samples is NOT identified (the counter does not record kappa).
  **Post hoc, informative only (not a verdict):** mean|log R̄| 0.1103 (Barker 0.1166, AS big 0.3895); light quarter 0.757 (Barker 0.775, AS 0.405); min R̄ 0.571 (Barker 0.126, AS 0.111); energy_w2 0.668
  (Barker 0.720, AS 0.552), so the energy_w2 excess over AS is shared with Barker, not specific to division.
- **Offline diagnosis (CPU, `scripts/division_diag.py`).** Hypothesis: estimated N_k wider than the true mode -> N_k / p unbounded -> overflow / infinite variance of r_k. The fit takes Sigma_k from the Laplace Hessian of E at the
  centre (not from samples), N_k = N(mu_k, 0.8 Sigma_k). On PROXY buffers (10 000 exact samples, and wrong-weight samples) the fit procedure gives shrink x maxeig(Sigma_true^{-1/2} Sigma_hat_k Sigma_true^{-1/2}) in
  [0.80000, 0.80011] (<< 1, bounded) for all 36 modes; centre error 1e-5; consistent with the logged per-version fit stats of the real run (xi in [0.5, 2] for 100 % of the buffer, xi_std 6e-5, centre_err_mean 1.2-1.7e-5,
  K = 36 at the three logged versions). max_k log u_k over the proxy samples: max 2.3-2.5 (overflow at 709), Pareto k_hat median 0.04-0.07 (one mode at 0.61 in the exact proxy, from ~40-sample tails). The hypothesis is therefore not
  supported for the fit procedure; the real buffer (`samples_final.pt`, outlying endpoints of the trained sampler) and the epoch-86 / 1479 versions were NOT available locally (not saved), so items 1-2 are not checked on them.
  **Real final buffer (epoch 2000, `samples_final.pt`, run on the cluster):** K = 36, centre error 1.2e-5; shrink x maxeig in [0.80000, 0.80012] (unshrunk max 1.00015, min 0.99998), so N_k / p_k is bounded for every mode;
  max_k log u_k over samples: median -0.56, p99.9 2.00, max 2.41 (no overflow possible); Pareto k_hat of u_k: median -0.047, max 0.842, one mode above 0.5 out of 36. Since log u_k is bounded above (max 2.4) and N_k is narrower than the
  mode, the single k_hat > 0.5 is read as a tail-fit artefact on a small per-mode sample (u_k cannot have infinite variance), not as a heavy tail. The width hypothesis is refuted on the real epoch-2000 buffer; the two non-finite samples
  (epochs 86, 1479) remain unexplained (versions at those epochs not saved; the counter does not record kappa). If the term must be made robust, the remaining candidate is the x_t-dependent factor 1/C_k(x_t) for modes
  k other than the sample's own, not the x1-dependent weight.
- **Cause of the non-finite samples: t = 0 exactly (found offline, `scripts/division_t_edge.py`).** `sample_t` uses `torch.rand` in float32, which returns exactly 0.0 with probability 2^-24 ~ 6e-8 per sample. At t = 0: kappa = 0,
  v = 0, x_t = 0, so Var Q_k = kappa^2 lambda + v = 0 and the term is 0/0 = NaN (log C_k, log Q_k, m_k, softmax, log-weights and term all non-finite for the 100 % of the samples at t = 0; every intermediate is finite at the smallest
  positive float32 t = 2^-24, at 1e-8, 1e-6, at kappa = kappa_max = 0.99 (t = 0.29504) and at t = 1). Both launches (same seed 0, same random stream until the first event) hit it at epoch 86: the first run, without guards, turned
  into NaN weights; the relaunch zeroed the sample (`div/nonfinite_frac` 1.95e-5 = 1/51 200 at epochs 86 and 1479). It was not 1/nu_{1|t} nor the widths. Also found: the cut was a product `term * lam` (0 * NaN = NaN); now a `torch.where`.
  Fix (not rerun): the term is computed only for kappa in [1e-6, kappa_max] (`KAPPA_MIN`), masked with `torch.where`; unit test `test_t_edges` at t = 0, 2^-24, 1e-12, kappa = kappa_max, t = 1 and a NaN raw term above the cut
  (fails without the fix at t = 0). The fixed term is exactly 0 for kappa < 1e-6 (a set of probability ~6e-8): the zero-mean property is untouched (any function of (x_t, t)). The pre-registered verdict stays "numerical failure".
- **Step 2 relaunch after the t = 0 fix (NEW run, commit fec80d1, seed 0; same pre-registered criteria and validity rule): VALID, verdict "MATCHES BARKER" (not success).** Validity: max `div/nonfinite_frac` = 0 over all 2000 logged
  points, mean `div/clipped_frac` = 0, max |term| component 31.9, run reached epoch 1999. mean|log R̄| (last 30 evals) = 0.10583: above the success threshold 0.05830 (0.5 x Barker), within ]0.5, 1] x Barker (0.11660), below AS big
  (0.38949). Other metrics (informative): light quarter 0.767 (Barker 0.775, AS 0.405); min R̄ 0.576 (Barker 0.126, AS 0.111); energy_w2 0.647 (Barker 0.720, AS 0.552). The earlier run a8e0c53 stays "numerical failure".
  Reading: the division CV reaches Barker's level on the frozen GMM, where L_k is exact, but does not beat it by the pre-registered margin; one seed, so this validates the implementation, not generality.
- **Barker seed-0 energy_w2 at d = 36:** 0.720 (division 0.668, AS big 0.552).

- **First launch of step 2: execution failure, no metric outcome.** The loss became NaN at epoch 86 and the job was killed; the root cause was not identified (hypothesis, to be checked with the new logs, not assumed: the 1/nu_{1|t} factor as kappa -> 1).
  The relaunch counts as THE SAME pre-registered run (seed 0, same criteria, same thresholds). Changes declared before the relaunch, none of which touches the criteria:
  (1) `refresh_every` 1 -> 5 (cost); (2) one batched division_term call for all parameter versions (per-sample gather, no Python loop; same maths, end-to-end zero-mean test unchanged: max |z| 2.1 / 5.5);
  (3) guards as counters: the term is computed in float64, finiteness checked in float64 and after the cast to the training dtype; non-finite sample terms set to 0, components beyond 1e4 rescaled; logged `div/nonfinite_frac`,
  `div/clipped_frac`, `div/term_absmax`; (4) per kappa bin the max ||term|| and the kappa of the max (`div/kappa[..]/term_norm_max`, `kappa_at_max`); (5) lambda(t) = 1{kappa_t <= kappa_max} multiplying the term
  (any function of (x_t, t) keeps the exact zero mean; unit test: term exactly 0 above kappa_max, unchanged below), kappa_max = 0.99; (6) diagnostics only, NOT in the target: max ||raw term|| removed by the cut, per bin
  [0.99, 0.999), [0.999, 1] (`div/cut_diag[..]/term_norm_max`); (7) validity rule (pre-registered): the run is valid only if `div/nonfinite_frac` == 0 throughout, mean `div/clipped_frac` < 1e-4 and the run reaches epoch 1999;
  otherwise the verdict is "numerical failure", not a method result.
- **kappa_max (chosen BEFORE the run, from the oracle only: frozen d = K = 36 GMM, exact mode posterior under the target; `scripts/division_kappa_max.py`, 200 000 samples per kappa).**
  Rule 1 (absolute: smallest kappa with E[1 - max_k pi*_k(x_t)] < 1e-3) has NO SOLUTION: the ambiguity does not go below the kappa = 1 floor 2.11e-2 (overlap of the modes, intrinsic to the frozen GMM).
  Curve E[1 - max pi*]: kappa 0.02: 0.907, 0.1: 0.863, 0.3: 0.753, 0.5: 0.604, 0.7: 0.387, 0.9: 0.116, 0.95: 0.0606, 0.97: 0.0421, 0.99: 0.0275, 0.999: 0.0221, 1: 0.0211; ambiguity(0) = 1 - max w = 0.928.
  Rule 2 (relative excess, adopted): smallest kappa with (ambiguity - floor) < 1% of (ambiguity(0) - floor) = 9.07e-3. Excess: 0.98: 1.31e-2, 0.985: 9.48e-3 (fails), 0.99: 6.2e-3 (passes) => kappa_max = 0.99 (grid step 0.005).

- **DW-4 division, step 0 (offline, `scripts/dw4_division_step0.py`; GO rule pre-registered in the script header, committed before the first run).** G-averaged Gaussian basin model (O(2) x S4 x Z2, 256-angle trapezoid exact
  to 1e-13 vs 1024 angles; the degree-2 trig interpolation matches direct evaluation to 9e-13), 5 quenched basins, shrink eta = 0.2. Reference-fitted (fit and Gelfand-Dey on the 10 000 reference samples; best case, model error only):
  b_hat vs ref: triangle+pendant 0.408 vs 0.476 (|log| 0.155), path of 2 0.503 vs 0.410 (0.204), 2 disjoint pairs 0.046 vs 0.059 (0.260), K4 minus an edge 0.037 vs 0.049 (0.272), star 0.0067 vs 0.0057 (0.169). SD within basin of
  log q_bar_a + E/tau: 1.76 / 1.83 / 2.23 / 2.05 / 1.82 nats (in-sample); coverage mean 1.00, median 0.83, p5 0.019, p95 2.17. Even with reference parameters the three largest basins are off by 0.155-0.26 > 0.1: the Gaussian
  (harmonic) basin model is too coarse for the quartic wells. **AS-fitted part (go rule proper; AS run `dw4_as_states_e1500`, ed8bb67, 10 000 model samples, 100 % quench-converged; basin n: 4335 / 3746 / 1134 / 622 / 163): VERDICT NO-GO.**
  b_hat(AS-fit) vs ref vs AS weight: triangle+pendant 0.629 / 0.476 / 0.434 (|log b/ref| 0.278 vs AS 0.094); path of 2 0.289 / 0.410 / 0.375 (0.351 vs 0.090); 2 disjoint pairs 0.0064 / 0.059 / 0.113 (2.231 vs 0.650);
  K4 minus an edge 0.067 / 0.049 / 0.062 (0.305 vs 0.239); star 0.0097 / 0.0057 / 0.0163 (0.533 vs 1.051). (1) |log b_hat/ref| <= 0.1 for the three largest basins: False (0.28, 0.35, 2.23); (2) closer to ref than AS for all five: False
  (worse in four of five; better only for the star). The estimator from AS samples is worse than the AS weights themselves; DW-4 division stops here, as pre-registered.

- **DW-4 division, variant B, step 0 (`scripts/dw4_division_b.py`, job 95063; rule and eta_a committed before the run; AS samples `results/dw4_e1500_ed8bb67_samples.pt`): VERDICT NO-GO, (c) fails.**
  eta_a (fixed from the model fitted on reference half A): triangle+pendant 0.3, path of 2 0.2, two disjoint pairs 0.8 and K4 minus an edge 0.8 (neither reaches < 1 % leakage of pair distances into [3.5, 4.5]: 8.2 % and 1.2 % at 0.8), star 0.8.
  Disclosure: a plumbing smoke test with reference samples standing in for AS ran steps 2-4 once before the job, so part (c) was seen before the formal run (it is identical in the job: it does not use AS samples).
  Leakage correction: f_a = 1.000 (0.999 for star) in every basin, so the correction changes nothing (b_hat with and without it are identical): quench leakage is not the problem.
  (c) ref-fitted (half A fit, half B weights), |log b_hat/ref|: triangle+pendant 0.174 (SE of b_hat 0.060), path of 2 0.161 (0.073), two disjoint pairs 0.066 (0.014), K4- 0.047, star 0.596: FAILS the <= 0.1 on the two largest basins.
  (d) AS-fitted: triangle+pendant 0.069 (AS 0.119), path of 2 0.110 (AS 0.073), two disjoint pairs 0.476 (AS 0.664), K4- 0.179 (0.246), star 0.261 (0.893): closer than AS in 4 of 5 (passes that clause), but two disjoint pairs 0.476 > 0.2: FAILS.
  (b) k_hat of u_a over all AS samples: 0.18, 0.22, 6.96, 1.14, 1.14 (restricted to the basin: 0.65, 0.57, 0.39, 0.78, 1.64): FAILS (three small basins). SD of log N_bar_a + E/tau on reference half B: 2.85, 2.76, 12.7, 16.2, 10.5.
  IS ESS (report only): ref-fitted proposal 6292 / 20000 (31.5 %), AS-fitted 2273 / 20000 (11.4 %).
  Pre-registered interpretation: (c) fails, so the single-Gaussian model is not enough; variant A is the user's call. Remarks (not part of the rule): the (c) deviations on the two big basins are of the order of the bootstrap SE
  (b_hat SE 0.06-0.07, i.e. ~15 % relative, with 2400 / 2000 fit samples and a within-basin SD of ~2.8 nats), so (c) is limited by estimator noise as much as by the model; the AS-fitted estimate (10 000 samples, SE 0.016)
  is closer to the reference than the ref-fitted one on the two big basins. The small, anharmonic basins (SD 10-16 nats) are where the single Gaussian breaks down. The AS file used here (AS weights 0.420 / 0.382 / 0.121 / 0.061 / 0.017)
  differs from `dw4_as_states_e1500/seed0` used in step 0 (0.434 / 0.375 / 0.113 / 0.062 / 0.016). Resolved: `results/dw4_as_states_e1500/seed0/samples_final.pt` was overwritten by run 94252 (same output folder), so the "AS-fitted" column of the FIRST division step 0 (`dw4_division_step0.py`, NO-GO) used AS run 94252, not ed8bb67; variant B used ed8bb67.

- **DW-4 division, post-hoc baselines (`scripts/dw4_division_b_baseline.py`, job 95510; report only, no rule, no training; AS samples ed8bb67, reference half B, N = 20000 exact draws from the step-4 proposal, 500-bootstrap SE).**
  Endpoint IS on FRESH draws of the proposal (self-normalised weights e^{-E}/proposal; the AS samples are not reweighted, they only fit the proposal; draws quenched), |log p/ref| per basin [triangle+pendant, path of 2, two disjoint pairs, K4-, star], next to |log AS/ref| = 0.119, 0.073, 0.664, 0.246, 0.893 (sum 1.995):
  - ref-fitted proposal: Pareto k_hat 0.630, ESS 6292 (31.5 %). p_IS 0.403 / 0.488 / 0.064 / 0.040 / 0.0053 (SE 0.006 / 0.006 / 0.005 / 0.003 / 0.0006); |log p_IS/ref| 0.161, 0.173, 0.031, 0.170, 0.285 (sum 0.819).
  - AS-fitted proposal: Pareto k_hat 0.724 (> 0.7: ESS 2273 / 11.4 % unreliable). p_IS 0.408 / 0.474 / 0.057 / 0.037 / 0.025 (SE 0.010 / 0.010 / 0.004 / 0.004 / 0.015); |log p_IS/ref| 0.148, 0.143, 0.096, 0.244, 1.261 (sum 1.892). The star value is 1 draw-weight-dominated tail (SE 0.015 on 0.025), not a result.
  Basin-level reweighting of the AS samples (p_a = b_hat_a, leakage-corrected Gelfand-Dey, no IS): |log b_hat/ref| = 0.069, 0.110, 0.476, 0.179, 0.261 (sum 1.095) with the AS-fitted model; 0.174, 0.161, 0.066, 0.047, 0.596 (sum 1.043) with the ref-fitted model.
  Reading (post hoc, not a rule): with the ref-fitted proposal, IS cuts the summed error from 1.995 (AS) to 0.82 and fixes the 'two disjoint pairs' basin (0.664 -> 0.031), but it is WORSE than raw AS on the two big basins (0.161 / 0.173 vs 0.119 / 0.073) whose bias exceeds the tiny bootstrap SE (0.006): the bootstrap only sees draw noise, not the model/fit error, so those SEs are not a measure of accuracy.
  With the AS-fitted proposal (the realistic one) IS gives no gain (1.892 vs 1.995, star dominates; without the star 0.631 vs 1.102). Basin-level reweighting with the AS-fitted model: 1.095 total, best on the big basins (0.069 / 0.110) and still bad on the two disjoint pairs (0.476).
  ESS caveat: the ref-fitted ESS was 6292 on the cluster but 329 (1.6 %) in a local run with the same seed and the same fit (b_hat equal to 4 digits); with k_hat ~ 0.6-0.7 the ESS is dominated by a few weights and depends on the torch random stream, so treat it as one noisy estimate.

- **Division, masses-perturbation hypothesis, step 0 (GMM D = K = 12, `scripts/division_mass_perturb.py`, local CPU 4 s; rule committed before the run): INCONCLUSIVE (declared rule: control cos <= 0.9).**
  Setting: oracle shapes and centres, L_k = -log w_k + delta_k (delta = +-0.5, draws 0-2) and a control delta = 0, wrong-weight buffer w^0.3 (w'/w ratio 5.6), 400 x_t x 60 exact posterior draws (24000) per kappa, c* = (M* - M^q)/nu_{1|t} exact.
  Pooled cos(E_q[term|x_t], c*) (scale = projection coefficient, 1 = exact correction), kappa = 0.05 / 0.1 / 0.2 / 0.3 / 0.5 / 0.7:
  control 0.66 / 0.68 / 0.72 / 0.76 / 0.79 / 0.83 (scale 0.95-1.02); draw 0 0.60 / 0.62 / 0.64 / 0.68 / 0.71 / 0.75 (scale 0.89 -> 0.74); draw 1 0.71 / 0.74 / 0.76 / 0.80 / 0.81 / 0.84 (scale 1.17 -> 1.04); draw 2 0.62 / 0.63 / 0.66 / 0.70 / 0.72 / 0.74 (scale 1.15 -> 0.92).
  q = pi check: R = ||e||^2 / (estimator variance) = 0.92-1.06 for the control and the three draws at every kappa, i.e. E_pi[term | x_t] = 0 within noise for any L.
  Reading: the pre-registered cos > 0.5 holds for all draws at all kappa, but the control (delta = 0) itself only reaches cos 0.66-0.83 (< 0.9): the Monte-Carlo noise of 60 draws per x_t caps the pooled cos, so by the rule declared before the run the verdict is inconclusive, not pass.
  Post hoc (not a rule): cos(draw) / cos(control) = 0.90-0.91 (draw 0), 1.08-1.10 (draw 1), 0.93-0.89 (draw 2), i.e. within ~10 % of the noise ceiling at every kappa, and the scale stays within [0.74, 1.19] of the exact correction. This is suggestive of the direction being preserved, not established.
  Step 1 (DW-4 refit) NOT run: it is conditional on a pass. A rerun with more draws per x_t (e.g. 400 x 600, seconds on CPU) would remove the noise ceiling; it needs a declared amendment of the 24000-draw protocol.

- **Division, masses-perturbation hypothesis, step 0 RERUN (400 x_t x 600 draws per kappa; amendment = draw count only, declared in the commit before the run; pass rule and control ceiling unchanged): PASS.**
  Pooled cos (debiased cos_corr, report only; scale) at kappa = 0.05 / 0.1 / 0.2 / 0.3 / 0.5 / 0.7:
  control 0.940 / 0.947 / 0.953 / 0.964 / 0.972 / 0.981 (corr 1.000-1.003; scale 0.98-1.00); draw 0 0.851 / 0.854 / 0.865 / 0.879 / 0.889 / 0.914 (corr 0.904-0.935; scale 0.86 -> 0.76); draw 1 0.924 / 0.930 / 0.932 / 0.944 / 0.948 / 0.953 (corr 0.962-0.968; scale 1.14 -> 1.08);
  draw 2 0.858 / 0.861 / 0.868 / 0.885 / 0.895 / 0.900 (corr 0.909-0.922; scale 1.12 -> 0.97). Control cos >= 0.94 > 0.9 at every kappa (ceiling ok); every perturbed draw has cos > 0.85 > 0.5 at every kappa. q = pi: R = 0.94-1.06 (zero mean within noise).
  Debiased cosine at 60 draws (report only): control 0.986-1.021, draws 0.885-0.993, i.e. consistent with the 600-draw values (the 60-draw cos was the noise ceiling, as diagnosed).
  The residual loss of direction for L perturbed by +-0.5 is real (cos_corr 0.90-0.97, not 1) and the scale moves by about -25 % / +16 %: the correction still points to the right weights but is not invariant to the masses (hence cos < 1), as the argument predicts through pi_hat.
  Step 1 (DW-4 refit) follows.

- **DW-4 division, step 1 (EM refit of the single-Gaussian G-averaged basin models, `scripts/dw4_division_em.py`, cluster job, AS samples ed8bb67; rule committed in 60f4107 before the run): FAIL (k_hat < 0.5 for every basin not met).**
  EM: 40 iterations (stopping tolerance 1e-4 not reached: last max change 2.1e-4, mean log-lik moves by < 3e-4 per iteration after iteration 30, dominated by the star basin), 49 s. Mean log-lik per basin, aligned initialisation -> EM: -7.244 -> -6.801 (triangle+pendant), -6.769 -> -6.355 (path of 2), -7.662 -> -6.591 (two disjoint pairs), -6.902 -> -6.163 (K4-), -7.061 -> -6.125 (star): the ML fit is much better than the hard-aligned one.
  eta_a (EM model, variant-B rule): 0.6, 0.3, 0.8 (flagged), 0.8, 0.8 (flagged).
  Pareto k_hat of u_a over ALL AS samples (EM / old aligned model): triangle+pendant -0.133 / 0.180, path of 2 0.088 / 0.216, two disjoint pairs 4.383 / 6.955, K4- 1.165 / 1.140, star 1.165 / 1.140: FAILS for the three small basins. In basin (diagnostic): -0.172, 0.223, 0.746, -0.024, 0.252.
  SD of log N_bar_a + E/tau on the AS samples (EM / old): 3.32 / 2.74, 1.90 / 2.74, 8.08 / 15.76, 7.17 / 13.53, 4.78 / 12.61 (on reference half B, EM model: 2.66, 1.68, 6.69, 6.46, 3.30): the EM fit cuts the spread of the three small basins by 2-3x but leaves it at 5-8 nats.
  Report only: b_hat (EM, AS-fitted, leakage-corrected) vs ref_B, |log b/ref|: 0.114, 0.139, 0.055, 0.127, 0.184, sum 0.618 (old aligned model 1.095, raw AS 1.995); SE 0.001-0.006. Improvement mostly on the small basins (two disjoint pairs 0.476 -> 0.055, K4- 0.179 -> 0.127, star 0.261 -> 0.184), the two big basins slightly worse than the old model (0.114 / 0.139 vs 0.069 / 0.110) and slightly worse than raw AS on path of 2 (0.139 vs 0.073).
  Observation (not tested): K4- and star have the SAME k_hat as in the old model (1.165 vs 1.140, identical to each other); with k_hat over all AS samples but in-basin k_hat of -0.02 / 0.25, the heavy tail of u_a for these basins comes from samples of OTHER basins (or from a single dominant sample, which makes the GPD fit degenerate), not from the basin itself. Not diagnosed: which samples.
  Per the user's instruction: stop after step 1, no matcher work.

- **DW-4 division, step 1', preparation job (cluster, AS samples ed8bb67): post hoc top-10 diagnostic of the step-1 tail (report only) and choice of c_a (model only).**
  Top-10 (untruncated EM model, step-1 eta_a), for the three failing basins. The user's prediction (high-energy samples with a stretched pair distance, well beyond 5.5) is NOT confirmed: the AS samples have max pair distance 6.31 overall (median 5.59, p99 6.05) and no sample is stretched;
  the top-u_a samples are ordinary, low-energy members of their own basin: two disjoint pairs top 100 in own basin 100 %, E -21.9 (median), max pair distance 5.55, share of sum_i u_a carried by top 1 / 10 / 100 = 0.028 / 0.227 / 0.776; K4-: 100 %, E -22.4, 5.01, 0.025 / 0.185 / 0.818;
  star: 95 % (5 of the top 100 are path-of-2 samples), E -19.9, 5.45, 0.093 / 0.593 / 0.999. Top 10 with max pair distance > 5.5: 5 / 0 / 0.
  Reading (post hoc): in the AS data, the weight of u_a is carried by a few tens of in-basin samples (sd of log u within basin 5-8 nats), not by far outliers; the far tail (e^{c d^4} vs e^{-d^2}) that the user's argument predicts is not sampled by AS at all (no pair distance beyond 6.3),
  so it cannot be what the empirical k_hat (fitted on the top 20 % of ALL AS samples, a mixture of in-basin and other-basin samples) measures here. The argument itself (E_pi[u_a^2] = infinity) is untested by these data.
  c_a (rule of step 1', model only, EM fit 40 iterations 43 s; scripts/dw4_division_trunc_params.json committed before the evaluation): max E over the box at c = 1.5 is -7.4 / -9.6 / -9.2 / -8.4 / -8.0 against E_min + 6 = -19.3 / -19.8 / -18.4 / -17.5 / -15.1: NO c in {1.5, 2, 2.5, 3} satisfies the rule in any basin
  (c = 2: +11.8, +11.5, +4.0, +9.7, +6.1). By the declared default c = 1.5 for all five basins, flagged. The box corners are at Mahalanobis radius c sqrt(6) = 3.7 at c = 1.5, which is where the 12-18 nats of E come from.

- **DW-4 division, step 1' (truncated components, `scripts/dw4_division_trunc.py`, cluster job 95634, AS samples ed8bb67; rule, c_a and eta_a committed before the run: b4f26a4): FAIL (k_hat < 0.5 for every basin not met).**
  Unit tests PASS (committed parameters == the file written by the preparation job: True): T1 normalisation |z| <= 1.46; T2 convolution identity |z| <= 2.70 (36 points, 2 components, kappa 0.2 / 0.5 / 0.8); T3 exact zero mean of r_a - r_b under the toy target |z| <= 1.53; positive control |z| = 5.94 at kappa 0.7 (psi = 1) but only 0.96 at kappa 0.3, so the control clears the declared threshold of 5 once of four (thin margin).
  c_a = 1.5 in all five basins (flagged: the rule has no solution), eta_a = 0.6, 0.3, 0.8, 0.8, 0.8. Fraction of AS samples of basin a inside the box of a: 71.3 %, 74.4 %, 73.0 %, 77.2 %, 81.9 % (so 18-29 % of own-basin samples carry u_a = 0).
  Pareto k_hat of u_a over ALL AS samples: triangle+pendant -0.170, path of 2 -0.206, two disjoint pairs 2.353, K4- 3.410, star 6.480 (n(u>0) 2991 / 2840 / 885 / 468 / 164): the same three basins fail. sup u / mean u (log): empirical max over the AS samples 2.9 / 2.6 / 5.7 / 5.5 / 6.8; model bound N_trunc(0) e^{E_max,box}: 28.2 / 25.5 / 27.6 / 28.7 / 28.5 (finite by construction, e^28 = 1e12: not informative).
  Report only: b_hat vs ref_B, |log b/ref| = 0.117, 0.143, 0.058, 0.131, 0.191, sum 0.640 (step 1 untruncated EM: 0.618; raw AS 1.995).
  Reading (post hoc): u_a is now bounded by construction, yet k_hat for the three small basins is still far above 0.5 (and larger than in step 1: 2.4 / 3.4 / 6.5 against 4.4 / 1.2 / 1.2, with fewer positive values). So the heavy-tail diagnostic is not caused by the unbounded far tail of e^{E} against a Gaussian, and
  truncation does not repair it; together with the top-10 diagnostic (the weight sits on a few tens of ordinary in-basin samples, within-basin sd of log u 5-8 nats) the evidence points to the mismatch of a single Gaussian shape with e^{-E} inside the basin, not to the far tail. Also, the GPD k_hat on 'the top 20 % of ALL samples' is a crude
  estimator on bounded, concentrated, multi-basin data (the finite-sample fit of a bounded variable can exceed 0.5); no alternative estimator was run. Stop after step 1' as instructed: no matcher work.

- **CONCLUSION, division CV on DW-4 (closed by the user after step 1'; no further DW-4 work).** Single-Gaussian G-averaged basin models (hard-aligned, EM, truncated) give weights u_a = N_bar_a e^{E/tau} with a within-basin SD of log u of 5-8 nats in the small basins (two disjoint pairs, K4 minus an edge, star),
  and Pareto k_hat far above 0.5 there in all three variants (4.4 / 1.2 / 1.2 EM, 2.4 / 3.4 / 6.5 truncated); the basin-reweighting error stays at 0.62-0.64 (EM, truncated) against 1.10 (aligned) and 1.99 (raw AS), without ever reaching the pre-registered passes. The far-tail hypothesis (e^{E} quartic against a Gaussian) is refuted by the
  top-10 diagnostic (no stretched samples) and by the truncated run (bounded u_a, same failure). Division requires convolution-tractable mode models that dominate pi pointwise, and this fails for anharmonic basins.
- **POSITIVE FINDING, GMM mass robustness of the division correction (offline, `scripts/division_mass_perturb.py`; oracle shapes and centres, masses perturbed by +-0.5 nat, wrong-weight buffer, 24000 x 10 exact posterior draws per kappa; amendment: 600 draws per x_t, declared before the rerun):** the direction of E_q[term | x_t] stays
  within the exact Tweedie correction c* up to the declared tolerance, cos_corr = 0.90-0.97 (pooled cos 0.85-0.95; control cos >= 0.94), scale -25 % / +16 %, kappa 0.05-0.7, 3 independent perturbations; E_pi[term | x_t] = 0 within noise for any masses. So the correction does not need exact masses on the GMM (basin reweighting, by contrast, inherits the mass error directly).

- **Division CV, learned coefficient (scalar beta_s(kappa) or matrix M(kappa) minimising the target variance) vs the derived coefficient 1 (`scripts/division_beta_opt.py`, GMM D = K = 12, offline, 13 s; rule committed before the run, metric amended before the full run after a 60-draw smoke test): VERDICT NOTHING (no variance gain) for both.**
  Setting: as the mass-perturbation test (oracle shapes, masses +-0.5 nat on 3 draws + control, buffer w^0.3), 400 x_t x 600 draws under q and under pi, kappa 0.05-0.7, coefficients fitted on half the x_t and evaluated on the other half. Metrics: V = within-x_t variance of f + c term (f = -adjoint1, the AS target);
  Bc = || E_q[c term | x_t] - c* ||^2 (c* the exact Tweedie correction), noise-floor corrected, relative to ||c*||^2. Amendment (declared before the full run): the f-based bias B(c) = ||E_q[f + c term] - E_pi[f]||^2 has no power at 60 or 600 draws (B(0) ~ 0.003-0.03, of the order of its noise floor), so Bc is primary and B(c) is a secondary column.
  Validity (control, c = 1: Bc(1)/Bc(0) < 0.1 at every kappa): met (|Bc(1)/Bc(0)| <= 0.005): with oracle masses the coefficient 1 reproduces the exact correction.
  Variance: the term hardly changes the regression noise: V(1)/V(0) = 0.96-0.996 and the best learned coefficient V(beta_s)/V(1) >= 0.99, V(M)/V(1) >= 0.99 (best 0.95/0.96 at kappa 0.7): no kappa reaches the declared gain of 10 % (0/6 for beta_s and M in every run). The variance of the target is carried by f, not by the term.
  beta_s is systematically below 1 (0.74 -> 0.67 control, 0.70-0.69, 0.68-0.61, 0.56-0.52 for the three perturbed draws, kappa 0.05 -> 0.7): variance minimisation SHRINKS the correction. In the control (oracle masses) this degrades the bias: Bc(beta_s)/Bc(0) = 0.07-0.12 against 0.00 for c = 1; M likewise 0.08-0.12.
  With perturbed masses c = 1 is no longer exact (Bc(1)/Bc(0) = 0.12-0.19 draw 0, 0.09-0.13 draw 1, 0.18-0.28 draw 2); the fitted M is lower than c = 1 in draws 0 and 2 (0.10-0.15 and 0.08-0.16) but higher in draw 1 and in the control: no consistent benefit, no claim.
  Conclusion: a learned coefficient is not worth it here: it gains at most a few % of regression variance and costs part of the bias correction. The derived coefficient 1 is kept. Not tested: a coefficient trained on a bias-aware objective (needs pi samples).

- **Portability note (found while preparing the DW-4 inflated-proposal test):** the V-coordinates (mu_a, Sigma_a) stored in `scripts/dw4_division_trunc_params.json` (and any fit made with `scripts/dw4_division_step0.py`) depend on the basis `B` of the zero-CoM subspace, taken from `torch.linalg.eigh` of a projector with a degenerate eigenvalue: this basis is machine/library dependent.
  On a local machine, E(mu_a) of the committed EM means is 101 / 52 / -5 / 89 / 61 instead of ~ -25 (a different B maps the same coordinates to other configurations). All the cluster results (steps 1, 1', unit tests) were produced and evaluated in the same environment and are internally consistent; a local run of `dw4_is_inflate.py` aborts on a sanity check.
  A local 2000-draw smoke test of that script (ESS ~ 3 of 2000) was run with this wrong basis before the check existed: it is meaningless and is not a result.

- **DW-4, post-hoc IS / independence-MH on endpoints with an ENCOMPASSING proposal (`scripts/dw4_is_inflate.py`, run LOCALLY after the cluster access was lost; rule committed in 808350a before the run; components refitted here by EM on the AS samples ed8bb67 with the new deterministic basis, commit after 808350a):**
  Proposal = mixture of G-averaged basin components (mixture weights = AS basin frequencies), Gaussian N(mu_a, s Sigma_a) or t5 with scale s Sigma_a; N = 20000 draws, w = e^{-E}/r. Pre-registered usable iff ESS > 50 % and k_hat < 0.5.
  Gauss s = 1 / 1.5 / 2 / 3: ESS 60.6 / 46.1 / 29.8 / 13.9 %, k_hat 0.128 / -0.040 / -0.013 / 0.348, IMH acceptance 0.64 / 0.45 / 0.33 / 0.21; t5 s = 1 / 1.5 / 2: ESS 61.6 / 46.2 / 32.5 %, k_hat -0.091 / -0.361 / -0.246.
  USABLE: gauss s = 1 and t5 s = 1 (2 of 7): an encompassing proposal IS usable on DW-4 by the pre-registered rule. Inflation does not help (ESS falls with s, as expected once the proposal already dominates pi).
  Basin proportions (sum_a |log p / ref_B|, raw AS 1.995): IS 0.579 / 0.577 / 0.546 / 0.508 (gauss) and 0.631 / 0.638 / 0.623 (t5); IMH 0.611 / 0.703 / 0.674 / 0.445 and 0.631 / 0.653 / 0.721. Per basin (gauss s = 1, IS): 0.146, 0.163, 0.011, 0.205, 0.053 (bootstrap SE of p_IS 0.001-0.005).
  Observation (not tested): the IS estimates of the two big basins (about 0.40 / 0.49) are stable across proposals and across this run and the earlier fitted-proposal IS (0.4025 / 0.4878), with ESS up to 60 % and SE 0.005, yet differ from ref_B (0.473 / 0.410) by 0.15-0.17 in log: far above the bootstrap SE and the reference sampling error (~0.015).
  This suggests that the reference MCMC proportions of the two dominant basins may not match e^{-E}/Z (or that every proposal misses the same region); the small basins (two disjoint pairs 0.064 vs 0.062, star 0.0053 vs 0.0070) agree with the reference. This is a hypothesis to check before using ref_B as the truth for the big basins.
  Change of the basis: `dw4_division_step0.basis()` is now deterministic (Helmert), so any fit made from here on is portable; the cluster JSON of step 1' is not valid in the new basis (it stays as the record of those cluster runs; all their results are basis independent).

- **DW-4, path-space importance weights of the AS controller (`scripts/dw4_path_weights.py`, local CPU 89 s; rule committed before the run; controller = AS run 94252, NOT ed8bb67 whose controller was overwritten; report only):**
  Exact discrete-time Girsanov weights of the simulated paths against P* = pi(x1) B_disc(path | x1) (99 Euler steps, 10000 paths). Unit test passed first (toy 2-D, |z| <= 2.02 for the exact formula; control with the sign of <u, xi> flipped: |z| = 335).
  log w: sd 1.10, range 9.3; ESS 3403 / 10000 (34.0 %); Pareto k_hat 0.320; max weight share 0.0024; independence-MH acceptance 0.469 (100 chains x 99 steps).
  Pre-registered (usable iff ESS > 50 % and k_hat < 0.5): NOT usable (ESS 34 %, k_hat passes). Informative anyway: sum_a |log p / ref_B| = 1.860 (raw samples) -> 0.809 (IS) / 0.707 (MH). Per basin [tri+pendant, path of 2, two disjoint pairs, K4-, star], |log p / ref_B|: raw 0.090, 0.094, 0.664, 0.191, 0.820; IS 0.138, 0.132, 0.107, 0.143, 0.289; MH 0.124, 0.087, 0.184, 0.061, 0.251.
  p_IS = 0.412 / 0.468 / 0.0695 / 0.0411 / 0.0093 (SE 0.008-0.009 on the big two), p_MH = 0.418 / 0.448 / 0.075 / 0.050 / 0.0090, ref_B = 0.473 / 0.410 / 0.062 / 0.047 / 0.0070, raw = 0.432 / 0.373 / 0.121 / 0.057 / 0.0159.
  The path-space weights fix the two small basins that AS gets wrong (two disjoint pairs 0.664 -> 0.107, star 0.820 -> 0.289) with no fitted density and no basin model, while the two dominant basins move from the reference (0.09 -> 0.13) exactly as in the endpoint IS with an encompassing proposal.
  Consistency (observation): the path-space IS (0.412 / 0.468) and the endpoint IS (0.40 / 0.49, all fitted proposals) are two independent estimators of the basin proportions of e^{-E}/Z that agree with each other and differ from ref_B on the two big basins (0.473 / 0.410), by 7 bootstrap SE: the reference proportions of the dominant pair may not be those of e^{-E}/Z; this has not been tested (it would need an independent long MCMC or a high-ESS estimate).

- **DW-4, minimum-to-minimum DARTING kernel as a Barker control variate (offline test `scripts/dw4_darting_test.py`, local CPU 84 s per run; GO rule in the header, committed unchanged before the runs: as received in the commit before 898fd6b, adapted in 898fd6b): VERDICT NO-GO (T1 fails; T0, T2, T3 pass).**
  Adaptations declared before the run (898fd6b, rule untouched): energy = the repo's convention (d = sqrt(|r|^2 + 1e-8)), asserted against `DoubleWellEnergy.eval`, its autograd and `dw4_quench.energy_grad_hess` on 100 reference samples (agreement 7e-15; the standalone formula of the script differed by 3.4e-8 in E and 1.0e-7 in the gradient because of that 1e-8 smoothing, so the 1e-8 requirement was not met by it);
  strict quench check ||grad|| < 1e-8 (non-converged samples excluded, non-converged landings counted invalid; in the runs 100 % converged: 3000/3000, 5000/5000, 26688/26688 and 48072/48072 landings); label mapping (edges + sorted degrees) asserted one-to-one against the repo names on the quenched reference (0 mismatches);
  AS samples read from `model_samples` (the first key of the dump is `buffer_x1`). A 700-sample smoke run on the reference was seen before the real runs (T1 failed at that small n).
  (a) ref = data/test_split_DW4.npy (3000 random), AS = ed8bb67 `model_samples` (5000), seed 0: T0 max |M_ba M_ab - Id| = 1.1e-15, round trip 1.6e-15 (n = 500): PASS. Valid landing 0.810, mean Barker weight 0.247. Reference weights: triangle+pendant 0.484, path of 2 0.411, two pairs 0.055, K4- 0.043, star 0.0077.
  T2 out-flux at kappa = 0 / 0.1 / 0.3 / 0.5 / 0.7: triangle+pendant 0.147 / 0.142 / 0.125 / 0.104 / 0.067; path of 2 0.127 / 0.123 / 0.111 / 0.088 / 0.060; two pairs 0.323 / ... / 0.153; K4- 0.532 / ... / 0.284; star 0.783 / ... / 0.466: PASS (>= 0.10 on the two dominant basins at kappa = 0, versus ~1 % for the 24 reflections).
  T1 flux balance (log ratio / SE), pairs with both fluxes > 1e-3: tri-path +0.312 / 0.042 (7.4 SE), tri-2pairs +0.333 / 0.100, tri-K4- -0.005 / 0.107, tri-star -0.058 / 0.237, path-2pairs +0.041 / 0.113, path-K4- -0.300 / 0.105, path-star -0.366 / 0.231, 2pairs-K4- -0.282 / 0.166, K4--star +0.048 / 0.247: FAILS (the dominant pair is out of balance by 7 SE).
  T3 (AS vs reference): entering basins two pairs (ref 0.0547, AS 0.1178, Delta_b -0.0179, z -8.2), K4- (0.0430 / 0.0604, -0.0076, z -3.9), star (0.0077 / 0.0144, -0.0047, z -3.5): all with the right sign and |z| >= 2: PASS. Dominant basins do not enter by the rule's definition; Delta_b = +0.0071 (z +3.4) for tri+pendant and +0.0231 (z +11.7) for path of 2.
  (b) control (ref = half A of the reference (3000 of 5000), AS = the OTHER half B (5000)): T0 and T2 as in (a) (valid landing 0.825); T1 the same imbalance on the dominant pair (+0.325 / 0.043, 7.6 SE; path-K4- -0.495 / 0.098; 2pairs-K4- -0.551 / 0.134); T3: no basin enters (as expected), but Delta_b is NOT ~0 for the dominant pair: triangle+pendant -0.0168 (z -8.3), path of 2 +0.0181 (z +9.4); other basins |z| <= 2.4.
  So the kernel moves the reference's own other half: mass flows from triangle+pendant to path of 2 even though both halves come from the same sampler. Two explanations, not distinguished here: (1) the kernel is not exactly pi-invariant for these two basins (a Jacobian / Hastings term), (2) the reference proportions of the two dominant basins are not those of e^{-E}/Z.
  Explanation (2) is quantitatively consistent with the independent IS estimates above (path-space IS and endpoint IS: 0.41-0.43 / 0.47-0.49 for the two dominant basins versus 0.48 / 0.41 in the reference): the log-ratio shift implied for the tri-path flux is ~ +0.29 (observed +0.31 and +0.33), and about +0.16 / -0.13 for the pairs with the small basins (observed within about 1.7 SE of that, with scatter). Not a proof; a decisive check would be the same flux balance on importance-weighted exact-pi samples (IS draws), which does not use the reference proportions.
  Consequence for the rule: the pre-registered verdict is NO-GO because of T1; the darting kernel itself reaches the 10 % out-flux and passes the AS signal test T3, so T1 and the control are the only open question.

- **POST HOC follow-up of the darting test (`scripts/dw4_darting_is_balance.py`, local CPU 109 s; declared post hoc, interpretation fixed before the run): the T1 imbalance is a property of the REFERENCE, not of the kernel.**
  The lumped flux balance of the same kernel (kappa = 0) recomputed on IMPORTANCE-WEIGHTED samples of e^{-E}/Z (6000 draws of the encompassing endpoint proposal, gaussian s = 1, EM components fitted on the AS samples ed8bb67, ESS 4074 = 67.9 %; no reference proportion used; 100 % quench convergence; bootstrap 300):
  tri-path -0.001 +- 0.037 (z -0.0) [reference: +0.312 +- 0.042], tri-2pairs +0.055 +- 0.071, tri-K4- +0.090 +- 0.072, tri-star +0.299 +- 0.134 (z +2.2), path-2pairs +0.080 +- 0.081, path-K4- +0.058 +- 0.069, path-star +0.242 +- 0.125 (z +1.9), 2pairs-K4- +0.025 +- 0.104: every pair |z| < 3.
  Declared interpretation: consistent with pi-invariance of the kernel on these samples; the imbalance of +0.31 found on the reference (and the Delta_b of the control) is a property of the reference: its dominant-pair proportions are not those of e^{-E}/Z.
  Weighted basin proportions of e^{-E}/Z (IS, 6000 draws): triangle+pendant 0.418, path of 2 0.477, two pairs 0.063, K4- 0.036, star 0.0052 (earlier IS and path-space IS: 0.40-0.43 / 0.47-0.49; reference ref_B: 0.473 / 0.410 / 0.062 / 0.047 / 0.0070).
  Caveats: the proposal is fitted on the AS samples (a region missed by all the IS runs would bias them in the same way; the path-space IS does not use a fitted density and agrees); the pairs with the star have z up to +2.2 (same sign, within the 3 SE).
  Consequence (to be handled, not yet done): every DW-4 metric of this log that treats the reference basin frequencies as the truth (the AS weight errors of section 4, the pre-registered Barker rule `dw4_barker_rule.py`, the continuation rule, the division go rules (c) and (d), the post hoc baselines) used frequencies of the two dominant basins that differ from e^{-E}/Z by about 0.15 in log:
  against the IS estimate, AS (ed8bb67: 0.420 / 0.382) is close to the target for triangle+pendant (0.418) but too low for path of 2 (0.382 vs 0.477, log -0.22), instead of the small deviation measured against the reference.

## Methodology notes

- Pair arms by seed: arms sharing a seed share their excursions, so seed-paired differences are the low-noise statistic.
- kappa_t > 0.7 for ~92 % of the regression times in this VE schedule.
- Compare time-averaged proportions (per-state, then TV), not snapshots: snapshot-to-snapshot TV moves by 0.03-0.08 (bootstrap SEs underestimate it).
- Read W&B history with the metric keys that are logged at every evaluation (a key logged less often silently drops the other rows); filter runs by `git_commit`.
- Hydra changes the working directory: paths given to `train.py` (`init_controller`, `init_buffer`) are resolved from the launch directory.

## Amendement : référence DW-4

Rules: `rules_reference_check.md` (commit 3d5b062, before any run of this series). Nothing above is rewritten. Everything ran locally on CPU (PT 52 s, analysis 2 s, T1' 81 s, Task 4 154 s), no deviation from the protocol (the PT run did not need to be shortened).
Scripts: `scripts/dw4_pt.py`, `dw4_reference_check.py`, `dw4_darting_T1prime.py`, `dw4_reference_rows.py`. PT states: `results/pt_samples_T1.pt` (not committed).

- **Design error declared (darting test T1).** T1 of `scripts/dw4_darting_test.py` treated the reference `data/test_split_DW4.npy` as samples of pi = e^{-E}/Z. The NO-GO of T1 stays the written verdict; the design error is that T1 was computed on a sample that, by Task 1 below, does not have the basin proportions of pi.
- **Task 1, independent parallel tempering: VALID, verdict "Référence biaisée".** 64 chains x 8 temperatures T = 5^(k/7), 2e5 steps (burn-in 4e4), 1600 saved T = 1 states per chain, E = `DoubleWellEnergy`, quench 100 % converged. Acceptance 0.30 at every T after the burn-in, swap acceptance 0.77-0.86.
  Validity: (i) split R-hat of the basin indicators 1.000-1.001 for the 5 basins, over the 64 chains and within each initial group, group means agree (|z_diff| <= 1.4); (ii) 34 508 transitions triangle+pendant <-> path of 2 between consecutive saved states (>= 500); (iii) KS on the T = 1 energies between the reference-started and AS-started groups: statistic 0.0030, p = 0.97 (the saved states are autocorrelated, so this p is optimistic).
  p_PT (SE over the 64 chains) = triangle+pendant 0.4127 (0.0021), path of 2 0.4792 (0.0021), two disjoint pairs 0.0644 (0.0007), K4 minus an edge 0.0374 (0.0007), star 0.0063 (0.0002); reference (10 000 samples) 0.4762 / 0.4099 / 0.0592 / 0.0490 / 0.0057; log(p_PT / p_ref) = -0.143, +0.156, +0.084, -0.270, +0.100.
  Rule on the two dominant basins: z(PT vs ref) = -12.2 and +12.3 (> 3), z(PT vs path-space IS of the controller 94252: 0.4118 / 0.4683) = +0.1 and +1.2 (< 3) => "Référence biaisée". (SE of the reference assumes independent samples; the reference MCMC may be autocorrelated, which would make it larger, but z ~ 12 is far from 3.) Not part of the rule: K4 minus an edge also differs from the reference (-0.27 in log, about 5 SE).
- **Task 2a, energy above the minimum per basin: "non conclusif" by the rule.** Mean E - E_min on the reference / PT (T = 1), z: triangle+pendant 2.816 / 2.829 (z -0.5), path of 2 2.917 / 2.784 (z +4.4), two disjoint pairs 3.525 / 3.482 (+0.5), K4- 2.649 / 2.726 (-1.0), star 2.070 / 2.097 (-0.2); ratios ref/PT 0.996, 1.048, 1.012, 0.972, 0.987. Only one basin has |z| > 3 (not 4 of 5 with the same sign; not all |z| < 3). Without a rule: the per-basin energy distributions agree to within a few percent, i.e. the reference does not show a different temperature or energy scale (a temperature of 2 would roughly double E - E_min); this does not say why its basin proportions differ.
- **Task 2b, provenance of the reference (no generator found in the repo):** `scripts/download.sh` lines 7 and 10-14 download `test_split_DW4.npy` from `https://raw.githubusercontent.com/jarridrb/DEM/main/data` (the DEM repository); `README.md` (line 40) says nothing about the data. The repo contains no code that generates the file.
  Reported by a delegated web-fetch agent (not independently checked by me; check the URLs): the DEM repo has no README or script for its `data/` directory (https://github.com/jarridrb/DEM/tree/main/data); `configs/energy/dw4.yaml` has `data_from_efm: true` and `data_path: data/test_split_DW4.npy`; `dem/energies/multi_double_well_energy.py` only loads the file and removes the mean, and defines the energy with a = 0.9, b = -4, c = 0, offset 4 and no tau and no 1/2 (same as ours);
  the DEM paper (https://arxiv.org/html/2402.06121, Appendix F.4.2, Eq. 28) writes E = (1/2 tau) sum_ij [a (d_ij - d0) + b (d_ij - d0)^2 + c (d_ij - d0)^4] with a = 0, b = -4, c = 0.9, tau = 1 (d0 not stated there) and says the validation and test sets are "from the MCMC samples in Klein et al. (2023b)" (the EFM paper, https://arxiv.org/html/2306.15030v2, Appendix C.2, Eq. 30, d0 = 4, tau = 1).
  Not stated in those sources (as read by the agent): the MCMC method, chain length, thinning, number of samples and chains for DW4 (the "1000 parallel chains, 10 000 steps after 200 000 burn-in" text of that appendix is for LJ13 and LJ55). One open point: the papers have a prefactor 1/(2 tau) = 1/2 that the code does not; if their sum runs over i < j the energy of the papers is E_code / 2 (temperature 2), if over all ordered pairs it equals E_code. This is not resolved here; Task 2a does not show a temperature difference.
- **Task 3, T1' on fresh PT samples: PASS (T1 stays NO-GO, reported separately).** 8000 PT T = 1 states (random without replacement), kernel as in the T1 run, 100 % quench convergence, valid landing 0.824, mean Barker weight 0.241. Flux balance log-ratio / SE (states bootstrap = the rule; chain bootstrap in brackets): tri-path +0.049 / 0.026 (z +1.9; 0.028), tri-2pairs +0.024 / 0.058, tri-K4- +0.035 / 0.062, tri-star +0.221 / 0.170, path-2pairs -0.007 / 0.065, path-K4- +0.025 / 0.060, path-star +0.189 / 0.163, 2pairs-K4- -0.002 / 0.087: every pair |log-ratio| < 3 SE (largest z +1.9). Compare T1 on the reference: tri-path +0.312 / 0.042.
  The darting kernel is therefore consistent with the detailed balance on samples of e^{-E}/Z; its T2 out-flux (0.147 and 0.127 at kappa = 0 for the two dominant basins) and T3 signal stand as reported above.
- **Task 4, errors recomputed with p_PT as the truth (no rule).** log(p_row / p_PT) +- SE_comb, columns: triangle+pendant, path of 2, two pairs, K4-, star; sum of |log| vs p_PT; sum vs the reference.
  AS ed8bb67: +0.017 +- 0.013, -0.228 +- 0.013, +0.632 +- 0.029, +0.482 +- 0.044, +0.999 +- 0.084; 2.358; 2.226.
  AS 94252: +0.049 +- 0.013, -0.246 +- 0.014, +0.566 +- 0.030, +0.508 +- 0.043, +0.951 +- 0.086; 2.320; 2.123.
  Endpoint IS gaussian s = 1: -0.011 +- 0.012, +0.008 +- 0.010, -0.020 +- 0.047, +0.032 +- 0.039, +0.053 +- 0.092; 0.123; 0.773.
  Endpoint IS t5 s = 1: -0.011 +- 0.012, +0.005 +- 0.010, -0.000 +- 0.030, +0.049 +- 0.042, +0.024 +- 0.084; 0.090; 0.744.
  Path-space IS: -0.002 +- 0.020, -0.023 +- 0.019, +0.076 +- 0.057, +0.094 +- 0.062, +0.394 +- 0.139; 0.589; 1.108.
  Path-space IMH: +0.012 +- 0.041, -0.068 +- 0.037, +0.152 +- 0.110, +0.298 +- 0.140, +0.357 +- 0.242; 0.887; 0.941.
  Reading: against p_PT the endpoint IS reproduces the truth within 0.05 per basin (sum 0.09-0.12 against 0.74-0.77 measured on the reference); AS errors are +0.5 to +1.0 (two pairs, K4-, star) and -0.23 to -0.25 on the path of 2, while triangle+pendant is correct (+0.02 to +0.05), so the sum of |log| of AS is about the same as measured on the reference (2.3 against 2.1-2.2) but its distribution among the basins is not;
  the path-space IS removes the small-basin errors of AS apart from the star (+0.39 +- 0.14); the path IMH is noisier (100 chains). The IS rows use proposals fitted on AS ed8bb67 (so they are not independent of AS); p_PT uses neither.
- **What this amendment does not do.** It does not recompute the verdicts of the earlier sections (AS weight errors of section 4, the Barker and continuation rules, the division go rules, the other baselines), which used the reference proportions; those verdicts stand as written, and the two dominant-basin frequencies of the reference are now known to differ from e^{-E}/Z by about 0.15 in log (and K4 minus an edge by 0.27).

## Réévaluation contre p_PT

Rules: `rules_reeval_pt.md` (commit 4a1342f, before any history VALUE was read; only the keys of a run summary had been listed). Script `scripts/dw4_reeval_pt.py` (+ `scripts/dw4_dem_splits.py`). Local CPU, 18 s + 5 s. The earlier verdicts above stay as written; the written rules are applied literally with p_PT as the only truth (thresholds, windows, SE, seeds and bootstrap counts unchanged).
Truth vectors (PT at T = 1, 102 400 states, SE over 64 chains): R (raw `state_ids`, 11 classes) path of 2 0.4742 (0.0021), triangle+pendant 0.4073 (0.0021), two disjoint pairs 0.0442 (0.0006), path of 3 (4 particles) 0.0250 (0.0005), K4 minus an edge 0.0379 (0.0007), star 0.0095 (0.0003), 1 pair 0.0011; Q (quenched basins) triangle+pendant 0.4127, path of 2 0.4792, two disjoint pairs 0.0644, K4- 0.0374, star 0.0063.
Declared deviation: the written rules compute their statistics on quenched BASIN proportions (`basin/<k>` columns, quenched final samples), whereas the requested adjustment was the raw-state truth. Both are pre-registered and reported (R = raw `state/<k>` columns and raw final samples, primary per the request; Q = basins, the letter of the rules); no rule combines them, and they agree.

- **Continuation (priority; paired, time-averaged): METHOD NOT DEMONSTRATED ALIVE, in both variants (as with the reference).** Runs dw4_cont_{C1, C2, G3, G10, G30}, commit aadf33b, last 20 evaluations (epochs 300-490), no NaN in the state columns, G30 stable (finite loss, eq_w2 0.62 vs 1.5 x controls 0.90).
  R: S = C1 0.3672 +- 0.0217, C2 0.2099 +- 0.0163, G3 0.3591 +- 0.0202, G10 0.3974 +- 0.0166, G30 0.3290 +- 0.0224; noise = |S_C1 - S_C2| + 2 SE_max = 0.1573 + 0.0448 = 0.2022; effect threshold min(S_C1, S_C2) - noise = 0.0077; effect at G3, G10, G30: False, False, False; S decreasing in gamma within noise: True.
  Q: S = 0.3632 +- 0.0220, 0.2029 +- 0.0185, 0.3557 +- 0.0202, 0.3953 +- 0.0166, 0.3256 +- 0.0226; noise 0.2056; threshold -0.0027; no effect at any gamma.
  Check of the code: with the reference as truth the same code reproduces the values of the earlier section (0.2246 / 0.1675 / 0.2171 / 0.2567 / 0.1870, noise 0.1023, threshold 0.0651).
  The AS arms are farther from p_PT than from the reference (S_C1 0.37 against 0.22), but the noise, dominated by the difference between the two AS seeds (0.157-0.160), grew with it: the verdict is unchanged. p_PT's own SE (0.002 per class) is not propagated by the rule and is negligible next to this noise.
- **Barker rule (snapshots; the continuation prevails in case of disagreement): primary FAILS in both variants (as before).** Final samples, 5000, bootstrap 200.
  R: TV(AS ed8bb67) 0.0983 +- 0.0056, TV(AS 94252) 0.1029 +- 0.0064, TV(CV) 0.1015 +- 0.0063; threshold min(AS) - 2 SE_comb = 0.0815 => fails. Q: 0.0932 +- 0.0059, 0.1006 +- 0.0067, CV 0.0893 +- 0.0060; threshold 0.0764 => fails (CV is below the AS minimum by 0.004, far from the 2 SE margin).
  Direction (b), lists derived from the truth (declared in the rules): FAILS in both (two disjoint pairs go up: CV 0.0824 vs AS 0.0795 in R, 0.1280 vs 0.1181 in Q; path of 3 (4 particles) up in R); path of 2, star and K4- go the right way; triangle+pendant (AS above the truth by 0.004-0.009) goes down.
  Mechanistic and sanity: no truth involved, not recomputed, earlier results unchanged; direction (a) with the hard-coded lists unchanged.
- **Report only: paired differences Barker - AS (G - C1, same seed 1), mean of the last 20 evaluations, bootstrap SE over the evaluations (this SE does not contain the run-to-run variability, which the rule's noise shows to be large), sign vs p_PT - p_C1 (R; Q is similar).**
  G30: path of 2 +0.0191 +- 0.0069 (right sign), two pairs -0.0119 +- 0.0018 (right), star -0.0003 +- 0.0007 (right, null), path of 3 -0.0046 +- 0.0013 (right), K4- -0.0098 +- 0.0027 (right), triangle+pendant +0.0073 +- 0.0078 (wrong sign, not significant).
  G10: path of 2 -0.0151 +- 0.0061 (WRONG), two pairs -0.0059 +- 0.0015 (right), star -0.0017 +- 0.0006 (right), path of 3 -0.0026 +- 0.0006 (right), triangle+pendant +0.0234 +- 0.0062 (WRONG), K4- +0.0020 +- 0.0019 (wrong).
  G3: all differences within 0.004 (path of 2 +0.0040 +- 0.0023, two pairs -0.0010 +- 0.0007, K4- -0.0028 +- 0.0012, triangle+pendant +0.0010 +- 0.0021).
  Reading (not a verdict): at gamma = 30 the small basins move toward p_PT (two pairs 6.6 SE, K4- 3.6 SE) and the path of 2 up (2.8 SE), but gamma = 10 does not follow (two states in the wrong direction by about 2.5-3.8 SE): no dose-response, consistent with the continuation verdict; the single seed pair behind these paired differences cannot separate them from run-to-run variability.
- **Report only: the DEM DW-4 splits against p_PT (basins, repo quench, 100 % converged; z with the binomial SE of the split and the SE of p_PT).** The bias is the same in all three splits: triangle+pendant test 0.4762 (z +11.7), train (100 000) 0.4755 (z +23.6), val 0.4797 (z +12.3) against 0.4127; path of 2 0.4099 (z -13.0), 0.4130 (z -25.7), 0.4101 (z -13.0) against 0.4792; K4 minus an edge 0.0490 (z +5.1), 0.0470 (z +9.8), 0.0398 (z +1.1) against 0.0374; two disjoint pairs 0.0592 (z -2.1), 0.0585 (z -5.7), 0.0647 (z +0.1) against 0.0644; star within 1 SE in all. The whole DEM DW-4 data set shares the proportions of the two dominant basins of the test split, which differ from e^{-E}/Z by about 0.15 in log; this does not depend on the sample size of the split (SE assumes independent samples).
- **Conclusion of the re-evaluation:** against the true target the verdicts of the written rules do not change: no effect of the Barker term at any gamma in the paired continuation (priority rule), and the snapshot rule fails; the AS errors were mis-distributed between the basins by the reference, not their absence.

## W2 et biais de la référence

- **W2 metrics and the reference bias (`scripts/dw4_w2_check.py`, local CPU 270 s; report only; interpretation fixed in the commit 3e03d69 before the run).** The metrics are the ones of the official AS evaluation (`DW4Evaluator`, n = 2000 samples against 2000 random reference samples, `energy_w2` and `eq_w2`); declared deviation: `eq_w2` with the vectorised `dist_point_clouds_fast` (the evaluator's `fast_eq`; same distances up to ties, `tests/test_eval_vectorized.py`) because the default double loop is too slow for ~100 evaluations.
  (a) W2(test split, PT): energy_w2 0.1527, eq_w2 0.5385 (the PT states are the `samples`, drawn at random over chains and times, seed 0). Extra, no rule, over 50 random draws: energy_w2 median 0.1407 [p5 0.0912, p95 0.2271], eq_w2 median 0.5127 [0.4336, 0.5998].
  (b) floor W2(PT_a, PT_b), 50 pairs of disjoint sub-samples from different chains (the 64 chains split at random into two groups of 32; 2000 states drawn from each): energy_w2 median 0.1180 [p5 0.0759, p95 0.1770]; eq_w2 median 0.3080 [p5 0.2857, p95 0.3745].
  (c) W2(AS ed8bb67, PT): energy_w2 0.3199, eq_w2 0.5569; W2(AS ed8bb67, test split): energy_w2 0.2760, eq_w2 0.4344 (single evaluations of the first 2000 `model_samples`).
  Interpretation (fixed before the run): energy_w2: (a) 0.1527 is inside [0.0759, 0.1770]: this metric does NOT see the bias of the reference proportions. eq_w2: (a) 0.5385 is above the p95 0.3745 (and above the whole floor range): this metric sees a difference between the test split and PT.
  Observation (not a rule): the eq_w2 floor between two independent PT sub-samples is 0.31, and the official eq_w2 values of the AS runs (0.49-0.61 in section 4) are of the size of the distance between the test split and the true distribution (0.54), so at this sample size eq_w2 does not separate an AS run from the reference's own deviation; AS is farther from PT than from the test split in both metrics (energy_w2 0.32 vs 0.28, eq_w2 0.56 vs 0.43).

## Démo 1D à deux modes (relance locale, rapport seul)

W&B already had the plain-AS runs of this demo (project `adjoint_sampler_cv`: `demo_as` gjt34av5 and `demo_as_2000ep` xin72vn4 / 9s8aafni: left-mode mass 0.53 / 0.53 / 0.41, mean of the last 10 evaluations; `demo_as_target_buffer` 081dq65h with exact samples: 0.118; the only MH run `demo_as_mh_g2.0_M4` 3u8q4wxk had crashed after 9 evaluations at 0.142; no Barker run, no image, no samples stored). The demo was therefore re-run locally on CPU with the same configs (`scripts/demo1d_local.sh`, `scripts/plot_demo1d.py`; figures `figures/demo1d_hist.png`, `figures/demo1d_leftmass.png`; evaluations `figures/demo1d_leftmass_evals.json`).
- **Target:** `GMM1D`, modes at -3 and +3, scale 0.5, weights 0.1 (left) / 0.9 (right); metric: left-mode mass (target 0.1).
- **Protocol (no tuning):** configs of `demo_as`, `demo_as_barker` (gamma 1, M = K = 2), `demo_as_mh` (gamma 2, M = 4) unchanged, seed 0, 1000 epochs (the W&B plain-AS left mass is stationary from epoch 100), evaluation every 100 epochs on 2000 samples, 10 000 final samples dumped. CPU, 0.26 / 0.47 / 0.40 s per epoch (AS / Barker / MH).
- **Results** (left-mode mass; evaluation at epoch 0 is the untrained model, 0.52 for all): the last 5 evaluations (epochs 500-900), mean +- SE over the evaluations (serially correlated, so optimistic): **AS 0.483 +- 0.056**, **AS + Barker 0.090 +- 0.007**, **AS + MH 0.122 +- 0.009**; final 10 000 samples: AS 0.362, Barker 0.104, MH 0.176. Evaluations of AS: 0.52, 0.72, 0.69, 0.70, 0.60, 0.66, 0.51, 0.43, 0.50, 0.32; Barker from epoch 100: 0.11, 0.09, 0.11, 0.13, 0.08, 0.08, 0.10, 0.08, 0.11; MH: 0.10, 0.11, 0.06, 0.10, 0.15, 0.13, 0.12, 0.11, 0.10.
- **Reading:** plain AS keeps 32-72 % of the mass on the mode whose weight is 10 % (W&B runs: 41-64 %), Barker brings it to the target within about 100 epochs and stays there; MH is close (about 0.12, slightly above). One seed, 1000 epochs.
- **Deviations to report:** (1) 1000 epochs instead of the configs' 2000 / 6000; (2) `eval_utils.fig2img` was patched for matplotlib 3.10 (`tostring_rgb` removed; same image from `buffer_rgba`); (3) the evaluation metrics are printed by the runs but not kept in `train.log` (wandb off): the left-mass series of `leftmass_evals.json` were copied from the console output of the runs (AS: typed from the output of this session; Barker and MH: parsed from the saved task outputs); (4) wall time of the Barker and MH runs was longer than their CPU time (the machine was shared); (5) one seed.

- **1D demo, same re-run with the DIVISION control variate instead of MH (requested; `configs/experiment/demo_as_division.yaml`, matcher defaults of `adjoint_ve_division.yaml`: gamma 1, shrink 0.8, refresh every 5, kappa_max 0.99, min_buffer 2000, no tuning for this target; seed 0, 1000 epochs, CPU, 0.5 s per epoch, 9 min).**
  Left-mode mass (target 0.1), same protocol as above: evaluations every 100 epochs 0.52 (epoch 0, untrained), 0.073, 0.054, 0.073, 0.106, 0.150, 0.093, 0.101, 0.114, 0.086; last 5 evaluations (epochs 500-900): **0.108 +- 0.011** (AS 0.483 +- 0.056, Barker 0.090 +- 0.007); final 10 000 samples: **0.093** (AS 0.362, Barker 0.104). The MH run above stays reported (0.122 +- 0.009), it is not in the figures any more: `figures/demo1d_hist.png` and `figures/demo1d_leftmass.png` now show AS, AS + Barker and AS + division.
  Check of what the division term has in this 1D case (`fit_modes` on the final samples of the plain-AS run): 2 modes found, centres -3.00003 and 2.99999, variances 0.25000 and 0.25000, softmax(-L) = 0.09999999 / 0.9000000 against the true weights 0.1 / 0.9: the fitted parameters are exact here, so this is the best case of the method (the masses L are exact), not a test of its robustness to mass errors.
  Same deviations as above (1000 epochs; the evaluation series are copied from the console output, `figures/demo1d_leftmass_evals.json`; one seed).

- **1D demo, WIDE target: modes at -5 and +5 (scale 0.5, weights 0.1 / 0.9), requested because the plain-AS blindness is more striking there (`configs/problem/demo_wide.yaml`, `configs/experiment/demo_wide_{as, as_barker, as_division}.yaml`, `GMM1D(loc=...)`; sigma_max 6 = about 2 x the std of this target (3.1), the same rule as the original demo (4 = 2 x 1.87); everything else as above: seed 0, 1000 epochs, CPU).**
  Note on the W&B runs: the plain-AS runs `demo_as` of W&B (commit f19aa5d) were produced with the target of the repository at that commit, modes at -3 and +3 (the history of `dist_utils.py` has [-1, 2] with weights 0.5 / 0.5 in the original code, then [-3, 3] with 0.1 / 0.9; no commit has modes at +-5): the +-5 target above is a new run, not a re-run of those.
  Left-mode mass (target 0.1), evaluations every 100 epochs (epoch 0 = untrained model, 0.0025): AS 0.85, 0.78, 0.81, 0.79, 0.78, 0.68, 0.58, 0.63, 0.46; Barker 0.13, 0.08, 0.10, 0.17, 0.09, 0.10, 0.10, 0.06, 0.10; division 0.06, 0.04, 0.14, 0.09, 0.18, 0.12, 0.10, 0.12, 0.08.
  Last 5 evaluations (epochs 500-900), mean +- SE over the evaluations (serially correlated, optimistic): **AS 0.627 +- 0.053**, **AS + Barker 0.092 +- 0.008**, **AS + division 0.120 +- 0.017**; final 10 000 samples: **AS 0.480** (50/50), Barker 0.095, division 0.090.
  Reading: plain AS puts 46-85 % of its mass on the mode whose weight is 10 % (8.5 times too much at epoch 100) and drifts down towards 0.5 at the end of the run; the final samples are 48 / 52 (`figures/demo1d_wide_AS.png`). Barker and division both bring the left mass to the target from the first evaluation (epoch 100) and keep it there. One seed, 1000 epochs, the AS series is not monotone (0.85 -> 0.46), so the "50/50" is the value of the last epoch, not a stable plateau.
  Figures: `figures/demo1d_wide_hist.png`, `demo1d_wide_leftmass.png` and one image per method `demo1d_wide_AS.png`, `demo1d_wide_barker.png`, `demo1d_wide_division.png`; series in `figures/demo1d_leftmass_evals.json` (keys `demo_wide_*`).

- **1D demo, NARROW target, plain AS only (no control variate; requested: "a last run at +-1 to see the evolution"): modes at -1 and +1 (scale 0.5, weights 0.1 / 0.9), `configs/problem/demo_narrow.yaml`, `configs/experiment/demo_narrow_as.yaml`, sigma_max 1.6 (about 2 x the target std 0.78, the same rule as the other demos), seed 0, 1000 epochs, an evaluation every 20 epochs (50 evaluations, 2000 samples each, saved with the optional `+dump_eval_samples=true` of `train.py`, off by default), CPU 5 min.**
  Left-mode mass (target 0.1): all 50 evaluations lie between 0.101 and 0.148; epoch 0 (after the first 100 gradient steps) 0.107; mean of the 50 evaluations 0.122 +- 0.002 (SE over the evaluations, serially correlated); epochs 500-900 (every 100, the window used for the other two targets) **0.125 +- 0.004**. The model is at the target from the first evaluation and stays about 20-25 % above it (0.12-0.13 for 0.10).
  Reading: with the modes this close the two bumps overlap and plain AS is NOT blind: there is no evolution towards a wrong proportion, the distribution matches the target at epoch 0 and stays there (`figures/demo1d_narrow_evolution.png`: snapshots at epochs 0, 20, 100, 300, 600, 980; `figures/demo1d_narrow_leftmass.png`: the 50 evaluations).
  Trend with the distance between the modes, plain AS, left mass (mean +- SE over the evaluations at epochs 500-900, target 0.1): modes at +-1 (distance 2) **0.125 +- 0.004**; at +-3 (distance 6) **0.483 +- 0.056**; at +-5 (distance 10) **0.627 +- 0.053** (`figures/demo1d_separation.png`). The proportion error of plain AS grows with the distance between the modes; one seed per target, and the +-3 and +-5 series are not stationary (see above), so the trend is indicative.
  Same deviations as the other 1D demos (1000 epochs, one seed); the +-3 and +-5 values come from runs with one evaluation every 100 epochs.

## Run de continuation avec le noyau de darting (un seul run, descriptif)

Rule: `rules_darting_run.md` (commit 2f622c2, before the run). Code: `adjoint_samplers/utils/dw4_darting.py` (batched torch kernel, 20 x faster than the numpy reference), `AdjointVEDW4DartingMatcher`, tests `tests/test_dw4_darting.py` and `tests/test_dw4_darting_matcher.py` (exact zero mean under exact-pi PT states: max |z| 1.62-2.66 against 13-15 for the positive control; round trip 2e-15; flux balance max |z| 1.64).
Run: continuation from the controller and buffer of the AS run 94252, seed 1, 500 epochs, nfe 100, sigma_max 2, evaluation every 10 epochs on 5000 samples, gamma = 3 (chosen in advance), compared with the existing AS arms `dw4_cont_C1` (seed 1) and `dw4_cont_C2` (seed 2). W&B run `CCDS_NTU/as/nz4fxg0y` (`dw4_cont_D3`, tag `code_commit` 5390154).
Execution history (all declared): (1) first launch on the user's Mac (CPU) was stopped by the user (a long local run had not been authorised): it had also produced NaN losses from epoch 1 because a model sample had a non-positive polar-chart factor f_vol (log of a negative number); fixed in a434329 (validity now requires f_src > 0 and f_tgt > 0, symmetric under reversal; exactness tests unchanged); (2) a first Vast.ai instance (RTX 4070, Xeon E5-2698 v3 on a loaded host, 14 s per epoch) was replaced after 40 epochs and destroyed; (3) the run reported here: Vast.ai RTX 3080 + i9-7960X, 3.5-5.8 s per epoch, 500 epochs, W&B on; without the CUDA-only speed options (`fast_sampling`, `fast_train`, `buffer_on_device`), which are equivalent to the default path and were used by C1/C2; OMP_NUM_THREADS = 4. Total cost about 0.15 USD (credit 10.00 -> 9.85); both instances destroyed.
Log and outputs: `results/dw4_cont_D3_vast_s1/` (not committed).

Pre-registered descriptive reading (S = sum of |p_bar - p_PT| over the 11 classes, mean of the last 20 evaluations, epochs 300-490; `scripts/dw4_darting_run_analysis.py`):
1. S: variant R (raw states) darting **0.1374** against C1 0.3672 and C2 0.2099: "below both"; variant Q (quenched basins) darting **0.1222** against 0.3632 and 0.2029: "below both".
2. Errors |log(p / p_PT)| (Q): path of 2 0.097 (C1 0.476, C2 0.238), triangle+pendant 0.042 (0.195, 0.028), two disjoint pairs 0.457 (0.475, 0.591), K4 minus an edge 0.394 (0.809, 0.573), star 0.648 (0.738, 0.872); darting better than BOTH AS runs on the three small basins in 3 of 3, in both variants (R: two pairs 0.358 against 0.373 / 0.516, K4- 0.395 against 0.797 / 0.563, star 0.607 against 0.656 / 0.777).
3. Sanity: loss finite all along (5.76-7.43, last 5.98); eq_w2 (last two) 0.459 against C1 0.614 and C2 0.589; energy_w2 0.385 against 0.345 and 0.329 (+14 % above the mean of the two, inside the 15 % band); untyped fraction 0; 8.24 valid proposals per sample.
4. Noise: |S_C1 - S_C2| = 0.157 (R) / 0.160 (Q). The darting run is below the better AS run (C2) by 0.073 (R) / 0.081 (Q), i.e. by about half of that noise: not interpretable relative to C2; it is below C1 by 0.23-0.24 (more than the noise), but C1 is the AS run that drifts away.
Descriptive extras (not part of the written rule): S in blocks of 100 epochs (Q), darting / C1 / C2: epochs 0-99 0.225 / 0.268 / 0.251; 100-199 0.264 / 0.255 / 0.258; 200-299 0.157 / 0.347 / 0.242; 300-399 0.170 / 0.355 / 0.205; 400-499 0.199 / 0.374 / 0.246. In the first 200 epochs the three runs are indistinguishable; the separation appears after epoch 200 (C1 rises, darting falls, C2 stays flat). Per-evaluation SD of S over epochs 300-490: 0.069 (darting), 0.095 (C1), 0.062 (C2).
Matcher statistics (mean over the run; they replace the estimates from PT states): out-flux at kappa < 0.1 0.186, kappa 0.1-0.7 0.138, **kappa > 0.7 0.0043** (against 0.033 measured on PT states); ||term|| / ||grad g|| 0.026, 0.037, **0.004**. The kappa > 0.7 bin holds 92 % of the training times: the term is negligible there (0.4 % of the gradient), so any effect has to come from the 8 % of the times with kappa < 0.7.
Conclusion (no verdict, by the rule): one run cannot separate an effect from the seed-to-seed noise (0.16): the darting run ends below both AS runs and better than both on the three small basins, which is consistent with an effect but does not establish it. The term is bounded and small at large kappa. A decision on 4 + 4 seeds (the design discussed before, rule to be written first) is separate.

## Darting γ = 3 sur les graines 1 et 2, avec les options rapides (phase 1 de `rules_darting_seeds_gamma.md`, commit f14cdf6)

Runs `dw4_cont_D3` graines 1 et 2 (W&B `CCDS_NTU/as`, tag `code_commit` f14cdf6), Vast.ai RTX 3080 + EPYC 7542 (offre 57084432), 2 runs en parallèle, ≈ 3,5 s par époque, ≈ 35 min ; coût ≈ 0,094 USD (crédit 9,85 → 9,756) ; instance détruite après récupération (logs, contrôleurs et échantillons dans `results/dw4_cont_D3_seed{1,2}_fast/`). 500 époques, 50 évaluations par run, pertes finies, aucun NaN. Écart déclaré : `fast_sampling` (graphes CUDA), Adam fusionné et `buffer_on_device` étaient actifs, mais `torch.compile` a échoué (pas de compilateur C dans l'image) : le modèle a été entraîné non compilé (repli prévu par le code, numériquement équivalent).

Lecture écrite d'avance (S contre p_PT, 20 dernières évaluations, comparaison appariée par graine) :
1. S (R / Q) : graine 1 : D3 0,1838 / 0,1787 contre C1 0,3672 / 0,3632, Δ = −0,183 / −0,185 ; graine 2 : D3 0,1385 / 0,1229 contre C2 0,2099 / 0,2029, Δ = −0,071 / −0,080. Δ négatif sur 2 graines sur 2. Le bruit rappelé (|S_C1 − S_C2|) vaut 0,157 / 0,160 : Δ de la graine 1 au bord hors du bruit, Δ de la graine 2 dans le bruit.
2. Petits bassins (deux paires disjointes, K4 moins une arête, étoile), D3(s) contre C(s) de même graine, erreur |log(p̄/p_PT)| (Q) : graine 1 : 0,401 / 0,686 / 0,470 contre C1 0,475 / 0,738 / 0,809 ; graine 2 : 0,437 / 0,795 / 0,402 contre C2 0,591 / 0,872 / 0,573. D3 bat son AS sur 3 petits bassins sur 3 pour les deux graines, en R comme en Q. Les grands bassins : chemin de 2 plus proche de p_PT que l'AS de même graine (0,206 contre 0,476 ; 0,124 contre 0,238), mais encore trop bas (p̄ 0,39 et 0,42 contre 0,479).
3. Réplication, graine 1 sans / avec options rapides : S_R 0,1374 / 0,1838, S_Q 0,1222 / 0,1787, soit un bruit d'exécution du darting de 0,046 / 0,057 à graine égale (même ordre que Δ de la graine 2).
4. Sanité : eq_w2 (deux dernières évaluations) 0,455 / 0,540 contre C1 0,614 / C2 0,589 ; energy_w2 0,266 / 0,334 contre 0,345 / 0,329 ; propositions valides par échantillon 8,32 / 8,30 ; ‖terme‖/‖grad‖ par plage de κ 0,026 / 0,037 / 0,004 (graine 1), 0,027 / 0,037 / 0,004 (graine 2), identique au run précédent ; flux sortant 0,18 / 0,14 / 0,004.
Énoncé permis : Δ négatif sur 2 graines sur 2, plus petit ou égal au bruit sur l'une. Aucun verdict de méthode (règle sans seuil d'effet, 2 graines).

## Darting, dose-réponse en γ, graine 1 (phase 2 de `rules_darting_seeds_gamma.md`, amendée graine 1 seule, commit 15c1453)

Runs `dw4_cont_D10` et `dw4_cont_D30` (graine 1, même protocole et mêmes options rapides que la phase 1 ; `torch.compile` en repli non compilé comme avant), Vast.ai RTX 3080 + EPYC 7542, 2 runs en parallèle, ≈ 3,4 s par époque, ≈ 30 min ; coût ≈ 0,090 USD (crédit 9,756 → 9,666) ; instance détruite après récupération (`results/dw4_cont_D{10,30}_seed1_fast/`). 500 époques, 50 évaluations, pertes finies, aucun NaN.

S contre p_PT (20 dernières évaluations, graine 1 ; γ = 0 est C1, γ = 3 est D3 graine 1 de la phase 1) :
| γ | S (R) | S (Q) |
|---|---|---|
| 0 (AS, C1) | 0,3672 | 0,3632 |
| 3 | 0,1838 | 0,1787 |
| 10 | 0,0972 | 0,0830 |
| 30 | 0,0612 | 0,0482 |
« S décroît avec γ » : VRAI en R et en Q (graine 1, sans test). Les pas de 3 → 10 (−0,087 R, −0,096 Q) et de 10 → 30 (−0,036 R, −0,035 Q) dépassent ou égalent le bruit d'exécution du darting (≈ 0,05) pour le premier, pas pour le second (à ne pas interpréter).
Erreurs |log(p̄/p_PT)| (Q), γ = 3 / 10 / 30 (C1) : chemin de 2 0,206 / 0,090 / 0,040 (0,476) ; deux paires 0,401 / 0,330 / 0,246 (0,475) ; étoile 0,686 / 0,557 / 0,402 (0,738) ; triangle + pendant 0,068 / 0,000 / 0,012 (0,195) ; K4 moins une arête 0,470 / 0,270 / 0,078 (0,809). Les trois petits bassins sont meilleurs que C1 à tout γ.
Sanité de stabilité : pertes finies (γ = 30 : 6,8–7,6, plus haut qu'à γ = 3 car le terme est plus grand) ; eq_w2 0,516 (γ = 10) et 0,550 (γ = 30) contre C1 0,614 : ≤ 1,5 × C1 ; energy_w2 0,248 et 0,231 contre 0,345. Dose mesurée, ‖terme‖/‖grad‖ par plage de κ ([0,0,1] / [0,1,0,7] / [0,7,1]) : γ = 3 0,026 / 0,037 / 0,004 ; γ = 10 0,087 / 0,119 / 0,014 ; γ = 30 0,255 / 0,350 / 0,041. La dose augmente avec γ, et à κ > 0,7 elle reste très en dessous du seuil de 0,5 de la règle : pas d'instabilité signalée.
Énoncé permis : sur la graine 1, S décroît avec γ en R et en Q et la dose mesurée augmente avec γ. Une seule graine pour γ = 10 et 30 : aucun verdict de méthode. Le chemin de 2 et les quatre autres bassins se rapprochent de p_PT en même temps ; le bassin star reste le plus éloigné (0,40 en log à γ = 30).

## Plancher de S et repères IS dans la métrique S (local, rapport seulement ; scripts `dw4_floor_S.py`, `dw4_is_S_*.py`, commit 07b422f)

a) Plancher : S d'un sampler parfait, c'est-à-dire 20 évaluations de 5000 tirages avec remise dans le pool PT à T = 1 (64 chaînes × 1600 états), moyennées puis comparées à p_PT (moyenne du pool), 50 répétitions. Q : médiane **0,0034**, p5 0,0014, p95 0,0060 ; R : médiane 0,0036, p5 0,0022, p95 0,0070. Avec l'incertitude de p_PT elle-même (pool rééchantillonné par chaînes, vérité inchangée) : Q 0,0058 (p5 0,0023, p95 0,0132) ; R 0,0069 (0,0034 ; 0,0136). Le plancher est donc d'environ 0,004 à 0,007 ; les tirages iid du pool ignorent l'autocorrélation des chaînes (limite).
b) IS dans la même métrique (un seul lot de N échantillons pondérés, donc plus bruité que 20 × 5000 ; SD bootstrap entre parenthèses) :
| estimateur | N, ESS | S_Q | S_R |
|---|---|---|---|
| IS d'endpoint, gaussienne s = 1 | 20000, 61 % | 0,0112 (0,0072) | 0,0156 (0,0069) |
| IS d'endpoint, t5 s = 1 | 20000, 62 % | 0,0143 (0,0079) | 0,0141 (0,0076) |
| IS de chemin (contrôleur 94252) | 12000, 32 % | 0,0111 (0,0090) | 0,0136 (0,0095) |
| échantillons bruts du contrôleur 94252 (non pondérés) | 12000 | 0,2087 | 0,2147 |
Rappel : darting γ = 30 (graine 1) 0,048 (Q) / 0,061 (R) ; AS C1 0,363 / 0,367, C2 0,203 / 0,210. Les estimateurs IS sont proches du plancher (0,01 à 0,015 contre 0,004 à 0,007) ; le darting γ = 30 est environ 12 fois au-dessus du plancher et 4 à 5 fois au-dessus des IS, et environ 4 fois en dessous des échantillons bruts de départ. Les propositions IS sont ajustées sur les échantillons AS ed8bb67 (voir plus haut) ; l'IS de chemin n'utilise aucune densité ajustée.

## Test de CONFIRMATION du darting γ = 30 (graines 2, 3, 4), `rules_darting_confirm.md` (règle commitée en 07b422f, amendée en 8ec415b avant la location)

Runs : D30 graines 2, 3, 4 et AS C2 (RELANCÉ), C3, C4, 6 runs en parallèle sur une instance Vast.ai (RTX 3080 + EPYC 7542, offre 57084432), mêmes options rapides, `torch.compile` en repli non compilé pour TOUS les bras (pas de compilateur C) ; 500 époques et 50 évaluations par run, pertes finies, aucun NaN ; durée ≈ 55 min ; coût réel ≈ **0,149 $** (crédit 9,666 → 9,517 ; plafond 0,35 $) ; instance détruite après récupération des 6 logs (`results/confirm/`). Écarts : amendements 1 et 2 (C2 relancé comme témoin de la graine 2, (iii) contre le max des C), tous deux écrits avant la location ; correction après coup d'un bug de lecture dans le script d'analyse (config W&B sans `git_commit` pour les nouveaux runs ; n'affecte ni la règle ni les chiffres).

| graine | S_Q D30 | S_Q C | Δ_Q | S_R D30 | S_R C | Δ_R |
|---|---|---|---|---|---|---|
| 2 | 0,0604 | 0,1970 | −0,137 | 0,0723 | 0,2047 | −0,132 |
| 3 | 0,0922 | 0,2555 | −0,163 | 0,0972 | 0,2614 | −0,164 |
| 4 | 0,0502 | 0,2206 | −0,170 | 0,0613 | 0,2256 | −0,164 |
Plancher (sampler parfait) : Q 0,0034, R 0,0036 ; repères IS : 0,011 à 0,014 en Q. Les D30 sont à 15 à 27 fois le plancher et 4 à 8 fois les IS, les C à 58 à 75 fois le plancher. Le C2 du cluster (rapporté seulement) : S_Q 0,2029, S_R 0,2099, eq_w2 0,589, energy_w2 0,329 : proche du C2 relancé (0,1970 / 0,2047), ce qui n'indique pas d'effet de plateforme.

Verdict écrit d'avance :
- (i) S_Q(D30_s) < S_Q(C_s) − 0,05 : VRAI sur les 3 graines (Δ −0,14, −0,16, −0,17).
- (ii) médiane S_Q(D30) 0,0604 < min(S_Q(C2, C3, C4)) 0,1970 : VRAI.
- (iii) eq_w2 et energy_w2 de D30_s ≤ 1,1 × max des C (eq_w2 max 0,574 → seuil 0,631 ; energy_w2 max 0,421 → seuil 0,463) : **FAUX sur la graine 2** : eq_w2(D30_s2) = 0,655 > 0,631 (energy_w2 0,261, correct) ; VRAI sur les graines 3 (0,416 ; 0,369) et 4 (0,556 ; 0,142).
- **VERDICT : « non confirmé »** (condition (iii) violée sur la graine 2, de 0,024 en eq_w2 ; pas de relâchement après coup).
Observation, hors règle : eq_w2 est très bruité d'une évaluation à l'autre (D30_s2 : 0,454 à 0,677 ; C2 : 0,439 à 0,706 ; D30_s4 : 0,300 à 0,812), plus que la marge de 10 % ; le garde-fou (iii) tel qu'écrit est donc sensible au bruit de cette mesure (la moyenne des deux dernières valeurs). Cela n'annule pas le verdict, mais en dit la nature : les conditions portant sur S (i, ii) sont satisfaites avec une marge de 2,7 à 3,4 fois le seuil de 0,05, et l'échec porte sur un contrôle de sanité bruité.
Tableau par bassin (erreur |log(p̄/p_PT)|, Q ; D30 / C) :
- graine 2 : chemin de 2 0,063 / 0,230 ; deux paires 0,248 / 0,613 ; étoile 0,568 / 0,839 ; triangle + pendant 0,002 / 0,017 ; K4− 0,177 / 0,569.
- graine 3 : 0,101 / 0,310 ; 0,277 / 0,618 ; 0,364 / 0,671 ; 0,031 / 0,080 ; 0,234 / 0,621.
- graine 4 : 0,026 / 0,262 ; 0,255 / 0,560 ; 0,471 / 0,720 ; 0,032 / 0,073 ; 0,067 / 0,496.
D30 bat son témoin sur les cinq bassins à chaque graine (15 comparaisons sur 15). Dose mesurée (‖terme‖/‖grad‖ par plage de κ) : 0,24–0,25 / 0,35 / 0,04 sur les trois graines, identique à la graine 1 ; propositions valides par échantillon 8,03.

### Test de confirmation : verdict maintenu, erreur de conception déclarée, et eq_w2 / energy_w2 en moyenne temporelle (report-only, `scripts/dw4_confirm_w2_timeavg.py`)

**Verdict : « non confirmé »**, tel qu'écrit dans `rules_darting_confirm.md` (amendé en 8ec415b) et appliqué sans relâchement : (i) et (ii) vrais, (iii) faux sur la graine 2 (eq_w2 0,655 > 0,631).
**Erreur de conception de la règle (celle de la proposition de l'utilisateur, pas de l'exécution) :** le garde-fou (iii) ne précisait ni la moyenne temporelle de eq_w2 ni une marge calibrée sur le bruit de eq_w2. La valeur 0,655 est la moyenne de DEUX évaluations isolées (époques 400 et 490 : 0,677 et 0,634, convention « deux dernières valeurs » du script d'analyse, héritée de `rules_darting_run.md`), pas une moyenne sur les époques 300 à 490 ; eq_w2 n'est évalué qu'aux époques 0, 100, 200, 300, 400 et 490, donc 3 valeurs seulement dans la fenêtre 300-490.

Moyennes temporelles sur les époques 300 à 490 (energy_w2 : 20 évaluations, SE par moyennes de 4 lots de 5 évaluations consécutives ; eq_w2 : 3 valeurs, SE = SD/√3, grossière) :
| bras | energy_w2 (SE) | eq_w2 (SE) |
|---|---|---|
| D30 graine 2 / 3 / 4 | 0,297 (0,033) / 0,335 (0,022) / 0,215 (0,030) | 0,588 (0,068) / 0,499 (0,083) / 0,627 (0,164) |
| C2 / C3 / C4 (relancés) | 0,398 (0,018) / 0,483 (0,039) / 0,294 (0,054) | 0,569 (0,012) / 0,553 (0,038) / 0,458 (0,024) |
| graine 1 : D3 / D10 / D30 (locaux) | 0,333 / 0,290 / 0,277 | 0,452 / 0,501 / 0,570 |
| C1 (cluster) | 0,445 (0,011) | 0,613 (0,074) |
Différence pairée D30 − C (SE = racine de la somme des carrés des deux SE ; hypothèse d'indépendance conservatrice) :
| graine | Δ energy_w2 (SE, z) | Δ eq_w2 (SE, z) |
|---|---|---|
| 2 | −0,101 (0,038, −2,68) | +0,019 (0,069, +0,28) |
| 3 | −0,148 (0,045, −3,27) | −0,054 (0,091, −0,60) |
| 4 | −0,079 (0,062, −1,27) | +0,170 (0,166, +1,02) |
| moyenne des graines 2-4 | −0,109 (0,028, z −3,84) | +0,045 (0,067, z +0,67) |
| (graine 1 contre C1 du cluster, hors moyenne : graine utilisée pour choisir γ) | −0,168 (0,032, −5,34) | −0,043 (0,123, −0,35) |
Lecture : D30 n'est PAS systématiquement plus haut que C en eq_w2 (signes +, −, +, aucun |z| > 2 ; moyenne +0,045 ± 0,067). En energy_w2, D30 est plus BAS que C sur les trois graines (même signe, |z| > 2 sur 2 graines sur 3, moyenne −0,109 ± 0,028). Les valeurs de eq_w2 sont à environ 0,3 du plancher de ce W2 à n = 2000 (0,31) : peu discriminant. Le seuil de (iii) est donc franchi sur la graine 2 par un écart de bruit sur deux évaluations. Cela ne change pas le verdict écrit.

## Exploration locale avant les pistes D100 / t_focus (report-only ; `dw4_S_trajectory.py`, `dw4_darting_flux.py`, `dw4_tfocus_check.py`)

**1a. S par évaluation (époques 0 à 490) et pente sur 300-490.** Attention : S d'UNE évaluation (5000 échantillons) est plus grand que le S de la moyenne des 20 évaluations (métrique des règles) : plancher d'une évaluation d'un sampler parfait 0,016 (Q) / 0,017 (R), p95 0,034 / 0,036. S moyen des évaluations de 300 à 490, Q (R très proche) : D30 graines 1 à 4 : 0,118 / 0,118 / 0,128 / 0,132 ; C1 (cluster) 0,365 ; C2 relancé 0,225 ; C3 0,265 ; C4 0,229.
Pente de S sur 300-490 (régression linéaire, unité S par 100 époques ; SE corrigée de l'autocorrélation AR(1) des résidus) :
| bras | pente (SE, z) Q | pente (SE, z) R |
|---|---|---|
| D30 s1 | +0,017 (0,035, +0,48) | +0,015 (0,034, +0,43) |
| D30 s2 | +0,000 (0,026, 0,00) | −0,005 (0,026, −0,20) |
| D30 s3 | −0,008 (0,030, −0,27) | −0,010 (0,030, −0,33) |
| D30 s4 | −0,025 (0,031, −0,79) | −0,023 (0,032, −0,71) |
| C1 (cluster) | +0,022 (0,046, +0,48) | +0,020 (0,045, +0,45) |
| C2 relancé | +0,030 (0,023, +1,29) | +0,027 (0,021, +1,27) |
| C3 | −0,006 (0,034, −0,17) | −0,002 (0,032, −0,06) |
| C4 | +0,025 (0,038, +0,66) | +0,026 (0,037, +0,70) |
Moyenne des pentes : D30 −0,004 (Q) / −0,006 (R) par 100 époques (SD entre bras 0,017 / 0,016) ; AS +0,018 / +0,018 (SD 0,016 / 0,013). Aucune pente n'est distinguable de zéro (|z| < 1,3) : D30 est sur un PLATEAU sur 300-490, pas en convergence lente. Moyennes par blocs de 100 époques (Q) D30 s1 : 0,125, 0,138, 0,132, 0,097, 0,139 ; les quatre D30 sont déjà à 0,12-0,15 dans les 100 premières époques (les AS : 0,21-0,27). Limite : la précision de la pente est de ±0,03 à ±0,05 par 100 époques, soit 25-40 % du niveau de S par évaluation : une dérive lente ne serait pas détectée.

**1b. Flux du noyau de darting par bassin** (probabilité par pas ; `out` = masse qui quitte le bassin, `in` = masse reçue par unité de masse du bassin, `net` = in − out ; échantillons PT stratifiés (6645 états) et AS ed8bb67 (10 000 échantillons) ; κ = 0, 0,5, 0,7 ; x_t = κ x1 + √v ξ, moyenne de 20 tirages de ξ).
| basin | out κ=0 | out κ=0,5 | out κ=0,7 | net κ=0 sur PT | net κ=0 sur AS |
|---|---|---|---|---|---|
| triangle + pendant | 0,154 | 0,0038 | 0,0003 | −0,000 | +0,016 |
| chemin de 2 | 0,132 | 0,0023 | 0,0001 | −0,001 | +0,063 |
| deux paires disjointes | 0,344 | 0,0057 | 0,0002 | −0,004 | −0,139 |
| K4 moins une arête | 0,523 | 0,0191 | 0,0016 | +0,011 | −0,114 |
| étoile | 0,789 | 0,0854 | 0,0105 | +0,038 | −0,420 |
Sur les échantillons PT (donc ~π), le flux est équilibré à κ = 0 (|net| ≤ 0,004 sur les trois premiers bassins ; étoile +0,038 sur 645 états, du bruit d'échantillonnage). Sur les échantillons AS (qui ont trop de petits bassins : étoile 0,017 contre 0,0063, deux paires 0,121 contre 0,064, K4− 0,061 contre 0,037), le noyau retire de la masse de l'étoile (−0,42), des deux paires (−0,14) et de K4− (−0,11) et en donne aux deux bassins dominants (+0,06 au chemin de 2) : c'est le sens correcteur. À κ = 0,5, seule l'étoile garde un flux notable (0,085) ; les deux paires tombent à 0,006 et K4− à 0,019 ; à κ = 0,7 tout est ≤ 0,0105 (étoile) et ≤ 0,0016 (les autres). Le noyau agit donc surtout aux petits κ (< 0,3), très peu aux deux paires dès κ = 0,5.

**1c. t_focus / t_focus_max.** Les deux paramètres existent dans la classe de base `AdjointVEMatcher` (signature `t_focus: float = 0.0, t_focus_max: float = 0.15`) ; le matcher de darting en hérite (`sample_t` hérité, `prepare_target` appelle `super().prepare_target`). Ils ne sont PAS dans les YAML `adjoint_ve.yaml` ni `adjoint_ve_dw4_darting.yaml` : il faut les passer avec un « + » (`+adjoint_matcher.t_focus=...`). Un instant « focalisé » est t × t_focus_max (uniforme sur [0, t_focus_max]) avec probabilité t_focus. Avec le schéma AS (sigma_max 2, sigma_min 0,001) : t(κ = 0,1) = 0,0069, t(0,5) = 0,0456, t(0,7) = 0,0792, t(0,9) = 0,1515 ; donc t_focus_max = 0,08 donne κ < 0,704 (0,15 donne κ < 0,898). Fraction des instants avec κ dans [0,1 ; 0,7] (aujourd'hui, t uniforme : 7,2 % ; [0, 0,1] 0,7 % ; [0,7, 1] 92,1 %) : avec t_focus_max = 0,08 et t_focus = 0,25 / 0,5 / 0,7 / 0,9 : 28 % / 49 % / 65 % / 82 % (dans [0,7, 1] : 69 % / 47 % / 28 % / 10 %) ; avec t_focus_max = 0,15 : 18 % / 28 % / 36 % / 44 %.

## Exploration sur la graine 1 : choix écrits AVANT le lancement (descriptive, sans règle d'effet)

Quatre bras, graine 1, continuation de 500 époques depuis 94252, même protocole et mêmes options rapides que D30 (nfe 100, sigma_max 2, évaluation toutes les 10 époques à 5000 échantillons) : (1) D100, sans focus ; (2) D30 + focus ; (3) AS + focus (`dw4_cont_CF`, témoin indispensable des bras avec focus : mêmes instants) ; (4) D100 + focus. Pour tous : eq_w2 toutes les 10 époques (`evaluator.eq_w2_every=10`, `fast_eq=true`), energy_w2 comme avant, statistiques du matcher (‖terme‖/‖grad‖ par plage de κ).
**Choix : t_focus = 0,5 et t_focus_max = 0,05 (et non 0,08).** Raison : le flux du noyau (1b) est presque éteint au-delà de κ = 0,5 (sauf l'étoile), alors que la symétrie annule le terme quand κ → 0 ; choisi à partir de 1b, avant tout run avec focus. Écart de formulation : κ(t = 0,05) = 0,532 (κ = 0,5 correspond à t = 0,0456) : « κ < 0,5 » est donc « κ < 0,53 » avec t_focus_max = 0,05.
Fraction des instants de régression avec κ dans [0,1 ; 0,5] : **40,6 %** avec t_focus = 0,5 et t_focus_max = 0,05 (dans [0, 0,1] 7,3 %, dans [0,5, 1] 52,1 %), contre **3,9 %** avec le réglage par défaut (t uniforme ; [0, 0,1] 0,7 %, [0,5, 1] 95,4 %). Pour comparaison : t_focus_max = 0,08 donnerait 26,1 %, 0,15 donnerait 14,8 %.
Mise en œuvre : `+adjoint_matcher.t_focus=0.5 +adjoint_matcher.t_focus_max=0.05` (les clés ne sont pas dans les YAML) ; la composition de la configuration a été vérifiée localement pour le matcher de darting et pour AS.
Plafond de dépense : 0,20 $ (au-delà, arrêt et demande). Avec une seule graine, l'écart minimal interprétable reste d'environ 0,05 (bruit d'exécution du darting) ; rien n'est conclu en dessous.

## Exploration sur la graine 1 : résultats (lecture descriptive, SANS verdict ; `scripts/dw4_explore_analysis.py`, choix écrits en 8e12df6 avant le lancement)

Exécution : 4 runs en parallèle sur Vast.ai, l'offre 57084432 n'étant plus disponible, remplacée par l'offre équivalente autorisée 60226930 (RTX 3080, Core i9-10850K, 0,136 $/h annoncé, 0,1417 $/h facturé avec le stockage) ; 500 époques, 50 évaluations par run, `eq_w2` à CHAQUE évaluation (`eq_w2_every=10`), pertes finies, aucun NaN ; `torch.compile` en repli non compilé (pas de compilateur C) ; coût réel ≈ **0,147 $** (crédit 9,517 → 9,370 ; plafond 0,20 $) ; instance détruite après récupération des 4 logs (`results/explore/`). Avec une seule graine, l'écart minimal interprétable reste d'environ 0,05 ; rien n'est conclu en dessous.

**S (moyenne 300-490) et pente.** Pente de S par évaluation sur 300-490, par 100 époques, SE corrigée AR(1) (z entre parenthèses) :
| bras | S_Q | S_R | pente (SE, z) |
|---|---|---|---|
| C1 (AS, cluster) | 0,363 | 0,367 | +0,022 (0,046, +0,48) |
| D3 | 0,179 | 0,184 | +0,042 (0,031, +1,36) |
| D10 | 0,083 | 0,097 | +0,030 (0,026, +1,14) |
| D30 | 0,048 | 0,061 | +0,017 (0,035, +0,48) |
| **D100** | 0,090 | 0,097 | +0,064 (0,055, +1,17) |
| **D30 + focus** (D30F) | 0,069 | 0,083 | +0,013 (0,028, +0,45) |
| **AS + focus** (CF, témoin) | 0,276 | 0,282 | +0,001 (0,036, +0,03) |
| **D100 + focus** (D100F) | 0,050 | 0,059 | +0,069 (0,048, +1,42) |
Plancher d'un sampler parfait : 0,0034 (Q), 0,0036 (R). Aucune pente n'est distinguable de zéro (|z| < 1,5 ; précision ±0,03 à ±0,06 par 100 époques). S par évaluation par blocs de 100 époques (Q) : D30 0,125 / 0,138 / 0,132 / 0,097 / 0,139 ; D100 0,215 / 0,198 / 0,169 / 0,171 / 0,268 ; D100F 0,349 / 0,223 / 0,253 / 0,192 / 0,242 ; D30F 0,174 / 0,132 / 0,154 / 0,151 / 0,140 ; CF 0,273 / 0,306 / 0,348 / 0,275 / 0,286. À γ = 100 le S de la moyenne est petit mais S par évaluation est plus grand qu'à γ = 30 (0,17-0,27 contre 0,10-0,14) : la dispersion d'une évaluation à l'autre est plus grande.

**Erreur |log(p̄/p_PT)| par bassin (Q ; p̄ entre parenthèses ; p_PT : chemin de 2 0,4792, deux paires 0,0644, étoile 0,0063, triangle + pendant 0,4127, K4− 0,0374) :**
| bras | chemin de 2 | deux paires | étoile | triangle + pendant | K4 moins une arête |
|---|---|---|---|---|---|
| C1 | 0,476 | 0,475 | 0,738 | 0,195 | 0,809 |
| D30 | 0,040 | 0,246 | 0,402 | 0,012 | 0,078 |
| D100 | 0,042 (0,4999) | 0,291 | 0,314 | 0,110 (0,3699) | 0,053 |
| D30F | 0,064 | 0,288 | 0,517 | 0,011 | 0,204 |
| CF | 0,339 | 0,593 | 0,841 | 0,092 | 0,697 |
| D100F | 0,005 | 0,224 | 0,385 | 0,062 | 0,078 |
(D3 : 0,206 / 0,401 / 0,686 / 0,068 / 0,470 ; D10 : 0,090 / 0,330 / 0,557 / 0,000 / 0,270.) L'étoile et les deux paires restent les bassins les plus éloignés de p_PT dans tous les bras darting (étoile 0,31-0,52 ; deux paires 0,22-0,29).

**Moyennes temporelles sur 300-490 (SE par moyennes de 4 lots de 5 évaluations ; `eq_w2` : 20 valeurs dans les nouveaux bras, 3 dans C1, D3, D10, D30 où SD/√3) :**
| bras | energy_w2 (SE) | eq_w2 (SE) |
|---|---|---|
| C1 | 0,445 (0,011) | 0,613 (0,074) |
| D30 | 0,277 (0,029) | 0,570 (0,098) |
| D100 | 0,324 (0,047) | 0,707 (0,063) |
| D30F | 0,385 (0,079) | 0,523 (0,014) |
| CF | 0,423 (0,039) | 0,536 (0,012) |
| D100F | 0,299 (0,035) | 0,671 (0,015) |

**Instabilité et dose.** Aucune perte non finie, aucun NaN. Les niveaux de perte ne sont pas comparables entre bras avec et sans focus (les instants focalisés ont des cibles plus grandes : CF 11,5 contre ≈ 6 pour C1) : D3 5,8, D10 6,0, D30 6,9, D100 15,2, D30F 14,1, CF 11,5, D100F 29,8 (dernière époque). ‖terme‖/‖grad‖ par plage de κ ([0, 0,1] / [0,1, 0,7] / [0,7, 1]) : D30 0,255 / 0,349 / 0,041 ; **D100 0,798 / 1,143 / 0,132** ; D30F 0,246 / 0,317 / 0,039 ; D100F 0,802 / 1,037 / 0,127 : à γ = 100 le terme est du même ordre que le gradient (≥ 1 sur [0,1, 0,7]), et reste à 0,13 sur κ > 0,7 (seuil de 0,5 de la règle de la phase 2 non atteint). Fractions des instants dans les trois plages : sans focus 0,007 / 0,072 / 0,921 ; avec focus 0,073 / 0,467 / 0,460 (0,406 dans [0,1, 0,5]). Propositions valides par échantillon ≈ 8,0.

**Constats descriptifs (aucun au-dessous de 0,05 n'est interprété).**
- Le focus change l'AS lui-même : S_Q(CF) = 0,276 contre 0,363 pour C1 (écart −0,087, au-dessus de 0,05 mais C1 vient du cluster et il n'y a qu'une graine) ; les comparaisons avec focus se font donc contre CF. Contre leur témoin de même focus : D30F − CF = −0,207, D100F − CF = −0,226 (S_Q) ; sans focus : D30 − C1 = −0,315, D100 − C1 = −0,274.
- Entre bras darting : D100 − D30 = +0,042, D30F − D30 = +0,021, D100F − D100 = −0,040 (S_Q) : tous sous 0,05, non interprétables. D100F a le plus petit S_Q des bras avec focus (0,050), au niveau de D30 (0,048).
- eq_w2 : D100 et D100F sont plus hauts que leur témoin ou que CF (D100F − CF = +0,135 pour des SE de 0,015 et 0,012 ; D100 0,707 contre C1 0,613 ± 0,074) ; D30F est au niveau de CF (−0,013). energy_w2 : D100 et D100F plus bas que CF (0,324 et 0,299 contre 0,423) ; D30F plus haut en bruit (0,385 ± 0,079).
- Pas de pente significative à 300-490 dans aucun bras (précision limitée).

## AS en fonction de (d, K, Δ) : pilote (une graine, rapport seul ; `predictions_dk.md` écrit avant, commits 3c4d490 et 1d5ad6e)

Cible `gmm_dk.py` (K ≤ d gaussiennes de variance 1 sur axes orthogonaux à distance Δ, poids géométriques max/min = 10, vérité = cellules de Voronoï). AS seul, `gmm_dk_as` (FourierMLP 4×128, nfe 200, 1500 époques, options rapides), sigma_max = 2√(1 + Δ²/(2d)), graine 0, évaluation toutes les 50 époques sur 5000 échantillons, 30 évaluations. Projet W&B `gmm_dk`. TV = moyenne des TV des 10 dernières évaluations ; couverture, ratios et β sur p̂ moyenné sur ces 10 évaluations ; IS de chemin = 20 000 trajectoires du contrôleur final (ESS, k̂ par ajustement Pareto, TV repondérée). Temps : 0,41 s par époque (médiane, 5 runs en parallèle sur une RTX 3070), 10,5 min par run. Perte finie partout.

| (d, K, Δ) | couverture (perdus, poids p_k) | TV (plancher) | β ± SE | ratios p̂/p | ESS IS | k̂ | TV repondérée |
|---|---|---|---|---|---|---|---|
| 8, 2, 8 | 1/2 (mode 2, 0,091) | 0,091 (0,002) | n.d. | 1,10 / 0,00 | 0,45 | 0,16 | 0,091 |
| 8, 8, 8 | 6/8 (modes 6 et 7, 0,058 et 0,042) | 0,160 (0,014) | −0,20 ± 0,12 | 0,82 1,02 1,21 1,74 1,27 0,003 0,007 1,37 | 0,39 | 0,25 | 0,099 |
| 32, 2, 8 | 1/2 (mode 2, 0,091 ; p̂ = 0,016, ratio 0,18) | 0,075 (0,002) | n.d. | 1,08 / 0,18 | 0,10 | 0,53 | 0,022 |
| 32, 8, 8 | 3/8 (modes 4 à 8) | 0,433 (0,014) | +3,04 ± 0,75 | 2,25 1,26 0,30 puis 0,001 (modes 4–8) | 0,066 | 0,49 | 0,324 |
| 32, 8, 4 | 8/8 | 0,028 (0,014) | −0,07 ± 0,02 | 0,95 0,95 1,06 1,03 0,98 1,08 1,09 1,07 | 0,17 | 0,34 | 0,015 |

Confrontation avec les prédictions (écrites avant) : (32, 8, 4) est dans toutes les fourchettes (couverture 8/8, TV 0,028 ∈ 0,02–0,08, β −0,07 ∈ −0,3–0,1, ESS 0,17 ∈ 0,005–0,3). À Δ = 8 les prédictions de couverture sont fausses dans 3 cellules sur 4 : (8, 2, 8) prédit 2/2, obtenu 1/2 ; (8, 8, 8) prédit 8/8 (7–8), obtenu 6/8 ; (32, 2, 8) prédit 2/2, obtenu 1/2 (ratio 0,18, juste sous le seuil 0,25) ; (32, 8, 8) prédit 5/8 (2–8), obtenu 3/8 (dans la fourchette). β : le signe prédit (négatif, aplatissement) n'est observé que pour (8, 8, 8) ; en (32, 8, 8) β = +3,04 (masse concentrée sur les modes lourds, sur 3 modes couverts seulement). ESS : prédictions tenues pour (8, 2, 8) et (32, 8, 4), fausses (ESS plus grand) pour (8, 8, 8) 0,39 contre 0,01–0,2, (32, 2, 8) 0,10 contre < 0,05 et (32, 8, 8) 0,066 contre < 10⁻³ avec k̂ 0,49 < 0,7. Prédictions de l'utilisateur : P2 vue sur les deux paires (d = 8 → 32 à Δ = 8 : ESS 0,45 → 0,10 pour K = 2, 0,39 → 0,066 pour K = 8). P3 : à Δ = 8 et d = 32, les cinq modes les plus légers sont perdus quand K = 8 (couverture 3/8) ; à d = 8, ce ne sont pas les plus légers qui sont perdus (modes 6 et 7, le plus léger, mode 8, est sur-représenté à 1,37). P1 : β ≈ −0,07 à Δ = 4 mais pas de dérive vers −1 à Δ = 8 pour d = 32.
Réserves : une graine, 1500 époques, sigma_max dépend de (d, Δ) par construction, β calculé sur les seuls modes couverts (K = 2 : pas de SE ; 3 modes en (32, 8, 8)). À Δ = 8 en d ≥ 8 AS perd des modes (le mode léger de K = 2 a une proportion de 0,000 en d = 8), alors qu'en 2D (K = 3, Δ jusqu'à 10,4) tous les modes sont couverts et les poids sont aplatis vers 1/3 : le comportement change entre d = 2 et d ≥ 8 (non expliqué).

**Lecture du pilote (HYPOTHÈSE, pas un résultat ; formulée par l'utilisateur, non testée).** Deux régimes : (proche) tous les modes trouvés et poids justes ; (éloigné) modes perdus, avec concentration sur les modes lourds à d = 32 (β = +3 sur 3 modes couverts) au lieu de l'aplatissement vers l'égalité observé en 2D (K = 3). Une graine, 5 cellules : ce n'est pas établi ; le balayage en Δ et l'ablation de sigma_max (lot suivant) servent à la mettre à l'épreuve.

## AS en fonction de (d, K, Δ) : balayage en Δ et ablation de sigma_max (une graine ; rapport seul ; addendum 1 de `predictions_dk.md`, commit b95fd97, écrit avant)

Protocole du pilote (1500 époques, 10 dernières évaluations sur 5000 échantillons, sigma_max = 2√(1 + Δ²/(2d)) sauf ablation). 15 runs nouveaux + 5 réutilisés du pilote + 4 ablations. Figure : `figures/gmm_dk_sweep.png` (couverture, β, ESS de l'IS en fonction de Δ) ; chiffres : `figures/gmm_dk_sweep_analysis_part{1,2}.json`. Premier essai avec 19 runs en parallèle : échec immédiat (le GPU de 8 Go sature, erreur de capture de graphe CUDA) ; relance avec au plus 7 runs en parallèle, protocole inchangé, perte finie partout. **L'IS de chemin n'a été calculé que pour les 5 runs du pilote** (le plafond de coût était atteint) : la panneau « ESS » de la figure n'a donc que ces 5 points.

Couverture (modes couverts / K), TV (plancher), β ± SE (K = 2 : pas de SE ; « n.d. » = moins de 2 modes couverts) :
| Δ | (8, 2) | (8, 8) | (32, 2) | (32, 8) |
|---|---|---|---|---|
| 4 | 2/2, 0,010, −0,04 | 8/8, 0,022, −0,03 ± 0,01 | 2/2, 0,009, −0,04 | 8/8, 0,028, −0,07 ± 0,02 |
| 5 | 2/2, 0,007, −0,03 | 8/8, 0,028, −0,03 ± 0,01 | 2/2, 0,004, −0,01 | 8/8, 0,037, −0,06 ± 0,04 |
| 6 | 2/2, 0,012, −0,04 | 8/8, 0,046, −0,05 ± 0,05 | 2/2, 0,034, +0,21 | 7/8 (perdu : mode 8, 0,031), 0,161, +0,20 ± 0,27 |
| 7 | 2/2, 0,112, −0,41 | 8/8, 0,077, −0,22 ± 0,05 | 2/2, 0,048, +0,35 | 3/8, 0,323, −0,10 ± 0,75 |
| 8 | 1/2, 0,091, n.d. | 6/8, 0,160, −0,20 ± 0,12 | 1/2, 0,075, n.d. | 3/8, 0,433, +3,04 ± 0,75 |
Planchers de TV : 0,002–0,003 (K = 2), 0,014 (K = 8).

Ablation de sigma_max (règle ; × 0,5 ; × 2) :
| (d, K, Δ) | sigma_max (règle, ×0,5, ×2) | règle : couverture, TV, β | × 0,5 | × 2 |
|---|---|---|---|---|
| (8, 8, 8) | 4,472 ; 2,236 ; 8,944 | 6/8, 0,160, −0,20 ± 0,12 | 8/8, 0,153, −0,34 ± 0,12 | 2/8, 0,535, +0,58 (sans SE) |
| (32, 8, 6) | 2,500 ; 1,250 ; 5,000 | 7/8, 0,161, +0,20 ± 0,27 | 8/8, 0,082, −0,02 ± 0,06 | 1/8, 0,694, n.d. |

Confrontation avec les prédictions de l'addendum (écrites avant) :
- Δ de la première perte de mode : (8, 2) prédit 7 (6–8), observé 8 ; (8, 8) prédit 7 (6–8), observé 8 ; (32, 2) prédit 7 (5–8), observé 8 ; (32, 8) prédit 6 (5–7), observé 6. Les quatre sont dans la fourchette ; deux à l'unité près au-dessus de ma valeur centrale (j'ai prédit la perte un cran trop tôt pour trois cellules).
- Zone « tous les modes trouvés et β < −0,3 » à d = 32 : prédite absente, observée absente (β ∈ [−0,07 ; +0,35] là où tous les modes sont trouvés). À d = 8 : (8, 2, 7) donne β = −0,41 avec tous les modes trouvés (sans SE) ; (8, 8, 7) donne −0,22 ± 0,05 (> −0,3) ; zone d'aplatissement faible présente en d = 8 pour K = 2 à Δ = 7 seulement.
- **sigma_max : prédiction FAUSSE, effet de signe contraire.** Prédit : × 2 augmente la couverture, × 0,5 la diminue. Observé : × 0,5 donne 8/8 aux deux cellules (contre 6/8 et 7/8 avec la règle) et × 2 provoque un quasi-effondrement (2/8 et 1/8, TV 0,54 et 0,69). L'explication « base trop étroite » est contredite par cette ablation (une graine, deux cellules). Conséquence pour la lecture du pilote : la perte de modes à Δ = 8 dépend de sigma_max, donc les « régimes » décrits plus haut sont confondus avec la règle sigma_max (qui croît avec Δ) ; la règle × 2 de l'écart-type est peut-être trop grande pour ces cibles. Aucune explication testée.
- Temps et coût : 0,41–0,50 s par époque avec 7 runs en parallèle (prédit 0,5–0,9) ; 19 runs en parallèle impossibles sur 8 Go. Coût du lot : l'instance a tourné de 18:25 à 19:34 (69 min) à 0,1339 $/h, soit environ 0,154 $, **légèrement au-dessus du plafond de 0,15 $** (le premier essai de 19 runs en parallèle et mes tentatives SSH bloquées y ont ajouté du temps). Instance détruite à la fin.
Réserves : une graine ; une seule cellule (8, 2, 7) montre β < −0,3 avec tous les modes trouvés, et sans SE ; IS de chemin du balayage non calculé.

## AS en fonction de (d, K, Δ) : sigma_max domine ; test de H_ρ (CPU local, sans GPU ; une graine ; rapport seul)

**Lecture à corriger (décision de l'utilisateur, 2026-11-19).** L'ablation de sigma_max (× 0,5 / × 2, deux cellules) a un effet de signe contraire à la prédiction et plus grand que celui de Δ à sigma_max fixé par la règle. La lecture « deux régimes (proche : tous les modes et poids justes ; éloigné : modes perdus, concentration sur les modes lourds à d = 32) » du pilote est donc **confondue avec la règle sigma_max** (qui croît avec Δ) : elle ne décrit pas une propriété d'AS indépendante du choix de sigma_max. Elle reste une hypothèse non testée à sigma_max contrôlé.

**Ablation sigma_max, β ± SE et ratios p̂/p par mode (4 runs).** Poids cibles p = (0,30 ; 0,22 ; 0,16 ; 0,11 ; 0,08 ; 0,06 ; 0,04 ; 0,03).
- (32, 8, 6), × 0,5 (sigma_max 1,25) : couverture 8/8, TV 0,082, β = −0,02 ± 0,06, ratios 1,18 0,90 0,80 0,98 0,95 0,98 1,12 1,01 (poids presque justes).
- (32, 8, 6), × 2 (5,0) : 1/8, TV 0,694, β non défini ; p̂_1 = 0,994 (ratio 3,31), les 7 autres modes à 0,0006–0,001 (ratios 0,004–0,032) : tout sur le mode le plus lourd.
- (8, 8, 8), × 0,5 (2,236) : 8/8, TV 0,153, β = −0,34 ± 0,12, ratios 0,76 0,67 1,10 1,43 1,01 1,45 2,09 1,19 (légèrement aplati).
- (8, 8, 8), × 2 (8,944) : 2/8, TV 0,535, β = +0,58 (sans SE, 2 modes) ; p̂ = 0,73 (mode 1, ratio 2,43), 0,26 (mode 3, ratio 1,66), mode 2 (poids 0,22) perdu : les modes trouvés ne sont pas les plus lourds dans l'ordre.

**ρ = √(d ν₁)/‖μ‖** (ν₁ = Σ g(t_k)² Δt sur `linspace(0,1,200)`, schéma réel ; ‖μ‖ = Δ/√2 ; `scripts/gmm_dk_rho.py`, tableau complet `figures/gmm_dk_rho_table.json`, figure `figures/gmm_dk_rho.png`, `scripts/gmm_dk_rho_figure.py`). H_ρ et son critère ont été écrits et commités avant (addendum 2 de `predictions_dk.md`, commit 4099d9f).
Résultats (24 runs ; TV/plancher ; l'ν₁ du schéma Euler est ≈ 4 % au-dessus de σ_max²) :
- **Seuil commun en ρ : NON soutenu.** Meilleur seuil ρ* ≈ 4,56 : 6 runs mal classés sur 24 (critère : au plus 2).
- Groupes (min d'erreurs, intervalle de ρ* qui l'atteint) : d = 8 : 2 erreurs, [2,89 ; 3,73] ; d = 32 : 4 erreurs, [4,56 ; 5,68] (aucun seuil à ≤ 1 erreur dans l'un ou l'autre groupe). Δ = 4 : 0 erreur, [4,56 ; 5,56] ; Δ = 5 : 0, [3,85 ; 4,85] ; Δ = 6 : 1, [2,45 ; 5,10] ; Δ = 7 : 1, [2,35 ; 4,10] ; Δ = 8 : 0, ρ* = 1,71 (entre 1,14 sans perte et 2,28 avec perte).
- **Le seuil dépend de Δ** : les intervalles de Δ = 4 ou 5 et de Δ = 8 sont disjoints, et le seuil décroît quand Δ croît (≥ 4,5 à Δ = 4 ; ≈ 1,7 à Δ = 8). Preuve directe : (8, ·, 4) et (32, ·, 8) ont le même sigma_max (2,828), le même ν₁ (8,32) et le même ρ (2,89) ; les premiers trouvent tous leurs modes (2/2 et 8/8), les seconds en perdent (1/2 et 3/8).
- **À (d, K, Δ) fixés, ρ va dans le sens de l'hypothèse** : × 0,5 (ρ divisé par 2) retrouve tous les modes, × 2 les perd (ablation, 2 cellules).
- Sous la règle sigma_max, ρ décroît quand Δ croît (d = 8 : 2,89 → 2,28 de Δ = 4 à 8 ; d = 32 : 4,56 → 2,89), alors que les pertes arrivent aux Δ élevés : c'était ma prédiction écrite avant (« seuil dépendant de Δ »).
Réserves : une graine ; 24 points dont 6 à Δ = 8 et 4 ablations ; le critère est un compte d'erreurs de classification (pas un test statistique) ; K = 2 et K = 8 mélangés ; aucune grandeur de remplacement n'est testée ici.

## Fenêtre de sigma_max en fonction de Δ (graine 0 ; rapport seul ; addendum 3 de `predictions_dk.md`, commit f581ea6, écrit avant)

16 runs nouveaux (+ 8 réutilisés) : (d, K) = (32, 8), Δ ∈ {4, 6, 8}, et (8, 8, 8), sigma_max = m × règle, m ∈ {0,25 ; 0,5 ; 0,75 ; 1 ; 1,5 ; 2}, 1500 époques, 7 runs en parallèle (0,42–0,50 s par époque), perte finie partout. IS de chemin (20 000 trajectoires) sur les 16 nouveaux contrôleurs et sur les 24 contrôleurs du balayage en Δ. Tableau complet (d, K, Δ, graine, m, sigma_max, ρ, couverture, TV, β, ESS, k̂, TV après IS) : `figures/gmm_dk_window_table.json` ; figures `figures/gmm_dk_window_d32_K8.png` et `figures/gmm_dk_window_d8_K8.png` ; scripts `scripts/gmm_dk_window_report.py`, `scripts/gmm_dk_analysis.py --is_only`.
**Lot 2 (graine 1) NON réalisé** : je l'ai lancé (7 runs) puis arrêté à 20:55 parce qu'il n'aurait pas fini avant le plafond de coût (0,95 s par époque avec l'IS en parallèle). Aucun résultat de graine 1 n'existe.

Couverture (modes couverts / 8) en fonction du multiplicateur m (sigma_max entre parenthèses) :
| cellule | × 0,25 | × 0,5 | × 0,75 | × 1 (règle) | × 1,5 | × 2 |
|---|---|---|---|---|---|---|
| (32, 8, Δ = 4) | 1 (0,56) | 8 (1,12) | 8 (1,68) | 8 (2,24) | 8 (3,35) | 8 (4,47) |
| (32, 8, Δ = 6) | 1 (0,63) | 8 (1,25) | 7 (1,88) | 7 (2,50) | 7 (3,75) | 1 (5,00) |
| (32, 8, Δ = 8) | 1 (0,71) | 1 (1,41) | 3 (2,12) | 3 (2,83) | 2 (4,24) | 1 (5,66) |
| (8, 8, Δ = 8) | 1 (1,12) | 8 (2,24) | 8 (3,35) | 6 (4,47) | 2 (6,71) | 2 (8,94) |
TV/plancher (plancher 0,014), β ± SE dans la fenêtre : (32, 8, 4) : 1,9 à 2,4 fois le plancher, β ∈ [−0,08 ; 0,00] pour m = 0,5 à 2 ; (32, 8, 6) : m = 0,5 : 5,8 fois, β = −0,02 ± 0,06 ; (8, 8, 8) : m = 0,5 : 11,1 fois, β = −0,34 ± 0,12 ; m = 0,75 : 8,7 fois, β = −0,09 ± 0,14.

Meilleur sigma_max par (d, K, Δ) (graine 0 ; couverture maximale puis TV minimal ; cellules du lot) : (32, 8, 4) : × 0,75 (1,677), 8/8, TV 0,027, β = −0,03 ± 0,03, ratios 1,00 0,94 1,00 1,02 1,07 1,09 0,96 1,05 ; (32, 8, 6) : × 0,5 (1,25), 8/8, TV 0,082, β = −0,02 ± 0,06, ratios 1,18 0,90 0,80 0,98 0,95 0,98 1,12 1,01 ; (32, 8, 8) : × 1 (2,83), 3/8 seulement, TV 0,433 (× 0,75 a la même couverture, 3/8, avec TV 0,611) : AUCUNE valeur de la grille ne couvre les 8 modes à Δ = 8 [CORRECTION du 2026-11-19 : la version précédente disait « × 0,75 ou × 1 » et attribuait un TV minimal à × 1,5 ; c'était faux, le meilleur est × 1] ; (8, 8, 8) : × 0,75 (3,354), 8/8, TV 0,120, β = −0,09 ± 0,14, ratios 1,15 0,75 0,72 0,97 1,66 0,84 1,31 0,96. Le tableau des autres cellules du balayage (Δ = 4 à 7, K = 2 ou 8, d = 8 ou 32, m = 1) est dans la sortie de `gmm_dk_window_report.py` et dans le JSON.

IS de chemin : dans la fenêtre l'ESS est élevé et décroît quand sigma_max augmente (par exemple (32, 8, 4) : 0,875 à × 0,5, 0,50 à × 0,75, 0,25 à × 1, 0,056 à × 1,5, 0,011 à × 2 ; k̂ de 0,11 à 0,76). L'ESS n'est PAS un indicateur fiable de perte de mode : (32, 8, 8) × 0,25 (couverture 1/8) a un ESS de 0,49 et un TV après IS de 0,69 ; (32, 8, 4) × 0,25 (1/8) a un ESS de 0,001 et k̂ = 1,27. Là où des modes manquent mais sont visités, l'IS corrige en partie ((8, 8, 8) × 1 : TV 0,160 → 0,095 ; (32, 8, 6) × 1 : 0,161 → 0,049) ; quand un mode est totalement absent des trajectoires, il ne corrige rien.

Confrontation avec les prédictions de l'addendum 3 (écrites avant) :
- Première perte en montant : (8, 8, 8) × 1 prédit et observé ✓ ; (32, 8, 6) prédit × 1, observé × 0,75 (7/8, un cran plus tôt) ✗ ; (32, 8, 4) prédit × 1,5, **observé aucune perte jusqu'à × 2 (sigma_max = 4,47)** ✗ ; (32, 8, 8) prédit × 1 avec × 0,75 sans perte, **observé aucune valeur de la grille avec les 8 modes** ✗.
- Critique bas (perte à × 0,25) : (32, 8, 8) oui ✓ ; (32, 8, 6) oui ✓ ; (8, 8, 8) oui ✓ ; (32, 8, 4) prédit non (60 %), **observé oui (1/8)** ✗. Les quatre cellules s'effondrent à × 0,25 (sigma_max entre 0,56 et 1,12).
- Fenêtre contenant × 0,5 et × 0,75 : vraie pour (32, 8, 4) et (8, 8, 8), fausse pour (32, 8, 6) (× 0,75 perd un mode) et (32, 8, 8).
- Hypothèse de travail « sigma_max critique haut dépend surtout de d, peu de Δ » : **contredite** à d = 32 : la fenêtre est large à Δ = 4 (m de 0,5 à 2, sigma_max de 1,1 à 4,5), étroite à Δ = 6 (m = 0,5 seulement sur la grille) et absente à Δ = 8. Ce qui est établi sur ces runs : la fenêtre en sigma_max se rétrécit quand Δ augmente et disparaît à Δ = 8 à d = 32 sur la grille testée (une graine).
- Signe de β dans la fenêtre : d = 32 : β ≈ 0 ✓ (−0,08 à 0,00 pour Δ = 4 ; −0,02 pour Δ = 6) ; d = 8 : négatif ✓ (−0,34 à m = 0,5, −0,09 à m = 0,75). 
- ESS prédit 0,1–0,4 à d = 32 et 0,3–0,6 à d = 8 : dans la fenêtre observé 0,25–0,88 (d = 32) et 0,47–0,57 (d = 8) : au-dessus pour d = 32 aux petits sigma_max.
- Coût et temps : instance de 20:05 à 20:56 (50,6 min) à 0,1339 $/h, soit **environ 0,113 $, sous le plafond de 0,15 $** (ma prédiction 0,10–0,13 $ ✓) ; le lot 2 aurait fait dépasser le plafond.
Réserves : une graine ; la grille de multiplicateurs est grossière (facteurs 1,33–2) ; à Δ = 8 les β (−3,4 ; +3,0 ; −5,5) portent sur 2–3 modes couverts et ne sont pas interprétables ; le critère de « meilleur sigma_max » est celui fixé par l'utilisateur et ne tient pas compte de l'ESS.

**Conclusion sur l'IS de chemin (décision de l'utilisateur, 2026-11-20).** Dans les 3 cellules à couverture 8/8 au meilleur sigma_max ((32, 8, 4) × 0,75, (32, 8, 6) × 0,5, (8, 8, 8) × 0,75), l'IS de chemin ramène le TV au niveau du plancher (0,007 ; 0,007 ; 0,012 contre un plancher d'environ 0,009–0,010 à 10 000 échantillons effectifs, extrapolation en 1/√N du plancher 0,014 à 5000). Les modes jamais visités ne sont pas corrigés : à (32, 8, 8) × 1 (3/8) le TV passe seulement de 0,433 à 0,324. Les ratios p̂/p par mode après IS n'avaient pas été sauvegardés.

## Critère de choix de sigma_max par log Ẑ de l'IS de chemin (règle de l'addendum 4 de `predictions_dk.md`, commit d7170f6, non modifiée ; une graine ; évaluation du 2026-11-20)

IS de chemin (20 000 trajectoires) sur les 24 contrôleurs des 4 cellules × 6 multiplicateurs, avec sauvegarde des log w (`is_logw.pt`, dans `results/gmm_dk_logz/logw.tar.gz`, hors dépôt) et des proportions pondérées par mode ; log Ẑ = logsumexp(log w) − log N, SE par la méthode delta ; scripts `scripts/gmm_dk_analysis.py --is_only`, `scripts/gmm_dk_logz_eval.py` ; JSON `figures/gmm_dk_logz_eval.json`, `figures/gmm_dk_logz_is_{0,1,2}.json`. Le log Z exact de la cible est 0. Coût : 10,4 min d'instance à 0,1339 $/h, soit environ 0,023 $ (dont 7,5 min perdues sur une première instance tombée à l'état `exited` sans ouvrir SSH, détruite).

Tableau (cellule (d, K, Δ), m = multiplicateur de la règle, sigma_max, couverture, log Ẑ ± SE, ESS, k̂, TV AS → TV après IS) :
| cellule | m | sigma_max | cov | log Ẑ ± SE | ESS | k̂ | TV AS → IS |
|---|---|---|---|---|---|---|---|
| (8, 8, 8) | 0,25 | 1,118 | 1/8 | −0,186 ± 0,103 | 0,005 | 4,53 | 0,765 → 0,119 |
| (8, 8, 8) | 0,5 | 2,236 | 8/8 | −0,001 ± 0,006 | 0,571 | 0,20 | 0,153 → 0,008 |
| (8, 8, 8) | 0,75 | 3,354 | 8/8 | −0,004 ± 0,007 | 0,488 | 0,18 | 0,120 → 0,005 |
| (8, 8, 8) | 1 | 4,472 | 6/8 | −0,103 ± 0,010 | 0,355 | 0,34 | 0,160 → 0,094 |
| (8, 8, 8) | 1,5 | 6,708 | 2/8 | −0,780 ± 0,012 | 0,266 | 0,31 | 0,552 → 0,541 |
| (8, 8, 8) | 2 | 8,944 | 2/8 | −0,542 ± 0,031 | 0,049 | 0,63 | 0,535 → 0,413 |
| (32, 8, 4) | 0,25 | 0,559 | 1/8 | +0,044 ± 0,242 | 0,001 | 1,16 | 0,637 → 0,323 |
| (32, 8, 4) | 0,5 | 1,118 | 8/8 | +0,002 ± 0,003 | 0,879 | 0,03 | 0,027 → 0,004 |
| (32, 8, 4) | 0,75 | 1,677 | 8/8 | −0,011 ± 0,007 | 0,499 | 0,21 | 0,027 → 0,011 |
| (32, 8, 4) | 1 | 2,236 | 8/8 | +0,000 ± 0,013 | 0,241 | 0,32 | 0,028 → 0,015 |
| (32, 8, 4) | 1,5 | 3,354 | 8/8 | −0,004 ± 0,031 | 0,050 | 0,53 | 0,028 → 0,043 |
| (32, 8, 4) | 2 | 4,472 | 8/8 | −0,046 ± 0,044 | 0,025 | 0,62 | 0,034 → 0,062 |
| (32, 8, 6) | 0,25 | 0,625 | 1/8 | +0,193 ± 0,333 | 0,000 | 1,50 | 0,673 → 0,234 |
| (32, 8, 6) | 0,5 | 1,250 | 8/8 | +0,000 ± 0,005 | 0,635 | 0,26 | 0,082 → 0,009 |
| (32, 8, 6) | 0,75 | 1,875 | 7/8 | −0,003 ± 0,020 | 0,115 | 0,35 | 0,111 → 0,021 |
| (32, 8, 6) | 1 | 2,500 | 7/8 | +0,012 ± 0,030 | 0,052 | 0,47 | 0,161 → 0,024 |
| (32, 8, 6) | 1,5 | 3,750 | 7/8 | +0,019 ± 0,058 | 0,015 | 0,69 | 0,072 → 0,049 |
| (32, 8, 6) | 2 | 5,000 | 1/8 | −52,258 ± 0,977 | 0,000 | 3,94 | 0,694 → 0,700 |
| (32, 8, 8) | 0,25 | 0,707 | 1/8 | −1,192 ± 0,007 | 0,533 | 0,20 | 0,697 → 0,694 |
| (32, 8, 8) | 0,5 | 1,414 | 1/8 | −0,071 ± 0,072 | 0,010 | 0,44 | 0,668 → 0,137 |
| (32, 8, 8) | 0,75 | 2,121 | 3/8 | −0,402 ± 0,022 | 0,094 | 0,33 | 0,611 → 0,324 |
| (32, 8, 8) | 1 | 2,828 | 3/8 | −0,435 ± 0,024 | 0,079 | 0,47 | 0,433 → 0,323 |
| (32, 8, 8) | 1,5 | 4,243 | 2/8 | −0,700 ± 0,048 | 0,021 | 0,58 | 0,597 → 0,480 |
| (32, 8, 8) | 2 | 5,657 | 1/8 | −2,178 ± 0,079 | 0,008 | 0,71 | 0,887 → 0,886 |

**Verdict de la règle (appliquée telle qu'écrite) : critère validé, 3 cellules sur 3.** (32, 8, 4) : tous les runs sont à égalité à 2 SE près (le max de log Ẑ est le run × 0,25 collapsé, +0,044 ± 0,242) ; départage par l'ESS : × 0,5 (sigma_max 1,118, 8/8, ESS 0,879) ; meilleur réel × 0,75 (1,677, 8/8, TV 0,027). (32, 8, 6) : égalités {× 0,25, 0,5, 0,75, 1, 1,5} ; choisi × 0,5 (1,25, 8/8, ESS 0,635) = meilleur réel. (8, 8, 8) : égalités {× 0,25, 0,5, 0,75} ; choisi × 0,5 (2,236, 8/8, ESS 0,571) ; meilleur réel × 0,75 (3,354, 8/8, TV 0,120). (32, 8, 8) (pas de 8/8 dans la grille, hors règle) : choisi × 0,5 (1,414, 1/8, ESS 0,010, seul run dans les 2 SE du max −0,071 ± 0,072), alors que le meilleur réel est × 1 (3/8) : le critère y choisit un mauvais run.
**Réserve essentielle : le succès vient du départage par l'ESS, pas de log Ẑ.** Dans (32, 8, 4) et (32, 8, 6), l'argmax de log Ẑ seul est le run effondré × 0,25 (1/8), avec une SE de 0,24 et 0,33 qui le rend indiscernable des runs à 8/8 ; sans le départage par l'ESS le critère échouerait dans ces deux cellules. Au choix du plus petit sigma_max de la fenêtre (le plus grand ESS), il réussit parce que, dans ces cellules, l'ESS est très faible au × 0,25 (0,001 et 0,000). Ma prédiction (65 % de validation, point faible attendu : un × 0,25 effondré avec un grand ESS) : validation observée ; le point faible n'apparaît que dans la cellule hors règle (32, 8, 8) × 0,25 (ESS 0,533, 1/8).
Au meilleur sigma_max réel des 3 cellules à 8/8, ratios p̂/p après IS de chemin : (32, 8, 4) × 0,75 : 1,00 1,01 1,03 0,95 0,97 1,02 1,04 0,92 ; (32, 8, 6) × 0,5 : 1,01 1,01 0,98 0,98 0,99 1,05 0,94 1,01 ; (8, 8, 8) × 0,75 : 1,00 1,01 1,00 1,02 1,00 0,96 0,96 1,00 (ratios AS correspondants : 1,00 0,94 1,00 1,02 1,07 1,09 0,96 1,05 ; 1,18 0,90 0,80 0,98 0,95 0,98 1,12 1,01 ; 1,15 0,75 0,72 0,97 1,66 0,84 1,31 0,96). Au meilleur sigma_max de (32, 8, 8) × 1 : 1,55 1,36 1,51 0,00 0,002 0,00 0,006 0,008 (les 5 modes les plus légers sont absents).

**Sans règle : log Ẑ contre le log de la masse cible couverte** (somme des p_k des modes couverts par l'AS, p̂ ≥ 0,25 p ; colonne « log Ẑ − log M » dans `figures/gmm_dk_logz_eval.json`). log Ẑ suit log M à ±0,05 pour 17 runs sur 24 (18 sur 24 à 2 SE près) : par exemple (8, 8, 8) × 1,5 : −0,780 contre −0,779 ; (32, 8, 8) × 0,25 : −1,192 contre −1,197 ; (32, 8, 8) × 2 : −2,178 contre −2,184. Le critère mesure donc bien la masse atteinte par les trajectoires quand les modes perdus sont réellement absents. Il s'écarte dans 6 runs : (i) × 0,25 des trois cellules à 1/8 et (32, 8, 8) × 0,5 : log Ẑ dépasse log M de +1,1 à +1,4 (parfois avec une SE de 0,07–0,10 en (8, 8, 8) et (32, 8, 8)) : l'IS atteint de la masse dans des modes que l'AS ne couvre pas au seuil 0,25 p (visites rares) ; (ii) (8, 8, 8) × 2 : +0,24 ± 0,03 ; (iii) (32, 8, 6) × 2 : log Ẑ = −52 ± 0,98 (k̂ = 3,94, ESS 0,000) : effondrement de l'estimateur, très au-dessous de log M = −1,20. Corrélation entre log Ẑ et log M (hors le run à −52) : 0,65. Un log Ẑ très négatif accompagne bien une perte de masse dans ces runs, mais un log Ẑ proche de 0 n'implique pas la couverture (× 0,25 : +0,04, +0,19, −0,19 avec 1/8).
Réserves : une graine ; 24 points ; la couverture « 0,25 p » est le critère de l'étude, pas celui d'Ẑ ; les ESS de départage varient de 0,000 à 0,879 sur une grille de multiplicateurs grossière.

**Lecture du critère log Ẑ (décision de l'utilisateur, 2026-11-20).** La règle passe (3 cellules sur 3), mais grâce à l'ESS : log Ẑ seul aurait échoué dans 2 cellules sur 3 ((32, 8, 4) et (32, 8, 6)), à cause de runs effondrés (× 0,25, 1/8) dont log Ẑ est sans biais mais de variance énorme (SE 0,24 et 0,33), donc indiscernable des runs à 8/8. Voir `rules_sigma_select.md` pour la validation sur la graine 1.

## Validation du critère de choix de sigma_max sur la graine 1 (`rules_sigma_select.md`, commits 09cf984 puis 39a9edc, écrits avant la location ; 24 runs de graine 1, IS de chemin 20 000 trajectoires sur chaque contrôleur ; 2026-11-20)

Protocole de la graine 0 (1500 époques, 4 cellules × 6 multiplicateurs), 7 runs en parallèle (≈ 13 min par vague de 7 runs, 0,4–0,5 s par époque), perte finie partout, IS avec sauvegarde des log w (`results/gmm_dk_s1/logw_all.tar.gz`, hors dépôt) et des proportions pondérées ; JSON `figures/gmm_dk_s1_{a0,a1,a2,b}.json` et `figures/gmm_dk_sigma_select_seed1.json` ; script `scripts/gmm_dk_sigma_select_eval.py` (testé avant sur la graine 0 : il en reproduit les choix).

**Verdict (règle appliquée telle qu'écrite) : critère validé, 3 cellules sur 3.**
| cellule | choisi (k̂ < 0,7, argmax log Ẑ, égalité à 2 SE → plus grand ESS) | couverture | TV après IS (seuil 2 × plancher ≈ 0,028) | ESS, k̂ | réussite |
|---|---|---|---|---|---|
| (8, 8, 8) | × 0,5 (sigma_max 2,236) ; égalités {0,5 ; 0,75} | 8/8 | 0,009 | 0,63, 0,24 | oui |
| (32, 8, 4) | × 0,5 (1,118) ; égalités {0,5 ; 0,75 ; 1 ; 2} | 8/8 | 0,009 | 0,87, 0,07 | oui |
| (32, 8, 6) | × 0,5 (1,25) ; égalités {0,5 ; 0,75 ; 1} | 8/8 | 0,006 | 0,63, 0,30 | oui |
| (32, 8, 8) (aucun run à 8/8 en graine 1 : hors règle) | × 0,5 (1,414) | 1/8 | 0,135 | 0,03, 0,31 | n.a. |
Tableau complet des 24 runs (couverture, log Ẑ ± SE, ESS, k̂, TV avant et après IS, log Ẑ − log de la masse couverte) dans la sortie du script et dans le JSON. Faits de la graine 1 qui diffèrent de la graine 0 : (32, 8, 6) × 1 couvre 8/8 (7/8 en graine 0) ; (32, 8, 4) × 0,25 couvre 5/8 (1/8 en graine 0) ; (32, 8, 8) × 0,75 : 3/8 avec log Ẑ = +1,080 ± 0,762 (k̂ 0,62, ESS 0,000), estimation non fiable.

**Rapporté sans règle.**
- Règle d'origine (d7170f6, sans filtre k̂) : même choix que la règle reformulée dans les 4 cellules (× 0,5 partout). « Max ESS seul » : × 0,5 dans 3 cellules (8/8) et × 1 à (32, 8, 8) (2/8).
- Rôle du filtre k̂ : sur la graine 1, l'argmax de log Ẑ SEUL (sans départage par l'ESS) avec le filtre k̂ choisit un run à 8/8 dans les 3 cellules ((8, 8, 8) × 0,5 ; (32, 8, 4) × 1 (TV après IS 0,014) ; (32, 8, 6) × 1 (0,016)). Sans le filtre, il échoue à (8, 8, 8) : il prend le run effondré × 0,25 (1/8, log Ẑ = +0,176 ± 0,156, k̂ = 4,75). Le filtre k̂ < 0,7 écarte les runs effondrés ((8, 8, 8) × 0,25 ; (32, 8, 4) × 0,25 : k̂ 0,83 ; (32, 8, 6) × 0,25 et × 2 : k̂ 1,38 et 4,45).
- Ratios p̂/p après IS de chemin au sigma_max choisi : (8, 8, 8) × 0,5 : 1,01 0,97 1,00 0,99 1,01 1,02 0,99 1,12 (AS : 1,01 0,59 0,87 1,17 0,98 1,75 1,85 1,38) ; (32, 8, 4) × 0,5 : 1,02 0,98 1,00 1,01 1,00 0,94 1,01 1,02 (AS : 0,96 1,04 1,00 1,00 1,01 1,00 1,05 1,00) ; (32, 8, 6) × 0,5 : 1,02 0,99 1,00 1,00 1,01 0,99 0,95 0,97 (AS : 0,96 1,08 0,90 1,01 1,06 1,10 1,24 0,66) ; (32, 8, 8) × 0,5 : 1,14 1,07 0,97 1,12 1,78 0,00 0,00 0,003 (les 3 modes les plus légers absents).
- Log Ẑ contre log de la masse couverte : proche (à ±0,1) pour la majorité des runs ; les écarts les plus grands sont les runs effondrés (× 0,25 à 1/8 : +1,1 à +1,7 ; l'IS atteint de la masse dans des modes visités rarement) et (32, 8, 6) × 2 (log Ẑ = −43,9 ± 1,0, k̂ = 4,45 : effondrement de l'estimateur).
Réserves : deux graines au total (0 et 1) ; la règle a donc passé son test sur données neuves, sans être validée au-delà ; (32, 8, 8) n'a aucun run à 8/8 donc la règle n'y est pas testée ; le départage par l'ESS reste nécessaire en graine 0 (sans lui, 2 cellules sur 3 échouaient) et le filtre k̂ le remplace en graine 1 ; le choix × 0,5 coïncide avec le plus petit sigma_max de la fenêtre dans les 3 cellules, et la règle choisit donc surtout le sigma_max de plus grand ESS.
**Coût : 0,189 $ facturés (instance 61596626, de 15:12:17 à 16:37:05, 84,8 min à 0,1339 $/h), au-dessus du plafond de 0,12 $.** Les 24 runs étaient finis et analysés vers 16:07, mais j'ai posé à l'utilisateur la question « finir ou arrêter ? » à 16:04 alors que l'instance continuait de tourner, et la réponse n'est arrivée qu'à 16:36 : environ 30 minutes d'instance inutile (≈ 0,067 $). Erreur de ma part : l'instance aurait dû être détruite dès la fin des runs, ou la question posée avant que le plafond soit atteint.

## Quadrature 1D (Failure Modes) : vitesse de correction des poids, AS seul contre AS + darting (CPU local, déterministe, sans Monte Carlo ; 2026-11-20)

Règles écrites avant le script : `rules_quadrature_1d.md` (c204938). Script `scripts/quadrature_1d_failure_modes.py`, résultats `figures/quadrature_1d_results.json`, figure `figures/quadrature_1d_logrho.png` (ln ρ en fonction de 1/(2ε²)). **Temps total mesuré : 15,1 s** (budget 60 s) ; échelon A de la grille (dx = ε/20, 400 pas, 48 nœuds de Gauss–Hermite par composante), pilote : projection 7,6 s. Aucun réseau, aucun job.
Contrôles : u = 0 donne TV(p(1), N(0, ν₁ − w_fin)) entre 8e-6 et 1,8e-5 (< 1e-4 ✓) ; la masse est conservée (1,000000) ; u = u_q donne ŵ = 0,50000 aux 6 ε ; vérification de t₀ à ε = 0,3 (ν_{t₀} divisé par 2) : |Δŵ| = 3e-14 (u_q) et 1,7e-6 (AS), sous le plancher ✓.

Tableau (ŵ ; ρ = (ŵ_m − ŵ_{u_q}) / (w − w') ; « plancher » = définition pré-enregistrée max(|ŵ_q − w'|, |ŵ_* − w|, 1e-6)) :
| ε | plancher | ŵ AS | ρ AS | ŵ CV η=0 | ρ CV η=0 | ŵ CV η=0,1 | ρ CV η=0,1 | ŵ CV η=0,3 | ρ CV η=0,3 | E_q[ω] (η = 0 ; 0,1 ; 0,3) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0,60 | 2,9e-2 | 0,82281 | 1,076 | 0,92718 | 1,424 | 0,92577 | 1,419 | 0,92246 | 1,408 | 0,0774 ; 0,0765 ; 0,0748 |
| 0,50 | 1,4e-2 | 0,75128 | 0,8376 | 0,90145 | 1,338 | 0,89985 | 1,333 | 0,89541 | 1,318 | 0,0782 ; 0,0778 ; 0,0757 |
| 0,40 | 3,7e-3 | 0,62188 | 0,4063 | 0,84051 | 1,135 | 0,83861 | 1,129 | 0,83254 | 1,108 | 0,0781 ; 0,0778 ; 0,0759 |
| 0,35 | 1,3e-3 | 0,55749 | 0,1916 | 0,80533 | 1,018 | 0,80337 | 1,011 | 0,79670 | 0,989 | 0,0774 ; 0,0772 ; 0,0756 |
| 0,30 | 2,9e-4 | 0,51628 | 0,0543 | 0,78312 | 0,9437 | 0,78115 | 0,9372 | 0,77417 | 0,9139 | 0,0765 ; 0,0763 ; 0,0747 |
| 0,25 | 2,6e-5 | 0,50180 | 0,0060 | 0,77936 | 0,9312 | 0,77752 | 0,9251 | 0,77059 | 0,9020 | 0,0753 ; 0,0751 ; 0,0735 |
Itération CV η = 0 à ε = 0,3 (ŵ_n pour n = 1…5, w'₀ = 0,5) : 0,78312 ; 0,79836 ; 0,79958 ; 0,79969 ; 0,79969 (|ŵ − w| = 0,0169 ; 0,0016 ; 0,0004 ; 0,0003 ; 0,0003). E_q[ω] : moyenne en temps t uniforme sur [t₀, t_fin].

**Verdicts (règles appliquées telles qu'écrites).**
- **H_AS : NON SOUTENUE.** Première partie vraie : sur les 6 ε (tous au-dessus de 10 × plancher), ln ρ_AS est affine en 1/(2ε²), pente −0,797 (dans [−1,5 ; −0,5]), R² = 0,992. Deuxième partie fausse : ρ_AS < 0,05 pour ε ≤ 0,35 n'est pas vérifié (ρ_AS = 0,19 à ε = 0,35 et 0,054 à ε = 0,30 ; 0,006 à ε = 0,25). La courbe est visiblement convexe en 1/(2ε²) malgré R² = 0,992.
- **H_CV : SOUTENUE.** ρ_CV(η = 0) ≥ 0,5 aux 6 ε (de 0,93 à 1,42) et l'itération atteint |ŵ − w| < 0,01 à la 2ᵉ itération (0,0016).
- **H_flux (descriptif).** ρ_CV varie peu avec η : à ε = 0,30, 0,944 (η = 0), 0,937 (0,1), 0,914 (0,3) ; à ε = 0,60, 1,424 ; 1,419 ; 1,408 ; E_q[ω] moyenné en temps ≈ 0,075–0,078, presque indépendant de η et de ε.

**Déviation à déclarer sur le plancher (mon opérationnalisation, non l'énoncé).** Le contrôle « u = u_* → ŵ ≈ w » est faux physiquement aux grands ε : ∫_{x>0} π = wΦ(1/ε) + (1−w)(1−Φ(1/ε)) vaut 0,771 (ε = 0,6), 0,786 (0,5), 0,796 (0,4), 0,7987 (0,35), 0,79974 (0,3), 0,79998 (0,25), à cause du recouvrement des deux composantes. Les ŵ_* calculés coïncident avec cette valeur exacte à 4,4e-5 près au plus (6,8e-6 à ε = 0,25) : l'erreur de discrétisation réelle est ≤ 4,4e-5. Mon plancher pré-enregistré (max avec |ŵ_* − w|) est donc gonflé par ce recouvrement aux ε ≥ 0,4 (2,9e-2 à 3,7e-3). Effet sur les verdicts : aucun, car les 6 ε restent au-dessus de 10 × plancher avec la définition corrigée (≤ 4,4e-4) et la partie 2 de H_AS ne dépend pas du plancher. De même, ρ = 1 ne correspond pas à « corrigé en un pas » aux grands ε : la correction parfaite vaut ρ = (ŵ_exact − 0,5)/0,3 = 0,904 (ε = 0,6), 0,955 (0,5), 0,988 (0,4), 0,996 (0,35), 0,999 (0,3), 1,000 (0,25) ; AS (ρ = 1,076) et CV (1,42) dépassent donc la cible à ε = 0,6 (la correction relative à ŵ_exact est 1,19 pour AS et 1,57 pour CV η = 0). Ces valeurs sont données ici comme lecture, sans changer les règles ni les verdicts. Le point fixe de l'itération CV (0,79969) est celui de la valeur exacte (0,79974) à 5e-5 près.
Réserves : l'indicateur de validité de darting rend l'intégrande discontinu en x₁ (Gauss–Hermite 48 nœuds, non raffiné) ; le p(1, ·) du texte est p à w_fin = (dx/10)² de la fin ; une seule paire (w, w') et un seul σ_max.

## Darting post hoc contre IS pour corriger les proportions des modes, en dimension croissante (CPU local, aucun nouvel entraînement ; 2026-11-20)

Règles pré-enregistrées : `rules_darting_vs_is.md` (f31f199, trois amendements écrits avant les exécutions qu'ils concernent : 77d0151, 1b6fe90, 8436d13). Scripts `scripts/darting_vs_is_stepA.py`, `scripts/darting_vs_is_stepB.py`. **Temps : étape A 8,0 s, étape B 11,9 s** (budget 1 min chacune). Résultats : `figures/darting_vs_is_stepA.json`, `figures/darting_vs_is_stepB.json`, figure `figures/darting_vs_is_stepA.png` (TV en fonction de d, A1 et A2) ; 1re exécution de l'étape A conservée : `figures/darting_vs_is_stepA_run1_100step_quench.{json,png}`.

### Étape A (K = 8, Δ = 8, δ = 0,1, n = 10⁴ ; une réalisation, graines fixées, sans barre d'erreur)
Plancher TV d'un sampler exact à n = 10⁴ : 0,0092 (A1), 0,0098 (A2). Coûts en évaluations d'énergie : IS(n) 10⁴ ; IS(10n) 10⁵ ; darting S = 1, 2, 5, 10 : 2·10⁴, 3·10⁴, 6·10⁴, 1,1·10⁵.
A1 (tout gaussien ; la carte de translation est exacte) :
| d | IS n : TV, ESS, k̂ | IS 10n : TV, ESS | darting TV (S = 1 / 2 / 5 / 10) | acceptation (S = 10) |
|---|---|---|---|---|
| 32 | 0,0217, 0,403, 0,16 | 0,0061, 0,407 | 0,1159 / 0,0541 / 0,0081 / 0,0059 | 0,582 |
| 64 | 0,0169, 0,245, 0,33 | 0,0046, 0,250 | 0,1232 / 0,0564 / 0,0055 / 0,0116 | 0,582 |
| 128 | 0,0299, 0,085, 0,51 | 0,0060, 0,096 | 0,1295 / 0,0565 / 0,0134 / 0,0098 | 0,585 |
| 256 | 0,0853, 0,026, 0,72 | 0,0077, 0,019 | 0,1258 / 0,0636 / 0,0162 / 0,0110 | 0,583 |
| 512 | 0,1561, 0,0023, 1,00 | 0,1055, 0,0009 | 0,1218 / 0,0537 / 0,0149 / 0,0099 | 0,583 |
A2 (Student pour k pair ; la carte de translation est fausse) :
| d | IS n : TV, ESS, k̂ | IS 10n : TV, ESS | darting TV (S = 1 / 2 / 5 / 10), sans oracle | acceptation (S = 10) | exploratoire : darting S = 10 avec centres ORACLE (TV ; acceptation) |
|---|---|---|---|---|---|
| 32 | 0,0210, 0,439, 0,11 | 0,0030, 0,439 | 0,1950 / 0,1545 / 0,1415 / 0,1405 | 0,319 | 0,1388 ; 0,318 |
| 64 | 0,0161, 0,274, 0,35 | 0,0049, 0,279 | 0,2153 / 0,1632 / 0,1145 / 0,1202 | 0,284 | 0,1154 ; 0,284 |
| 128 | 0,0235, 0,097, 0,51 | 0,0137, 0,124 | non réalisable (4 minima sur 8) | | 0,0914 ; 0,263 |
| 256 | 0,0473, 0,036, 0,67 | 0,0269, 0,016 | non réalisable (4 minima sur 8) | | 0,0746 ; 0,256 |
| 512 | 0,1072, 0,0082, 0,93 | 0,0366, 0,0034 | non réalisable (4 minima sur 8) | | 0,0829 ; 0,256 |
Contrôle δ = 0 (exploratoire) : IS ESS ≈ 0,667 (A1) et 0,672 (A2) à d = 32 et 512, TV_IS 0,007–0,009 ; darting A1 (S = 1/2/5/10) 0,127/0,047/0,0096/0,0119 (d = 32), 0,122/0,054/0,0069/0,0093 (d = 512) ; A2 d = 32 : 0,181/0,128/0,063/0,058.

**Verdicts (règles telles qu'écrites).**
- **R1 (le darting gagne) : VRAIE.** À d = 512 en A1 : ESS_IS(n) = 0,0023 < 5 %, TV_darting(S = 5) = 0,0149 et TV_darting(S = 10) = 0,0099 sont ≤ 0,5 × TV_IS(10⁵) = 0,0528 et ≤ 2 × plancher = 0,0184 (hits (512, 5) et (512, 10) ; variante S = 10 seul : vraie aussi). À d = 256 (ESS 0,026 < 5 %), le darting (0,0110) ne passe pas le seuil 0,5 × 0,0077 (IS à 10⁵ y est encore bon). L'IS à 10⁵ ne devient mauvais qu'à d = 512 (TV 0,1055, ESS = 0,09 % soit environ 90 échantillons effectifs sur 10⁵).
- **R2 (le darting perd) : (a) NON ÉVALUABLE, (b) VRAIE ; R2 non établie telle qu'écrite.** (a) L'acceptation à d = 512 ne peut pas être mesurée : sans oracle, la quench ne retrouve que 4 minima sur 8 à d ≥ 128 (les modes gaussiens), donc le darting ne tourne pas. (b) TV_darting(S = 10) vaut 0,1405 et 0,1202 à d = 32 et 64, soit bien plus que TV_IS(10⁵) = 0,0030 et 0,0049 (et que TV_IS(10⁴) = 0,0210 et 0,0161) : le darting perd déjà à d = 32.
- **R3 (réel) : VRAIE.** Voir l'étape B.

**Pourquoi A2 casse l'étape de découverte des minima (diagnostic, vérifié numériquement).** Le gradient analytique est exact (écart < 1e-8 avec les différences finies). Pour d ≳ 141 (0,226 d > 32), la densité en m_k des queues gaussiennes des modes voisins (≈ e^(−32 − 0,919 d)) dépasse le pic de la composante Student normalisée à Hessienne I (e^(−1,145 d)) : le mélange n'a plus de minimum local au centre des composantes Student, et la descente de gradient converge vers le mode gaussien dominant (238 des 253 échantillons Student convergent vers le mode 1 à d = 128). Les Hessiennes du mélange A2 aux minima ne sont de plus pas égales à I (écart ≈ 3e-3, responsabilités non nulles des composantes voisines à queues lourdes), donc la carte complète avec jacobien √(det H_k/det H_j) a dû être utilisée en A2 (amendement 1).
**Lecture exploratoire (oracle).** Avec les centres connus et la translation, l'acceptation ne s'effondre pas avec d : 0,318, 0,284, 0,263, 0,256, 0,256 (rapport d = 512 / d = 32 = 0,81, au-dessus du seuil 0,5 de R2(a)) ; la TV du darting reste à 0,07–0,14, très au-dessus du plancher (0,0098), et est supérieure à celle de l'IS(10⁵) à tout d (0,0030 à 0,0366). Donc A2 fait perdre le darting par un biais de forme (la chaîne n'équilibre pas la forme intra-mode) et non par une chute d'acceptation. Aucune de ces valeurs n'entre dans un verdict.

### Étape B (graine 0, sigma_max × 0,5, cellules à 8/8 ; 10⁴ trajectoires simulées sur CPU, mêmes 10⁴ points terminaux pour l'IS de chemin et le darting)
**La graine 1 est exclue de la comparaison** : ses contrôleurs n'ont pas été conservés (seuls les log w et les JSON), le darting y est impossible sans réentraînement, exclu. Pour information, l'IS de chemin sauvegardé de la graine 1 (20 000 trajectoires) donne TV 0,009 (ESS 0,629), 0,009 (0,874) et 0,006 (0,626) pour (8,8,8), (32,8,4), (32,8,6) × 0,5.
| cellule | TV AS | IS de chemin : TV, ESS, k̂ | darting S = 1 : TV, acc | S = 5 : TV, acc | S = 10 : TV | plancher (n = 10⁴) | carte |
|---|---|---|---|---|---|---|---|
| (8, 8, 8) | 0,1909 | 0,0096, 0,554, 0,32 | 0,0638, 0,646 | 0,0110, 0,590 | 0,0053 | 0,0095 | translation (H = I à 1e-6) |
| (32, 8, 4) | 0,0383 | 0,0076, 0,879, 0,06 | 0,0206, 0,533 | 0,0096, 0,529 | 0,0097 | 0,0098 | Hessiennes (recouvrement des modes) |
| (32, 8, 6) | 0,0830 | 0,0094, 0,643, 0,27 | 0,0254, 0,548 | 0,0061, 0,553 | 0,0052 | 0,0095 | Hessiennes |
**R3 : VRAIE** : TV_darting(S = 5) = 0,0110, 0,0096, 0,0061 contre 2 × TV_IS = 0,0193, 0,0152, 0,0188 (rapports darting/IS = 1,15 ; 1,26 ; 0,65). Les deux méthodes ramènent la TV au niveau du plancher.
Pour mémoire, (32, 8, 8) × 0,5 (modes absents, exclu de la comparaison) : AS 1/8 modes (TV 0,671) ; l'IS de chemin atteint 5/8 modes (TV 0,197, ESS 0,034) ; sans oracle la quench ne retrouve que 2 minima sur 8 ; le darting entre ces 2 minima donne 2/8 modes couverts (TV 0,528) : il ne peut pas créer les modes absents.

### Déviations et réserves
- L'étape A a été exécutée 3 fois à cause des amendements 1 à 3 (7,5 s, 7,8 s puis 8,0 s ; résultats identiques d'une exécution à l'autre, l'amendement 2 n'ayant eu aucun effet) ; le script B a d'abord déclaré à tort le darting « n/a » pour (32,8,4) et (32,8,6) (oubli de l'appel de la carte générale quand H ≠ I à 1e-6, corrigé sans changement de règle).
- Une seule réalisation par configuration (graines fixées) : pas de barre d'erreur ; les écarts de TV de l'ordre de 0,003 à 0,005 autour du plancher (0,009–0,010) ne sont pas interprétables. δ = 0,1 et Δ = 8 uniques. L'IS utilise les poids exacts π/q (meilleur cas) ; l'IS de chemin réel fait moins bien en général (étape B : même niveau que le darting).
- Le darting corrige les poids des modes mais pas la forme intra-mode ; la métrique est la TV des proportions de cellules de Voronoï.

## Test décisif de la forme 1 : AS entraîné AVEC la variable de contrôle darting (GMM-dk, graine 0, sigma_max × 0,5 ; 2026-11-20)

Règles écrites avant la location : `rules_train_darting.md` (b3a8c18 puis amendement 1, d96d4c4). Accord de l'utilisateur : offre 57456544, plafond dur 0,10 $, 6 runs de 500 époques : (a) AS seul, (b30) AS + darting γ = 30 (bras principal), (b1) AS + darting γ = 1 (secondaire, descriptif) ; aucune autre valeur de γ. Cellules (8,8,8) (sigma_max 2,236) et (32,8,6) (1,25). Départ : contrôleurs de la graine 0 (buffer initial = 10⁴ points terminaux du contrôleur de départ, identique pour les 6 runs), mêmes hyperparamètres et options que l'entraînement d'origine, graine 0, tous les t (t_focus = 0). Matcher `AdjointVEGMMDartingMatcher` (minima découverts sur le buffer, rafraîchis toutes les 25 époques ; 8 minima sur 8 et max |H − I| ≤ 9e-6 dans les 4 runs ; fraction de termes non finis : 0). Résultats : `figures/gmm_dk_train_darting.json`, figure `figures/gmm_dk_train_darting_curves.png`, contrôleurs finaux `figures/gmm_dk_cont_controllers/`, scripts `scripts/gmm_dk_train_darting_report.py`, `scripts/vast_gmm_dk_train_darting_remote.sh`.

**Vérification demandée (point 1), faite dans le code avant le lancement :** à chaque époque, `train_one_epoch` simule 512 NOUVEAUX chemins avec le contrôleur COURANT (`ControlledSDE(ref_sde, controller)`, EMA désactivée) et les ajoute au buffer (1000 // 512 = 1 appel de `populate_buffer`), qui conserve les 10 000 plus récents : même boucle que l'entraînement d'origine ; le matcher darting ne change que la cible de régression.

**Incident déclaré.** La 1re exécution des 4 runs darting a divergé en NaN : à l'époque 126 pour (8,8,8) (b1 et b30 en même temps, même graine) et vers l'époque 301 pour (32,8,6). Cause (vérifiée numériquement) : un tirage t = 0 exactement donne κ = 0, v = 0 et un 0/0 dans le facteur gaussien du terme (probabilité ≈ 10⁻⁷ par tirage, quelques millions de tirages par run). Correctif : v ≥ 10⁻³⁰ et un garde-fou qui remplace tout terme non fini par 0 et le compte (`dart/nonfinite_frac` = 0 dans les runs finaux). Les 4 runs darting ont été relancés DEPUIS LE DÉBUT avec le correctif (les logs des tentatives divergentes sont gardés hors dépôt) ; les 2 runs (a), qui n'ont pas de terme darting, viennent du 1er lancement. Aucune valeur de γ n'a été changée.
**Coût : 0,065 $** (instance 61614278, 29,1 min à 0,1339 $/h, de 17:22:37 à 17:51:44), sous le plafond de 0,10 $ ; la tentative divergente et la relance y sont comprises. Instance détruite dès les résultats rapatriés.

**Tableau (échantillons finaux du modèle, n = 10⁴ ; plancher TV = 0,0098 / 0,0097 ; IS de chemin = 20 000 trajectoires du contrôleur final).** Départ : (8,8,8) TV 0,1909, couverture 8/8, max |log| 0,842, après IS TV 0,008, ESS 0,571, k̂ 0,20 ; (32,8,6) TV 0,0830, 8/8, 0,331, après IS 0,009, ESS 0,635, k̂ 0,26.
| cellule | bras | TV brut | couverture | max \|log(p̂/p)\| | TV après IS | ESS | k̂ |
|---|---|---|---|---|---|---|---|
| (8,8,8) | (a) AS seul | 0,1688 | 8/8 | 0,649 | 0,0087 | 0,585 | 0,23 |
| (8,8,8) | (b30) | 0,1050 | 8/8 | 0,415 | 0,0065 | 0,605 | 0,17 |
| (8,8,8) | (b1) | 0,0591 | 8/8 | 0,299 | 0,0079 | 0,664 | 0,22 |
| (32,8,6) | (a) AS seul | 0,0467 | 8/8 | 0,220 | 0,0091 | 0,672 | 0,10 |
| (32,8,6) | (b30) | 0,1398 | 8/8 | 0,614 | 0,0139 | 0,576 | 0,22 |
| (32,8,6) | (b1) | 0,0341 | 8/8 | 0,161 | 0,0120 | 0,671 | 0,26 |

**Verdicts (règles telles qu'écrites ; verdict principal = b30).**
- **R_train (b30) : FAUX dans les deux cellules.** (8,8,8) : 0,1050 > 0,5 × 0,1688 = 0,0844. (32,8,6) : 0,1398, soit bien au-dessus de 0,5 × 0,0467 = 0,0234, et PIRE que (a) (0,0467) et que le départ (0,0830). La couverture reste 8/8.
- **R_fort (b30) : FAUX** (seuil 3 × plancher ≈ 0,029).
- **R_sûreté (b30) : VRAI dans les deux cellules** (TV après IS 0,0065 et 0,0139 ≤ 2 × plancher ≈ 0,0195 ; ESS(b30)/ESS(a) = 1,03 et 0,86 ≥ 0,5).
- **b1 (descriptif).** R_train : vrai en (8,8,8) (0,0591 ≤ 0,0844), faux en (32,8,6) (0,0341 > 0,0234) donc faux globalement ; R_fort faux ; R_sûreté vrai.

**Dose ‖terme‖/‖cible‖ moyenne par tranche de κ, à la fin** (moyenne des valeurs imprimées aux évaluations des époques 350, 400, 450) : b30 : (8,8,8) 5,78 pour κ < 0,1, 2,16 pour [0,1 ; 0,7), 0,0094 au-delà ; (32,8,6) 8,63, 1,62, 4e-6. b1 : (8,8,8) 0,197, 0,077, 3e-4 ; (32,8,6) 0,28, 0,054, 5e-7. À γ = 30, le terme est donc 2 à 9 fois plus grand que la cible aux κ < 0,7.

**Vitesse (descriptif) : TV brut aux évaluations (époques 0, 50, …, 450).** (8,8,8) : (a) 0,158, 0,164, 0,146, 0,121, 0,147, 0,153, 0,151, 0,139, 0,159, 0,190 ; (b30) 0,209, 0,107, 0,045, 0,059, 0,153, 0,078, 0,062, 0,039, 0,074, 0,086 ; (b1) 0,140, 0,111, 0,062, 0,046, 0,077, 0,053, 0,030, 0,050, 0,055, 0,103. (32,8,6) : (a) 0,061, 0,068, 0,079, 0,032, 0,076, 0,045, 0,044, 0,069, 0,051, 0,072 ; (b30) 0,266, 0,134, 0,149, 0,081, 0,105, 0,139, 0,128, 0,202, 0,090, 0,113 ; (b1) 0,047, 0,051, 0,032, 0,033, 0,067, 0,037, 0,053, 0,062, 0,039, 0,051.
Lecture : b30 **oscille et ne plafonne jamais près du plancher** : minimum 0,039 (époque 350) en (8,8,8), 0,081 (époque 150) en (32,8,6), jamais sous 3 × plancher. Exploratoire (moyenne des 3 dernières évaluations, 350–450) : (8,8,8) a 0,163, b30 0,066, b1 0,069 ; (32,8,6) a 0,064, b30 0,135, b1 0,050. En (8,8,8), AS seul reste autour du départ (0,12–0,19) alors que les deux bras darting descendent à 0,03–0,10 (≈ moitié) ; en (32,8,6), AS seul descend de lui-même de 0,083 à 0,03–0,08 (moyenne 0,0585 sur les évaluations ≥ époque 100, écart-type 0,016), b1 ne se distingue pas de lui (0,047), b30 est environ deux fois plus mauvais (0,126).
Réserves : une graine, 500 époques ; le TV brut d'une évaluation à 10⁴ échantillons varie de 0,012 à 0,036 (écart-type entre évaluations) bien au-dessus du plancher 0,0098 (le modèle n'est pas stationnaire), donc le critère R_train sur les seuls échantillons finaux est bruité ; les agrégats ci-dessus sont exploratoires et ne remplacent pas les verdicts. Écart théorie/pratique (b1) : la quadrature 1D prédisait ρ ≈ 1 (correction presque complète par itération externe), mais en entraînement la dose à γ = 1 est modeste (≈ 20 % de la cible pour κ < 0,1, 5–8 % pour [0,1 ; 0,7)) et l'effet observé est au plus d'une division par deux du TV en (8,8,8) et nul à la limite du bruit en (32,8,6).

## Phase 0 : darting post hoc contre IS sur les benchmarks MoG et MoS de Beyond ELBOs (CPU local, une graine ; 2026-11-20)

Règles écrites avant le script : `rules_mog_mos_phase0.md` (cb52a93). Scripts `scripts/mog_mos_phase0.py`, `scripts/mog_mos_phase0_table.py` ; résultats `figures/mog_mos_phase0_*.json`, `figures/mog_mos_truth_*.json`, moyennes `data/mog_mos_means.npz`. Aucun entraînement, aucune location. **Temps par exécution : 0,6 à 31 s** (le plus long : MoS d = 200, 31 s ; budget 60 s ; aucune troncature du comparateur à coût égal).

### 0.1 Définitions et sources (lues, non devinées)
- **MoG (Beyond ELBOs, `targets/gmm40.py`, ID `gaussian_mixture40`, commit 32de621 du 2024-12-09, jax 0.4.28)** : K = 40, composantes N(μ_k, I), poids 1/K, μ_k = uniform(−1, 1, (K, d), PRNGKey(0)) × 40, d « any » ; l'article évalue d ∈ {2, 50, 200}. À d = 2 : μ₁ = (−15,7582 ; 18,1165), μ₂ = (−34,8089 ; 21,2255) (jeu distinct des moyennes du GMM-40 de FAB, tirées avec torch, que NETS liste : μ₁ = (−0,2995 ; 21,4577)). `targets/gaussian_mixture.py` (K = 10, Wishart) est une autre cible du dépôt, non retenue.
- **MoS (`targets/student_t_mixture.py`)** : K = 10 (code et annexe C.5 ; **la légende de la Figure 3 de l'article dit 15 : incohérence signalée, 10 retenu**), produit de t de Student 1D indépendantes de degrés de liberté 2 et d'échelle 1, centrées en μ_k = uniform(−10, 10, (K, d), PRNGKey(0)), poids 1/K, d ∈ {2, 50, 200}. **Toutes les composantes ont les mêmes degrés de liberté et la même échelle : une translation relie exactement les modes.**
- **NETS : EXCLU.** L'article (Section 5.3) dit seulement « 8-mode target GMMs in d = 36, 64, 128, 200 » sans moyennes, covariances ni poids ; l'annexe définit le GMM-40 de FAB (d = 2), le funnel 10-d et le MoS 50-d de Blessing (10 composantes) ; **aucun dépôt n'a été trouvé** (aucun lien dans l'article, `malbergo` sans dépôt NETS, recherche GitHub vide). Je ne devine pas les paramètres.
Vérité (10⁶ tirages) : proportions de Voronoï de 0,0245 à 0,0254 (MoG, 40 cellules) et 0,093 à 0,105 (MoS d = 2) puis ≈ 0,100 (MoS d = 50, 200) ; Voronoï et argmax de la densité des composantes (mode descriptors de Beyond ELBOs) coïncident à 100 % (MoG), 94,5 % (MoS d = 2) et 99,8–99,9 % (MoS d = 50, 200). Séparation des moyennes : MoG d = 2 min 1,78 (modes qui se touchent), d = 50 min 162 ; MoS d = 2 min 2,36, d = 50 min 48,7 (en unités d'échelle).
Plancher TV (n = 10⁴, sampler exact) : 0,0251 / 0,0251 / 0,0248 (MoG d = 2 / 50 / 200), 0,0113 / 0,0123 / 0,0113 (MoS d = 2 / 50 / 200), 0,0078 / 0,0083 / 0,0080 (MoG « extra » d = 2 / 5 / 10).

### 0.2 Tableau (n = 10⁴ ; TV_darting (S) / TV_IS à coût égal [ESS de cet IS] ; G = règle de la phase 0 pour ce S)
| cible | d | minima trouvés (500 éch.) | IS(n = 10⁴) : TV, ESS, k̂ | S = 1 | S = 5 | S = 10 |
|---|---|---|---|---|---|---|
| MoG | 2 | 32/40 | 0,0266, 0,64, −0,63 | 0,111 / 0,0081 [0,64] | 0,102 / 0,0042 [0,64] | 0,091 / 0,0031 [0,64] |
| MoG | 50 | 30/40 | 0,0483, 0,31, 0,12 | 0,162 / 0,0103 [0,30] | 0,162 / 0,0059 [0,30] | 0,162 / 0,0047 [0,30] |
| MoG | 200 | 30/40 | 0,1078, 0,044, 0,44 | 0,159 / 0,0343 [0,034] | 0,159 / 0,0210 [0,032] | 0,159 / 0,0138 [0,031] |
| **MoS** | 2 | 10/10 | 0,0682, 0,021, 0,82 | 0,028 / 0,048 [0,0024] | 0,020 / 0,027 [0,0055] | **0,0103 / 0,0264 [0,0053] : G vrai** |
| **MoS** | 50 | 10/10 | 0,3138, 0,0015, 1,49 | 0,032 / 0,256 [0,0001] | **0,0074 / 0,647 [0,0000] : G vrai** | **0,0108 / 0,531 [0,0000] : G vrai** |
| **MoS** | 200 | 10/10 | 0,6470, 0,0003, 1,97 | 0,033 / 0,677 [0,0000] | **0,0108 / 0,639 [0,0000] : G vrai** | **0,0137 / 0,629 [0,0000] : G vrai** |
| MoG extra (K = 5, mode rare 10⁻³) | 2, 5, 10 | 4/5, 4/5, 4/5 | 0,053 / 0,007 / 0,011, ESS 0,02–0,03, k̂ ≈ 1 | TV_darting = 0,199 (mode rare jamais découvert) | idem | idem ; IS à coût égal 0,0034 / 0,0016 / 0,0024 |
Coûts (évaluations d'énergie et de gradient, y compris quench, Hessiennes et réglage de MALA) : ≈ 1,9·10⁵ (S = 1), 6,6–6,8·10⁵ (S = 5), 1,26–1,27·10⁶ (S = 10) ; acceptation des sauts : 0,73–0,93 (MoG d = 2, 50, 200 à 500 éch. : 0,85–0,91) à 1,00 (MoS d = 50, 200 ; 0,73–0,84 à d = 2) ; acceptation MALA : 0,41–0,71 (réglage à 0,6 puis dérive quand les chaînes changent de mode).
Variante exploratoire (5000 échantillons de découverte, coût compté) : MoG 39/40, 40/40, 40/40 minima ; darting TV 0,020 / 0,023 / 0,027 à S = 10 (au niveau du plancher 0,025) mais IS à coût égal 0,0030 / 0,0045 / 0,0125 : G reste faux. MoG extra : 5/5 minima seulement à d = 5 (darting 0,0020 contre IS 0,0011 : G faux).

### 0.3 Verdicts G
- **G vrai : MoS d = 2 (S = 10 seulement), MoS d = 50 (S = 5 et 10), MoS d = 200 (S = 5 et 10).** G faux partout ailleurs : MoG à d = 2, 50, 200 (protocole standard et variante à 5000 échantillons), MoG extra à d = 2, 5, 10, et S = 1 pour toutes les cibles.
- **Pourquoi MoS gagne :** les composantes ont la même forme (translation exacte), donc l'acceptation des sauts est de 1,00 et les poids sont corrigés indépendamment de l'erreur de forme ; l'IS exact avec un sampler à queues gaussiennes trop légères a des poids de variance infinie (k̂ de 0,82 à 1,97, ESS ≤ 2 % à d = 2 et ≈ 0 à d ≥ 50, TV_IS entre 0,53 et 0,68 à coût égal).
- **Pourquoi MoG perd :** (i) la quench sur 500 échantillons à poids de rapport 10 ne retrouve que 30 à 32 modes sur 40 (les modes légers ont moins de 5 échantillons) et le darting ne peut pas atteindre un mode non découvert ; (ii) même avec 40/40 minima, la TV du darting est bornée par le plancher d'un sampler à n = 10⁴ (≈ 0,025) alors que l'IS à coût égal dispose de ≈ 3·10⁴ échantillons effectifs à d = 200 (ESS 3 %) et atteint 0,0125 : G exigerait un IS à ESS ≲ 0,2 % au coût de S = 10 (d ≳ 300 pour ce δ = 0,1).
- Cas à mode rare : la découverte sans oracle (500 puis 5000 échantillons) échoue à trouver un mode de poids 10⁻³ (4/5 minima sauf à d = 5 avec 5000) ; le darting ne peut pas redonner de la masse à un mode absent de son ensemble de minima.

### Déviations et réserves (à lire)
- Le coût mesuré inclut le gradient en chaque saut accepté (état mis en cache pour MALA) : un peu au-dessus de la formule n(2 + 11 S) des règles ; le comparateur utilise le coût mesuré.
- « MoG extra » (K = 5, w = 0,2, mode rare 10⁻³) est mon opérationnalisation du cas « w_k ≥ 0,1 », impossible avec les 40 composantes de poids 0,025.
- Le MoS à d = 2 utilise la carte de Hessiennes (recouvrement des queues), non la translation (écart des Hessiennes ≥ 10⁻⁶).
- Comparateur : IS EXACT avec une proposition synthétique ; l'IS de chemin d'AS ne sera pas meilleur ; la phase 1 le mesure sur de vrais contrôleurs. Le sampler synthétique du MoS (gaussiennes de même Hessienne) a des queues trop légères par construction, ce qui est favorable à G ; un vrai AS peut ne pas avoir cette erreur.
- Comptage du coût : seules les évaluations d'énergie et de gradient sont comptées (le coût en passages de réseau de l'IS de chemin, ≈ nfe = 200 par trajectoire, ne l'est pas) : à discuter avant la phase 1.
- Une graine, un seul δ = 0,1.

## Pilote AS sur le MoS d = 50 et variante MoS-w (2026-11-20)

Règles écrites avant tout résultat : `rules_mos_pilot.md` (8771d10). Scripts `scripts/mos_pilot_eval.py`, `scripts/dart_lib.py`, `scripts/vast_mos_pilot_remote.sh` (324c656) ; résultats `figures/mos_pilot_results.json`, `figures/mos_pilot_eval.log`, contrôleurs `figures/mos_pilot_controllers/` (3 fichiers). Offre 57456546 (RTX 3070, 0,1339 $/h), plafond dur 0,05 $.
**Coût : au plus 0,028 $ (12,4 min comptées à 19:18:12).** Le lot a fini à 19:14 (8,2 min après la création) ; la commande de destruction a bloqué sur sa confirmation (appel de ma part dont le délai a expiré) et l'instance n'existait plus à 19:19:16 : l'instant exact de la destruction n'est donc pas relevé, la borne haute est 0,028 $. Aucun incident d'entraînement (perte finie, pas de NaN).

### Pilote (MoS standard d = 50, AS seul, 300 époques, graine 0, 3 valeurs de σ_max : 5,8 ; 11,6 ; 23,2 ; 10⁴ échantillons terminaux ; 20 000 trajectoires pour l'IS de chemin)
| σ_max | couverture | TV brut | IS de chemin : TV, ESS, k̂, log Ẑ (vrai : 0) | darting |
|---|---|---|---|---|
| 5,8 | 1/10 | 0,888 | 0,900, 0,0002, 3,05, −10,6 ± 0,5 | 1 minimum découvert : inapplicable |
| 11,6 | 1/10 | 0,817 | 0,900, 0,0001, 6,45, −48,9 ± 1,0 | 1 minimum découvert : inapplicable |
| **23,2** | **10/10** | **0,267** | 0,786, 0,0001, **7,43**, −70,6 ± 0,7 | 10 minima ; voir ci-dessous |
Rapports p̂/p à σ_max = 23,2 : 0,42 0,93 0,40 0,92 2,63 0,41 1,31 0,40 1,73 0,84.
**P1 : VRAI** (un seul σ_max, le plus grand de la grille, atteint 10/10 ; la grille s'arrête là : rien n'est connu au-delà de 23,2). Entraînement non stationnaire : couverture aux évaluations (époques 0, 50, …, 250) pour σ_max = 23,2 : 10, 3, 10, 10, 10, 10 et TV brut 0,248 / 0,809 / 0,091 / 0,072 / 0,069 / 0,150, puis 0,267 à la fin (les poids se dégradent entre les époques 200 et 300) ; pour 5,8 et 11,6, la couverture chute de 8 ou 10 modes à 2 modes au cours de l'entraînement.
**Contrôleur retenu : σ_max = 23,2** (seul à 10/10 ; aucun 10/10 avec k̂ < 0,7 : règle « plus grand ESS », signalée).
**P2 (question clé) : k̂ = 7,43 à σ_max = 23,2 (> 0,7).** Le vrai AS a des queues beaucoup trop légères pour la cible Student : ESS = 10⁻⁴, TV après IS de chemin 0,786 (plus mauvais que le TV brut 0,269 ; TV non pondéré des mêmes 20 000 points 0,269), log Ẑ = −70,6 ± 0,7 (le log Z exact vaut 0). Confirme (et dépasse) le sampler synthétique de la phase 0.

### Darting post hoc sur le contrôleur retenu (descriptif ; sur les 10⁴ premiers points terminaux ; pas de MALA réglé une fois ; carte de translation)
| S | TV darting | acceptation des sauts / MALA | coût (E + ∇) | IS de chemin à coût égal : n_IS, TV, ESS, k̂ | G |
|---|---|---|---|---|---|
| 1 | 0,105 | 0,46 / 0,59 | 1,07·10⁶ | 1,07·10⁶, 0,698, 0,0000, 5,79 | faux |
| 5 | 0,080 | 0,45 / 0,42 | 1,53·10⁶ | 1,53·10⁶, 0,752, 0,0000, 5,61 | faux |
| 10 | 0,072 | 0,47 / 0,35 | 2,10·10⁶ | 2,10·10⁶, 0,760, 0,0000, 5,59 | faux |
Le darting fait environ 10 fois mieux que l'IS de chemin à coût égal (TV 0,07–0,10 contre 0,70–0,76) et divise par 3,7 le TV brut (0,267 → 0,072), mais **G est faux car TV_darting reste très au-dessus de 2 × plancher = 0,0226** (le plancher à n = 10⁴ vaut 0,0113). L'acceptation des sauts vaut 0,46 (1,00 sur le sampler synthétique : la forme du vrai AS n'est pas celle d'une Student) ; le TV baisse encore de S = 5 à S = 10 (0,080 → 0,072). Le coût inclut 1,0·10⁶ évaluations de gradient pour la quench (500 échantillons, 4000 pas maximum, la tolérance de convergence n'ayant pas été atteinte).
Coût secondaire : l'IS de chemin à coût égal utilise 2,1·10⁸ à 4,2·10⁸ passages de réseau (200 par trajectoire) alors que le darting n'en fait aucun ; temps réel : IS de chemin 62 s sur GPU pour 2,1·10⁶ trajectoires, darting 21 s de CPU de l'instance (3 balayages cumulés et quench), IS de 20 000 trajectoires 0,5 s.

### MoS-w (poids inégaux log w ~ N(0,1), graine 2026 ; q à poids uniformes, queues gaussiennes ; phase 0.2 refaite sur CPU)
Poids de la cible de 0,012 à 0,33. d = 50 : IS(n) TV 0,320, ESS 0,0006, k̂ 1,26 ; darting TV 0,162 (S = 1), 0,029 (S = 5), **0,0070 (S = 10)** contre IS à coût égal 0,195 / 0,331 / 0,365 ; acceptation des sauts 0,74 / 0,63 / 0,59. d = 200 : IS(n) TV 0,481, ESS 0,0005, k̂ 1,94 ; darting 0,154 / 0,034 / **0,0105** contre IS 0,589 / 0,508 / 0,446 ; acceptation 0,73 / 0,62 / 0,59. Plancher 0,0108 et 0,0106. **G vrai aux deux dimensions à S = 10 seulement** (S = 5 : 0,029 et 0,034 > 2 × plancher). Acceptation MALA 0,59 à 0,71. Le test n'est plus trivial (acceptation < 1) et le darting le passe encore, mais il lui faut 10 balayages pour converger. Figures/fichiers : `figures/mog_mos_phase0_mosw_d{50,200}_disc500.json`.

### Réserves
Une graine, 300 époques (recette d'origine : 1500), grille de σ_max s'arrêtant au bord là où le seul 10/10 apparaît ; couverture mesurée sur 10⁴ échantillons ; le pilote ne vise pas à choisir σ_max au-delà de 23,2. Le TV brut final (0,267) est bien pire que celui observé aux époques 100 à 200 (≈ 0,07) : un entraînement plus court ou un arrêt anticipé aurait donné un meilleur sampler brut.

## Étape 1 CPU sur le contrôleur MoS d = 50, σ_max = 23,2 : sauts du darting, témoin MALA seul, IS robustes (2026-11-20)

Règles écrites avant tout calcul : `rules_mos_step1.md` (257767a). Scripts `scripts/mos_step1.py`, `scripts/dart_lib.py` ; résultats `figures/mos_step1_{a_n10000,a_n5000,b_n10000,c}.json`. **Temps par exécution :** (a) 17,3 s (n = 10⁴) et 11,7 s (essai à n = 5000, plancher 0,0166, résultats concordants) ; (b) 11,2 s ; (c) 7,1 s (2·10⁴ trajectoires) puis 30,8 s (10⁵ trajectoires). Aucune location. **Correction du rapport du pilote :** le plancher du MoS d = 50 à n = 10⁴ est 0,0123 (pas 0,0113) ; 2 × plancher = 0,0246 ; aucun verdict ne change. Échantillons de (a) et (b) : les 10⁴ échantillons terminaux du contrôleur (le TV brut est 0,2674).

**(a) Darting (n = 10⁴, translation, 10 minima découverts, pas de MALA 0,781 réglé à 0,6) :**
| S | TV | acceptation des sauts | acceptation MALA | coût (E + ∇) | couverture |
|---|---|---|---|---|---|
| 10 | 0,0740 | 0,466 | 0,384 | 2,03·10⁶ | 10/10 |
| 20 | 0,0687 | 0,502 | 0,325 | 3,18·10⁶ | 10/10 |
| 40 | 0,0523 | 0,553 | 0,277 | 5,50·10⁶ | 10/10 |
(Essai à n = 5000 : 0,0732 / 0,0543 / 0,0508, plancher 0,0166.) Le TV baisse avec S mais reste de 4 fois le plancher à S = 40 ; l'acceptation des sauts monte de 0,47 à 0,55 quand les échantillons se rapprochent de la cible. Le coût inclut 8,8·10⁵ évaluations avant les balayages (quench de 500 échantillons qui n'atteint pas la tolérance, Hessiennes, réglage de MALA, état initial).

**(b) Témoin MALA seul (mêmes échantillons, même pas 0,781, mêmes coûts totaux) :** TV 0,2679 (100 pas, coût 2,03·10⁶), 0,2645 (157 pas), **0,2614 (273 pas, coût 5,50·10⁶)**, acceptation 0,33 → 0,26 ; couverture 10/10. Le TV brut est 0,2674 : MALA seul ne corrige PRESQUE RIEN des masses.
**H_saut : VRAIE** (TV_darting(S = 40) = 0,0523 ≤ 0,5 × 0,2614 = 0,1307, soit 5 fois mieux) : la correction des masses vient des sauts.

**(c) IS de chemin et variantes robustes (trajectoires resimulées sur CPU : les log w du pilote n'avaient pas été sauvegardés ; autres graines) :**
| n trajectoires | IS simple : TV, ESS, k̂ | PSIS : TV, ESS, k̂ | IS tronqué (√n × moyenne) : TV, ESS |
|---|---|---|---|
| 20 000 | 0,8997, 5·10⁻⁵, 6,79 | 0,8758, 1,1·10⁻⁴, 6,79 | 0,8990, 1,0·10⁻⁴ |
| 100 000 | 0,8990, 1·10⁻⁵, 6,34 | 0,8697, 3·10⁻⁵, 6,34 | 0,8166, 3·10⁻⁵ |
log Ẑ = −67,1 (20 000) et −61,8 (10⁵), vrai : 0 ; TV non pondéré des mêmes points : 0,264 à 0,266. L'IS simple à 25 000 et 50 000 trajectoires : 0,900 et 0,796. Aucune variante ne corrige : k̂ reste à 6,3–6,8 (PSIS n'est pas fiable pour k̂ > 1), le TV pondéré est le plus souvent PIRE que le TV non pondéré (0,87–0,90 contre 0,26), et l'IS tronqué ne descend qu'à 0,82 à n = 10⁵. Meilleure variante de l'IS à n = 10⁵ : 0,8166 (« comparateur tronqué » : l'IS à coût égal du darting, 2·10⁶ à 5,5·10⁶ trajectoires, n'est pas atteignable sur CPU ; le comparateur GPU du pilote à 1,07·10⁶ à 2,10·10⁶ trajectoires donnait 0,698 à 0,760).

**G (règle inchangée) : FAUX pour S = 10, 20 et 40, contre l'IS simple de coût égal ET contre la meilleure variante.** La première condition est remplie avec une marge large (TV_darting ≤ 0,5 × TV_IS : 0,074 / 0,069 / 0,052 contre un seuil de 0,41 à 0,38), mais la seconde ne l'est jamais (TV_darting ≤ 2 × plancher = 0,0246 : 0,0523 à S = 40, soit 2,1 fois le seuil).

**Réserves :** une graine, un seul contrôleur ; log w resimulés (pas ceux du pilote) ; PSIS = approximation de l'ajustement de Pareto par `scipy.stats.genpareto` ; comparateur robuste évalué à n = 10⁵ seulement ; le TV du darting à S = 40 (0,052) n'a pas encore atteint un plateau.

### Étape 2 (proposition seulement, rien n'est loué ; code prêt et testé en local)
Pré-requis ajoutés : `save_controller_epochs` dans `train.py` (instantanés `controller_ep100.pt`, `ep200`, `ep300`), `configs/problem/mosw.yaml` (MoS-w d = 50, poids 0,037 / 0,104 / 0,012 / 0,331 / 0,155 / 0,061 / 0,060 / 0,111 / 0,063 / 0,065), `configs/experiment/mosw_as.yaml`.

## Étape 2 (MoS-w d = 50, AS) et diagnostic du darting sur le contrôleur MoS de l'étape 1 (2026-11-21)

Règles écrites avant la location et avant tout résultat : `rules_mos_step1.md` amendement 1 (08e1b43). Scripts `scripts/mosw_step2_eval.py`, `scripts/vast_mosw_step2_remote.sh`, `scripts/mos_diag.py` ; résultats `figures/mosw_step2_results.json`, `figures/mosw_step2_eval.log`, log w + cellules `figures/mosw_step2_logw/is_logw_ep{100,200,300}.npz`, contrôleurs `figures/mosw_step2_controllers/mosw_ep{100,200,300}.pt`, diagnostic `figures/mos_diag_*.json`.
**Coût de l'étape 2 : 0,029 $** (instance 61664746, 13,0 min à 0,1339 $/h, de 23:57:01 à ≈ 00:10), sous le plafond dur de 0,06 $ ; arrêt d'urgence (18 min) non déclenché ; instance détruite dès le rapatriement.

### Étape 2 : MoS-w d = 50 (poids 0,037 / 0,104 / 0,012 / 0,331 / 0,155 / 0,061 / 0,060 / 0,111 / 0,063 / 0,065), AS, σ_max = 23,2, 300 époques, graine 0 ; plancher TV (n = 10⁴) 0,0108
Couverture 10/10 aux trois époques (à l'époque 50 d'entraînement : 3/10 ; TV brut aux évaluations 0,237 / 0,717 / 0,226 / 0,258 / 0,183 / 0,247). Pour chaque contrôleur : 20 000 trajectoires (GPU) ; darting S ∈ {10, 40} sur les 10⁴ premiers points terminaux ; MALA seul au coût total de chaque S.
| époque | TV brut | IS simple : TV, ESS, k̂ | PSIS : TV | IS tronqué : TV | darting S = 10 : TV (acc. sauts) | darting S = 40 : TV (acc. sauts) | MALA seul (coût de S = 40) | G | H_saut |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 0,234 | 0,727, 1,4·10⁻⁴, 6,44 | 0,682 | 0,527 | 0,200 (0,14) | 0,190 (0,13) | 0,235 | faux | FAUX (0,190 > 0,117) |
| 200 | 0,208 | 0,845, 5·10⁻⁵, 1,26 | 0,467 | 0,842 | 0,119 (0,23) | 0,089 (0,27) | 0,211 | faux | vrai (0,089 ≤ 0,106) |
| 300 | 0,291 | 0,826, 5·10⁻⁵, 8,08 | 0,511 | 0,480 | 0,144 (0,25) | 0,121 (0,28) | 0,307 | faux | vrai (0,121 ≤ 0,154) |
Rapports p̂/p de l'époque 300 : 2,06 0,80 4,69 0,50 1,87 0,57 2,16 0,61 0,82 0,66 (le mode de poids 0,012 est sur-représenté d'un facteur 4,7). Darting : 10 minima aux époques 100 et 200, **8 seulement à l'époque 300** (deux modes non découverts) ; pas de MALA 0,996 / 0,781 / 0,781 ; coût 1,9–2,0·10⁶ (S = 10) et 5,1–5,4·10⁶ (S = 40) ; CPU de l'instance 41–45 s par contrôleur. IS de chemin à coût égal (époque 300, GPU, IS simple) : **2,02·10⁶ trajectoires : TV 0,746, k̂ 6,05 ; 4,40·10⁶ sur 5,41·10⁶ demandées (tronqué) : TV 0,841**, 116 s ; passages de réseau 4,0·10⁸ (S = 10) et 1,1·10⁹ (S = 40) contre aucun pour le darting. log Ẑ de l'IS de chemin : −97,0 / −52,7 / −78,5 (vrai : 0).
**Verdict :** G faux aux trois époques, pour S = 10 et 40, contre l'IS simple de coût égal et contre la meilleure variante (TV_darting entre 0,089 et 0,200, très au-dessus de 2 × plancher = 0,0216). Le darting divise le TV brut par 1,2 à 2,3 (0,234 → 0,190, 0,208 → 0,089, 0,291 → 0,121) alors qu'il atteignait le plancher sur le sampler synthétique MoS-w (phase 0, G vrai à S = 10) ; l'acceptation des sauts n'est que de 0,13 à 0,28 (0,59–0,74 avec le sampler synthétique). PSIS fait mieux que l'IS simple aux époques 200 et 300 (0,47 et 0,51 contre 0,84 et 0,83) mais reste à 20 fois le plancher ; PSIS n'est pas fiable pour k̂ > 1. MALA seul ne corrige rien à l'époque 100 et 300 (TV 0,235 et 0,307 = TV brut) ; H_saut est faux à l'époque 100 (le darting y n'améliore presque rien : 0,234 → 0,190).

### Diagnostic CPU (descriptif) sur le contrôleur MoS d = 50, σ_max = 23,2 de l'étape 1
1. **Sauts refusés** (10⁵ propositions sur 10 balayages ; 10 minima) : acceptés 46,6 % ; **invalides (l'image tombe hors de la cellule j) 13,1 %** ; **valides mais refusés par MH 40,3 %**. Log π(y)/π(x) des refusés par MH, quantiles 1 / 10 / 25 / 50 / 75 / 90 / 99 % : −88,6 / −68,8 / −56,3 / **−41,9** / −27,0 / −13,9 / −1,7 (attendu ≈ 0 au cœur des modes : FAUX, les refusés sont en queue) ; des acceptés : médiane 0,0, quantiles 1–99 % de −0,11 à 5,6. L'acceptation par balayage monte de 0,46 à 0,50.
2. **Distance de x à son minimum** : médiane **108,8 pour les échantillons d'AS contre 14,7 pour des échantillons exacts de π**. Les déciles de la distance exacte étant [0 ; 10,4 ; … ; 26,6 ; ∞), **100 % des échantillons d'AS tombent dans le dernier décile** (> 26,6, qui ne contient que 10 % de π). Acceptation des sauts depuis x (premier saut, sans MALA) : exacts 0,997 au total (1,000 dans les 9 premiers déciles, 0,971 dans le dernier), AS 0,456 ; invalides : AS 0,144, exacts 0,002.
3. **Décomposition du TV du darting (S = 40)** : vérité calculée de la même façon (10⁴ échantillons exacts, quench de 150 pas, 92,5 % convergés) : TV entre proportions de Voronoï et par bassin **0,0002** (débordement de queues sous π : 0,02 % des échantillons). Darting S = 40 : avec 150 pas de quench la convergence n'est atteinte que pour 2,4 % des échantillons (ils sont très loin des minima) : assignation par bassin non fiable ; variante exploratoire à 800 pas (n = 1000, convergés 67 %) : **28,8 % des échantillons ont une cellule de Voronoï différente de leur bassin** (0,2 % sous π) ; TV de Voronoï 0,048, TV par bassin 0,044 (contre la vérité par bassin), TV entre les deux jeux de proportions du darting 0,030. (i) L'erreur de masse par bassin (≈ 0,044) explique l'essentiel du TV ; (ii) le débordement de queues d'une cellule à l'autre est présent (29 % des échantillons) mais ses effets se compensent en grande partie (0,030). Le bruit d'échantillonnage à n = 1000 est de l'ordre de 0,04 : décomposition indicative.
4. **Stationnarité** : darting complet (S = 10 et 40) démarré depuis 10⁴ échantillons exacts de π : TV 0,0144 et 0,0147 (TV brut 0,0126, plancher 0,0123), acceptation 0,997–0,998 : le noyau préserve π (au plancher), comme attendu.

**Lecture d'ensemble (à confirmer, non établie par une règle) :** le noyau est correct (stationnarité), mais les échantillons d'AS sont 7 fois plus loin de leurs modes que ceux de π (distance 109 contre 15) : ce ne sont pas de « bonnes formes aux mauvais poids » ; ils sont dans la queue lointaine, où les sauts sont souvent invalides (13 %) ou refusés (40 %, log-rapport médian −42), ce qui limite la correction (TV 0,052 à S = 40 sur le MoS, 0,09–0,19 sur le MoS-w). La mesure de masse par cellules de Voronoï sur de tels échantillons est elle-même peu informative sur les modes.
Réserves : une graine, 300 époques, un seul contrôleur par époque ; les diagnostics 1 à 4 portent sur le seul contrôleur MoS de l'étape 1 ; la décomposition 3 est bruitée (n = 1000 à 3000) ; comparateur à coût égal tronqué pour S = 40 (4,4·10⁶ sur 5,4·10⁶ trajectoires).

**Note de lecture (à retenir pour les étapes 1 et 2) :** à σ_max = 23,2 et 300 époques, AS n'a **pas appris** le MoS : distance médiane au minimum 109 contre 15 sous π. Les comparaisons darting / IS des étapes 1 et 2 portent donc sur un sampler raté ; elles restent dans ce fichier mais **ne testent pas le régime « bonnes formes, mauvais poids »**, qui est celui où le darting est censé aider. L'étape 3 (`rules_mosw_step3.md`) mesure d'abord si un contrôleur a appris (critère Q) avant toute comparaison.

## Étape 3 : AS MoS-w à la recette d'origine (1500 époques, d = 50 et d = 10), critère de qualité Q (2026-11-21)

Règles écrites et commitées avant la location et avant tout résultat : `rules_mosw_step3.md` (a793a82). Scripts `scripts/mosw_step3_eval.py`, `scripts/basin_lib.py`, `scripts/vast_mosw_step3_remote.sh` ; résultats `figures/mosw_step3/step3_d{50,10}.json`, journaux `figures/mosw_step3_eval_d{50,10}.log`, contrôleurs `figures/mosw_step3/controllers/` (6 par dimension), log w + cellules `figures/mosw_step3/logw/` (20 000 trajectoires par contrôleur ; points terminaux gardés hors dépôt dans `results/mosw_step3/`).
**Coût : 0,048 $** (instance 61755466, offre RTX 3070 57456544 à 0,1339 $/h, 21,6 min de 13:02:42 à 13:24:17), sous le plafond dur de 0,10 $ (estimation avant location : ≈ 0,078 $). Arrêts de la minute 36 / 39 non déclenchés. Instance détruite dès le rapatriement (0 instance restante).
Incident : la première version des deux processus d'évaluation a planté à la première lecture (`sample_exact`, poids float32 ne sommant pas exactement à 1 pour `numpy.random.choice`) ; les entraînements n'étaient pas touchés ; correctif de normalisation (426a134, définition de Q inchangée), évaluations relancées avant la fin de l'époque 250 : aucun résultat perdu.

### Entraînement
MoS-w, σ_max = 23,2 aux deux dimensions (justification dans les règles : rapport σ_max / écart-type des moyennes par coordonnée = 4, indépendant de d), 1500 époques, graine 0, perte finie partout, aucun NaN. Références mesurées avec les mêmes minima et la même quench sur 10⁴ échantillons exacts de π : médiane de la distance au minimum le plus proche **14,66 (d = 50)** et **5,23 (d = 10)** ; TV par bassin exact contre la vérité 0,0107 (plancher 0,0108) et 0,0111 (plancher 0,0120) ; quench convergée à 100 % ; seuils de Q1 : 22,0 et 7,8.

### Critère Q sur chaque contrôleur (10⁴ points terminaux ; quench convergée à 100 % partout, donc la clause « ≥ 95 % » de Q2 est toujours satisfaite)
| d | époque | médiane distance AS (rapport à π) | Q1 | bassins atteints (≥ 1 échantillon) / couverts (p̂ ≥ 0,25 p) | Q2 | **Q** | TV brut Voronoï / bassins |
|---|---|---|---|---|---|---|---|
| 50 | 250 | 116,5 (7,95) | faux | 10 / 10 | vrai | **faux** | 0,258 / 0,301 |
| 50 | 500 | 147,7 (10,07) | faux | 10 / 9 | faux | **faux** | 0,321 / 0,522 |
| 50 | 750 | 137,6 (9,38) | faux | 10 / 6 | faux | **faux** | 0,436 / 0,605 |
| 50 | 1000 | 148,4 (10,12) | faux | 10 / 6 | faux | **faux** | 0,419 / 0,695 |
| 50 | 1250 | 148,1 (10,10) | faux | 10 / 2 | faux | **faux** | 0,439 / 0,785 |
| 50 | 1500 | 149,8 (10,22) | faux | 10 / 1 | faux | **faux** | 0,457 / 0,817 |
| 10 | 250 | 16,8 (3,21) | faux | 10 / 7 | faux | **faux** | 0,167 / 0,240 |
| 10 | 500 | 12,6 (2,42) | faux | 10 / 5 | faux | **faux** | 0,342 / 0,388 |
| 10 | 750 | 10,3 (1,96) | faux | 10 / 4 | faux | **faux** | 0,321 / 0,348 |
| 10 | 1000 | 8,68 (1,66) | faux | 10 / 5 | faux | **faux** | 0,287 / 0,310 |
| 10 | 1250 | 7,79 (**1,49**) | **vrai** | 10 / 4 | faux | **faux** | 0,378 / 0,390 |
| 10 | 1500 | 7,77 (**1,49**) | **vrai** | 9 / 5 | faux | **faux** | 0,278 / 0,301 |
**Verdict : aucun des 12 contrôleurs ne passe Q ; aucune comparaison darting / IS n'a été calculée (règle : « si aucun ne passe : rapport et arrêt »).**

### Faits (descriptifs)
- **d = 50 : AS n'apprend pas le MoS-w à σ_max = 23,2, à aucun moment.** La distance médiane au minimum reste de 8 à 10 fois celle de π de l'époque 250 à 1500 (116 → 150), et la masse se concentre avec l'entraînement sur le bassin du plus grand poids (poids 0,331 : p̂/p = 2,0 / 4,3 / 4,9 / 5,5 / 6,1 / 6,3 aux époques 250 à 1500), les autres bassins tombant à 0,01–0,09 × leur poids à l'époque 1500. Le TV brut par bassin passe de 0,30 à 0,82 ; par cellules de Voronoï de 0,26 à 0,46. Plus d'entraînement dégrade les masses (pas de convergence vers la cible).
- **d = 10 : les formes sont apprises (rapport 3,2 → 1,49 de l'époque 250 à 1250, Q1 vrai aux époques 1250 et 1500, de justesse : 1,49 contre un seuil de 1,5), mais pas les poids** : seuls 4 ou 5 bassins sur 10 ont p̂ ≥ 0,25 p. À l'époque 1500, p̂/p : 0,02 0,38 0,01 1,13 1,08 2,91 0,03 2,18 0,03 0,00 (modes de poids 0,037 / 0,012 / 0,060 / 0,063 / 0,065 quasi absents, le mode de poids 0,061 sur-représenté d'un facteur 2,9). Neuf bassins ont au moins un échantillon, le dixième aucun. C'est le régime « bonnes formes, mauvais poids » (au bord de Q1), mais Q2 y est faux : le critère pré-enregistré exclut donc ces contrôleurs de la comparaison.
- Les évaluations d'entraînement (5000 échantillons, tous les 50 époques) donnent la même tendance : d = 50 couverture de modes 10 puis 9/10 avec TV brut 0,24–0,44 ; d = 10 couverture 10 → 5/10, TV brut 0,24–0,43.

### Réserves
Une graine ; une seule valeur de σ_max par dimension (23,2, sans grille, comme demandé : rien ne dit qu'une autre valeur ne ferait pas passer Q, en particulier à d = 10 où 23,2 est peut-être trop grand : non testé) ; Q2 utilise la couverture p̂ ≥ 0,25 p (définition du projet) et rejette des contrôleurs qui ont ≥ 1 échantillon dans 9 ou 10 bassins ; Q1 à d = 10 est franchi avec une marge de 0,01 ; la quench convergée utilise un préconditionnement (direction P ∇ log π, descente monotone), qui envoie un point loin dans la queue vers le minimum de la composante la plus responsable (accord avec Voronoï 99,9 % sous π, 98,8 % à d = 10).

## Étape 4 : GMM-dk en grande dimension avec un vrai AS, (d, K, Δ) = (128, 8, 6) et (256, 8, 6) (2026-11-21)

Règles écrites et commitées avant la location et avant tout résultat : `rules_gmmdk_step4.md` (2527419). Scripts `scripts/gmmdk_step4_eval.py` (sous-commandes `q` et `cmp`), `scripts/basin_lib.py`, `scripts/vast_gmmdk_step4_remote.sh` ; résultats `figures/gmmdk_step4/{q_*.json, cmp_d128.json, cmp_d256.json}`, journaux, contrôleurs (24, `controllers/`), log w et cellules (24, `logw/`) ; points terminaux gardés hors dépôt (non rapatriés).
**Coût : 0,047 $** (instance 61761965, offre RTX 3070 57456544 à 0,1339 $/h, 20,9 min de 14:04:11 à 14:25:05 ; estimation avant location 0,05 $ central, 0,08 $ pessimiste ; plafond dur 0,15 $). Arrêts des minutes 50 / 58 non déclenchés ; instance détruite dès le rapatriement (0 restante). Aucun incident ; perte finie, aucun NaN.
Entraînement : recette d'origine des GMM-dk, 1500 époques, graine 0, σ_max = m × 2√(1 + Δ²/(2d)), m ∈ {0,5 ; 0,75} (d = 128 : 1,068 et 1,602 ; d = 256 : 1,0346 et 1,5518), 4 runs en parallèle (≈ 0,7 s par époque). Références (10⁴ échantillons exacts, même quench) : médiane de la distance au minimum le plus proche 11,29 (d = 128) et 15,98 (d = 256) ; plancher TV (n = 10⁴) 0,0100 ; quench convergée à 100 % partout.

### Filtre Q sur les 24 contrôleurs (quench convergée à 100 % : la clause « ≥ 95 % » de Q2 est toujours satisfaite)
| run | Q1 (rapport de la distance médiane à π) | bassins couverts (p̂ ≥ 0,25 p) aux époques 250 / 500 / 750 / 1000 / 1250 / 1500 | Q | k̂, ESS de l'IS de chemin (contrôleurs qui passent Q) |
|---|---|---|---|---|
| d = 128, × 0,5 | vrai (1,03–1,04) | 6 / 7 / 7 / 7 / **8 / 8** | passe à 1250 et 1500 | 1250 : 0,16, 0,240 ; 1500 : 0,22, 0,193 |
| d = 128, × 0,75 | vrai (1,04–1,07) | 8 / 8 / 8 / 8 / 8 / 8 | passe aux 6 | k̂ 0,97–1,27, ESS 2·10⁻⁴ à 2·10⁻³ |
| d = 256, × 0,5 | vrai (1,04) | 7 / 7 / **8 / 8 / 8 / 8** | passe de 750 à 1500 | k̂ 0,19–0,34, ESS 0,144–0,195 (1500 : 0,34, 0,162) |
| d = 256, × 0,75 | vrai (1,30–1,31) | 8 / 8 / 8 / 8 / 8 / 8 | passe aux 6 | k̂ 3,3–3,8, ESS 1–2·10⁻⁴ |
TV brut par cellules de Voronoï / par bassins à l'époque 1500 : d = 128 : 0,408 / 0,413 (× 0,5), 0,200 / 0,121 (× 0,75) ; d = 256 : 0,304 / 0,308 (× 0,5), 0,184 / 0,099 (× 0,75).

### Choix de σ_max (règle habituelle, appliquée aux candidats = dernier contrôleur sauvegardé qui passe Q, ici l'époque 1500 des deux runs de chaque dimension)
d = 128 : seul × 0,5 a k̂ < 0,7 (0,22 contre 0,97 pour × 0,75) → **× 0,5** (époque 1500). d = 256 : seul × 0,5 a k̂ < 0,7 (0,34 contre 3,28) → **× 0,5** (époque 1500). Comparaison sur ces deux contrôleurs seulement.

### Comparaison sur le candidat choisi (20 000 trajectoires pour l'IS ; darting sur les 10⁴ premiers points ; plancher 0,0100)
| | d = 128, × 0,5, ép. 1500 | d = 256, × 0,5, ép. 1500 |
|---|---|---|
| TV brut (Voronoï / bassins) | 0,408 / 0,413 | 0,304 / 0,308 |
| IS simple : TV (bassins), ESS, k̂, log Ẑ (vrai 0) | 0,0100 (0,0112), **0,193**, 0,22, −0,005 | 0,0204 (0,0200), **0,162**, 0,34, −0,030 |
| PSIS / tronqué : TV | 0,0112 / 0,0100 | 0,0212 / 0,0204 |
| darting S = 1 : TV (bassins), acc. sauts | 0,217 (0,220), 0,42 | 0,157 (0,161), 0,47 |
| S = 5 | 0,0281 (0,0290), 0,50 | 0,0147 (0,0159), 0,52 |
| S = 10 | 0,0098 (0,0078), 0,52 | 0,0080 (0,0086), 0,53 |
| MALA seul, coût de S = 1 / 5 / 10 | 0,399 / 0,377 / 0,346 | 0,297 / 0,282 / 0,263 |
| IS de chemin à coût égal (n = 1,83·10⁵ / 6,44·10⁵ / 1,22·10⁶) : TV | 0,0059 / 0,0019 / **0,0012** | 0,0119 / 0,0047 / **0,0046** |
| passages de réseau de l'IS à S = 10 | 2,4·10⁸ | 2,4·10⁸ |
Rapports p̂/p de l'IS simple : d = 128 : 1,02 1,00 0,99 0,96 1,05 0,98 0,96 0,95 ; d = 256 : 1,04 1,00 0,93 1,05 0,91 0,99 0,99 0,99. Découverte : 8 minima sur 8 aux deux dimensions ; CPU du darting 5 s et 11 s ; IS à coût égal 68 s et 71 s sur GPU (non tronqué).

### Règles
- **Q : VRAI aux deux dimensions** (candidat choisi : Q1 1,03 et 1,04, 8/8 bassins).
- **I (ESS < 5 %) : FAUX aux deux dimensions** (ESS 0,193 et 0,162).
- **G : FAUX aux deux dimensions** (pour S = 1, 5, 10, l'IS de coût égal est meilleur que le darting : à S = 10, 0,0012 contre 0,0098 à d = 128 et 0,0046 contre 0,0080 à d = 256). Rapporté à côté, sans changer le verdict : contre la meilleure variante à 20 000 trajectoires, G serait vrai à d = 256 pour S = 10 (0,0080 ≤ 0,5 × 0,0204 = 0,0102), faux à d = 128 ; l'IS à coût égal (1,2·10⁶ trajectoires) fait mieux que ce comparateur, d'où G faux.
- **H_saut : VRAI aux deux dimensions** (par cellules et par bassins) : le MALA seul ne corrige rien (0,35 et 0,26), les sauts font la correction, mais l'IS de chemin fait mieux au même coût.
**Décision écrite à l'avance : « Q passe et ESS ≥ 5 % à d = 256 » → l'IS tient ; la piste darting est fermée.** (« Q vrai, I vrai, G vrai » n'est réalisé à aucune dimension.)

### Faits à côté de la décision (descriptifs ; non comparés par la règle)
- Les runs × 0,75 passent aussi Q (couverture 8/8, rapport 1,05 et 1,30) mais ont un ESS de l'IS de 10⁻⁴ à 2·10⁻³ et un k̂ de 1,0 à 3,8 : le cas « Q vrai, I vrai » existe donc dans les données aux deux dimensions. La règle de sélection de σ_max (k̂ < 0,7) les a écartés au profit du × 0,5, dont l'IS tient ; **le darting n'a pas été calculé sur ces contrôleurs** (contrôleurs et log w sont sauvegardés). Leur TV brut est plus bas que celui du × 0,5 (0,20 / 0,12 à d = 128, 0,18 / 0,10 à d = 256 par Voronoï / bassins, contre 0,41 et 0,30), mais l'IS de chemin y est inutilisable : c'est le profil « sampler plus large que la cible » déjà vu au MoS (queues trop légères pour l'IS, rapport de distance plus élevé à d = 256 : 1,30).
- Le TV brut du × 0,5 (0,30–0,41) est élevé alors que l'IS le ramène au plancher : AS apprend les formes et couvre les 8 bassins mais pas les poids (« bonnes formes, mauvais poids »), et là l'IS de chemin suffit (ESS 16–19 %, k̂ < 0,35).
- Entraînement non stationnaire : le × 0,5 ne passe Q qu'à partir de l'époque 1250 (d = 128) et 750 (d = 256) ; avant, 1 à 2 bassins manquent.
Réserves : une graine ; deux valeurs de σ_max par dimension ; Q2 à 8/8 est franchi par des runs qui ont un ou deux bassins faibles ; le comparateur de G à coût égal est un IS simple sans tronquage au-delà du standard ; une seule dimension intermédiaire (d = 128, 256), pas d'essai à d = 512.

## Étape 5 : dernier lot de la piste darting, GMM-dk (128, 256) et MoS-dk (32, 128) (2026-11-21)

Règles écrites et commitées avant la location et avant tout résultat : `rules_dk_step5.md` (c47fe51). Scripts `scripts/dk_step5_eval.py` (sous-commandes `q`, `cmp`, `truth`), `scripts/mosdk_truth.py`, `scripts/vast_dk_step5_remote.sh`, cible `MoSDK` (`dist_utils.py`, `configs/problem/mosdk.yaml`, `configs/experiment/mosdk_as.yaml`) ; résultats `figures/dk_step5/` (q_*.json, cmp_mos_d{32,128}.json, cmp_gmm_d{128,256}_supp.json, truth_*.json, journaux, 24 contrôleurs, log w et cellules).
**Coût : 0,063 $** (instance 61768924, offre RTX 3070 57456544 à 0,1339 $/h, 28,1 min de 15:12:44 à 15:40:52 ; estimation avant location 0,09 $ central, 0,14 $ pessimiste ; plafond dur 0,25 $). Arrêts des minutes 80 / 95 non déclenchés ; instance détruite dès le rapatriement (0 restante). Aucun incident, aucun NaN (≈ 0,65 s par époque, 4 runs en parallèle).
**Écarts déclarés à la demande (écrits avant la location, `rules_dk_step5.md` §1) :** (i) les 4 runs GMM-dk (128 et 256 × σ_max {0,5 ; 0,75}) ne sont pas réentraînés : ce sont ceux de l'étape 4 (même recette, même graine) ; seul le darting à S = 40 est ajouté, par réévaluation des deux contrôleurs retenus à l'étape 4 (× 0,5, époque 1500) ; (ii) proportions par bassin de la vérité : 2·10⁵ tirages exacts (et non 10⁶), rapportées à côté.

### Cibles et vérités
MoS-dk : K = 8, moyennes (Δ/√2) e_k, poids géométriques de rapport 10, composantes = produits de Student 1D ν = 3, s² = 4/3 (Hessienne I au mode). Plancher TV (n = 10⁴) 0,0096 (MoS-dk) et 0,0099 (GMM-dk). Médiane de la distance au minimum sous π : MoS-dk 9,57 (d = 32), 20,67 (d = 128) ; GMM-dk 11,29, 15,98. **Fait à noter : pour le MoS-dk, les bassins de la quench ne sont pas les cellules de Voronoï, même sous π** (queues lourdes) : TV(bassin, Voronoï) = 0,087 à d = 32 et 128 (0,004 pour le GMM-dk) ; proportions par bassin de la vérité 0,325 0,219 0,156 0,110 0,077 0,053 0,036 0,025 contre 0,265 0,198 0,150 0,115 0,090 0,072 0,059 0,050 par Voronoï. Les TV « par bassin » du MoS-dk, calculés contre la vérité de Voronoï (convention des étapes 3 et 4), ont donc un plancher ≈ 0,087 et ne sont pas comparables au TV de Voronoï ; ils sont rapportés ci-dessous contre la vérité par bassin.

### Q sur les 24 contrôleurs MoS-dk (quench convergée à 100 % ; Q1 vrai partout : rapports de 0,74 à 1,04)
| run | bassins couverts (p̂ ≥ 0,25 p) aux époques 250 / 500 / 750 / 1000 / 1250 / 1500 | Q | k̂, ESS (contrôleurs qui passent Q) |
|---|---|---|---|
| d = 32, × 0,5 | 8 / 8 / 8 / 8 / 8 / 8 | passe aux 6 | k̂ 0,73 → 0,59–0,63, ESS 0,034–0,068 (1500 : 0,63 ; 0,067) |
| d = 32, × 0,75 | 8 / 8 / 8 / 8 / 8 / 8 | passe aux 6 | k̂ 0,60–0,67, ESS 0,007 → 0,164 (1500 : 0,60 ; 0,164) |
| d = 128, × 0,5 | 3 / 1 / 1 / 1 / 1 / 1 | ne passe jamais (effondrement : TV brut Voronoï 0,44–0,54) | |
| d = 128, × 0,75 | 7 / 7 / 7 / 7 / 7 / **8** | **passe à 1500 seulement** | k̂ 1,19, ESS 0,0008 |
TV brut Voronoï à l'époque 1500 : d = 32 : 0,067 (× 0,5), 0,024 (× 0,75) ; d = 128 : 0,518 (× 0,5), 0,025 (× 0,75).

### Choix de σ_max et comparaison
- **MoS-dk d = 32** : les deux candidats (époque 1500) ont k̂ < 0,7 ; argmax de log Ẑ : **× 0,75** (−0,100 ± 0,016 contre −0,369 ± 0,026).
- **MoS-dk d = 128** : seul candidat : × 0,75, époque 1500, **k̂ = 1,19 ≥ 0,7 : cas « aucun k̂ < 0,7 », plus grand ESS, signalé** (un seul candidat).
| | MoS-dk d = 32, × 0,75, ép. 1500 | MoS-dk d = 128, × 0,75, ép. 1500 |
|---|---|---|
| TV brut (Voronoï / bassins vs vérité Voronoï / bassins vs vérité bassin) | 0,0242 / 0,126 / 0,041 | 0,0245 / 0,097 / 0,033 |
| IS simple : TV, ESS, k̂, log Ẑ | 0,0172, **0,164**, 0,60, −0,100 | 0,2933, **0,0008**, 1,19, −1,02 |
| PSIS / tronqué : TV (n = 20 000) | 0,0176 / 0,0172 | 0,3073 / 0,0511 |
| IS (bassins vs vérité bassin) : simple / PSIS / tronqué | 0,021 / 0,022 / 0,021 | 0,318 / 0,333 / 0,047 |
| darting S = 1 : TV (bassins vs vérité bassin), acc. sauts | 0,0176 (0,018), 0,50 | **0,0119** (0,015), 0,53 |
| S = 5 | 0,0091 (0,008), 0,51 | **0,0059** (0,014), 0,52 |
| S = 10 | 0,0053 (0,006), 0,52 | 0,0199 (0,018), 0,52 |
| S = 40 | 0,0068 (0,009), 0,52 | 0,0082 (0,008), 0,52 |
| MALA seul, coût de S = 1 / 5 / 10 / 40 : TV | 0,0227 / 0,0193 / 0,0110 / 0,0083 | 0,0214 / 0,0144 / 0,0176 / 0,0106 |
| IS de chemin à coût égal (n = 1,9·10⁵ / 6,5·10⁵ / 1,2·10⁶ / 4,7·10⁶) : TV simple | 0,0201 / 0,0165 / 0,0074 / 0,0113 | 0,1117 / 0,1556 / 0,1126 / 0,1172 |
| idem, meilleure variante (tronqué) | 0,0173 / 0,0087 / 0,0042 / 0,0023 | 0,0242 / 0,0123 / 0,0098 / 0,0146 |
Découverte : 8 minima sur 8 aux deux ; Hessiennes inégales (écart 0,06) : carte générale ; CPU du darting 24 s et 87 s ; IS à coût égal non tronqué (131 s et 152 s sur GPU). Proportions p̂/p par bassin du darting (S = 1, d = 128) : 1,03 1,01 0,93 1,04 0,97 1,00 0,99 0,96. IS simple d = 128, p̂/p par bassin : 0,72 0,62 0,71 2,53 0,64 1,15 2,40 0,11.

### Règles
- **MoS-dk d = 32 : Q vrai, I FAUX** (ESS 0,164 ≥ 5 %, k̂ 0,60), **G FAUX** (aux S = 5, 10, 40 l'IS tronqué de coût égal est meilleur : 0,0087, 0,0042, 0,0023 contre 0,0091, 0,0053, 0,0068 ; à S = 1, 0,0176 > 0,5 × 0,0173), **H_saut FAUX** (0,0068 > 0,5 × 0,0083 : le MALA seul fait presque autant, le TV brut étant déjà à 2,5 × le plancher).
- **MoS-dk d = 128 : Q VRAI** (candidat de l'époque 1500 : rapport 0,86, 8/8 bassins), **I VRAI** (ESS 0,0008 < 5 %, k̂ 1,19), **G VRAI** : à S = 1, TV_darting 0,0119 ≤ 0,5 × 0,0242 = 0,0121 (meilleure variante = tronqué) ≤ 0,5 × 0,1117 (IS simple), et ≤ 2 × plancher = 0,0192 ; à S = 5, 0,0059 ≤ 0,5 × 0,0123 = 0,0062 ; faux à S = 10 (0,0199 > 0,0192) et S = 40 (0,0082 > 0,0073). **H_saut FAUX** (0,0082 > 0,5 × 0,0106). Contre la meilleure variante à 20 000 trajectoires : vrai à S = 1, 5, 40.
- **GMM-dk (supplément S = 40 ; Q, choix et IS de l'étape 4)** : d = 128 : Q vrai, I faux (ESS 0,193), **G faux** (darting S = 1, 5, 10, 40 : 0,217, 0,0281, 0,0098, 0,0134 contre l'IS de coût égal 0,0059, 0,0019, 0,0012, 0,0010), H_saut vrai ; d = 256 : Q vrai, I faux (ESS 0,162), **G faux** (0,157, 0,0147, 0,0080, 0,0041 contre 0,0119, 0,0047, 0,0044, 0,0016), H_saut vrai.
**Décision écrite à l'avance (§7) : « rouverte seulement si, pour au moins une cible, Q vrai, I vrai et G vrai ». MoS-dk d = 128 réalise les trois : la règle, telle qu'écrite, ROUVRE la piste. Les trois autres cibles ne la rouvrent pas.**

### Marges et réserves sur le seul cas qui rouvre (faits ; la règle n'est pas modifiée)
- **G est vrai de très peu et ne l'est qu'à S = 1 et 5** : 0,0119 contre un seuil de 0,0121, et 0,0059 contre 0,0062 ; à S = 10, TV_darting 0,0199 (faux, non monotone en S) et à S = 40, 0,0082 (faux). Les TV de l'ordre de 0,006 à 0,02 sont au niveau du plancher (0,0096) : TV_darting(S = 5) est même sous le plancher d'un sampler exact ; avec une seule réalisation, ces écarts ne sont pas interprétables.
- Le TV brut du contrôleur est déjà de 0,0245 (2,5 × plancher) : le sampler est quasi correct en proportions de Voronoï ; **H_saut est faux** (le MALA seul fait 0,0214 à S = 1 et 0,0106 à S = 40 : il fait aussi presque tout, les sauts n'apportent pas le facteur 2).
- **Q est vrai de peu** : à l'époque 1500 seulement ; aux époques 250 à 1250 le même run ne couvre que 7/8 bassins (un bassin sous 0,25 p) ; le candidat est donc le seul contrôleur du run × 0,75 qui passe Q, et il n'y a pas de candidat k̂ < 0,7 (k̂ 1,19, cas signalé).
- **I vrai** est net : ESS 8·10⁻⁴, TV de l'IS simple 0,29 ; l'IS tronqué (biaisé) tombe cependant à 0,0098–0,0146 à coût égal (n ≥ 1,2·10⁶), du niveau du darting ; la comparaison G se fait donc contre un IS tronqué qui est déjà près du plancher, et le gain du darting repose sur S = 1 et 5.
- **Proportions par bassin** (vérité par bassin, 2·10⁵ tirages) : darting 0,015 (S = 1), 0,014 (S = 5), 0,018 (S = 10), 0,008 (S = 40), contre 0,033 pour AS brut, 0,318 pour l'IS simple, 0,047 pour l'IS tronqué à 20 000 trajectoires, 0,026 / 0,020 / 0,020 / 0,014 pour le MALA seul ; la version par bassins de G à coût égal n'est pas disponible (l'IS à coût égal n'a gardé que les cellules de Voronoï).
- Une graine, une seule valeur retenue de σ_max (× 0,75), un seul contrôleur ; au MoS-dk d = 32, où AS apprend aussi, l'IS tient (ESS 0,164) et bat le darting à S ≥ 5. Les runs × 0,5 du MoS-dk d = 128 se sont effondrés (1/8 bassin).

## Contrôle CPU de H_saut par bassins, MoS-dk d = 128 × 0,75, époque 1500 (suite de l'étape 5 ; 2026-11-21)

Règle écrite et commitée avant toute chaîne : `rules_hsaut_basin.md` (9d16c5a). Script `scripts/hsaut_basin_cpu.py` ; résultats `figures/hsaut_cpu/` (chain{0..4}_{dart,mala}.json, aggregate.json). CPU du Mac, aucune location, chaque exécution < 1 min (darting 41 s, MALA 28 s ; chaîne d'essai de graine 99 : 40,6 s, résultats jetés, d'où n = 10⁴).
5 chaînes indépendantes (jeux de 10⁴ points terminaux d'AS simulés sur CPU, graines 1000 + i ; darting graine 100 + i ; MALA seul graine 200 + i, même pas, mêmes points de départ, coût mesuré de chaque S). TV par bassin contre la vérité par bassin (2·10⁵ tirages), TV de Voronoï contre la vérité de Voronoï (10⁶), moyenne ± écart-type (n − 1) sur les 5 chaînes. Planchers (multinomial, n = 10⁴) : **bassin 0,0093, Voronoï 0,0099**.
| | TV brut d'AS | darting S = 1 | S = 5 | S = 40 | MALA seul (coût de S = 1) | (S = 5) | (S = 40) |
|---|---|---|---|---|---|---|---|
| par bassin | 0,0376 ± 0,0044 | 0,0143 ± 0,0032 | 0,0097 ± 0,0026 | **0,0116 ± 0,0036** | 0,0261 ± 0,0038 | 0,0174 ± 0,0009 | **0,0116 ± 0,0020** |
| par Voronoï | 0,0298 ± 0,0062 | 0,0120 ± 0,0028 | 0,0099 ± 0,0040 | 0,0105 ± 0,0031 | 0,0268 ± 0,0030 | 0,0190 ± 0,0027 | 0,0103 ± 0,0024 |
Coûts moyens 1,9·10⁵, 6,5·10⁵, 4,7·10⁶ ; acceptation des sauts 0,51. Rappel (GPU, 20 000 trajectoires, étape 5) : IS tronqué par bassin 0,047 contre la vérité par bassin (0,091 contre la vérité de Voronoï), Voronoï 0,051.
**H_saut_bassin : FAUX** (TV_bassin darting S = 40 = 0,0116 contre 0,5 × 0,0116 = 0,0058 pour le MALA seul : rapport 1,00 ; version Voronoï, descriptive : 0,0105 contre 0,5 × 0,0103, faux aussi). **Décision écrite : la réouverture tient aux pas locaux, pas aux sauts ; la piste darting est fermée, définitivement.**
Descriptif, sans verdict : à S = 40 le darting et le MALA seul sont au même niveau, tous deux au plancher (0,0093 par bassin) ; aux petits S le darting est meilleur que le MALA seul de même coût (S = 1 : 0,0143 contre 0,0261 ; S = 5 : 0,0097 contre 0,0174, rapports 0,55 et 0,56 par bassin), c'est-à-dire que les sauts accélèrent la correction au début sans rien apporter à S = 40, où le MALA seul rattrape. Réserves : un seul contrôleur et une graine d'entraînement ; les 5 chaînes ont des points terminaux indépendants mais viennent du même contrôleur ; écarts-types sur 5 chaînes (0,002 à 0,004) du même ordre que les différences à S = 40.

## Test GPU : Newton-Stein contre amortissement contre AS, GMM-dk d = 32 (2026-11-22)

Règle écrite et commitée avant la location : `rules_ns_damping.md` (656b270). Code : `AdjointVEDampedMatcher` (η = 0,5, éq. 176 de *Failure Modes*, contrôleur gelé au début de chaque époque) et `AdjointVENewtonSteinMatcher` (Ĥ = I, Λ_t = 1/(ν_{1|t} + κ_t)), tests `tests/test_newton_stein_matcher.py` (moyenne nulle sous π : max |z| 2,66 ; contrôle positif 51,8). Évaluation `scripts/ns_damping_eval.py`, verdicts `scripts/ns_damping_rules.py`, script distant `scripts/vast_ns_damping_remote.sh`. Résultats : `figures/ns_damping/` (q_*.json, verdicts.json, log w is_logw_*.npz, journaux, 48 contrôleurs + configs).
Graine 0, 1500 époques, recette `gmm_dk_as`. **Bras (a) AS : les 4 runs existants sont réutilisés, pas relancés** (balayage 788a0e8 pour × 1, fenêtre f581ea6 pour × 2 ; voie AS du code inchangée depuis) ; ils n'ont que le contrôleur final, donc R1 et R2 portent sur l'époque 1500 pour les trois bras, et les sauvegardes intermédiaires n'existent que pour (b) et (c). 8 runs nouveaux. IS de chemin sur 20 000 trajectoires (graine 0), même script pour les trois bras.
Coût : RTX 3070 (offre 60139709, 24 cœurs), 0,1211 $/h, création → destruction 19,8 min, **0,040 $** (estimation 0,06 $ central, plafond 0,15 $). torch.compile indisponible sur l'image (pas de compilateur C) : entraînement en mode eager, 0,62–0,67 s par époque à 8 runs en parallèle.

Époque 1500 (Q = filtre de l'étape 4 ; couverture = bassins avec p̂ ≥ 0,25 p) :
| cellule | σ_max | (a) AS : Q, couverture, ESS, k̂ | (b) amortissement | (c) Newton-Stein |
|---|---|---|---|---|
| (32, 8, 4) | × 1 (2,236) | vrai, 8/8, 0,2405, 0,29 | vrai, 8/8, 0,2214, 0,30 | vrai, 8/8, **0,2934**, 0,24 |
| (32, 8, 4) | × 2 (4,472) | vrai, 8/8, 0,0047, 0,70 | vrai, 8/8, 0,0078, 0,76 | vrai, 8/8, **0,0264**, 0,54 |
| (32, 8, 8) | × 1 (2,828) | faux, 3/8, 0,0711, 0,43 | faux, 2/8, 0,0067, 0,75 | faux, 2/8, 0,0006, 0,91 |
| (32, 8, 8) | × 2 (5,657) | faux, 1/8, 0,0088, 0,73 | faux, 1/8, 0,0109, 0,66 | faux, 3/8, 0,0039, 0,88 |
TV brut (Voronoï) à (32, 8, 4) : × 1 : 0,026 / 0,024 / 0,012 ; × 2 : 0,032 / 0,037 / 0,033. Q1 vrai partout (rapport 1,01) ; à Δ = 8 seule la couverture échoue. Réévaluation de (a) avec le même script : ESS 0,24 à × 1 (0,25 dans la fenêtre σ_max) et 0,0047 à × 2 (0,011 dans la fenêtre : l'ESS à ce niveau est bruité).

### Règles
- **R1 FAUX.** À × 2, les deux conditions tiennent : ESS (c) = 5,6 × (a) et 3,4 × (b). À × 1, aucune ne tient : 1,22 × (a) (seuil 2) et 1,33 × (b) (seuil 1,5). La règle exige les deux σ_max.
- **R2 FAUX** : aucun bras ne passe Q à (32, 8, 8), ni à × 1 ni à × 2 ; Newton-Stein couvre 2/8 et 3/8 bassins, comme AS (3/8, 1/8) et l'amortissement (2/8, 1/8).
- **R3 (descriptif)** : sur les dumps des époques 1000 à 1450, les erreurs de moyenne par mode sont au niveau du bruit Monte-Carlo pour presque tous les runs (z médian 1,5 à 2,3 : « signal trop faible »). Seule exception pour (a) : (32, 8, 8) × 1, « oscille » (O = 0,6 sur 5 paires seulement, z médian 2,9). Sur les sauvegardes de 250 en 250 époques, (b) et (c) n'oscillent pas (O ≤ 0,2). Les oscillations prédites par la théorie gaussienne ne sont donc pas visibles dans ces runs ; pour (a), la mesure n'est possible qu'à × 1.
- **Décision écrite à l'avance : ni R1 ni R2 → on arrête.** Newton-Stein n'est pas poursuivi.

### Descriptif (sans verdict)
- ESS des sauvegardes 250 → 1500, (32, 8, 4) × 1 : (b) 0,077 0,118 0,157 0,159 0,206 0,221 ; (c) 0,085 0,119 0,199 0,273 0,277 0,293 (≥ (b) aux 6 sauvegardes, rapport 1,0 à 1,7). × 2 : (b) 0,0014 0,0021 0,0077 0,0080 0,0114 0,0078 (7/8 bassins aux époques 250 et 500) ; (c) 0,0040 0,0171 0,0074 0,0073 0,0165 0,0264 (8/8 partout).
- Lecture : l'avantage de Newton-Stein apparaît là où la théorie gaussienne prédit l'instabilité d'AS (× 2 : ρ = σ_max² = 20, multiplicateur 1 − F(ρ) = −2,15) et pas au seuil (× 1 : ρ = 5, −1,01), mais il est mesuré sur une graine, avec des ESS de 0,005 à 0,03 (100 à 500 échantillons effectifs sur 20 000), donc bruités. Il ne change pas la découverte des modes (Δ = 8), comme le prédit le Théorème C du mémo.
- Mes prédictions écrites (§5 de la règle) : Q passé par les trois bras à Δ = 4 ✓ ; ESS (b) × 1 dans 0,2–0,5 ✓, × 2 sous 0,02–0,15 ✗ (0,0078) ; ESS (c) × 1 dans 0,2–0,6 ✓, × 2 juste sous 0,03–0,3 ✗ (0,026) ; R1 faux ✓, R2 faux ✓, décision « on arrête » ✓ ; R3 « signal trop faible » pour (a) × 1 ✓.
Réserves : une graine ; (a) réutilisé sans sauvegardes intermédiaires ni dumps à × 2 ; l'ESS sur 20 000 trajectoires est bruité aux petites valeurs ; Ĥ = I est exact dans les modes mais pas pour le mélange global aux petits t.

## Newton-Stein dans le régime d'instabilité extrême : MoS-w, σ_max = 23,2 (2026-11-22)

Règle écrite et commitée avant la location : `rules_mosw_ns.md` (e5c8b5d), avec l'étape 0 (CPU). Bras : (a) AS, les 2 runs de l'étape 3 réutilisés et réévalués avec le même script (chiffres identiques à ceux de l'étape 3) ; (b) amortissement η = 0,5 ; (c) Newton-Stein Ĥ = 1,5. MoS-w d = 50 et d = 10, graine 0, 1500 époques, 4 runs nouveaux. Mesures : `scripts/mosw_step3_eval.py --q_only`. Verdicts : `scripts/mosw_ns_rules.py`. Résultats : `figures/mosw_ns/` (out.json par run, verdicts.json, log w, journaux, 24 contrôleurs + configs). Coût : RTX 3070 (57456544), 0,1339 $/h, 27,7 min, **0,062 $** (plafond 0,10 $). Entraînement à environ 1,0 s par époque, l'estimation pessimiste.

**Étape 0.** A = 1 − F(ν₁Ĥ) : MoS et MoS-w, A de −3,0 à −5,7 (ρ de 50 à 807) ; MoS-dk, A de −0,07 à −0,76 (stables). A ne range pas les issues : le seul run du pilote MoS à 10/10 modes est le plus « instable » (σ_max = 23,2), et le MoS-dk le plus stable (d = 128, × 0,5, A = −0,07) s'est effondré. SSS (annexe C.1 et C.5) : GMM16 et GMM50, coordonnées 0,2 N(−1,5 ; 0,5²) + 0,8 N(2,5 ; 0,5²) ; leur AS utilise le processus d'Ornstein–Uhlenbeck de SSS (référence N(0, I), horizon T = 4) au lieu du processus VE, et aucun σ_max n'est donné. **Le A de leurs runs n'est pas calculable à partir de ce qui est écrit** ; extrapolation signalée comme telle : ρ = 4, A = −0,85.

Distance médiane au minimum / π (Q1 si ≤ 1,5 : *) ; bassins couverts sur 10 ; TV par bassin, aux sauvegardes 250 → 1500 :
| run | 250 | 500 | 750 | 1000 | 1250 | 1500 |
|---|---|---|---|---|---|---|
| d = 50 (a) AS | 7,95 / 10 / 0,30 | 10,07 / 9 / 0,52 | 9,38 / 6 / 0,61 | 10,12 / 6 / 0,70 | 10,10 / 2 / 0,79 | 10,22 / 1 / 0,82 |
| d = 50 (b) amortissement | 8,61 / 10 / 0,35 | 9,90 / 8 / 0,52 | 9,82 / 6 / 0,61 | 9,43 / 7 / 0,55 | 9,40 / 7 / 0,61 | 10,86 / 10 / 0,20 |
| d = 50 (c) Newton-Stein | 11,16 / 10 / 0,25 | 11,10 / 10 / 0,25 | 11,08 / 10 / 0,24 | 11,09 / 10 / 0,24 | 11,08 / 10 / 0,24 | 11,08 / 10 / 0,24 |
| d = 10 (a) AS | 3,21 / 7 / 0,24 | 2,42 / 5 / 0,39 | 1,96 / 4 / 0,35 | 1,66 / 5 / 0,31 | 1,49* / 4 / 0,39 | 1,49* / 5 / 0,30 |
| d = 10 (b) amortissement | 3,20 / 8 / 0,26 | 1,89 / 5 / 0,42 | 1,56 / 2 / 0,48 | 1,40* / 2 / 0,48 | 1,36* / 3 / 0,46 | 1,36* / 3 / 0,44 |
| d = 10 (c) Newton-Stein | 12,62 / 10 / 0,19 | 12,64 / 10 / 0,20 | 12,53 / 10 / 0,19 | 12,56 / 10 / 0,20 | 12,56 / 10 / 0,19 | 12,62 / 10 / 0,15 |

### Règles
- **S1 FAUX** : à d = 50, aucun bras ne passe Q1 à aucune sauvegarde (Newton-Stein : rapport 11,1 partout).
- **S2 VRAI tel qu'écrit** : à d = 10, époque 1500, Newton-Stein a 10/10 bassins couverts, contre 5 pour AS et 3 pour l'amortissement.
- **Décision écrite à l'avance : S2 vrai → la règle rouvre Newton-Stein.**
- **Amendement daté (2026-11-22, `rules_mosw_ns.md`) : « S2 exige Q1 en plus de la couverture ».** Le verdict d'origine (S2 vrai, Newton-Stein rouvert) reste écrit ci-dessus. Avec l'amendement, Newton-Stein d = 10 a un rapport Q1 de 12,62, donc S2 est faux, et **la piste Newton-Stein est fermée**. Indépendamment de S2, l'étape 0 suffit : A ne prédit pas les échecs (le run le plus instable est le seul à 10/10, le plus stable s'effondre).

### Réserve majeure (vérification a posteriori, CPU, 2 s ; la règle n'est pas modifiée)
Le S2 vrai ne vient pas d'un sampler qui garde les modes : **le sampler Newton-Stein est resté à la loi de base**. La loi de base N(0, 23,2² I) seule, mesurée de la même façon (10⁴ tirages, mêmes minima, même quench), donne : à d = 10, distance médiane 64,1 (Newton-Stein 66,0 ; π 5,2), 10/10 bassins couverts, TV par bassin 0,22 ; à d = 50, 159,3 (Newton-Stein 162,5 ; π 14,7), 10/10, TV 0,25. Une loi diffuse couvre tous les bassins parce que la quench répartit ses points selon le volume des bassins. Le critère de couverture sans Q1 ne distingue donc pas « garder les modes » de « ne rien apprendre ».
Cause (lecture, cohérente avec le Corollaire A1 du mémo) : avec Ĥ = 1,5 et ν₁ = 538, Λ_t = 1/(1,5 ν_{1|t} + κ_t) vaut environ 1/800 aux petits t. La courbure globale du mélange (moyennes uniformes sur [−10, 10]) est bien plus faible que 1,5, ce qui donne ρ ≪ 1 dans les directions globales : la mise à jour est sous-relaxée d'un facteur de l'ordre de 800, et les dérives des premiers instants, qui placent les points, n'apprennent presque pas en 1500 époques. Newton-Stein ne stabilise pas AS ici : il le gèle. L'amortissement (b) n'apprend pas non plus la forme à d = 50 et perd des bassins à d = 10 comme AS.
Lecture d'ensemble : sur le MoS-w, aucun des trois bras ne passe Q à aucune sauvegarde. Le seul critère rempli l'est par un sampler égal à la loi de base. Rien ici ne montre que l'échec d'AS vienne de l'instabilité ; la décision formelle (« rouvert ») est celle de la règle écrite.
Mes prédictions (§ de la règle) : S1 faux ✓ ; S2 vrai (prédit à 20 %) ✗ ; décision « on ferme » (70 %) ✗, pour la raison ci-dessus.

## Phase 0

### Tâche 1/4 : DW-4, garde-fou (iii) de la confirmation du darting refait avec assez de puissance (réanalyse déclarée, 2026-11-22)

Règle écrite et commitée avant tout calcul : `rules_dw4_guard_power.md` (d943965). Script `scripts/dw4_guard_power.py`, résultats `figures/dw4_guard_power/` (floor, arm_*, calib_s*, ctrl_s*, verdict.json). CPU du Mac, aucune location, aucun entraînement.

**Le verdict d'origine de `rules_darting_confirm.md` reste « non confirmé » (iii faux sur la graine 2), tel qu'écrit.** Cette réanalyse remplace (iii) par G3′ pour elle seule.

**Données et protocole.**
- Données : `model_samples` finaux (10⁴) des bras D30 et AS (C2 relancé, C3, C4), graines 2, 3, 4. Référence : le pool PT.
- Métriques : energy_w2 et eq_w2 de l'évaluation officielle, à **n = 2000, sous-échantillonnage déclaré**. À n = 10⁴, une évaluation dépasserait la minute ; ce n'a pas été tenté.
- Bootstrap : 20 rééchantillonnages par bras, avec une référence commune par rééchantillonnage.
- Temps par exécution : 12,6 s et 36,6 s (plancher) ; 25,1 à 26,2 s par demi-bras ; étalonnage 25,8 / 26,4 / 20,4 s ; contrôle positif 25,8 à 26,9 s. Aucune exécution au-dessus d'une minute.

**Plancher (PT contre PT, chaînes disjointes, 20 répétitions).** eq_w2 : médiane 0,301, p95 0,342, écart-type 0,021. energy_w2 : médiane 0,112, p95 0,191, écart-type 0,038.

| graine | eq_w2 D30 (± sd) | eq_w2 C (± sd) | Δ_s eq_w2 (sd) | energy_w2 D30 (± sd) | energy_w2 C (± sd) | Δ_s energy_w2 (sd) |
|---|---|---|---|---|---|---|
| 2 | 0,476 ± 0,051 | 0,653 ± 0,046 | −0,177 (0,069) | 0,657 ± 0,071 | 0,705 ± 0,066 | −0,047 (0,097) |
| 3 | 0,676 ± 0,038 | 0,475 ± 0,033 | **+0,201 (0,050)** | 0,225 ± 0,064 | 0,382 ± 0,076 | −0,157 (0,099) |
| 4 | 0,392 ± 0,038 | 0,690 ± 0,038 | −0,297 (0,054) | 0,295 ± 0,079 | 0,411 ± 0,084 | −0,117 (0,116) |

### G3′
- **eq_w2 : G3′ FAUSSE.** La moyenne Δ̄ = −0,091 est bien ≤ 2 × SE (SE 0,034), mais **la graine 3 a Δ_s = +0,201 > 3 × 0,050 = 0,151** (environ 4 écarts-types). Les graines 2 et 4 ont Δ_s négatif, à −2,6 et −5,5 écarts-types : D30 y est meilleur que l'AS.
- **energy_w2 (descriptif) : G3′ vraie.** Δ̄ = −0,107 (SE 0,060), D30 plus bas sur les 3 graines, aucune graine au-dessus de 3 écarts-types.

### Contrôle positif
D30 bruité, σ fixé par la procédure écrite : graine 2 σ = 0,5 ; graine 3 σ = 0,5, signalé (aucune valeur de la grille n'atteint le double de l'écart) ; graine 4 σ = 0,3. G3′ sur la version dégradée : Δ_s = +0,209 / +0,423 / −0,135, Δ̄ = +0,166 > 2 × 0,024, deux graines au-dessus de 3 écarts-types. **G3′ fausse : le contrôle positif est détecté**, la garde a de la puissance.

**Décision écrite à l'avance : G3′ fausse → « non confirmé ».** Verdict maintenu tel qu'écrit, sans modification.

**Lecture (décision de l'utilisateur, 2026-11-22).**
- La garde utilise un écart-type de bootstrap, c'est-à-dire le bruit de W2 à modèle fixé. Elle ignore la variabilité du modèle d'une évaluation à l'autre (eq_w2 de 0,30 à 0,81 pour D30 graine 4).
- Avec un seul instantané par run, la dégradation de la graine 3 n'est pas distinguable d'une fluctuation : **étiquette I (inconclusif) sur la géométrie**.
- **Erreur de spécification (de l'utilisateur) : la règle aurait dû exiger plusieurs instantanés par run.**
- Les proportions de bassins et energy_w2 sont meilleures sur les 3 graines.
- On ne relance rien sur DW-4.
- **Leçon pour toutes les règles suivantes** : tout écart-type utilisé dans un seuil doit inclure la variabilité entre évaluations du modèle (plusieurs instantanés, ou moyennes temporelles), pas seulement le bootstrap à modèle fixé. Le verdict d'origine ne change pas, mais sa raison change. Ce n'est plus du bruit sur deux évaluations de la graine 2, où D30 est en fait meilleur (Δ −0,18) : c'est une dégradation réelle d'eq_w2 sur la graine 3, à environ 4 écarts-types.

### Lecture et réserves (sans changer la décision)
- **La dégradation concerne la géométrie, pas l'énergie.** Le darting γ = 30 améliore les proportions de bassins (critère (i), 3 graines sur 3) et energy_w2 (3 graines sur 3). Mais sur la graine 3, la géométrie des configurations du modèle final est nettement plus loin du PT que celle de l'AS de même graine.
- **La réanalyse porte sur un seul instantané par run**, les échantillons du modèle final. Le bootstrap mesure le bruit de W2 à modèle fixé, pas la variabilité du modèle d'une évaluation à l'autre. Celle-ci est grande : eq_w2 par évaluation pendant l'entraînement allait de 0,30 à 0,81 pour D30 s4. Les valeurs ne sont donc pas comparables aux moyennes temporelles de la section de confirmation, qui étaient calculées contre la référence DEM (graine 3 : D30 0,499 contre C3 0,553).
- Une seule graine dégradée sur trois suffit à faire échouer G3′ telle qu'écrite.

### Tâche 2/4 : AS pondéré par IS tempéré (idée I1), quadrature 1D (2026-11-22)

Règle écrite et commitée avant tout calcul : `rules_tempered_is_quadrature.md` (356ec61). Script `scripts/tempered_is_quadrature.py`, résultats `figures/tempered_is_quadrature.json`.

**Méthode.** La régression pondérée par r^α est exactement la mise à jour d'AS évaluée sur q_α ∝ q^{1−α} π^α. Cette loi est calculée directement : la postérieure réciproque de q_α s'obtient par quadrature sur grille, puis par Gauss–Hermite quand la vraisemblance devient étroite. Elle est propagée par le Fokker–Planck de `cv_failure_modes_checks.py`, dans les unités de l'article (ν₁ = 1, σ² = 0,05).

**Temps.** g 9,6 s, w 15,5 s, m 13,4 s, ESS < 0,1 s. Aucune séparation nécessaire.

**Contrôle α = 0 (AS) : vrai sur les trois panneaux**, à l'identique du mémo : A = −2,15125, poids gauche 0,50025, masse gauche 2,1905·10⁻⁵.

| panneau | α | résultat | prédiction | verdict | ESS/N |
|---|---|---|---|---|---|
| P_g (instabilité ; A_α) | 0 | −2,1513 | −2,1530 | ✓ | 1 |
| | 0,25 | −1,6134 | −1,6147 | ✓ | 0,988 |
| | 0,5 | −1,0756 | −1,0765 | ✓ | 0,951 |
| | 0,9 | −0,2151 | −0,2153 | ✓ | 0,850 |
| | 1 | −0,0000 | 0 | ✓ | 0,819 |
| P_w (mauvais poids ; poids gauche) | 0 | 0,50025 | 0,5000 | (contrôle) | 1 |
| | 0,25 | 0,63984 | 0,6401 (Δlogit −0,001) | ✓ | 0,927 |
| | 0,5 | 0,75926 | 0,7597 (−0,003) | ✓ | 0,788 |
| | 0,9 | 0,88774 | 0,8882 (−0,005) | ✓ | 0,624 |
| | 1 | 0,90869 | 0,9091 (−0,005) | ✓ | 0,599 |
| P_m (mode absent ; masse gauche / q_α(x < 0)) | 0 | 2,19·10⁻⁵ / 3,83·10⁻⁶ | (contrôle) | | 1 |
| | 0,25 | 3,30·10⁻⁵ / 7,31·10⁻⁶ | (hors règle) | | 1,000 |
| | 0,5 | 8,72·10⁻⁵ / 2,35·10⁻⁵ (**rapport 3,71**) | facteur 3 | **✗** | 0,500 |
| | 0,9 | 0,0731 / 0,0266 (rapport 2,75) | facteur 3 | ✓ | **1,0·10⁻²⁵** |
| | 1 | 0,5000 / 0,5000 | [0,47 ; 0,53] | ✓ | **7,2·10⁻³⁵** |

**Verdicts tels qu'écrits.**
- **P_g vraie** : A_α = (1 − α) A_m à 2·10⁻³ près, variance terminale inchangée (rapport 0,9998 à 1,0000).
- **P_w vraie** : l'erreur de logit est multipliée par (1 − α), |Δlogit| ≤ 0,005.
- **P_m fausse** : à α = 0,5, la masse gauche vaut 3,7 × q_α(x < 0), au-delà du facteur 3 ; vraie à α = 0,9 et 1.
- Contrôle vrai.

**Lecture (sans changer les verdicts).**
- L'IS tempéré se comporte exactement comme un **amortissement en logit** : une fraction α de l'écart à π est corrigée par itération, que ce soit sur la moyenne (P_g) ou sur les poids (P_w). AS seul ne corrige pas les poids : c'est le Th. 3.8.
- Sur le mode absent, une itération d'AS **amplifie** la masse gauche de q_α d'un facteur 2,7 à 5,7 (rapport > 1 à tout α < 1). L'échec de P_m à α = 0,5 vient de là ; c'était le point fragile annoncé dans la règle (bosse de q_α centrée sur la barrière).
- **La vraie limite est l'ESS**, conforme à l'attente écrite : sur le mode absent, il n'est utile qu'à α ≤ 0,5, où q_α(x < 0) est d'environ 2·10⁻⁵ et ne transporte presque rien. Le gain n'apparaît qu'à α ≥ 0,9, où l'ESS vaut 10⁻²⁵ à 10⁻³⁴. Avec de vrais échantillons, l'IS tempéré ne peut donc pas découvrir un mode absent, en accord avec le Théorème C du mémo. Il corrige les poids et l'instabilité des modes déjà visités, avec un ESS raisonnable (0,6 à 0,95).

### Tâche 4/4 : GMM isotrope (section 1), Barker contre IS de chemin post hoc : ARRÊTÉE (2026-11-22)

**Les contrôleurs finaux n'existent plus localement.** Il s'agit d'AS big et de Barker γ = 3 big (graines 0 à 2, d = 24 et 36) et de la division (d = 36, graine 0). Recherche effectuée dans le dépôt (`results/`, historique git complet), dans `~/Desktop/NTU`, dans `~/Downloads` et sur le disque. Ces runs ont été entraînés sur le cluster NTU, inaccessible depuis le 2026-11-16.

Il ne reste localement que des **échantillons finaux** (`buffer_x1`, `model_samples`, 10⁴), dans `~/Desktop/NTU/data/{final, samples_final}/`, et seulement pour une partie des runs : `gmm36d_as_big_cap` graine 0, `gmm36d_as_barker_g2_thr_big` (γ = 2, pas γ = 3), `gmm36d_as_barker_g{2,3}_thr` (petit réseau), `gmm36d_as_cmp`, ainsi que des variantes oracle et k-means à γ = 1 à 3. Sans contrôleur, les poids de chemin ne sont pas calculables. Conformément à la consigne, la tâche est arrêtée, sans réentraînement et sans règle écrite (aucun calcul fait).

**Point 4, coût en énergies (lecture des configurations et du code, aucun run ; demandé après l'arrêt).**
- **Constantes** (`gmm_as.yaml`, `train.py`) : B = 512 nouveaux échantillons par époque (resample_size 1000 // resample_batch_size 512 = 1 appel de `populate_buffer`), U = 100 mises à jour de gradient par époque, buffer 10⁴. Le buffer est entièrement renouvelé en 10⁴/512 ≈ 19,5 époques, soit environ 1950 mises à jour.
- **AS** : un gradient ∇E par nouvel échantillon, aucune énergie, soit **B/U = 5,12 gradients par mise à jour** (le « ≈ 5 » de la section 1).
- **Barker tel qu'implémenté** (`AdjointVEBarkerMatcher` / `AdjointVEMHMatcher.populate_buffer`, configuration figée γ = 3, centres à seuil, 12 pas de débruitage). **Tout est calculé une fois, à l'insertion de l'échantillon dans le buffer, et stocké** (`E1`, `EY`, `xd`). Par nouvel échantillon :
  - 1 gradient (∇E, déjà fait par AS) ;
  - 1 énergie E(x₁) ;
  - f_v (M − 1) énergies aux translatés T_r(x₁), où M = nombre de centres (≈ K = 24 ou 36) et f_v la fraction de translatés valides (≤ 1, non connue sans les journaux du cluster) ;
  - débruitage BB : 1 à 7 énergies (recherche linéaire d'Armijo) et 1 à 12 gradients.
- **Fusion par barrière des centres** (une fois par époque) : au plus M(M − 1)/2 paires × 9 points + M énergies. Sur le simplexe de rayon 4,25, toutes les paires sont à 6,1 < 8 × rayon, donc 2508 (d = 24) et 5706 (d = 36) énergies par époque.
- **Par mise à jour de gradient** (débruitage minimal, f_v = 1) :
  - d = 24 : 5,12 × (2 + 23) + 25 ≈ **153 énergies** + 10 gradients (jusqu'à 184 énergies et 67 gradients au débruitage maximal) ;
  - d = 36 : 5,12 × (2 + 35) + 57 ≈ **247 énergies** + 10 gradients (jusqu'à 277 et 67).
  - Le « 140 à 200 » de la section 1 (venu de notes, origine non documentée) concorde à d = 24. À d = 36, il est en dessous de cette borne, ce qui correspond à f_v ≈ 0,75 ou à moins de centres : non vérifiable sans les journaux.
- **Par renouvellement complet du buffer** (10⁴ échantillons, 19,5 époques) : AS 10⁴ gradients. Barker d = 24 ≈ 10⁴ × 25 + 19,5 × 2508 ≈ **3,0·10⁵ énergies** (+ 2·10⁴ gradients) ; d = 36 ≈ 10⁴ × 37 + 19,5 × 5706 ≈ **4,8·10⁵ énergies**.
- **Avec mise en cache des énergies aux images T_r(x₁)** : **c'est déjà le cas**. Les énergies des translatés ne dépendent pas de x_t et sont calculées une seule fois par échantillon du buffer. Le coût ci-dessus est donc le coût avec cache. Sans cache (énergies recalculées à chaque échantillon de régression), il faudrait 512 × (M − 1) énergies par mise à jour, soit 1,2·10⁴ (d = 24) et 1,8·10⁴ (d = 36). Le cache actuel les divise déjà par 512/5,12 = 100.
- **Ce qui reste** : un surcoût de 30 à 50 fois AS, qui vient des M − 1 énergies par **nouvel** échantillon et de la fusion par barrière. Pour le réduire, il faudrait évaluer moins de translatés par échantillon (sous-orbite aléatoire, au prix de la variance du terme ; non testé) et faire la fusion moins souvent.

**Conclusion (utilisateur, 2026-11-22).** Le cache existe déjà. Le surcoût de la VC à sauts (30 à 50 × AS en énergies) vient des M − 1 images par nouvel échantillon, et ne se réduit qu'en sous-échantillonnant les partenaires (non testé).

### Tâche 3/4 : division sur DW-4, des gabarits plus riches réduisent-ils la dispersion de log u ? ARRÊT par la règle de cohérence (2026-11-22)

Règle écrite et commitée avant tout calcul : `rules_dw4_template_variance.md` (18a9a28). Script `scripts/dw4_template_variance.py`, résultats `figures/dw4_template_variance/` (prep.json, eval_J1.json, verdict.json). Tout est ajusté et évalué localement, dans la base déterministe de Helmert ; aucun paramètre du cluster.

**Données.**
- Échantillons AS ed8bb67 (10⁴ ; quench convergé, bassins par `state_ids`). Moitiés d'ajustement et d'évaluation par bassin : 2177/2176, 1878/1878, 589/589, 307/307 et 86/85, l'étoile ayant 171 échantillons.
- PT B : chaînes 32 à 63, au plus 2000 états par bassin ; l'étoile en a 309, K4 moins une arête 1866.
- Rapporté seulement : la moitié B de la référence DEM (2364 / 2052 / 312 / 237 / 35).
- **Temps** : prep 0,4 s, prepref 0,2 s, EM J = 1 23,2 s (40 itérations ; la tolérance 10⁻⁴ n'est pas atteinte, comme à l'étape 1, l'étoile bouge encore), évaluation 3,3 s.

**J = 1 (gabarit gaussien, comme l'étape 1).** η_a par la règle de la variante B : 0,6 / 0,4 / 0,8* / 0,8* / 0,8* (* signalé : aucun η < 1 % de fuite). À l'étape 1, η valait 0,6 / 0,3 / 0,8 / 0,8 / 0,8.

| bassin | SD AS-éval (k̂) | SD PT B (k̂) | SD réf. DEM B | étape 1 (réf. DEM B) | écart PT B | écart réf. DEM B |
|---|---|---|---|---|---|---|
| triangle + pendant | 3,18 (−0,19) | 2,51 (−0,17) | 2,61 | 2,66 | −6 % | −2 % |
| chemin de 2 | 2,44 (0,05) | 1,56 (−0,10) | 2,07 | 1,68 | −7 % | **+23 %** |
| deux paires disjointes | 9,09 (0,74) | 8,27 (0,91) | 6,58 | 6,69 | **+24 %** | −2 % |
| K4 moins une arête | 6,18 (0,19) | 5,79 (0,23) | 5,64 | 6,46 | −10 % | −13 % |
| étoile | 4,83 (0,43) | 5,92 (0,35) | 3,58 | 3,30 | **+79 %** | +9 % |

**Cohérence (règle : J = 1 à 15 % près sur PT B) : FAUSSE** (deux paires +24 %, étoile +79 %). **Décision écrite : arrêt.** J = 3, 5 et 10 ne sont ni ajustés ni évalués ; V1, V2 et le contrôle négatif ne sont pas évalués. La lecture fixée d'avance (« fermée (R) quel que soit J », ou réouverture) **n'est pas atteinte**.

**Faits, sans changer la décision.**
- **Sur la moitié B de la référence DEM, où les valeurs de l'étape 1 avaient été mesurées**, J = 1 reproduit 4 bassins sur 5 à 15 % près. L'exception est le chemin de 2 (+23 %), où le η choisi diffère (0,4 contre 0,3 à l'étape 1). L'implémentation locale reproduit donc l'étape 1 là où l'étape 1 avait été mesurée, sauf un bassin à η différent.
- **Les écarts sur PT B viennent des données**, pas de l'implémentation : les mêmes gabarits donnent des dispersions plus grandes sur PT B que sur la référence DEM, dans les deux petits bassins concernés (deux paires 8,27 contre 6,58 ; étoile 5,92 contre 3,58).
- **La référence DEM n'a que 35 états d'étoile** dans sa moitié B, contre 309 en PT B : la valeur 3,30 de l'étape 1 reposait sur un petit échantillon. Les différences entre l'ajustement sur une moitié des AS et l'ajustement de l'étape 1 sur tous les AS jouent aussi (étoile : 86 contre 164 échantillons).
- **Dès J = 1**, k̂ dans le bassin dépasse 0,5 pour les deux paires (0,74 sur la moitié AS d'évaluation, 0,91 sur PT B), et la dispersion y est de 8 à 9 nats.

#### Tâche 3/4, reprise après l'amendement daté (037ae34, écrit et commité avant la reprise)

L'arrêt d'origine reste écrit ci-dessus. Il venait d'une erreur de spécification de l'utilisateur : les valeurs de référence avaient été mesurées sur la moitié B de la référence DEM, pas sur PT B.
- **Vérification d'implémentation** (amendée) : J = 1 reproduit l'étape 1 sur la moitié B DEM à 15 % près dans **4 bassins sur 5** (−2 %, +23 %, −2 %, −13 %, +9 %) : **vraie**. L'écart du chemin de 2 s'explique par η = 0,4 au lieu de 0,3.
- Référence pour la suite : J = 1 sur PT B (2,50 / 1,56 / 8,27 / 5,79 / 5,91).
- Rétrécissement par composante (η_j, règle de la variante B, appliquée à chaque composante).

**Temps.** EM : J = 3 en 45,4 + 24,2 s ; J = 5 en 46,1 + 45,8 + 22,7 s ; J = 10 en 45,6 + 45,1 + 50,2 + 50,3 + 33,4 s ; 40 itérations à chaque J, sans atteindre la tolérance 10⁻⁴. Évaluations : 3,3 / 7,6 / 15,1 / 24,3 s. Aucune exécution au-delà d'une minute.

Écart-type de log u_a (rétréci, η_j de la règle) sur la moitié AS d'évaluation / sur PT B, avec k̂ dans le bassin entre parenthèses ; dernière colonne : η_j.
| bassin | J = 1 | J = 3 | J = 5 | J = 10 | η_j (J = 5) |
|---|---|---|---|---|---|
| triangle + pendant | 3,18 (−0,19) / 2,50 (−0,17) | 4,57 (0,58) / 3,60 (0,46) | 2,83 (1,10) / 2,12 (1,13) | 3,36 (0,85) / 2,44 (1,02) | 0,8 0,2 0,8 0,2 0,8 |
| chemin de 2 | 2,44 (0,05) / 1,56 (−0,10) | 3,23 (0,50) / 2,17 (0,54) | 2,85 (0,77) / 1,77 (0,78) | 3,04 (0,66) / 1,85 (0,57) | 0,2 0,8 0,8 0,8 0,2 |
| **deux paires** | 9,09 (0,74) / 8,27 (0,91) | 6,34 (0,15) / 6,19 (0,38) | **5,94 (0,83) / 5,40 (0,91)** | 5,90 (0,79) / 5,64 (1,01) | 0,8 ×4, 0,4 |
| **K4 moins une arête** | 6,18 (0,19) / 5,79 (0,23) | 5,68 (0,37) / 5,16 (0,53) | **5,38 (0,18) / 5,28 (0,47)** | 4,29 (0,44) / 4,15 (0,66) | 0,8 ×5 |
| **étoile** | 4,83 (0,43) / 5,92 (0,35) | 4,16 (0,08) / 5,69 (0,90) | **3,54 (0,41) / 5,98 (0,67)** | 3,79 (−0,02) / 5,83 (0,97) | 0,8 ×5 |
Aux petits bassins, presque tous les η_j valent 0,8 et sont signalés (aucun η < 1 % de fuite), c'est-à-dire des covariances × 0,2.

**Verdicts tels qu'écrits.**
- **V1 FAUSSE** : à J = 5 sur PT B, 5,40 / 5,28 / 5,98 nats, très au-dessus de 2.
- **V2 FAUSSE** : k̂ sur la moitié AS d'évaluation 0,83 (deux paires), 0,18 et 0,41.
- **Contrôle négatif FAUX, et en sens contraire** : le gabarit J = 5 à covariances × 2 donne un écart-type **plus petit** (2,14 / 2,12 / 2,65 contre 5,40 / 5,28 / 5,98). Comme écrit dans la règle, la mesure ne discrimine pas dans le sens attendu : c'est signalé.
- **Lecture fixée d'avance : V1 et V2 fausses → « division fermée (R) quel que soit J ».**

**Mécanisme** (à partir des écarts-types sans rétrécissement, η = 0, déclarés « rapportés » dans la règle ; ils ne changent pas les verdicts). Moitié AS d'évaluation / PT B :
| bassin | J = 1 | J = 3 | J = 5 | J = 10 |
|---|---|---|---|---|
| triangle + pendant | 0,77 / 0,68 | 0,51 / 0,42 | 0,42 / 0,32 | 0,45 / 0,32 |
| chemin de 2 | 1,15 / 0,74 | 0,69 / 0,37 | 0,48 / 0,36 | 0,49 / 0,35 |
| deux paires | 1,22 / 1,16 | 0,63 / 0,61 | 0,64 / 0,62 | 0,68 / 0,70 |
| K4 moins une arête | 0,78 / 0,68 | 0,58 / 0,57 | 0,59 / 0,59 | 0,57 / 0,62 |
| étoile | 0,74 / 0,93 | 0,64 / 0,91 | 0,64 / 0,90 | 0,65 / 0,95 |

**Lecture.**
- **Les gabarits ne sont pas en cause.** Non rétrécis, ils suivent π à moins d'un nat près dès J = 1 dans 4 bassins sur 5, et à moins de 0,95 nat partout dès J = 3.
- **La dispersion de 5 à 9 nats vient du rétrécissement** qu'impose la règle de fuite : P(paire dans [3,5 ; 4,5]) < 1 %, sans solution dans les petits bassins, d'où η = 0,8 et des covariances × 0,2. Ce rétrécissement sert à garder N̄_a dominé par π et confiné à son bassin, condition de la division. C'est aussi pourquoi le contrôle « covariances × 2 » réduit la dispersion : il annule en partie le rétrécissement.
- **L'échec de la division sur DW-4 tient donc à la contrainte de domination et de confinement**, pas à la richesse des gabarits. Enrichir les gabarits (J) n'y change rien, ce que dit aussi la lecture fixée d'avance. Le contrôle négatif, mal conçu, ne permet pas de dire plus.

## Darting post hoc dans le régime où l'IS de chemin s'effondre (réouverture n° 2 de l'audit, élargie ; 2026-11-23)

Règle écrite et commitée avant tout calcul : `rules_darting_large_sigma.md` (f8957c1). Script `scripts/darting_large_sigma.py`, résultats `figures/darting_large_sigma/` (floor, raw, dart, mala, is, verdict.json). CPU du Mac, aucune location, aucun entraînement.

**Données et protocole.**
- **Contrôleurs** :
  - **d128** = GMM-dk (128, 8, 6), σ_max × 0,75, époque 1500 ;
  - **d256** = (256, 8, 6), × 0,75, époque 1500 (étape 4) ;
  - **d32** = (32, 8, 4), × 2, bras AS du test Newton-Stein.
- **Configuration des runs × 0,75** : celle du run × 0,5, avec σ_max remplacé par celui du JSON de Q (déclaré).
- **Jeux de points** : 3 jeux indépendants de 10⁴ points terminaux par contrôleur, simulés sur CPU.
- **Temps** : simulation 2,6 à 6,9 s par jeu ; darting 0,9 à 6,3 s ; MALA 1,0 à 4,8 s ; bassins < 1 s ; IS < 0,5 s. Aucune exécution au-delà de 7 s.
- **Planchers** (n = 10⁴) : 0,0093 (d32), 0,0100 (d128, d256).
- **Comparateur de coût égal** (déviation déclarée) : TV_eq = min(TV de l'IS sur les 3 × 10⁴ trajectoires réunies, borne LB au nombre d'échantillons effectifs à coût égal). **Pour les trois contrôleurs et à tous les S, c'est LB qui est retenue.**

Moyenne ± écart-type sur les 3 jeux ; TV par Voronoï / par bassin.
| | d128 | d256 | d32 |
|---|---|---|---|
| AS brut | 0,195 ± 0,003 / 0,110 ± 0,004 | 0,194 ± 0,009 / 0,102 ± 0,007 | 0,036 ± 0,003 / 0,074 ± 0,007 |
| IS de chemin (10⁴) : TV | 0,202 ± 0,115 / 0,202 ± 0,114 | 0,536 ± 0,129 / 0,531 ± 0,127 | 0,106 ± 0,064 / 0,131 ± 0,013 |
| IS de chemin (10⁴) : ESS, k̂ | 3,7·10⁻³ ± 1,7·10⁻³, 1,00 ± 0,08 | 3,0·10⁻⁴ ± 1,1·10⁻⁴, 3,67 ± 0,33 | 8,9·10⁻³ ± 5,1·10⁻³, 0,67 ± 0,08 |
| PSIS (10⁴) : TV Voronoï | 0,209 ± 0,077 | 0,352 ± 0,101 | 0,069 ± 0,020 |
| IS réuni (3 × 10⁴) : TV, ESS, k̂ | 0,127 / 0,125, 2,7·10⁻³, 0,97 | 0,487 / 0,467, 1,1·10⁻⁴, 3,40 | 0,073 / 0,094, 6,9·10⁻³, 0,66 |
| darting S = 1 | 0,055 ± 0,003 / 0,047 ± 0,003 | 0,058 ± 0,005 / 0,048 ± 0,005 | 0,011 ± 0,003 / 0,062 ± 0,004 |
| darting S = 5 | **0,012 ± 0,005** / 0,012 ± 0,003 | **0,012 ± 0,007** / 0,012 ± 0,005 | 0,010 ± 0,001 / 0,062 ± 0,004 |
| darting S = 10 | **0,008 ± 0,003** / 0,009 ± 0,002 | **0,009 ± 0,002** / 0,012 ± 0,003 | 0,011 ± 0,001 / 0,063 ± 0,006 |
| MALA seul, coût de S = 1 | 0,119 ± 0,004 / 0,109 ± 0,003 | 0,109 ± 0,006 / 0,098 ± 0,006 | 0,017 ± 0,002 / 0,064 ± 0,007 |
| MALA seul, coût de S = 5 | 0,100 ± 0,001 / 0,097 ± 0,000 | 0,100 ± 0,004 / 0,094 ± 0,004 | 0,010 ± 0,002 / 0,061 ± 0,000 |
| MALA seul, coût de S = 10 | 0,087 ± 0,002 / 0,084 ± 0,003 | 0,097 ± 0,004 / 0,092 ± 0,003 | 0,011 ± 0,006 / 0,062 ± 0,001 |
| TV_eq = LB (S = 1 / 5 / 10) | 0,045 / 0,024 / 0,017 | 0,223 / 0,120 / 0,087 | 0,026 / 0,014 / 0,010 |
Coûts du darting (évaluations d'énergie et de gradient) : 1,8·10⁵ (S = 1), 6,4·10⁵ (S = 5), 1,2·10⁶ (S = 10) ; 8 minima sur 8 découverts dans les 9 jeux.

### Verdicts tels qu'écrits
- **d128 : G VRAI** (S = 5 et 10 ; 2 jeux sur 3 seuls), **H_saut_bassin VRAI** (0,009 contre 0,5 × 0,084 ; 3 jeux sur 3). **Lecture : le darting tient là où l'IS s'effondre**, et on le garde comme bras d'entraînement de la famille 2.
- **d256 : G VRAI** (S = 5 et 10 ; 3 jeux sur 3), **H_saut_bassin VRAI** (0,012 contre 0,5 × 0,092 ; 3 jeux sur 3). **Même lecture.**
- **d32 : G VRAI** (S = 1 seulement ; 2 jeux sur 3), **H_saut_bassin FAUX** (3 jeux sur 3). **Lecture : la correction vient des pas locaux.**

### Réserves (faits ; les verdicts ne changent pas)
- **G repose sur la borne LB**, pas sur un IS de coût égal simulé. LB utilise l'ESS mesuré sur 3 × 10⁴ trajectoires avec k̂ de 0,97 (d128) et 3,40 (d256), une estimation peu fiable, présumée surestimée, donc défavorable au darting. Le darting atteint de toute façon le plancher (0,008 à 0,012) à S = 5 et 10 sur d128 et d256, alors que l'IS réellement calculé reste à 0,13 et 0,49.
- **Erreur de spécification de ma part (d32)** : la règle affirmait TV(bassin, Voronoï) = 0,004 pour le GMM-dk ; c'est vrai à Δ = 6, faux à Δ = 4. Mesuré en descriptif sur 10⁴ échantillons exacts, le TV par bassin contre la vérité de Voronoï vaut **0,066 à d = 32, Δ = 4** (0,010 à d = 128). À d32, darting et MALA sont tous deux à ce niveau : H_saut_bassin n'y est pas informatif. Par Voronoï (descriptif), darting et MALA seul sont aussi au même niveau à S = 10 (0,011 contre 0,011) : la lecture « pas locaux » ne change pas. À d32, l'IS n'est d'ailleurs qu'à moitié effondré (ESS 7·10⁻³, k̂ 0,66), et l'AS brut est déjà à 0,036.
- **Ce que montrent d128 et d256** : sur des modes gaussiens à forme apprise (Q vrai) avec un sampler trop large, les sauts entre minima ramènent les proportions au plancher. Le MALA seul ne corrige presque rien (0,08 à 0,10), et l'IS de chemin est inutilisable (k̂ ≥ 1). C'est le cas « Q vrai, I vrai » de l'étape 4, jamais comparé jusqu'ici.
- Un seul contrôleur par cible (une graine d'entraînement) ; les 3 jeux mesurent la variabilité de l'échantillonnage à contrôleur fixé, pas la variabilité de l'entraînement.


## Étape A : robustesse du darting post hoc dans le régime où l'IS s'effondre (autres instantanés) et variance de log W par bassin (2026-11-23)

Règle écrite et commitée avant tout calcul : `rules_darting_large_sigma_snapshots.md` (bd7743c). Même protocole et même script que la section précédente (`scripts/darting_large_sigma.py`). Résultats `figures/darting_large_sigma/` (`verdict_snapshots.json`, `lwvar.json`). CPU, aucune location ; toutes les exécutions durent moins de 7 s.

Moyennes sur 3 jeux de 10⁴ points terminaux (± écart-type entre jeux), plancher 0,0100.
| d | époque | AS brut Voronoï / bassin | IS réuni (3 × 10⁴) : TV, ESS, k̂ | darting S = 5 (Voronoï) | darting S = 10 Voronoï / bassin | MALA seul S = 10 Voronoï / bassin | TV_eq S = 5 / 10 | G H |
|---|---|---|---|---|---|---|---|---|
| 128 | 750 | 0.197 / 0.113 | 0.296, 1.4e-04, 1.07 | 0.0098 ± 0.0007 | 0.0076 ± 0.0011 / 0.0070 | 0.089 / 0.085 | 0.104 / 0.076 | ✓ ✓ |
| 128 | 1000 | 0.211 / 0.125 | 0.106, 1.1e-03, 1.05 | 0.0100 ± 0.0027 | 0.0091 ± 0.0012 / 0.0106 | 0.103 / 0.100 | 0.037 / 0.027 | ✓ ✓ |
| 128 | 1250 | 0.198 / 0.113 | 0.112, 3.2e-03, 0.89 | 0.0083 ± 0.0008 | 0.0104 ± 0.0013 / 0.0108 | 0.088 / 0.084 | 0.022 / 0.016 | ✓ ✓ |
| 128 | 1500 | 0.194 / 0.110 | 0.127, 2.7e-03, 0.97 | 0.0115 ± 0.0047 | 0.0076 ± 0.0030 / 0.0090 | 0.087 / 0.084 | 0.024 / 0.017 | ✓ ✓ |
| 256 | 750 | 0.185 / 0.097 | 0.771, 3.5e-05, 3.48 | 0.0098 ± 0.0041 | 0.0081 ± 0.0021 / 0.0093 | 0.086 / 0.082 | 0.210 / 0.153 | ✓ ✓ |
| 256 | 1000 | 0.182 / 0.093 | 0.387, 1.7e-04, 3.19 | 0.0106 ± 0.0062 | 0.0095 ± 0.0032 / 0.0127 | 0.087 / 0.082 | 0.096 / 0.070 | ✓ ✓ |
| 256 | 1250 | 0.177 / 0.087 | 0.519, 8.0e-05, 3.38 | 0.0111 ± 0.0035 | 0.0116 ± 0.0023 / 0.0146 | 0.078 / 0.074 | 0.139 / 0.101 | ✓ ✓ |
| 256 | 1500 | 0.194 / 0.102 | 0.487, 1.1e-04, 3.40 | 0.0119 ± 0.0070 | 0.0091 ± 0.0022 / 0.0122 | 0.097 / 0.092 | 0.120 / 0.087 | ✓ ✓ |

**Règle A1 : VRAIE.** G et H_saut_bassin sont vrais sur **3 instantanés sur 3** (750, 1000, 1250) à d = 128 comme à d = 256, et aussi à 1500. Le darting post hoc ramène le TV au plancher (0,007 à 0,015 à S = 10) à toutes les époques, avec un MALA seul à 0,07–0,10 et un IS de chemin dont l'ESS va de 3·10⁻⁵ à 3·10⁻³. Le résultat de l'époque 1500 est robuste d'un instantané à l'autre. Réserve inchangée : G repose sur la borne LB, retenue à chaque S.

**Variance de log W par bassin (descriptif).** Pour × 0,75, ce sont les bassins des jeux ci-dessus ; pour × 0,5, les log w de l'étape 4 avec la cellule de Voronoï comme bassin (déclaré).
| run, époque | Var(log W) | intra-bassin | inter-bassins (part) | ESS(W^0,25) | ESS(W^0,5) | ESS(W) |
|---|---|---|---|---|---|---|
| x0.75_d128_ep750 | 19.37 | 19.35 | 0.026 (0.1 %) | 0.390 | 2.87e-02 | 1.4e-04 |
| x0.75_d128_ep1000 | 18.33 | 18.30 | 0.031 (0.2 %) | 0.414 | 5.59e-02 | 1.1e-03 |
| x0.75_d128_ep1250 | 14.57 | 14.53 | 0.034 (0.2 %) | 0.470 | 7.96e-02 | 3.2e-03 |
| x0.75_d128_ep1500 | 13.89 | 13.86 | 0.032 (0.2 %) | 0.491 | 8.41e-02 | 2.7e-03 |
| x0.5_d128_ep250 | 1.33 | 1.10 | 0.229 (17.2 %) | 0.915 | 6.74e-01 | 6.4e-02 |
| x0.5_d128_ep500 | 1.35 | 0.91 | 0.441 (32.6 %) | 0.906 | 6.36e-01 | 1.4e-01 |
| x0.5_d128_ep750 | 1.33 | 0.88 | 0.443 (33.4 %) | 0.910 | 6.53e-01 | 1.6e-01 |
| x0.5_d128_ep1000 | 1.28 | 0.86 | 0.416 (32.6 %) | 0.914 | 6.52e-01 | 3.5e-02 |
| x0.5_d128_ep1250 | 1.24 | 0.88 | 0.367 (29.5 %) | 0.919 | 6.95e-01 | 2.4e-01 |
| x0.5_d128_ep1500 | 1.28 | 0.84 | 0.433 (33.9 %) | 0.914 | 6.75e-01 | 1.9e-01 |
| x0.75_d256_ep750 | 154.46 | 154.42 | 0.040 (0.0 %) | 0.005 | 8.39e-05 | 3.5e-05 |
| x0.75_d256_ep1000 | 149.76 | 149.73 | 0.031 (0.0 %) | 0.010 | 5.26e-04 | 1.7e-04 |
| x0.75_d256_ep1250 | 150.39 | 150.34 | 0.050 (0.0 %) | 0.009 | 3.76e-04 | 8.0e-05 |
| x0.75_d256_ep1500 | 148.14 | 148.09 | 0.052 (0.0 %) | 0.013 | 6.47e-04 | 1.1e-04 |
| x0.5_d256_ep250 | 1.66 | 1.61 | 0.056 (3.3 %) | 0.903 | 6.64e-01 | 2.0e-01 |
| x0.5_d256_ep500 | 1.74 | 1.51 | 0.228 (13.1 %) | 0.895 | 6.35e-01 | 1.5e-01 |
| x0.5_d256_ep750 | 1.74 | 1.43 | 0.316 (18.1 %) | 0.892 | 6.21e-01 | 1.4e-01 |
| x0.5_d256_ep1000 | 1.67 | 1.44 | 0.225 (13.5 %) | 0.899 | 6.50e-01 | 1.9e-01 |
| x0.5_d256_ep1250 | 1.65 | 1.42 | 0.227 (13.8 %) | 0.901 | 6.55e-01 | 1.9e-01 |
| x0.5_d256_ep1500 | 1.68 | 1.41 | 0.277 (16.5 %) | 0.898 | 6.43e-01 | 1.6e-01 |

Lecture (objection de Jensen) :
- **À × 0,75, la variance de log W est presque entièrement intra-bassin** : 14 à 19 nats² à d = 128, environ 150 à d = 256, avec **0,0 à 0,2 % entre bassins** (0,03 à 0,05 nat²). L'information sur les masses que portent les poids (l'écart moyen de log W entre bassins) est donc noyée sous un bruit environ 500 fois plus grand.
- À α = 0,5, l'ESS de W^α vaut encore 0,03 à 0,08 à d = 128, mais 10⁻⁴ à 6·10⁻⁴ à d = 256.
- **À × 0,5**, la variance est de 1,2 à 1,7 nat², dont 13 à 34 % entre bassins, avec un ESS de W^0,5 de 0,62 à 0,69.
- Une correction des masses par poids de trajectoire tempérés ne peut donc marcher que dans le second régime, ce qui est la prédiction écrite de l'étape B.

## Étape B : correction des masses à l'entraînement, darting (sans poids) contre IS tempéré (poids de chemin) (2026-11-23)

Règle écrite et commitée avant la location : `rules_train_weightfree.md` (8592ed2). Code :
- `AdjointVETemperedISMatcher` (log W stocké à la génération, normalisation sur tout le buffer ; `tests/test_tempered_is_matcher.py` T1–T4 réussis) ;
- poids de perte dans `train_loop.py` ;
- buffer initial avec log W (`scripts/wf_make_start.py`) ;
- script distant `scripts/vast_train_weightfree_remote.sh` ; analyse locale `scripts/wf_analyse.py`.

Résultats : `figures/train_weightfree/` (analysis_*.json, verdict.json, final_ess_d128.json ; journaux, configurations et contrôleurs des époques 100 à 500 et finaux des 9 runs). Les échantillons d'évaluation restent hors dépôt (`results/train_weightfree_runs/`).

**Coût : 0,063 $** (RTX 3070, offre 57456548, 0,1339 $/h, 28,3 min ; estimation 0,06 à 0,09 $, plafond 0,20 $). Instance détruite dès le rapatriement ; aucun arrêt déclenché, aucun NaN.
- Vitesses (vague 1, 6 runs) : (a) 0,8 s par époque, (t) 1,2–1,3 s, (d) 1,9 s.
- **Énergies par mise à jour de gradient** (code) :
  - (a) 5,12 gradients ;
  - (t) 5,12 gradients + 5,12 énergies (fin de chemin) ;
  - (d) 5,12 gradients + **(M + 1) × 512 = 4608 énergies**, calculées à chaque échantillon de régression, sans cache, plus la découverte des minima toutes les 25 époques.

TV brut sans poids (moyenne ± écart-type sur les évaluations des époques 350, 400 et 450 ; 10⁴ échantillons chacune), Voronoï / bassin ; couverture 8/8 partout ; Q1 = rapport de distance à π.
| cellule | (a) AS | (t) IS tempéré α = 0,5 | (d) darting γ = 1,75 | Q1 (a / t / d) |
|---|---|---|---|---|
| (8, 8, 8) × 0,5 | 0,182 ± 0,009 / 0,182 | **0,034 ± 0,015** / 0,034 | **0,043 ± 0,012** / 0,043 | 1,017 / 1,017 / 1,015 |
| (32, 8, 6) × 0,5 | 0,041 ± 0,003 / 0,041 | 0,034 ± 0,002 / 0,037 | 0,042 ± 0,008 / 0,043 | 1,014 / 1,014 / 1,014 |
| (128, 8, 6) × 0,75 | 0,196 ± 0,009 / 0,106 | **0,113 ± 0,007** / 0,080 | 0,190 ± 0,009 / 0,100 | 1,046 / 1,057 / 1,046 |

### Verdicts tels qu'écrits
- **F1 FAUX** : vrai en (8, 8, 8) (0,043 ≤ 0,5 × 0,182), faux en (32, 8, 6) (0,042 > 0,5 × 0,041 ; l'AS y est déjà à 0,04).
- **F2 FAUX** : en famille 2, le darting à l'entraînement ne corrige presque rien (0,190 contre 0,196 pour AS ; par bassin 0,100 contre 0,106), et il est moins bon que l'IS tempéré (0,113).
- **F3 VRAI** dans les 3 cellules (Q1 de (t) 1,057 contre 1,046 à d = 128, dans la marge de 10 %).
- **Lecture fixée d'avance : F2 faux.** La correction sans poids ne marche pas à l'entraînement dans le régime où elle marche après coup (A1). La contribution prévue n'est pas établie.

### Prédictions écrites
- F1 vraie à 40 % ✗ (fausse) ; F2 vraie à 55 % ✗ (fausse) ; F3 vraie à 70 % ✓.
- **« (t) se dégrade en famille 2 » : FAUX.** (t) est le meilleur bras à d = 128 (0,113 contre 0,196 par Voronoï ; 0,080 contre 0,106 par bassin), malgré un ESS de départ de W^0,5 de 0,089.

### Faits descriptifs (sans changer les verdicts)
- **Dose réelle du darting à l'entraînement.** Rapport ‖terme‖/‖cible‖ moyen par plage de κ ([0 ; 0,1] / [0,1 ; 0,7] / [0,7 ; 1]) :
  - d8 : 0,35 / 0,12 / 0,0007 ;
  - d32 : 0,60 / 0,10 / 0,0000 ;
  - d128 : 0,15 / 0,06 / 0,0001.
  Les κ > 0,7 font 92 % des instants de régression. 8 minima sur 8 découverts, aucun terme non fini.
  **Hypothèse, non testée** : à d = 128, le terme est trop faible aux petits κ pour déplacer les masses en 500 époques ; et l'excès de TV de Voronoï y tient surtout à la largeur des échantillons (rapport Q1 1,046), que le darting post hoc corrige par ses pas MALA et que le terme d'entraînement ne touche pas.
- **Ce que fait l'IS tempéré** (hypothèse cohérente avec l'étape A : 99,8 % de la variance de log W est intra-bassin à × 0,75) : les poids corrigent surtout la largeur dans chaque bassin, ce qui améliore le TV de Voronoï, et le TV par bassin aussi (0,106 → 0,080).
- **Mais (t) rend ses poids de chemin moins utilisables après coup.** ESS de l'IS de chemin des contrôleurs finaux à d = 128, sur 10⁴ trajectoires, CPU : W 0,0044 (a), 0,0015 (t), 0,0049 (d) ; W^0,5 0,114 / 0,065 / 0,117.
- **Défaut de journalisation** : les statistiques ESS du matcher (t) n'ont été imprimées qu'une fois, au départ (ESS de W^0,5 0,089 à d128), d'où la mesure finale ci-dessus.
- **Famille 1** : à (8, 8, 8), les deux corrections divisent le TV d'AS par 4 à 5. À (32, 8, 6), AS est déjà à 4 fois le plancher et aucun bras ne fait mieux de façon nette.

Réserves : une graine d'entraînement, 500 époques de reprise, une valeur de α et une de γ ; les moyennes sur 3 évaluations donnent la variabilité entre évaluations, pas entre graines.
