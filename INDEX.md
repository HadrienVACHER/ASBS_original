# Index du projet : variable de contrôle de symétrie pour Adjoint Sampling

Ce fichier est une carte. Le journal détaillé (chiffres, règles, verdicts, écarts) est [RESULTS.md](RESULTS.md) ; rien n'y est réécrit, les corrections y sont ajoutées en fin de fichier sous forme d'amendements.
Branches : `barker` (travail Barker, sections 1 à 4 de RESULTS.md), `div` (division, puis tous les tests DW-4 récents ; environ 200 commits d'avance sur `main`, poussée). Les branches `cv`, `dang`, `mh_test`, `test`, `testt` ne sont pas revues ici.
Conventions de travail : règles pré-enregistrées dans l'en-tête du script ou dans un fichier `rules_*.md`, commitées avant le run ; seuils jamais relâchés après coup ; une seule graine (0) sauf mention.

## 1. Le fil des conclusions

| Cible | Méthode | Verdict | Où |
|---|---|---|---|
| GMM isotrope (d = 24, 36) | Barker (orbite, γ = 3, centres par seuil) | **Marche** : −86 % (d = 24, 0,235 → 0,034) et −68 % (d = 36, 0,351 → 0,112) sur mean\|log R̄\|, 3 graines, meilleur à chaque graine | RESULTS §1 |
| GMM isotrope d = 36, graine 0 | Division (Gelfand-Dey) | Égale Barker : 0,106 contre 0,117 (AS 0,389) ; pas de succès au seuil fixé (0,058). Un premier run avait échoué (NaN à t = 0 exactement, corrigé) | RESULTS §5 |
| GMM 1D à deux modes (poids 0,1 / 0,9) | AS contre AS + Barker, + division (et + MH) (relance locale, 1000 époques, graine 0) | AS garde 0,48 de masse sur le mode de poids 0,1 (W&B : 0,41 à 0,64) ; Barker 0,09, division 0,11, MH 0,12 : cible atteinte en ~100 époques ; les paramètres de la division y sont exacts (meilleur cas) | RESULTS §« Démo 1D à deux modes », `figures/demo1d_*.png` |
| GMM 1D, modes à ±5 (poids 0,1 / 0,9) | Même comparaison, cible plus large (relance locale) | AS : 0,63 ± 0,05 sur le mode de poids 0,1 (échantillons finaux 0,48, soit 50/50) ; Barker 0,09 et division 0,12 | RESULTS §« Démo 1D à deux modes », `figures/demo1d_wide_*.png` |
| GMM 1D, modes à ±1 / ±3 / ±5 | AS seul : proportion du mode de poids 0,1 selon la distance entre les modes (relances locales) | ±1 : 0,125 ± 0,004 (pas de cécité) ; ±3 : 0,48 ± 0,06 ; ±5 : 0,63 ± 0,05 : l'erreur croît avec la distance | RESULTS §« Démo 1D à deux modes », `figures/demo1d_narrow_*.png`, `demo1d_separation.png` |
| DW-4 | Darting (un run, continuation 500 époques, graine 1, Vast.ai) | S contre p_PT : 0,137 (R) / 0,122 (Q), sous les deux runs AS (0,367 / 0,210 ; 0,363 / 0,203) ; meilleur que les deux AS sur les 3 petits bassins ; mais bruit entre graines AS 0,16 : indicatif, pas un verdict. Le terme est négligeable à κ > 0,7 (0,4 % du gradient) | RESULTS §« Run de continuation avec le noyau de darting », `rules_darting_run.md` |
| GMM isotrope | Division : robustesse aux masses | Positif : une erreur de ±0,5 nat sur les masses garde la direction de la correction (cos_corr 0,90 à 0,97) | RESULTS §5 |
| GMM hétérogène | Barker | Aucune géométrie informative : abandonné | RESULTS §2 |
| ManyWell-32 | Barker | Fermé : AS ne visite pas les puits légers de la moitié des coordonnées, Barker ne l'améliore pas | RESULTS §3 |
| DW-4 | Barker (24 réflexions) | Pas d'effet démontré, confirmé contre la vraie cible p_PT (continuation pairée et règle des instantanés) | RESULTS §4, « Réévaluation contre p_PT » |
| DW-4 | Division (4 variantes : alignée, correction de fuite, EM, tronquée) | Échec : SD intra-bassin de log u de 5 à 8 nats, k̂ de Pareto ≫ 0,5 dans les petits bassins ; la queue lointaine est réfutée | RESULTS §5 |
| GMM | Coefficient appris sur le terme de division | Rien : au plus 5 % de variance gagnée, et la correction est rétrécie | RESULTS §5 |
| DW-4 | Noyau de darting (minimum à minimum) | Validé hors ligne (exactitude, flux sortant 15 % et 13 % contre ~1 % pour les réflexions, signal AS, T1′ sur échantillons PT) ; **jamais testé en entraînement** | RESULTS §5 (entrées darting, avant « Methodology notes ») et §« Amendement : référence DW-4 » (T1′) |
| DW-4 | Correction a posteriori | Marche : IS d'endpoint avec proposition englobante, erreur sommée 0,09 à 0,12 contre p_PT (AS brut 2,36) ; IS de chemin 0,59 ; MH de chemin 0,89 | RESULTS §« Amendement » |
| DW-4 | **Référence DEM** | **Biaisée** sur les deux bassins dominants (≈ ±0,15 en log, z ≈ ±12) ; les splits train et val aussi | RESULTS §« Amendement : référence DW-4 » |

## 2. Faits à ne pas perdre

- **Vérité DW-4 :** p_PT (parallel tempering indépendant, valide) = triangle + pendant 0,4127, chemin de 2 0,4792, deux paires disjointes 0,0644, K4 moins une arête 0,0374, étoile 0,0063 (SE 0,002 sur les deux premiers). La référence donne 0,4762 / 0,4099 / 0,0592 / 0,0490 / 0,0057. Fichier : `results/reference_check_pt.json` ; recalculable en 52 s avec `scripts/dw4_pt.py`.
- **Les anciens verdicts DW-4 ont été rendus contre la référence** : ils restent écrits tels quels ; la réévaluation contre p_PT ne les change pas (« Réévaluation contre p_PT »). Mais l'ampleur des erreurs d'AS était mal répartie entre bassins : AS est correct sur triangle + pendant, trop bas sur le chemin de 2 (−0,23 en log) et trop haut sur les trois petits bassins (+0,5 à +1,0).
- **`energy_w2` ne voit pas le biais** de la référence, `eq_w2` le voit (mais son plancher à n = 2000 est de 0,31, du même ordre que les valeurs officielles d'AS) : RESULTS §« W2 et biais de la référence ».
- **Deux runs AS DW-4 :** ed8bb67 (échantillons dans `results/dw4_e1500_ed8bb67_samples.pt`, **pas de contrôleur** : son dossier a été écrasé) et 94252 / c166b5d (échantillons et contrôleur `dw4_e1500_94252_*.pt`). Le dossier `dw4_as_states_e1500/seed0` du cluster est le run 94252.
- **Énergie DW-4 du dépôt :** E = Σ_{i<j} [0,9 (d−4)⁴ − 4 (d−4)²], τ = 1, distance lissée √(r² + 1e-8). Les articles DEM ont un facteur 1/(2τ) que le code n'a pas ; non résolu (RESULTS §« Amendement », provenance). La méthode de génération des données DEM n'est pas documentée.
- **Portabilité :** les (μ, Σ) de `scripts/dw4_division_trunc_params.json` sont dans une base du sous-espace de centre de masse nul qui dépendait de la machine (`eigh`) ; la base est maintenant déterministe (`dw4_division_step0.basis`), le JSON reste le registre des runs du cluster.
- **Limites générales :** une seule graine pour la plupart des tests DW-4 et division ; le résultat de la division à d = 36 valide l'implémentation, pas la généralité.

## 3. Règles pré-enregistrées (fichiers)

- `rules_reference_check.md` (commit 3d5b062) : PT indépendant, origine de l'écart, T1′ du darting, recalcul des erreurs.
- `rules_reeval_pt.md` (commit 4a1342f) : réévaluation des règles Barker et continuation contre p_PT.
- En-têtes des scripts : `dw4_barker_rule.py`, `dw4_continuation_rule.py`, `dw4_darting_test.py`, `dw4_division_*.py`, `dw4_is_inflate.py`, `dw4_path_weights.py`, `dw4_w2_check.py`, `division_*.py`, `manywell_*_rule.py`, `hetero_geometry_rule.py`.

## 4. Reproduire les tests récents (CPU, local)

```bash
python scripts/dw4_pt.py                                   # PT, 52 s -> results/pt_samples_T1.pt
python scripts/dw4_reference_check.py                      # validité, p_PT, verdict sur la référence, énergies par bassin
python scripts/dw4_darting_T1prime.py                      # T1′ du darting sur les états PT (~80 s)
python scripts/dw4_reference_rows.py                       # tableau des erreurs contre p_PT (~2,5 min)
python scripts/dw4_reeval_pt.py                            # réévaluation continuation et Barker (W&B en lecture)
python scripts/dw4_w2_check.py                             # W2 contre PT, plancher (~4,5 min)
python scripts/dw4_is_inflate.py --samples results/dw4_e1500_ed8bb67_samples.pt   # IS/IMH englobant (~2 min)
python scripts/dw4_path_weights.py                         # poids de chemin du contrôleur 94252 (~90 s)
python scripts/dw4_darting_test.py --ref data/test_split_DW4.npy --as results/dw4_e1500_ed8bb67_samples.pt --n-ref 3000 --n-as 5000 --nu1 4.0 --seed 0
```

## 5. Catalogue des scripts (`scripts/`, 58 scripts Python, 47 PBS, 7 shell)

- **GMM et Barker :** `analyse_anneal.py`, `analyse_buffer_centres.py`, `analyse_capacity.py`, `plot_gmm_final.py`, `committed_mass.py`, `diagnose_mode_flux.py`, `exact_iteration.py`, `hetero_geometry_rule.py`, `analyse_hetero.py`.
- **Division sur la GMM :** `analyse_division_gmm.py`, `division_kappa_max.py`, `division_diag.py`, `division_t_edge.py`, `division_mass_perturb.py`, `division_beta_opt.py`.
- **ManyWell-32 :** `analyse_manywell.py`, `manywell_rule.py`, `manywell_*_rule.py`, `manywell_posthoc.py`, `manywell_model_quality.py`, `manywell_pretrain_*.py`, `analyse_correction.py`.
- **IMH / MH sur les endpoints (GMM) :** `imh_demo.py`, `imh_fitted_demo.py`, `imh_fitted_gmm.py`, `imh_refined_gmm.py`, `as_then_imh*.pbs`.
- **DW-4, runs AS et Barker :** `analyse_dw4.py`, `analyse_dw4_quench.py`, `dw4_barker_rule.py`, `dw4_continuation_rule.py`, `dw4_reflection_rb.py`, `dw4_reflection_test3.py`, `dw4_flux_kappa.py`, `dw4_lumped_chain.py` (`dw4_barker_diagnostic.py` : abandonné).
- **DW-4, division :** `dw4_division_step0.py`, `dw4_division_b.py`, `dw4_division_b_baseline.py`, `dw4_division_em.py`, `dw4_division_trunc.py`, `dw4_division_top10.py`.
- **DW-4, vérité et post hoc :** `dw4_pt.py`, `dw4_reference_check.py`, `dw4_reference_rows.py`, `dw4_reeval_pt.py`, `dw4_dem_splits.py`, `dw4_w2_check.py`, `dw4_is_inflate.py`, `dw4_path_weights.py`.
- **DW-4, darting :** `dw4_darting_test.py`, `dw4_darting_is_balance.py`, `dw4_darting_T1prime.py`.
- **Autres :** `bench_sampler.py`, `check_fast_dw4.py`, `ab_compile_report.py` (performance de l'entraînement).
- **Tests unitaires** (`tests/`) : `test_barker_matcher.py`, `test_mh_matcher.py`, `test_division_cv.py`, `test_division_matcher.py`, `test_dw4_*.py`, `test_manywell.py`, `test_girsanov.py`, `test_imh_buffer.py`, `test_threshold_centres.py` ; les tests de zéro moyen exact sont la garantie de validité des termes.

## 6. Données et artefacts : ce qui n'est pas dans git

`results/*` est ignoré par git. Copies locales actuelles : `dw4_e1500_ed8bb67_samples.pt`, `dw4_e1500_94252_samples.pt`, `dw4_e1500_94252_controller.pt`, `dw4_as_barker_e1500_seed0_samples_final.pt`, `pt_samples_T1.pt` (recalculable), `reference_check_pt.json`.
Copie du cluster : `~/Desktop/NTU/cluster_backup/ASBS_original/` (55 Mo : dossiers de runs DW-4, sorties de job `.o*`). **Les échantillons et contrôleurs DW-4 n'existent nulle part ailleurs que sur ce Mac** (le cluster n'est plus accessible) : à sauvegarder hors de la machine.
W&B : projets `CCDS_NTU/as` (DW-4), `CCDS_NTU/adjoint_sampler_cv` (GMM, ManyWell) et `CCDS_NTU/div` (runs de division sur la GMM) ; seulement les métriques, pas les échantillons.
