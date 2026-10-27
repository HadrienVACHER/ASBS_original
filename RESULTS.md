# Results log — Barker control variate for Adjoint Sampling (branch `barker`)

One section per target: setup, pre-registered rule, verdict, key numbers, checks. W&B project `CCDS_NTU/adjoint_sampler_cv` (GMM, ManyWell) and `CCDS_NTU/as` (DW-4); runs carry their commit as tag `git:<hash>`.
Single seed (0) unless stated. Pre-registered thresholds were never relaxed after seeing a result.

---

## 1. GMM, isotropic (regular simplex, d = K)

- **Setup.** Spread (regular-simplex) layout, radius 4.25, log-uniform weights; big controller 256x6 for AS and Barker; seeds 0-2; d = 24 and 36; metrics time-averaged over the last 30 evaluations (per-mode ratios averaged first). Frozen method (commit `ec12405`, configs `gmm_as_final`, `gmm_as_barker_final`): Barker orbit control variate, gamma = 3, threshold centres (Barzilai-Borwein basin minima of the buffer, radius = 1 x median mode width, capacity 128, 12 denoising steps).
- **Rule.** gamma = the single global value minimising mean |log R̄| over d = 24 and 36 (3 seeds). Centres from the buffer only (zero mean stays exact).
- **Verdict.** Works.
- **Key numbers** (mean |log R̄|, Barker vs AS-big): d = 36: 0.112 vs 0.351 (-68 %), better at every seed; d = 24: 0.034 vs 0.235 (-86 %). Capacity ablation (d = 36, gamma = 2): -47 % with the small network, -55 % with the big one: the big network helps both and the Barker advantage persists.
- **Costs / caveats.** ~140-200 energy calls per gradient update (d = 24 / 36) against ~5 for AS. energy_w2 is worse with Barker (small net effect, dominated by a common 3-5 % mode-width excess that shrinks with capacity; +0.19 at d = 36 reported as is). gamma annealing was not adopted.
- **Checks.** Exact zero mean by orbit enumeration and under the exact posterior (tests/test_barker_matcher.py, test_mh_matcher.py, test_threshold_centres.py); oracle-centre intervention showed that k-means centres limited Barker, which motivated the threshold centres; the weight deficit of the simplex is monotone in the weight (no mode-specific effect).

## 2. GMM, heterogeneous

- **Setup.** `HeteroGMM` (unequal radii / covariances, affine orbit with the Jacobian in the weights, commit `3f672a7`).
- **Rule (AS alone, Barker never read before the choice).** A geometry is informative iff AS-big has light-quarter ratio < 0.6 (3-seed mean) AND modes_covered >= 0.9. Candidates G1-G4, extension E1-E4 (`a58e478`), last candidate F1 (`70a375f`), fixed stopping rule.
- **Verdict.** No informative regime found; hetero GMM dropped.
- **Key numbers.** v1 (radii 7-11): AS found 2-3 of 24 modes (results void, unread). G1-G4 and E1-E4: AS finds 8-18 % of the modes. F1 (isotropic simplex positions and weights, heterogeneous covariances): light quarter 0.785, modes_covered 1.0, i.e. too easy.
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

## Methodology notes

- Pair arms by seed: arms sharing a seed share their excursions, so seed-paired differences are the low-noise statistic.
- kappa_t > 0.7 for ~92 % of the regression times in this VE schedule.
- Compare time-averaged proportions (per-state, then TV), not snapshots: snapshot-to-snapshot TV moves by 0.03-0.08 (bootstrap SEs underestimate it).
- Read W&B history with the metric keys that are logged at every evaluation (a key logged less often silently drops the other rows); filter runs by `git_commit`.
- Hydra changes the working directory: paths given to `train.py` (`init_controller`, `init_buffer`) are resolved from the launch directory.
