# Results log — Barker control variate for Adjoint Sampling (branch `barker`)

One section per target: setup, pre-registered rule, verdict, key numbers, checks. W&B project `CCDS_NTU/adjoint_sampler_cv` (GMM, ManyWell) and `CCDS_NTU/as` (DW-4); runs carry their commit as tag `git:<hash>`.
Single seed (0) unless stated. Pre-registered thresholds were never relaxed after seeing a result.

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

## 5. Division (Gelfand-Dey) control variate, isotropic GMM (branch `div`; step 1 done, step 2 run done: numerical failure)

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

## Methodology notes

- Pair arms by seed: arms sharing a seed share their excursions, so seed-paired differences are the low-noise statistic.
- kappa_t > 0.7 for ~92 % of the regression times in this VE schedule.
- Compare time-averaged proportions (per-state, then TV), not snapshots: snapshot-to-snapshot TV moves by 0.03-0.08 (bootstrap SEs underestimate it).
- Read W&B history with the metric keys that are logged at every evaluation (a key logged less often silently drops the other rows); filter runs by `git_commit`.
- Hydra changes the working directory: paths given to `train.py` (`init_controller`, `init_buffer`) are resolved from the launch directory.
