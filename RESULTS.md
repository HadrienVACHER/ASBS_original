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
