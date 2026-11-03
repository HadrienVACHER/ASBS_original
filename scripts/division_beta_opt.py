""" Division CV: is a LEARNED coefficient on the term (scalar beta(kappa) or matrix M(kappa), any function of (x_t, t) keeps the exact zero mean under pi) better than the derived coefficient 1?
    Offline, GMM unit-test setting, CPU, no training, no matcher. The coefficient that minimises the variance of the regression target is the classical one; the derived coefficient 1 is the one that makes
    E_q[term | x_t] the exact Tweedie correction. Under q != pi these two objectives can disagree: variance minimisation can shrink the correction that removes the bias of q. This script measures both.

      python scripts/division_beta_opt.py [--draws 600]

    Setting (same as scripts/division_mass_perturb.py, fixed before running): isotropic GMM D = K = 12 (frozen: spread, loguniform ratio 20, radius 4.25, seed 0), sigma_max = 2 coord_std(D, 4.25), sigma_min = 0.001, shrink 0.8,
    oracle shapes and centres, masses L_k = -log w_k + delta_k: control (delta = 0) and 3 draws of delta_k = +-0.5 (seeds 0, 1, 2); buffer q = w^0.3 normalised (wrong weights).
    kappa in {0.05, 0.1, 0.2, 0.3, 0.5, 0.7}; 400 x_t ~ q_t per kappa, and for each x_t 600 exact posterior draws of x1 under q AND 600 under pi (same x_t).
    Regression target of the matcher: F_beta = f + beta term, f = -adjoint1 = -(grad E + base score)(x1) (ScoreGradTermCost, as in prepare_target), term = division_term (version = the run's masses).
    Per kappa and run, the 400 x_t are split in two halves, coefficients are FITTED on half A (x_t index < 200) and EVALUATED on half B:
      scalar   beta_s = - sum <f_c, term_c> / sum ||term_c||^2           (f_c, term_c centred within x_t; the minimiser of the within-x_t variance of f + beta term)
      matrix   M      = - C_ft C_tt^{-1} (D x D, ridge 1e-6 tr(C_tt) / D)    (minimiser of the within-x_t variance of f + M term)
    Criteria on half B:
      V(c)  = mean over (x_t, draws) of || (f - mean f) + c (term - mean term) ||^2        (regression noise variance; c in {0, 1, beta_s, M})
      B(c)  = mean over x_t of || E_q[f + c term | x_t] - E_pi[f | x_t] ||^2 minus its Monte-Carlo noise floor (estimator variances / 600): the bias of the fixed point (0 = the correct control)
      Bc(c) = mean over x_t of || E_q[c term | x_t] - c*(x_t) ||^2 minus its Monte-Carlo noise floor, c*(x_t) = (M* - M^q) / nu_{1|t} the exact Tweedie correction of the Gaussian mixture (no noise on the target side);
              Bc(0) = mean ||c*||^2. PRIMARY bias metric (the one used by the validity check and the verdict).
    AMENDMENT of the metric, declared BEFORE the full run: a 60-draw smoke test (not a result) showed that B(c) above (target f compared with E_pi[f | x_t]) has no power, because the true difference E_q[f|x_t] - E_pi[f|x_t]
    (B(0) ~ 0.02) is of the order of the Monte-Carlo noise floor; the bias is therefore measured on the correction itself (Bc), and B(c) is kept as a secondary column.
    VALIDITY (positive control of the metric): control run (delta = 0), c = 1: Bc(1) / Bc(0) < 0.1 at every kappa. If it fails, the bias analysis is invalid (reported as such, no verdict on the coefficient).
    PRE-REGISTERED VERDICT for a learned coefficient L in {beta_s, M} (all 4 runs, all 6 kappa):
      variance gain      : V(L) <= 0.9 V(1) at >= 4 of the 6 kappa in every run;
      bias not degraded  : Bc(L) <= 1.1 Bc(1) + 0.02 Bc(0) at every kappa in every run;
      FREE LUNCH = both; TRADE-OFF = variance gain but bias degraded somewhere; NOTHING = no variance gain.
    Also reported (no rule): beta_s itself, and Bc/Bc(0) for the constant coefficients 0.5 and 1.5, and the secondary B(c)/B(0).
"""
import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import division_mass_perturb as mp  # noqa: E402
from adjoint_samplers.components.sde import VESDE  # noqa: E402
from adjoint_samplers.components.term_cost import ScoreGradTermCost  # noqa: E402
from adjoint_samplers.energies.dist_energy import DistEnergy  # noqa: E402
from adjoint_samplers.utils.dist_utils import GMM, Delta  # noqa: E402
from adjoint_samplers.utils.division_cv import division_term, make_params  # noqa: E402

KAPPAS = mp.KAPPAS


def post_draws(mu, w, xt, kap, v, n, gen):
    pi, m, s2 = mp.post(None, mu, w, xt, kap, v)
    D = mu.shape[1]
    j = torch.multinomial(pi, n, replacement=True, generator=gen)
    mj = torch.gather(m, 1, j[..., None].expand(-1, -1, D))
    return mj + math.sqrt(s2) * torch.randn(len(xt), n, D, dtype=torch.float64, generator=gen)


def f_of(x1d, termcost):
    T, N, D = x1d.shape
    return -termcost(x1d.reshape(-1, D).float()).double().reshape(T, N, D)


def nv(a, c=None):
    """ within-x_t sum of component variances of a (T, N, D): (T,) """
    return a.var(1, unbiased=True).sum(-1)


def evaluate(f, tm, fpi, cstar):
    """ fit on x_t < T/2, evaluate on the rest; returns dict name -> (V, B) and beta_s """
    T, N, D = f.shape
    h = T // 2
    mf, mt, mpi = f.mean(1), tm.mean(1), fpi.mean(1)
    fc, tc = f - mf[:, None], tm - mt[:, None]
    n = h * N
    Cft = torch.einsum("tnd,tne->de", fc[:h], tc[:h]) / n
    Ctt = torch.einsum("tnd,tne->de", tc[:h], tc[:h]) / n
    beta = float(-torch.trace(Cft) / torch.trace(Ctt))
    M = -Cft @ torch.linalg.inv(Ctt + 1e-6 * torch.trace(Ctt) / D * torch.eye(D, dtype=torch.float64))
    coefs = {"0": lambda z: 0 * z, "0.5": lambda z: 0.5 * z, "1": lambda z: z, "1.5": lambda z: 1.5 * z, "beta_s": lambda z: beta * z, "M": lambda z: z @ M.T}
    out = {}
    for name, c in coefs.items():
        Fc = fc[h:] + c(tc[h:])                                                      # centred target of half B
        V = float((Fc ** 2).sum(-1).mean())
        d = mf[h:] + c(mt[h:]) - mpi[h:]
        floor = float(((Fc ** 2).sum(-1).sum(1) / (N - 1) / N + nv(fpi[h:]) / N).mean())          # variance of the two conditional-mean estimators
        dc = c(mt[h:]) - cstar[h:]
        floorc = float((c(tc[h:]) ** 2).sum(-1).sum(1).div(N - 1).div(N).mean())
        out[name] = (V, float((d ** 2).sum(-1).mean()) - floor, float((dc ** 2).sum(-1).mean()) - floorc)
    out["c0"] = float((cstar[h:] ** 2).sum(-1).mean())
    return out, beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=600)
    a = ap.parse_args()
    N, D = a.draws, 12
    g = GMM(dim=D, K=D, seed=0, layout="spread", weight_law="loguniform", weight_ratio=20.0, radius=4.25)
    energy = DistEnergy(g)
    mu, w = g.means.double(), g.weights.double()
    wq = w ** mp.POWER
    wq = wq / wq.sum()
    ref = VESDE(sigma_min=0.001, sigma_max=2 * GMM.coord_std(D, 4.25))
    nu1 = ref.total_var
    termcost = ScoreGradTermCost(source=Delta(dim=D), ref_sde=ref, energy=energy)
    cov = torch.eye(D, dtype=torch.float64).expand(D, D, D)
    runs = {"control": torch.zeros(D, dtype=torch.float64)}
    for s in range(3):
        gd = torch.Generator().manual_seed(s)
        runs[f"draw {s}"] = 0.5 * (2 * torch.randint(0, 2, (D,), generator=gd).double() - 1)
    pars = {n: make_params(mu, cov, -torch.log(w) + d, 0.8) for n, d in runs.items()}
    mp.N_DRAW = N
    res = {n: {} for n in runs}
    for kap in KAPPAS:
        v = nu1 * kap * (1 - kap)
        nu1t = nu1 * (1 - kap)
        gen = torch.Generator().manual_seed(1000 + int(1000 * kap))
        xt, x1q = mp.draw_xt_x1(g, mu, wq, kap, v, gen)
        x1p = post_draws(mu, w, xt, kap, v, N, gen)
        fq, fp = f_of(x1q, termcost), f_of(x1p, termcost)
        T = len(xt)
        xtr = xt[:, None, :].expand(-1, N, -1).reshape(T * N, -1)
        E1 = energy.eval(x1q.reshape(T * N, -1).float()).double()
        pi_s, m_, _ = mp.post(None, mu, w, xt, kap, v)
        pi_q, _, _ = mp.post(None, mu, wq, xt, kap, v)
        cstar = ((pi_s - pi_q)[..., None] * m_).sum(1) / nu1t
        for n, par in pars.items():
            tm = division_term(xtr, x1q.reshape(T * N, -1), E1, kap, v, nu1t, par).reshape(T, N, D)
            res[n][kap] = evaluate(fq, tm, fp, cstar)
    print(f"D = K = {D}; {len(xt)} x_t x {N} draws per kappa and per law (q: w^{mp.POWER}; pi); fit on half A, evaluation on half B")
    valid = True
    gain_ok, bias_ok = {"beta_s": True, "M": True}, {"beta_s": True, "M": True}
    for n in runs:
        print(f"\n== {n}: V(c)/V(0) for c = 1, beta_s, M | Bc(c)/Bc(0) for c = 0.5, 1, 1.5, beta_s, M (primary) | B(1)/B(0), B(0) (secondary, f-based)")
        print(f"{'kappa':>6} {'beta_s':>8} | {'V(1)':>7}{'V(bs)':>7}{'V(M)':>7} | {'Bc(.5)':>7}{'Bc(1)':>7}{'Bc(1.5)':>8}{'Bc(bs)':>7}{'Bc(M)':>7}  {'Bc(0)':>7} | {'B(1)':>7}{'B(0)':>8}")
        ngain = {"beta_s": 0, "M": 0}
        for kap in KAPPAS:
            o, beta = res[n][kap]
            V0, B0, C0 = o["0"][0], o["0"][1], o["c0"]
            rv = lambda k: o[k][0] / V0
            rc = lambda k: o[k][2] / C0
            print(f"{kap:6.2f} {beta:8.3f} | {rv('1'):7.3f}{rv('beta_s'):7.3f}{rv('M'):7.3f} | {rc('0.5'):7.3f}{rc('1'):7.3f}{rc('1.5'):8.3f}{rc('beta_s'):7.3f}{rc('M'):7.3f}  {C0:7.4f} | {o['1'][1] / B0 if abs(B0) > 1e-9 else float('nan'):7.3f}{B0:8.4f}")
            if n == "control" and not rc("1") < 0.1:
                valid = False
            for k in ("beta_s", "M"):
                ngain[k] += o[k][0] <= 0.9 * o["1"][0]
                if o[k][2] > 1.1 * o["1"][2] + 0.02 * C0:
                    bias_ok[k] = False
        for k in ngain:
            if ngain[k] < 4:
                gain_ok[k] = False
        print(f"  kappa with V(L) <= 0.9 V(1): beta_s {ngain['beta_s']}/6, M {ngain['M']}/6")
    print(f"\nVALIDITY (control, c = 1: Bc(1)/Bc(0) < 0.1 at every kappa): {valid}")
    if not valid:
        print("VERDICT: INVALID (the Bc metric does not reproduce the exact correction on the control; no verdict on the coefficient)")
    else:
        for k in ("beta_s", "M"):
            print(f"{k}: variance gain (>= 4/6 kappa, every run): {gain_ok[k]}; bias not degraded (every kappa and run): {bias_ok[k]}  =>  " + ("FREE LUNCH" if gain_ok[k] and bias_ok[k] else "TRADE-OFF" if gain_ok[k] else "NOTHING (no variance gain)"))


if __name__ == "__main__":
    main()
