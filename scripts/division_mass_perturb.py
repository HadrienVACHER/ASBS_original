""" Division CV, step 0 of the "fixed point does not depend on the masses L" hypothesis (GMM unit-test setting, CPU, no training, no matcher).

    Hypothesis (argument, untested): the correction E_q[term | x_t] keeps pointing in the direction of the exact Tweedie correction c* = (M* - M^q) / nu_{1|t}
    when the masses L_k of the frozen version are wrong, whereas a basin reweighting inherits the error of the masses directly.

    Setting (fixed here, before running): isotropic GMM D = K = 12 (frozen: layout spread, loguniform weights ratio 20, radius 4.25, seed 0, as tests/test_division_matcher.py),
    sigma_max = 2 coord_std(D, 4.25), sigma_min = 0.001, kappa = nu_t / nu_1, v = nu_1 kappa (1 - kappa), nu_{1|t} = nu_1 (1 - kappa), shrink 0.8.
    Oracle shapes and centres (Sigma_k = I, mu_k = GMM means); masses L_k = -log w_k + delta_k  (exp(-L_k) = Z w_k with Z = 1: E = -log p),
    delta_k = +-0.5 with equal probability (3 independent draws, seeds 0, 1, 2) + a control delta = 0.
    q = wrong-weight buffer, w'_k propto w_k^0.3 (as the positive control of the unit test). Exact posterior sampling: x_t ~ q_t (marginal of q at kappa), then x1 ~ q(x1 | x_t)
    (mixture over k with weights propto w'_k Q_k(x_t), component N(m_k, v / (v + kappa^2) I)): 400 x_t x 60 draws = 24000 draws per kappa and per q.
    Per x_t: e(x_t) = mean over the 60 draws of term(x_t, x1); c*(x_t) = (M*(x_t) - M^q(x_t)) / nu_{1|t}, M* = sum_k pi*_k m_k (pi* propto w_k Q_k), M^q: pi^q propto w'_k Q_k  (exact).
    Statistics, per kappa in {0.05, 0.1, 0.2, 0.3, 0.5, 0.7}:  cos = sum_x <e, c*> / (||e||_F ||c*||_F)  (pooled over the 400 x_t; PRIMARY),
    scale = sum_x <e, c*> / sum_x ||c*||^2  (1 = exact correction), median over x_t of the per-x_t cosine (secondary).
    The control delta = 0 gives the Monte-Carlo ceiling (cos ~ 1, scale ~ 1 expected). Noise inflates ||e||, hence lowers cos: the test is conservative.
    q = pi check: same with exact-pi endpoints, E_pi[term | x_t] = 0 for any L: report R = ||e||_F^2 / (sum_x tr Cov_x / 60), expected ~ 1 if the mean is 0 (and the scale ||e_pi|| / ||e_q||).

    PRE-REGISTERED PASS: cos > 0.5 at every kappa in the list for ALL 3 perturbed draws (direction preserved => the correction still points to the right weights).
    If the control delta = 0 fails to give cos > 0.9 at some kappa, the Monte Carlo is too noisy at that kappa and the verdict there is "inconclusive" (declared now), not pass.
    AMENDMENT (user-approved after the first run, declared before the rerun): the draw count only, 400 x_t x 600 exact posterior draws per kappa (`--draws 600`); the pass rule
    (cos > 0.5 for all 3 draws at every kappa) and the control ceiling (control cos >= 0.9) are unchanged. The first run (60 draws, 24000 per kappa) was INCONCLUSIVE (control cos 0.66-0.83).
    Secondary, report only: the noise-debiased cosine cos_corr = <e, c*> / (||c*|| sqrt(||e||^2 - tr Var_hat(e))) with the estimator variance already used for R (c* is exact, only ||e||^2 is biased),
    reported for the control and the 3 draws at 60 and at 600 draws per x_t.
    Additional, report only (not in the pass): D = K = 36 (the experimental setting), `--dim 36`.

      python scripts/division_mass_perturb.py [--dim 12]
"""
import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.components.sde import VESDE  # noqa: E402
from adjoint_samplers.energies.dist_energy import DistEnergy  # noqa: E402
from adjoint_samplers.utils.dist_utils import GMM  # noqa: E402
from adjoint_samplers.utils.division_cv import division_term, make_params  # noqa: E402

KAPPAS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
N_XT, N_DRAW = 400, 60
POWER = 0.3


def post(g, mu, w, xt, kap, v):
    """ posterior over modes pi (B, K) and component means m (B, K, D) at x_t for weights w; posterior variance s2 """
    var = kap ** 2 + v
    logQ = -0.5 * ((xt[:, None, :] - kap * mu[None]) ** 2).sum(-1) / var
    pi = torch.softmax(torch.log(w)[None] + logQ, dim=-1)
    m = mu[None] + kap / var * (xt[:, None, :] - kap * mu[None])
    return pi, m, v / var


def draw_xt_x1(g, mu, w, kap, v, gen):
    """ x_t ~ q_t, then N_DRAW exact posterior draws of x1 for each x_t """
    D = mu.shape[1]
    k = torch.multinomial(w, N_XT, replacement=True, generator=gen)
    x1 = mu[k] + torch.randn(N_XT, D, dtype=torch.float64, generator=gen)
    xt = kap * x1 + math.sqrt(v) * torch.randn(N_XT, D, dtype=torch.float64, generator=gen)
    pi, m, s2 = post(g, mu, w, xt, kap, v)
    j = torch.multinomial(pi, N_DRAW, replacement=True, generator=gen)                            # (N_XT, N_DRAW)
    mj = torch.gather(m, 1, j[..., None].expand(-1, -1, D))
    x1d = mj + math.sqrt(s2) * torch.randn(N_XT, N_DRAW, D, dtype=torch.float64, generator=gen)
    return xt, x1d


def mean_term(par, energy, xt, x1d, kap, v, nu1t):
    B = xt.shape[0] * N_DRAW
    xtr = xt[:, None, :].expand(-1, N_DRAW, -1).reshape(B, -1)
    x1 = x1d.reshape(B, -1)
    E1 = energy.eval(x1.float()).double()
    term = division_term(xtr, x1, E1, kap, v, nu1t, par).reshape(xt.shape[0], N_DRAW, -1)
    return term.mean(1), term.var(1, unbiased=True).sum(-1) / N_DRAW                            # e(x_t) and the variance of its estimator, summed over components


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=12)
    ap.add_argument("--draws", type=int, default=60)
    a = ap.parse_args()
    D = a.dim
    global N_DRAW
    N_DRAW = a.draws
    g = GMM(dim=D, K=D, seed=0, layout="spread", weight_law="loguniform", weight_ratio=20.0, radius=4.25)
    energy = DistEnergy(g)
    mu, w = g.means.double(), g.weights.double()
    wq = w ** POWER
    wq = wq / wq.sum()
    ref = VESDE(sigma_min=0.001, sigma_max=2 * GMM.coord_std(D, 4.25))
    nu1 = ref.total_var
    cov = torch.eye(D, dtype=torch.float64).expand(D, D, D)
    L0 = -torch.log(w)
    deltas = {"control (delta = 0)": torch.zeros(D, dtype=torch.float64)}
    for s in range(3):
        gd = torch.Generator().manual_seed(s)
        deltas[f"draw {s} (delta = +-0.5)"] = 0.5 * (2 * torch.randint(0, 2, (D,), generator=gd).double() - 1)
    pars = {n: make_params(mu, cov, L0 + d, 0.8) for n, d in deltas.items()}
    print(f"D = K = {D}; q: w^{POWER} normalised (max w'/w ratio {float((wq / w).max() / (wq / w).min()):.2f}); {N_XT} x_t x {N_DRAW} draws per kappa; nu_1 = {nu1:.3f}")
    print(f"{'kappa':>6} {'|c*|_F':>8} | " + " | ".join(f"{n[:22]:>22}: cos  corr scale med" for n in deltas) + " | q=pi: R (control, draws)")
    passed, inconclusive = True, False
    for kap in KAPPAS:
        v = nu1 * kap * (1 - kap)
        nu1t = nu1 * (1 - kap)
        gen = torch.Generator().manual_seed(1000 + int(1000 * kap))
        xt, x1d = draw_xt_x1(g, mu, wq, kap, v, gen)
        pi_s, m, _ = post(g, mu, w, xt, kap, v)
        pi_q, _, _ = post(g, mu, wq, xt, kap, v)
        cstar = ((pi_s - pi_q)[..., None] * m).sum(1) / nu1t
        gen2 = torch.Generator().manual_seed(2000 + int(1000 * kap))
        xtp, x1p = draw_xt_x1(g, mu, w, kap, v, gen2)
        row, rs = [], []
        for n, par in pars.items():
            e, var_e = mean_term(par, energy, xt, x1d, kap, v, nu1t)
            ip = (e * cstar).sum()
            cos = float(ip / (e.norm() * cstar.norm()))
            scale = float(ip / (cstar ** 2).sum())
            med = float(torch.nn.functional.cosine_similarity(e, cstar, dim=-1).median())
            den = float((e ** 2).sum() - var_e.sum())
            ccorr = float(ip / (cstar.norm() * math.sqrt(den))) if den > 0 else float("nan")
            row.append(f"{cos:6.3f} {ccorr:6.3f} {scale:5.2f} {med:5.2f}")
            ep, varp = mean_term(par, energy, xtp, x1p, kap, v, nu1t)
            rs.append(f"{float((ep ** 2).sum() / varp.sum()):.2f}")
            if n.startswith("control"):
                if cos <= 0.9:
                    inconclusive = True
            elif cos <= 0.5:
                passed = False
        print(f"{kap:6.2f} {float(cstar.norm()):8.3f} | " + " | ".join(f"{r:>38}" for r in row) + " | " + ", ".join(rs))
    ver = "PASS" if passed and not inconclusive else ("INCONCLUSIVE (control cos <= 0.9 at some kappa)" if passed else "FAIL")
    print(f"PRE-REGISTERED RULE (cos > 0.5 at every kappa for all 3 draws): {'met' if passed else 'not met'}; control ceiling ok: {not inconclusive}  =>  {ver}")


if __name__ == "__main__":
    main()
