""" DW-4 division, STEP 1 of the "masses do not enter the fixed point" plan (offline, CPU, no matcher, no training): refit the single-Gaussian G-averaged basin models by MAXIMUM LIKELIHOOD of the
    invariant model (EM with a latent group element) instead of hard alignment, and look at the heavy tails of u_a = N_bar_a e^{E/tau} on the AS samples.

      python scripts/dw4_division_em.py --samples results/dw4_e1500_ed8bb67_samples.pt         # AS run ed8bb67 (model_samples)

    Model (same as scripts/dw4_division_step0.py): q_bar_a(x) = (1/48) sum_h (1/2pi) int N(R_{-theta} h^{-1} x; mu_a, Sigma_a) dtheta on the zero-CoM subspace V (dim 6), h in S4 x Z2 (48), theta on the
    256-angle trapezoid grid (the Mahalanobis form is a trig polynomial of degree <= 2 in theta: exact from 8 angles, as in log_qbar).
    EM, per basin a, on the AS samples of basin a (basin = state of the quenched configuration; the 5 types of the quenched reference):
      E-step: responsibilities r_{i,h,theta} propto N(R_{-theta} h^{-1} x_i; mu, Sigma) over the 48 x 256 grid; (sample, h) pairs with total responsibility < 1e-7 are dropped from the moments.
      M-step: weighted mean and covariance of the aligned points y = R_{-theta} h^{-1} x in the canonical frame; covariance shrinkage toward the pooled covariance with weight n0 = 50 samples:
              Sigma_a = (n_a Sigma_ML,a + n0 Sigma_pool) / (n_a + n0),  Sigma_pool = sum_a n_a Sigma_ML,a / sum_a n_a (all basins, current iteration);
              tangent-variance floor: along the (normalised) rotation tangent t of mu_a the variance is raised to 5e-3 if smaller (t' Sigma t < 5e-3 -> Sigma += (5e-3 - t' Sigma t) t t').
      Initialisation (hard alignment only here): mu_a = lowest-energy quenched reference minimum of type a, Sigma_a = covariance of the aligned samples (scripts/dw4_division_step0.fit_basin).
      Stopping: at most 40 iterations, or |change of the mean log-likelihood| < 1e-4 for every basin. J = 1 (a single Gaussian per basin).
    eta_a: same rule as variant B, computed from the EM-fitted model only (20000 exact draws of N_bar_a with covariance (1 - eta) Sigma_a, P(some pair distance in [3.5, 4.5]) < 1 % for the smallest eta in
    {0.2, ..., 0.8}, 0.8 and flagged if none). It is computed and printed before any weight, k_hat or b_hat is looked at, and does not depend on them.
    Declared choices: the fit is on the AS samples only (the pass concerns the AS samples); the reference is used only for the 5 types, the initial mu_a and, report-only, ref_a (half B, odd indices,
    as in variant B); the leakage-corrected Gelfand-Dey estimator is that of variant B (f_a from 20000 quenched draws, 500 bootstraps). The old (aligned) model with the variant-B eta file is printed beside for comparison.

    PRE-REGISTERED PASS (written before the first run): Pareto k_hat of u_a = N_bar_a e^{E/tau} over ALL AS samples (top 20 % by value, generalized Pareto fit, scripts/dw4_division_b.khat) < 0.5 for EVERY basin.
    Report only (no rule): within-basin SD of log N_bar_a + E/tau, b_hat (AS-fitted, leakage-corrected) vs ref_a with the basin-reweighting error sum_a |log(b_hat_a / ref_a)| next to |log(AS_a / ref_a)|.
    Stop after this step and report: no matcher work.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dw4_division_b as b  # noqa: E402
import dw4_division_step0 as s0  # noqa: E402
from adjoint_samplers.utils.dw4_quench import quench  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, state_ids  # noqa: E402

N0, TFLOOR, MAX_IT, TOL, CHUNK = 50.0, 5e-3, 40, 1e-4, 200
ANG = torch.arange(s0.N_ANG) * (2 * math.pi / s0.N_ANG)
TH8 = torch.arange(8) * (2 * math.pi / 8)


def estep(x, mu_c, Sigma):
    """ returns S0 (total kept responsibility), S1 (6), S2 (6, 6) (weighted sums of the aligned points y) and the mean log-likelihood """
    P = torch.linalg.inv(Sigma)
    logdet = torch.logdet(2 * math.pi * Sigma)
    S0, S1, S2, ll = 0.0, torch.zeros(6), torch.zeros(6, 6), 0.0
    for lo in range(0, len(x), CHUNK):
        imgs = s0.group_act(s0.centre(x[lo:lo + CHUNK]))                                   # (c, 48, 4, 2)
        w8 = s0.to_v(s0.rot(imgs[:, :, None], TH8[None, None, :])) - mu_c                  # (c, 48, 8, 6)
        m8 = torch.einsum("nhaj,jk,nhak->nha", w8, P, w8)
        cf = torch.fft.rfft(m8, dim=-1)
        cf = torch.nn.functional.pad(cf, (0, s0.N_ANG // 2 + 1 - cf.shape[-1]))
        m = torch.fft.irfft(cf, n=s0.N_ANG, dim=-1) * (s0.N_ANG / 8)                       # (c, 48, 256)
        lr = -0.5 * m
        lse = torch.logsumexp(lr, dim=(1, 2))
        ll += float((lse - 0.5 * logdet - math.log(48 * s0.N_ANG)).sum())
        r = torch.exp(lr - lse[:, None, None])
        ci, hi = (r.sum(-1) > 1e-7).nonzero(as_tuple=True)
        y = s0.to_v(s0.rot(imgs[ci, hi][:, None], ANG[None, :]))                          # (p, 256, 6)
        rr = r[ci, hi]                                                                      # (p, 256)
        S0 += float(rr.sum())
        S1 += torch.einsum("pt,ptj->j", rr, y)
        S2 += torch.einsum("pt,ptj,ptk->jk", rr, y, y)
    return S0, S1, S2, ll / len(x)


def tangent(mu_c):
    mu8 = (s0.B @ mu_c).reshape(4, 2)
    t = s0.to_v(torch.stack([-mu8[:, 1], mu8[:, 0]], -1))
    return t / t.norm()


def fit_em(xs, mus, verbose=True):
    """ xs: {a: samples (n_a, 8)}, mus: {a: initial mu_c}; returns {a: (mu_c, Sigma)} and a log """
    st = {a: s0.fit_basin(xs[a], _mu8(mus[a]))[:2] for a in xs}
    n = {a: len(xs[a]) for a in xs}
    hist = {a: [] for a in xs}
    for it in range(MAX_IT):
        ml = {}
        for a in xs:
            S0, S1, S2, ll = estep(xs[a], *st[a])
            mu = S1 / S0
            ml[a] = (mu, S2 / S0 - torch.outer(mu, mu), ll)
            hist[a].append(ll)
        pool = sum(n[a] * ml[a][1] for a in xs) / sum(n.values())
        new = {}
        for a in xs:
            mu, Sml, _ = ml[a]
            Sg = (n[a] * Sml + N0 * pool) / (n[a] + N0)
            Sg = 0.5 * (Sg + Sg.T)
            t = tangent(mu)
            vt = float(t @ Sg @ t)
            if vt < TFLOOR:
                Sg = Sg + (TFLOOR - vt) * torch.outer(t, t)
            new[a] = (mu, Sg)
        st = new
        delta = max(abs(hist[a][-1] - hist[a][-2]) for a in xs) if it > 0 else float("inf")
        if verbose:
            print(f"  EM iteration {it + 1}: mean log-lik " + ", ".join(f"{hist[a][-1]:.3f}" for a in xs) + f"   max |change| {delta:.2e}", flush=True)
        if delta < TOL:
            break
    return st, hist, it + 1


def _mu8(mu_c):
    return s0.centre((s0.B @ mu_c).reshape(1, 8))[0]


def pick_eta(fit, gen):
    res = {}
    for a, (mu, Sg) in fit.items():
        ps = [float(b.pair_in_window(b.draw(mu, (1 - e) * Sg, b.N_DRAW, gen)).double().mean()) for e in b.ETAS]
        ok = [e for e, p in zip(b.ETAS, ps) if p < 0.01]
        res[a] = (ok[0] if ok else b.ETAS[-1], not ok, ps)
    return res


def gd_mass(fit, etas, x, lab, types, E, seed):
    """ leakage-corrected Gelfand-Dey (variant B): b_hat, SE (500 bootstraps), f_a, per-basin log values """
    gen = torch.Generator().manual_seed(seed)
    T = len(types)
    logm, f, ind, lv = [], [], [], []
    for a in types:
        mu, Sg = fit[a]
        Sh = (1 - etas[a]) * Sg
        idx = (lab == a).nonzero().flatten()
        v = s0.log_qbar(x[idx], mu, Sh) + E[idx] / b.TAU
        lv.append(v)
        logm.append(b.logmeanexp(v))
        qs, _, _ = quench(b.draw(mu, Sh, b.N_DRAW, gen))
        landed = (state_ids(qs.double()) == a).double()
        ind.append(landed)
        f.append(float(landed.mean()))
    logm, f_t = torch.stack(logm), torch.tensor(f)
    bh = lambda lm, ff: torch.softmax(torch.log(ff.clamp(min=1e-12)) - lm, 0)
    g2 = torch.Generator().manual_seed(seed + 1)
    boots = []
    for _ in range(b.N_BOOT):
        lm = torch.stack([b.logmeanexp(v[torch.randint(len(v), (len(v),), generator=g2)]) for v in lv])
        ff = torch.stack([ind[i][torch.randint(b.N_DRAW, (b.N_DRAW,), generator=g2)].mean() for i in range(T)])
        boots.append(bh(lm, ff))
    return bh(logm, f_t), torch.stack(boots).std(0), f_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--max-per-basin", type=int, default=0, help="smoke test only: subsample each basin")
    a = ap.parse_args()
    t0 = time.time()
    ref, q, lab, types, mu8 = b.load_reference()
    names = [NAMES[KEYS[k]] for k in types]
    Bi = torch.arange(1, len(ref), 2)
    refB = torch.stack([(lab[Bi] == k).double().mean() for k in types])
    model = torch.load(a.samples, map_location="cpu", weights_only=False)["model_samples"].double()
    qm, gm, _ = quench(model)
    lab_m = state_ids(qm.double())
    asw = torch.stack([(lab_m == k).double().mean() for k in types])
    Em = s0.energy(model)
    print(f"AS samples: {a.samples}: {len(model)} ({100 * float((gm < 1e-4).double().mean()):.1f} % quench-converged); per-basin n: " + ", ".join(f"{n} {int((lab_m == k).sum())}" for n, k in zip(names, types)))
    xs, mus = {}, {k: s0.to_v(mu8[k]) for k in types}
    for k in types:
        idx = (lab_m == k).nonzero().flatten()
        if a.max_per_basin:
            idx = idx[:a.max_per_basin]
        assert len(idx) >= b.MIN_N, (names[types.index(k)], len(idx))
        xs[k] = model[idx]
    print("EM (hard-alignment initialisation), columns = basins in the order above")
    fit, hist, nit = fit_em(xs, mus)
    old = b.fit_set(model, lab_m, types, mu8)                                              # the aligned (variant B) model on the same samples
    print(f"EM stopped after {nit} iterations ({time.time() - t0:.0f} s). Mean log-lik per basin: EM vs aligned model (old, initialisation = after 0 iterations)")
    for i, k in enumerate(types):
        print(f"  {names[i]:26} EM {hist[k][-1]:9.3f}   first iteration {hist[k][0]:9.3f}   tangent variance {float(tangent(fit[k][0]) @ fit[k][1] @ tangent(fit[k][0])):.4f}")
    gen = torch.Generator().manual_seed(0)
    eta = pick_eta(fit, gen)
    etas = {k: eta[k][0] for k in types}
    print("eta_a (EM model; rule of variant B): " + ", ".join(f"{names[i]} {etas[k]}" + (" (flagged: none < 1 %)" if eta[k][1] else "") for i, k in enumerate(types)))
    eta_old = json.loads(b.ETA_FILE.read_text())
    etas_old = {k: eta_old[NAMES[KEYS[k]]]["eta"] for k in types}
    # k_hat and SD
    print(f"\nPareto k_hat of u_a = N_bar_a e^(E/tau) over ALL AS samples (top 20 %); in basin = diagnostic; SD of log N_bar_a + E/tau within basin: on the AS samples and on reference half B (EM model fitted on AS)")
    print(f"{'basin':26}{'k_hat(all AS)':>15}{'k_hat(in basin)':>17}{'SD (AS)':>9}{'SD (ref B)':>12}{'|  old: k_hat(all)':>20}{'old SD (AS)':>13}")
    EB = s0.energy(ref[Bi])
    khs = []
    for i, k in enumerate(types):
        row = []
        for f_, e_ in ((fit, etas), (old, etas_old)):
            mu, Sg = f_[k]
            lu = s0.log_qbar(model, mu, (1 - e_[k]) * Sg) + Em / b.TAU
            u = torch.exp(lu - lu.max()).numpy()
            sel = (lab_m == k).nonzero().flatten()
            row.append((b.khat(u), b.khat(u[(lab_m == k).numpy()]), float(lu[sel].std())))
        mu, Sg = fit[k]
        selB = (lab[Bi] == k).nonzero().flatten()
        sdB = float((s0.log_qbar(ref[Bi][selB], mu, (1 - etas[k]) * Sg) + EB[selB] / b.TAU).std())
        khs.append(row[0][0])
        print(f"{names[i]:26}{row[0][0]:15.3f}{row[0][1]:17.3f}{row[0][2]:9.3f}{sdB:12.3f}{row[1][0]:20.3f}{row[1][2]:13.3f}")
    ok = all(kk < 0.5 for kk in khs)                                                           # NaN -> False
    # b_hat
    print("\nreport only: basin-reweighting b_hat (leakage-corrected Gelfand-Dey, AS-fitted) vs ref_a (half B)")
    bh, se, f = gd_mass(fit, etas, model, lab_m, types, Em, 20)
    bo, so, fo = gd_mass(old, etas_old, model, lab_m, types, Em, 20)
    print(f"{'basin':26}{'ref_B':>8}{'AS':>8}{'b_hat(EM)':>11}{'SE':>8}{'|log b/ref|':>13}{'f_a':>7}{'|  b_hat(old)':>15}{'|log b/ref|':>13}{'|log AS/ref|':>14}")
    for i, k in enumerate(types):
        print(f"{names[i]:26}{float(refB[i]):8.4f}{float(asw[i]):8.4f}{float(bh[i]):11.4f}{float(se[i]):8.4f}{float(torch.log(bh[i] / refB[i]).abs()):13.3f}{float(f[i]):7.3f}{float(bo[i]):15.4f}{float(torch.log(bo[i] / refB[i]).abs()):13.3f}{float(torch.log(asw[i].clamp(min=1e-12) / refB[i]).abs()):14.3f}")
    print(f"  sum_a |log b_hat/ref|: EM {float(torch.log(bh / refB).abs().sum()):.3f}, old {float(torch.log(bo / refB).abs().sum()):.3f}; raw AS {float(torch.log(asw.clamp(min=1e-12) / refB).abs().sum()):.3f}")
    print(f"\nPASS RULE (k_hat < 0.5 for every basin): {ok}\nVERDICT: " + ("PASS" if ok else "FAIL") + f"   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
