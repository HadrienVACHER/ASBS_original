""" DW-4 division, STEP 1' (offline, CPU, no matcher, no training): TRUNCATED components. Hypothesis (user): e^{E} grows like e^{c d^4} (E quartic in the pair distances) while any Gaussian N_bar decays like
    e^{-d^2}, so u_a = N_bar_a e^{E/tau} is unbounded far from the basin and E_pi[u_a^2] = infinity even under pi (the heavy tails of step 1 would then be intrinsic, not a fit issue). Cure tested here: compact support.

      python scripts/dw4_division_trunc.py --stage model --samples results/dw4_e1500_ed8bb67_samples.pt   # c_a from the model only; writes scripts/dw4_division_trunc_params.json -> COMMIT it BEFORE --stage eval
      python scripts/dw4_division_trunc.py --stage test                                                     # unit tests (normalisation, convolution identity, exact zero mean of r_a)
      python scripts/dw4_division_trunc.py --stage eval --samples results/dw4_e1500_ed8bb67_samples.pt     # runs the unit tests first, then the evaluation

    Model (J = 1, EM-fitted mu_a, Sigma_a of scripts/dw4_division_em.py on the AS samples ed8bb67; Sigma_a = V diag(lam) V^T in V, the 6-dim zero-CoM subspace):
        N_trunc_a(y) = N(z; 0, (1 - eta_a) diag(lam)) restricted to the box |z_i| <= c_a sqrt(lam_i) and renormalised (z = V^T (y - mu_a); product of 1D truncated normals, mass erf(c / sqrt(2 (1 - eta)))^6 removed),
        N_bar_a(x)  = mean over h in S4 x Z2 (48) and the 256 rotation angles theta_l of N_trunc_a(R_{-theta_l} h^{-1} x - mu_a)   (finite average of normalised densities: exactly normalised on V),
        C_bar_a(x_t) = mean over the same (h, l) of C_trunc_a(R_{-theta_l} h^{-1} x_t), C_trunc_a(w) = int N_trunc_a(u - mu_a) N(w; kappa u, v I) du = prod_i N(w_i; kappa-shifted, kappa^2 S_i + v) [Phi((b_i - m_i)/sigma_i) - Phi((-b_i - m_i)/sigma_i)] / mass
        (S_i = (1 - eta) lam_i, b_i = c sqrt(lam_i), m_i = kappa S_i w_i / (kappa^2 S_i + v), sigma_i^2 = S_i v / (kappa^2 S_i + v)): exact, in log space.
    eta_a: the values of step 1 (EM model, variant-B rule; recorded in RESULTS.md): triangle + pendant 0.6, path of 2 0.3, two disjoint pairs 0.8, K4 minus an edge 0.8, star 0.8 (not re-chosen).
    c_a (MODEL ONLY, before any mass estimate, k_hat or b_hat): largest c in {1.5, 2, 2.5, 3} such that max E over the box is <= E_min_a + 6, with E_min_a = energy of the reference quenched minimum of type a, and
    max E over the box = max over 20000 exact draws of N_trunc_a (canonical frame; E is invariant under h, theta) and of a local maximisation (300 steps of normalised-gradient ascent projected on the box, step 0.05 b_i 0.98^step,
    from the 50 worst draws). If none satisfies it, c = 1.5 and the basin is flagged. The EM fit, eta_a, c_a and the maxima are written to scripts/dw4_division_trunc_params.json and COMMITTED before the evaluation (stage eval refuses to run without it).

    Unit tests (stage test; thresholds fixed here, |z| < 3.5 for every z-score; a failure invalidates the run):
      T1 normalisation: uniform draws in the box, mean of vol x N_trunc = 1 (z-score), each basin.
      T2 convolution identity: for 6 points x_t (4 drawn from x1 -> x_t of the component, 2 "far", from another basin's component), kappa in {0.2, 0.5, 0.8}, v = nu_1 kappa (1 - kappa) (nu_1 = 4, sigma_max = 2):
         E_{x1 ~ N_bar_a} [N(x_t; kappa x1, v I)] (4e5 exact draws) = C_bar_a(x_t) (z-score). This identity is what gives E_pi[r_a | x_t] = 1 / (Z p_t(x_t)) for every a, i.e. the exact zero mean of the term, for any target.
      T3 end to end: toy target e^{-E} = 0.5 N_bar_a + 0.5 N_bar_b (two basins), E_{x1 ~ pi, x_t ~ B}[(r_a - r_b) psi(x_t)] = 0 for psi = 1 and psi = the first coordinate of x_t in V, kappa in {0.3, 0.7}
         (2e4 draws); POSITIVE CONTROL: the same statistic with x1 drawn from 0.9 N_bar_a + 0.1 N_bar_b but r computed with the 0.5/0.5 target must give |z| > 5 (at least one of them).

    PRE-REGISTERED PASS (written before the first run): k_hat of u_a = N_bar_a e^{E/tau} over ALL AS samples (ed8bb67) < 0.5 for EVERY basin AND sup u_a / mean_q u_a finite and reported.
      k_hat: generalized Pareto fit (scipy, floc = 0) to the excesses over the (m+1)-th largest value of u / max u, m = min(max(int(0.2 n), 20), n_pos - 1) with n_pos = number of samples with u > 0
      (samples outside every box have u = 0; 'top 20 %' becomes 'all positive values but the smallest' if fewer than 20 % are positive; NaN if n_pos < 40, which fails). mean_q u_a = mean over ALL AS samples (q = the AS samples);
      sup reported two ways: the empirical maximum over the AS samples, and the model bound N_trunc_a(0) e^{E_max,box} (E is invariant, so the sup over the union of rotated boxes is at most this).
    Report only: b_hat (leakage-corrected Gelfand-Dey of variant B on the AS samples, f_a from 20000 exact draws of N_bar_a, 500 bootstraps) vs ref_a (half B) with the basin-reweighting error next to |log AS/ref| and
    the old (untruncated EM) values of step 1; the fraction of AS samples of basin a that fall inside the box of a. Stop after this step: no matcher work.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import genpareto

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dw4_division_b as b  # noqa: E402
import dw4_division_em as em  # noqa: E402
import dw4_division_step0 as s0  # noqa: E402
from adjoint_samplers.utils.dw4_quench import energy_grad_hess, quench  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, state_ids  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAR_FILE = ROOT / "scripts" / "dw4_division_trunc_params.json"
ETA_EM = {"triangle + pendant": 0.6, "path of 2 (3 particles)": 0.3, "2 disjoint pairs": 0.8, "K4 minus an edge": 0.8, "star (3 edges)": 0.8}
CS, MARGIN, NU1, ZTHR = [1.5, 2.0, 2.5, 3.0], 6.0, 4.0, 3.5
ANG = em.ANG
LOG_GRID = math.log(48 * s0.N_ANG)


class Comp:
    """ one truncated component: mu (6), eigenframe V (6, 6), lam (6), shrink s = 1 - eta, box half-width c """
    def __init__(self, mu, Sigma, eta, c):
        lam, V = torch.linalg.eigh(0.5 * (Sigma + Sigma.T))
        self.mu, self.V, self.lam, self.eta, self.c = mu, V, lam.clamp(min=1e-10), eta, c
        self.s = 1.0 - eta
        self.S = self.s * self.lam                                                    # variances of the Gaussian (shrunk)
        self.bw = c * torch.sqrt(self.lam)                                            # box half-widths
        self.lmass = 6 * math.log(math.erf(c / math.sqrt(2 * self.s)))                # log of the Gaussian mass of the box (equal in the 6 directions)
        self.lnorm = -0.5 * float(torch.log(2 * math.pi * self.S).sum()) - self.lmass  # log N_trunc at z = 0


def log_ntrunc(z, k):
    """ log N_trunc at box coordinates z (..., 6): -inf outside the box """
    lp = -0.5 * (z ** 2 / k.S).sum(-1) + k.lnorm
    return torch.where((z.abs() <= k.bw).all(-1), lp, torch.full_like(lp, -float("inf")))


def _images(x, chunk_slice=None):
    imgs = s0.group_act(s0.centre(x))                                                 # (n, 48, 4, 2)
    return s0.to_v(s0.rot(imgs[:, :, None], ANG[None, None, :]))                      # (n, 48, 256, 6): R_{-theta_l} h^{-1} x in V


def log_nbar(x, k, chunk=50):
    """ log N_bar(x) for x (n, 8): finite average over (h, l); -inf if no box contains the aligned point """
    out = []
    for lo in range(0, len(x), chunk):
        z = (_images(x[lo:lo + chunk]) - k.mu) @ k.V
        out.append(torch.logsumexp(log_ntrunc(z, k), dim=(1, 2)) - LOG_GRID)
    return torch.cat(out)


def log_diff_ndtr(u1, u2):
    """ log(Phi(u2) - Phi(u1)) for u2 > u1, stable in both tails """
    flip = (u1 + u2) > 0
    a = torch.where(flip, -u2, u1)
    c_ = torch.where(flip, -u1, u2)                                                   # a < c_, a + c_ <= 0
    la, lc = torch.special.log_ndtr(a), torch.special.log_ndtr(c_)
    lower = lc + torch.log(-torch.expm1(la - lc))
    mid = torch.log1p(-torch.special.ndtr(a) - torch.special.ndtr(-c_))
    return torch.where(c_ <= 0, lower, mid)


def log_cbar(xt, k, kappa, v, chunk=50):
    """ log C_bar(x_t) for x_t (n, 8) (exact convolution of N_bar with N(x_t; kappa x1, v I_6)) """
    out = []
    var_t = kappa ** 2 * k.S + v
    sig = torch.sqrt(k.S * v / var_t)
    for lo in range(0, len(xt), chunk):
        w = (_images(xt[lo:lo + chunk]) - kappa * k.mu) @ k.V                          # (c, 48, 256, 6)
        m = kappa * k.S * w / var_t
        ld = log_diff_ndtr((-k.bw - m) / sig, (k.bw - m) / sig)
        t = -0.5 * w ** 2 / var_t - 0.5 * torch.log(2 * math.pi * var_t) + ld - math.log(math.erf(k.c / math.sqrt(2 * k.s)))
        out.append(torch.logsumexp(t.sum(-1), dim=(1, 2)) - LOG_GRID)
    return torch.cat(out)


def draw_z(k, n, gen):
    """ exact draws of the box coordinates z of N_trunc (n, 6) """
    lo = torch.special.ndtr(-k.bw / torch.sqrt(k.S))
    u = lo + (1 - 2 * lo) * torch.rand(n, 6, generator=gen, dtype=torch.float64)
    return torch.sqrt(k.S) * torch.special.ndtri(u)


def draw_nbar(k, n, gen):
    """ exact draws of N_bar (n, 8): x = h R_{theta_l} y, y = mu + V z, h uniform in the 48, l uniform in the 256 """
    y = (k.mu + draw_z(k, n, gen) @ k.V.T) @ s0.B.T
    th = ANG[torch.randint(len(ANG), (n,), generator=gen)]
    z = s0.rot(y.reshape(n, 4, 2), -th)
    h = torch.randint(48, (n,), generator=gen)
    return s0.group_act(z)[torch.arange(n), h].reshape(n, 8)


def box_maxE(mu, Sigma, eta, c, gen):
    k = Comp(mu, Sigma, eta, c)
    Eof = lambda z: s0.energy((mu + z @ k.V.T) @ s0.B.T)
    z = draw_z(k, b.N_DRAW, gen)
    E = Eof(z)
    best = float(E.max())
    zz = z[E.topk(50).indices].clone()
    for step in range(300):
        x8 = (mu + zz @ k.V.T) @ s0.B.T
        gz = ((energy_grad_hess(x8)[1] @ s0.B) @ k.V)
        zz = (zz + 0.05 * 0.98 ** step * k.bw * gz / gz.norm(dim=1, keepdim=True).clamp(min=1e-12)).clamp(-k.bw, k.bw)
        best = max(best, float(Eof(zz).max()))
    return best


def load_as_fit(samples, verbose=False):
    """ reference types, AS samples and the EM fit of step 1 (same code path as scripts/dw4_division_em.py) """
    ref, q, lab, types, mu8 = b.load_reference()
    model = torch.load(samples, map_location="cpu", weights_only=False)["model_samples"].double()
    qm, gm, _ = quench(model)
    lab_m = state_ids(qm.double())
    xs = {k: model[(lab_m == k).nonzero().flatten()] for k in types}
    fit, hist, nit = em.fit_em(xs, {k: s0.to_v(mu8[k]) for k in types}, verbose=verbose)
    return dict(ref=ref, lab=lab, types=types, mu8=mu8, model=model, lab_m=lab_m, fit=fit, names=[NAMES[KEYS[k]] for k in types], nit=nit)


def stage_model(samples, out):
    t0 = time.time()
    d = load_as_fit(samples)
    gen = torch.Generator().manual_seed(0)
    res = {}
    print(f"EM fit done ({d['nit']} iterations, {time.time() - t0:.0f} s). max E over the box (20000 draws + local ascent) vs E_min + {MARGIN:g}")
    print(f"{'basin':26}{'eta':>5}{'E_min':>9}" + "".join(f"{'c=' + str(c):>10}" for c in CS) + "   chosen")
    for k, name in zip(d["types"], d["names"]):
        mu, Sigma = d["fit"][k]
        Emin = float(s0.energy(d["mu8"][k].reshape(1, 8))[0])
        mx = {c: box_maxE(mu, Sigma, ETA_EM[name], c, gen) for c in CS}
        ok = [c for c in CS if mx[c] <= Emin + MARGIN]
        chosen = max(ok) if ok else CS[0]
        res[name] = dict(eta=ETA_EM[name], c=chosen, flagged=not ok, Emin=Emin, maxE={str(c): mx[c] for c in CS}, mu=mu.tolist(), Sigma=Sigma.tolist())
        print(f"{name:26}{ETA_EM[name]:5.1f}{Emin:9.3f}" + "".join(f"{mx[c]:10.3f}" for c in CS) + f"   {chosen}" + ("  (NONE <= E_min + 6: 1.5 by default, flagged)" if not ok else ""))
    Path(out).write_text(json.dumps(res, indent=1))
    print(f"written {out}")
    print("==== BEGIN PARAMS JSON ====")
    print(json.dumps(res, indent=1))
    print("==== END PARAMS JSON ====")


def load_comps(path):
    r = json.loads(Path(path).read_text())
    return {name: Comp(torch.tensor(v["mu"], dtype=torch.float64), torch.tensor(v["Sigma"], dtype=torch.float64), v["eta"], v["c"]) for name, v in r.items()}, r


def zscore(est, se, ref):
    return float((est - ref) / se)


def stage_test(comps, n_unif=2_000_000, n_conv=400_000, n_e2e=20_000):
    gen = torch.Generator().manual_seed(123)
    names = list(comps)
    bad = []
    print("T1 normalisation (uniform draws in the box, mean of vol x N_trunc = 1):")
    for nm in names:
        k = comps[nm]
        z = (2 * torch.rand(n_unif, 6, generator=gen, dtype=torch.float64) - 1) * k.bw
        val = torch.exp(log_ntrunc(z, k)) * (2 * k.bw).prod()
        zs = zscore(val.mean(), val.std() / n_unif ** 0.5, 1.0)
        print(f"  {nm:26} c = {k.c}: mean {float(val.mean()):.4f}  z = {zs:+.2f}")
        bad.append(abs(zs))
    print("T2 convolution identity E_{x1~N_bar}[N(x_t; kappa x1, v I)] = C_bar(x_t):")
    for ia, ib in ((0, 1), (2, 4)):
        a, o = comps[names[ia]], comps[names[ib]]
        x1 = draw_nbar(a, n_conv, gen)
        y1 = x1 @ s0.B
        for kap in (0.2, 0.5, 0.8):
            v = NU1 * kap * (1 - kap)
            pts = [kap * draw_nbar(a, 1, gen) + math.sqrt(v) * (torch.randn(1, 6, generator=gen, dtype=torch.float64) @ s0.B.T) for _ in range(4)]
            pts += [kap * draw_nbar(o, 1, gen) + math.sqrt(v) * (torch.randn(1, 6, generator=gen, dtype=torch.float64) @ s0.B.T) for _ in range(2)]
            zs = []
            for xt in pts:
                dens = torch.exp(-0.5 * ((xt @ s0.B - kap * y1) ** 2).sum(-1) / v - 3 * math.log(2 * math.pi * v))
                C = float(torch.exp(log_cbar(xt, a, kap, v))[0])
                zs.append(zscore(dens.mean(), dens.std() / n_conv ** 0.5, C))
            print(f"  {names[ia]:26} kappa {kap}: z = " + " ".join(f"{z:+.2f}" for z in zs) + "   (last 2: x_t from another basin)")
            bad += [abs(z) for z in zs]
    ctl_max, ex = test_e2e(comps, gen, n_e2e)
    bad += ex
    ok = max(bad) < ZTHR and ctl_max > 5
    print(f"unit tests: max |z| (T1, T2, exact T3) = {max(bad):.2f} (< {ZTHR}), positive control max |z| = {ctl_max:.1f} (> 5): {'PASS' if ok else 'FAIL (run invalid)'}")
    return ok


def test_e2e(comps, gen, n_e2e, kappas=(0.3, 0.7)):
    names = list(comps)
    print("T3 end-to-end exact zero mean of r_a - r_b under the toy target e^{-E} = 0.5 N_bar_a + 0.5 N_bar_b (a = first, b = last basin):")
    a, o = comps[names[0]], comps[names[-1]]
    bad, ctl_max = [], 0.0
    for kap in kappas:
        v = NU1 * kap * (1 - kap)
        for label, wa in (("exact pi", 0.5), ("positive control (x1 from 0.9/0.1)", 0.9)):
            na = int(round(wa * n_e2e))
            x1 = torch.cat([draw_nbar(a, na, gen), draw_nbar(o, n_e2e - na, gen)])
            xt = kap * x1 + math.sqrt(v) * (torch.randn(n_e2e, 6, generator=gen, dtype=torch.float64) @ s0.B.T)
            la, lo_ = log_nbar(x1, a), log_nbar(x1, o)
            lpi = torch.logsumexp(torch.stack([la, lo_]), 0) - math.log(2)
            ra = torch.exp(la - lpi - log_cbar(xt, a, kap, v))
            rb = torch.exp(lo_ - lpi - log_cbar(xt, o, kap, v))
            dr = ra - rb
            zs = []
            for psi in (torch.ones(n_e2e, dtype=torch.float64), (xt @ s0.B)[:, 0]):
                f = dr * psi
                zs.append(zscore(f.mean(), f.std() / n_e2e ** 0.5, 0.0))
            print(f"  kappa {kap} {label:38}: z = " + " ".join(f"{z:+.2f}" for z in zs))
            if wa == 0.5:
                bad += [abs(z) for z in zs]
            else:
                ctl_max = max(ctl_max, max(abs(z) for z in zs))
    return ctl_max, bad


def khat_pos(u):
    u = np.sort(np.asarray(u, dtype=float))
    u = u / u.max()
    npos = int((u > 0).sum())
    if npos < 40:
        return float("nan"), 0
    m = min(max(int(0.2 * len(u)), 20), npos - 1)
    thr = u[-m - 1]
    ex = u[-m:] - thr
    return (float(genpareto.fit(ex, floc=0)[0]) if ex.max() > 0 else float("nan")), m


def stage_eval(samples, params, run_tests=True):
    if not Path(params).exists():
        sys.exit("run --stage model and COMMIT scripts/dw4_division_trunc_params.json first")
    comps, raw = load_comps(params)
    if run_tests and not stage_test(comps):
        sys.exit("unit tests failed: the evaluation is invalid")
    ref, q, lab, types, mu8 = b.load_reference()
    names = [NAMES[KEYS[k]] for k in types]
    Bi = torch.arange(1, len(ref), 2)
    refB = torch.stack([(lab[Bi] == k).double().mean() for k in types])
    model = torch.load(samples, map_location="cpu", weights_only=False)["model_samples"].double()
    qm, gm, _ = quench(model)
    lab_m = state_ids(qm.double())
    asw = torch.stack([(lab_m == k).double().mean() for k in types])
    Em = s0.energy(model)
    print(f"\nAS samples: {samples}: {len(model)}; c_a: " + ", ".join(f"{n} {comps[n].c}" for n in names))
    gen = torch.Generator().manual_seed(30)
    print(f"{'basin':26}{'n AS':>6}{'inside box':>12}{'n(u>0)':>8}{'m':>6}{'k_hat':>8}{'sup/mean (AS)':>15}{'sup/mean (bound)':>18}")
    lvs, logm, f, ind, khs = [], [], [], [], []
    for i, kk in enumerate(types):
        k = comps[names[i]]
        ln = log_nbar(model, k)
        lu = ln + Em / b.TAU
        sel = (lab_m == kk).nonzero().flatten()
        u = torch.exp(lu - lu.max()).numpy()
        kh, m = khat_pos(u)
        khs.append(kh)
        log_mean = float(torch.logsumexp(lu, 0) - math.log(len(lu)))
        lbound = k.lnorm + raw[names[i]]["maxE"][str(k.c)] - log_mean
        print(f"{names[i]:26}{len(sel):6d}{100 * float(torch.isfinite(ln[sel]).double().mean()):11.1f}%{int((u > 0).sum()):8d}{m:6d}{kh:8.3f}{float(lu.max()) - log_mean:15.3f} (log){lbound:12.3f} (log)", flush=True)
        lvs.append(lu[sel])
        logm.append(float(torch.logsumexp(lu[sel], 0) - math.log(len(sel))))
        qs, _, _ = quench(draw_nbar(k, b.N_DRAW, gen))
        landed = (state_ids(qs.double()) == kk).double()
        ind.append(landed)
        f.append(float(landed.mean()))
    logm, f_t = torch.tensor(logm), torch.tensor(f)
    bh = lambda lm, ff: torch.softmax(torch.log(ff.clamp(min=1e-12)) - lm, 0)
    bhat = bh(logm, f_t)
    g2 = torch.Generator().manual_seed(31)
    boots = []
    for _ in range(b.N_BOOT):
        lm = torch.stack([torch.logsumexp(v[torch.randint(len(v), (len(v),), generator=g2)], 0) - math.log(len(v)) for v in lvs])
        ff = torch.stack([ind[i][torch.randint(b.N_DRAW, (b.N_DRAW,), generator=g2)].mean() for i in range(len(types))])
        boots.append(bh(lm, ff))
    se = torch.stack(boots).std(0)
    print("\nreport only: basin-reweighting b_hat (leakage-corrected Gelfand-Dey, AS-fitted truncated model) vs ref_a (half B); step 1 (untruncated EM): 0.114, 0.139, 0.055, 0.127, 0.184 (sum 0.618)")
    print(f"{'basin':26}{'ref_B':>8}{'AS':>8}{'b_hat':>8}{'SE':>8}{'f_a':>7}{'|log b/ref|':>13}{'|log AS/ref|':>14}")
    for i in range(len(types)):
        print(f"{names[i]:26}{float(refB[i]):8.4f}{float(asw[i]):8.4f}{float(bhat[i]):8.4f}{float(se[i]):8.4f}{float(f_t[i]):7.3f}{float(torch.log(bhat[i] / refB[i]).abs()):13.3f}{float(torch.log(asw[i].clamp(min=1e-12) / refB[i]).abs()):14.3f}")
    print(f"  sum_a |log b_hat/ref| = {float(torch.log(bhat / refB).abs().sum()):.3f}; raw AS {float(torch.log(asw.clamp(min=1e-12) / refB).abs().sum()):.3f}")
    ok = all(kh < 0.5 for kh in khs)                                                           # NaN -> False; sup/mean is finite by construction
    print(f"\nPASS RULE (k_hat < 0.5 for every basin AND sup/mean finite): {ok}\nVERDICT: " + ("PASS" if ok else "FAIL"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["model", "test", "eval"], required=True)
    ap.add_argument("--samples", default=None)
    ap.add_argument("--params", default=str(PAR_FILE))
    ap.add_argument("--no-test", action="store_true", help="plumbing smoke test only")
    a = ap.parse_args()
    if a.stage == "model":
        stage_model(a.samples, a.params)
    elif a.stage == "test":
        stage_test(load_comps(a.params)[0])
    else:
        stage_eval(a.samples, a.params, not a.no_test)


if __name__ == "__main__":
    main()
