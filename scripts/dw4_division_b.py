""" DW-4 division, VARIANT B, step 0 (offline, no matcher, no training): fix the basin MASSES with a leakage-corrected Gelfand-Dey estimator, keeping the coarse single-Gaussian G-averaged basin model
    of scripts/dw4_division_step0.py (reused: alignment, tangent variance, trapezoid on 256 angles in log space).

      python scripts/dw4_division_b.py --stage eta                                    # step 1 (model only; reference half A): writes scripts/dw4_division_b_eta.json -> COMMIT it
      python scripts/dw4_division_b.py --stage mass --samples <AS samples_final.pt>   # steps 2-4 (needs the committed eta file)

    Split of the reference (data/test_split_DW4.npy) by index parity: A (even, fit) and B (odd, evaluation). Every sample is quenched (Newton) and labelled by its basin type a (the 5 types of the quenched
    reference). ref_a = frequency of type a in half B (for both runs). Model: for each a, mu_a = lowest-energy quenched minimum of type a, Sigma_a = covariance of the aligned samples of basin a
    (best h of 48, closed-form 2D Procrustes angle, tangent variance := median vibrational eigenvalue); N_bar_a with covariance (1 - eta_a) Sigma_a; q_bar_a = unshrunk.
    Declared choices where the specification left room: minimum 20 samples per basin to fit (the star basin has ~28 samples in half A); eta_a is computed ONCE from the model fitted on half A and used for both runs;
    mu_a comes from the reference quench in both runs; in step 3 the weights u_a are over ALL AS samples (top 20 % by value, generalized Pareto fit by scipy), and the same k_hat restricted to the samples of
    basin a is printed as a diagnostic (not used by the rule); the AS-fitted model is used for k_hat; the IS ESS is reported for both fitted models.

    Step 1: eta_a = smallest value in {0.2, 0.3, ..., 0.8} such that 20000 exact draws of N_bar_a (h and theta uniform, Gaussian in the canonical frame) have some pair distance in [3.5, 4.5]
    with probability < 1 % (0.8 if none, flagged). Written to scripts/dw4_division_b_eta.json and committed BEFORE step 2.
    Step 2: E_{pi|a}[N_bar_a e^{E/tau}] = f_a / (Z w_a), f_a = P_{N_bar_a}(quench lands in a) (20000 draws, quenched). m_a = mean over the samples of basin a of N_bar_a e^{E/tau} (logsumexp);
    Z w_a = f_a / m_a; b_hat = normalised. Also b_hat with f_a = 1. SE: bootstrap 500 over the samples and over the f_a draws. (c) ref-fitted: mu, Sigma, m_a from half A, against ref_a of half B;
    (d) AS-fitted: mu, Sigma, m_a from the AS samples.
    Step 3: Pareto k_hat of u_a = N_bar_a e^{E/tau} over ALL AS samples; SD of log N_bar_a + E/tau per basin on reference half B.
    Step 4 (report only): IS effective sample size, proposal sum_a b_hat_a q_bar_a (exact draws, unshrunk), weights e^{-E}/proposal.

    PRE-REGISTERED GO RULE (written and committed BEFORE running steps 2-4). GO iff all three hold:
      (b) k_hat < 0.5 for every basin;
      (c) ref-fitted: |log(b_hat_a / ref_a)| <= 0.1 for the three largest basins (triangle + pendant, 2-edge path, two disjoint pairs);
      (d) AS-fitted: |log(b_hat_a / ref_a)| <= 0.2 for those three basins AND |log(b_hat_a / ref_a)| < |log(AS_a / ref_a)| for at least 4 of the 5 basins.
    Interpretation fixed in advance: if (c) fails the single-Gaussian model is not enough (variant A would be the next question, the user's call); if (c) passes and (d) fails the AS shape error biases
    Gelfand-Dey; if GO the next step is the DW-4 division matcher, the S_psi end-to-end test, then the paired continuation test. The corrected estimate (f_a) is the one used by the rule.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import genpareto

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dw4_division_step0 as s0  # noqa: E402
from adjoint_samplers.utils.dw4_quench import quench  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, pair_distances, state_ids  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ETA_FILE = ROOT / "scripts" / "dw4_division_b_eta.json"
ETAS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MIN_N, N_DRAW, N_BOOT, TAU = 20, 20000, 500, 1.0


def load_reference():
    ref = torch.tensor(np.load(ROOT / "data" / "test_split_DW4.npy", allow_pickle=True)).double()
    q, g, _ = quench(ref)
    q = q.double()
    lab = state_ids(q)
    freq = torch.bincount(lab, minlength=len(KEYS)).double() / len(lab)
    types = sorted([k for k in range(len(KEYS)) if freq[k] > 0.001], key=lambda k: -float(freq[k]))
    E = s0.energy(q)
    mu8 = {}
    for k in types:
        idx = (lab == k).nonzero().flatten()
        mu8[k] = s0.centre(q[idx[E[idx].argmin()][None]])[0]
    return ref, q, lab, types, mu8


def fit_set(x, lab, types, mu8):
    """ per type: (mu_c, Sigma unshrunk) or None """
    out = {}
    for a in types:
        idx = (lab == a).nonzero().flatten()
        out[a] = s0.fit_basin(x[idx], mu8[a])[:2] if len(idx) >= MIN_N else None
    return out


def draw(mu_c, Sigma, n, gen):
    """ exact draws from the G-average of N(mu_c, Sigma): x = h R_theta y, y ~ N(mu, Sigma) (n, 8) """
    Lc = torch.linalg.cholesky(Sigma)
    y = (mu_c + torch.randn(n, 6, generator=gen) @ Lc.T) @ s0.B.T
    th = torch.rand(n, generator=gen) * 2 * math.pi
    z = s0.rot(y.reshape(n, 4, 2), -th)                                          # R_theta y
    h = torch.randint(48, (n,), generator=gen)
    return s0.group_act(z)[torch.arange(n), h].reshape(n, 8)


def pair_in_window(x, lo=3.5, hi=4.5):
    d = pair_distances(x)
    return ((d >= lo) & (d <= hi)).any(1)


def stage_eta():
    ref, q, lab, types, mu8 = load_reference()
    A = torch.arange(0, len(ref), 2)
    fits = fit_set(ref[A], lab[A], types, mu8)
    gen = torch.Generator().manual_seed(0)
    res = {}
    print("step 1: eta_a from the model fitted on reference half A (20000 exact draws per value); P(some pair distance in [3.5, 4.5])")
    print(f"{'basin':26}" + "".join(f"{e:>8.1f}" for e in ETAS) + "   chosen")
    for a in types:
        ps = []
        for e in ETAS:
            Sigma = fits[a][1]
            ps.append(float(pair_in_window(draw(fits[a][0], (1 - e) * Sigma, N_DRAW, gen)).double().mean()))
        ok = [e for e, p in zip(ETAS, ps) if p < 0.01]
        chosen = ok[0] if ok else ETAS[-1]
        res[NAMES[KEYS[a]]] = dict(eta=chosen, probs=dict(zip(map(str, ETAS), ps)), flagged=not ok)
        print(f"{NAMES[KEYS[a]]:26}" + "".join(f"{p:8.4f}" for p in ps) + f"   {chosen}" + ("  (NONE < 1 %: 0.8 by default)" if not ok else ""))
    ETA_FILE.write_text(json.dumps(res, indent=1))
    print(f"written {ETA_FILE}")


def logmeanexp(v):
    return torch.logsumexp(v, 0) - math.log(len(v))


def khat(u, frac=0.2):
    u = np.sort(np.asarray(u, dtype=float))
    u = u / u.max()
    m = max(int(frac * len(u)), 20)
    if len(u) < 40:
        return float("nan")
    thr = u[-m - 1]
    ex = u[-m:] - thr
    return float(genpareto.fit(ex, floc=0)[0]) if ex.max() > 0 else float("nan")


def mass_run(name, x_fit, lab_fit, types, mu8, etas, ref_w, as_w, rng_seed):
    """ fits on (x_fit, lab_fit); returns dict with b_hat (corrected / uncorrected), SE, f_a, fits and per-basin log-values """
    fits = fit_set(x_fit, lab_fit, types, mu8)
    E = s0.energy(x_fit)
    gen = torch.Generator().manual_seed(rng_seed)
    T = len(types)
    logm, f, ind, lv = [], [], [], []
    for i, a in enumerate(types):
        mu_c, Sigma = fits[a]
        Sh = (1 - etas[a]) * Sigma
        idx = (lab_fit == a).nonzero().flatten()
        v = s0.log_qbar(x_fit[idx], mu_c, Sh) + E[idx] / TAU
        lv.append(v)
        logm.append(logmeanexp(v))
        xs = draw(mu_c, Sh, N_DRAW, gen)
        qs, _, _ = quench(xs)
        landed = (state_ids(qs.double()) == a).double()
        ind.append(landed)
        f.append(float(landed.mean()))
    logm = torch.stack(logm)
    f_t = torch.tensor(f)

    def bhat(lm, ff, corr=True):
        z = (torch.log(ff.clamp(min=1e-12)) if corr else 0.0) - lm
        return torch.softmax(z, 0)

    b_c, b_u = bhat(logm, f_t), bhat(logm, f_t, False)
    boots, boots_u = [], []
    g2 = torch.Generator().manual_seed(rng_seed + 1)
    for _ in range(N_BOOT):
        lm = torch.stack([logmeanexp(v[torch.randint(len(v), (len(v),), generator=g2)]) for v in lv])
        ff = torch.stack([ind[i][torch.randint(N_DRAW, (N_DRAW,), generator=g2)].mean() for i in range(T)])
        boots.append(bhat(lm, ff))
        boots_u.append(bhat(lm, ff, False))
    se, se_u = torch.stack(boots).std(0), torch.stack(boots_u).std(0)
    return dict(fits=fits, b=b_c, b_u=b_u, se=se, se_u=se_u, f=f_t, logm=logm)


def stage_mass(samples):
    if not ETA_FILE.exists():
        sys.exit("run --stage eta first and commit scripts/dw4_division_b_eta.json")
    eta_j = json.loads(ETA_FILE.read_text())
    ref, q, lab, types, mu8 = load_reference()
    etas = {a: eta_j[NAMES[KEYS[a]]]["eta"] for a in types}
    names = [NAMES[KEYS[a]] for a in types]
    A, Bi = torch.arange(0, len(ref), 2), torch.arange(1, len(ref), 2)
    refB = torch.stack([(lab[Bi] == a).double().mean() for a in types])
    model = torch.load(samples, map_location="cpu", weights_only=False)["model_samples"].double()
    qm, gm, _ = quench(model)
    lab_m = state_ids(qm.double())
    asw = torch.stack([(lab_m == a).double().mean() for a in types])
    print(f"reference half B: {len(Bi)} samples; AS: {len(model)} samples ({100 * float((gm < 1e-4).double().mean()):.1f} % quench-converged); eta_a: " + ", ".join(f"{n} {etas[a]}" for n, a in zip(names, types)))
    runs = {"(c) ref-fitted": mass_run("c", ref[A], lab[A], types, mu8, etas, refB, asw, 10),
            "(d) AS-fitted": mass_run("d", model, lab_m, types, mu8, etas, refB, asw, 20)}
    d_as = torch.log(asw.clamp(min=1e-12) / refB).abs()
    verdicts = {}
    for title, r in runs.items():
        print(f"\n{title}: Gelfand-Dey with leakage correction, against ref_a (half B)")
        print(f"{'basin':26}{'ref_B':>8}{'AS':>8}{'b_hat':>8}{'SE':>8}{'b_hat f=1':>11}{'SE':>8}{'f_a':>8}{'eta_a':>7}{'|log b/ref|':>13}{'|log AS/ref|':>14}")
        dh = torch.log(r["b"] / refB).abs()
        for i, a in enumerate(types):
            print(f"{names[i]:26}{float(refB[i]):8.4f}{float(asw[i]):8.4f}{float(r['b'][i]):8.4f}{float(r['se'][i]):8.4f}{float(r['b_u'][i]):11.4f}{float(r['se_u'][i]):8.4f}{float(r['f'][i]):8.3f}{etas[a]:7.1f}{float(dh[i]):13.3f}{float(d_as[i]):14.3f}")
        r["dh"] = dh
    big3 = refB.argsort(descending=True)[:3]
    c_ok = bool((runs["(c) ref-fitted"]["dh"][big3] <= 0.1).all())
    dd = runs["(d) AS-fitted"]["dh"]
    d_ok = bool((dd[big3] <= 0.2).all()) and int((dd < d_as).sum()) >= 4
    # step 3
    print("\nstep 3: Pareto k_hat of u_a = N_bar_a e^{E/tau} over ALL AS samples (top 20 %); diagnostic: restricted to the samples of basin a; SD of log N_bar_a + E/tau on reference half B (ref-fitted model)")
    Em, EB = s0.energy(model), s0.energy(ref[Bi])
    kh = []
    print(f"{'basin':26}{'k_hat(all AS)':>15}{'k_hat(in basin)':>17}{'SD (ref B)':>12}{'n_B':>7}")
    for i, a in enumerate(types):
        mu_c, Sigma = runs["(d) AS-fitted"]["fits"][a]
        lu = s0.log_qbar(model, mu_c, (1 - etas[a]) * Sigma) + Em / TAU
        u = torch.exp(lu - lu.max()).numpy()
        k_all, k_in = khat(u), khat(u[(lab_m == a).numpy()])
        kh.append(k_all)
        mc, Sc = runs["(c) ref-fitted"]["fits"][a]
        sel = (lab[Bi] == a).nonzero().flatten()
        sd = float((s0.log_qbar(ref[Bi][sel], mc, (1 - etas[a]) * Sc) + EB[sel] / TAU).std())
        print(f"{names[i]:26}{k_all:15.3f}{k_in:17.3f}{sd:12.3f}{len(sel):7d}")
    b_ok = all(k < 0.5 for k in kh)                                                    # NaN -> False
    # step 4
    print("\nstep 4 (report only): IS ESS with proposal sum_a b_hat_a q_bar_a (unshrunk, exact draws); weights e^{-E} / proposal")
    gen = torch.Generator().manual_seed(7)
    for title, r in runs.items():
        comp = torch.multinomial(r["b"], N_DRAW, replacement=True, generator=gen)
        xs = torch.zeros(N_DRAW, 8)
        for i, a in enumerate(types):
            sel = (comp == i).nonzero().flatten()
            if len(sel):
                xs[sel] = draw(*r["fits"][a], len(sel), gen)
        logq = torch.logsumexp(torch.stack([torch.log(r["b"][i]) + s0.log_qbar(xs, *r["fits"][a]) for i, a in enumerate(types)], 1), 1)
        lw = -s0.energy(xs) - logq
        w = torch.exp(lw - lw.max())
        ess = float(w.sum() ** 2 / (w ** 2).sum())
        print(f"  {title}: ESS = {ess:.0f} / {N_DRAW} ({100 * ess / N_DRAW:.1f} %)")
    print(f"\nGO rule: (b) k_hat < 0.5 for every basin: {b_ok}  (c) ref-fitted <= 0.1 on the three largest: {c_ok}  (d) AS-fitted <= 0.2 on the three largest and closer than AS for >= 4 of 5: {d_ok}")
    print("VERDICT: " + ("GO" if (b_ok and c_ok and d_ok) else "NO-GO") + ("" if c_ok else "  [(c) fails: the single-Gaussian model is not enough]") + ("  [(c) passes, (d) fails: AS shape error biases Gelfand-Dey]" if (c_ok and not d_ok) else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["eta", "mass"], required=True)
    ap.add_argument("--samples", default=None)
    a = ap.parse_args()
    if a.stage == "eta":
        stage_eta()
    else:
        stage_mass(a.samples)


if __name__ == "__main__":
    main()
