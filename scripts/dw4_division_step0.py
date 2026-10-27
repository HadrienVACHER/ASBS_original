""" DW-4 division control variate, STEP 0 (offline, CPU, no matcher, no training): does a G-averaged Gaussian basin model recover the reference basin weights from AS samples?

      python scripts/dw4_division_step0.py --samples results/dw4_as_states/seed0/samples_final.pt        # AS run ed8bb67 (model_samples)
      python scripts/dw4_division_step0.py                                                                # reference-fitted part only

    Model, per quenched basin type a (the 5 types present in the quenched reference), G = O(2) x S4 on the zero-CoM subspace V (dim 6), |S4 x Z2| = 48:
        q_bar_a(x) = (1/48) sum_{h in S4 x Z2} (1/2pi) int N(R_{-theta} h^{-1} x; mu_a, Sigma_a) dtheta
    For fixed h the theta-integrand is exp(trig polynomial of degree 2): the Mahalanobis form is a trig polynomial of degree <= 2, obtained EXACTLY from 8 angles (trigonometric interpolation)
    and evaluated on a 256-angle trapezoid rule in log space (logsumexp); checked against 1024 angles.
    Fit: mu_a = lowest-energy quenched minimum of type a (canonical frame); Sigma_a = empirical covariance of the samples of basin a after aligning each to mu_a (best h in 48 and the
    closed-form 2D Procrustes angle), the variance along the rotation tangent of mu_a being set to the median vibrational eigenvalue (the 5 others). Shrunk N_bar_a: same with (1 - eta) Sigma_a, eta = 0.2.
    Gelfand-Dey per basin: exp(L_a) = mean over the samples in basin a of N_bar_a(x) exp(E(x)/tau);  b_hat = softmax(-L).
    Basin of a sample = state of its quenched configuration (graph of short pairs after the quench).

    PRE-REGISTERED GO RULE (written before the first run; reference basin weights = frequencies of the 10 000 quenched reference samples; "three largest" = by reference weight):
        GO  iff, with AS-fitted parameters, |log(b_hat_a / ref_a)| <= 0.1 for the three largest basins
            AND |log(b_hat_a / ref_a)| < |log(AS_a / ref_a)| for all five basins.
        A basin with fewer than 30 AS samples cannot be fitted: b_hat_a is undefined and the rule fails (the verdict is then NO-GO, stated as such). Otherwise: stop and report.
"""
import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils.dw4_quench import energy_grad_hess, quench  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, state_ids  # noqa: E402

torch.set_default_dtype(torch.float64)
ETA, TAU, N_ANG, MIN_N = 0.2, 1.0, 256, 30
PERMS = list(itertools.permutations(range(4)))                                   # S4 (24); with the reflection y -> -y: 48 elements


def basis():
    """ B (8, 6): orthonormal basis of the zero-CoM subspace of R^8 (4 particles x 2) """
    P = torch.eye(8) - torch.kron(torch.ones(4, 4) / 4, torch.eye(2))
    w, V = torch.linalg.eigh(P)
    return V[:, w > 0.5]


B = basis()


def centre(x):
    z = x.reshape(-1, 4, 2)
    return z - z.mean(1, keepdim=True)


def group_act(z):
    """ all 48 images h z of the centred configurations z (n, 4, 2): (n, 48, 4, 2) """
    out = []
    for p in PERMS:
        zp = z[:, list(p), :]
        out.append(zp)
        out.append(zp * torch.tensor([1.0, -1.0]))
    return torch.stack(out, 1)


def rot(z, theta):
    """ R_{-theta} z for z (..., 4, 2) and theta (..., ) broadcastable on the leading dims """
    c, s = torch.cos(theta)[..., None], torch.sin(theta)[..., None]
    return torch.stack([c * z[..., 0] + s * z[..., 1], -s * z[..., 0] + c * z[..., 1]], -1)


def to_v(z):
    return z.reshape(*z.shape[:-2], 8) @ B


def log_qbar(x, mu_c, Sigma, n_ang=N_ANG):
    """ log of q_bar for x (n, 8) (any translation) with mu_c (6,) and Sigma (6, 6) in the coordinates of V """
    P = torch.linalg.inv(Sigma)
    logdet = torch.logdet(2 * math.pi * Sigma)
    imgs = group_act(centre(x))                                                   # (n, 48, 4, 2)
    th8 = torch.arange(8) * (2 * math.pi / 8)
    w = to_v(rot(imgs[:, :, None], th8[None, None, :])) - mu_c                  # (n, 48, 8, 6)
    m8 = torch.einsum("nhaj,jk,nhak->nha", w, P, w)                             # Mahalanobis at 8 angles: trig polynomial of degree <= 2 in theta
    c = torch.fft.rfft(m8, dim=-1)                                              # exact trigonometric interpolation onto n_ang angles
    c = torch.nn.functional.pad(c, (0, n_ang // 2 + 1 - c.shape[-1]))
    m = torch.fft.irfft(c, n=n_ang, dim=-1) * (n_ang / 8)                       # (n, 48, n_ang)
    return torch.logsumexp(-0.5 * m - 0.5 * logdet, dim=(1, 2)) - math.log(48 * n_ang)


def align(x, mu8):
    """ x (n, 8) -> R_{-theta} h^{-1} x closest to mu8 (4, 2) (centred): best h among 48, closed-form 2D Procrustes angle """
    imgs = group_act(centre(x))                                                   # (n, 48, 4, 2)
    zc = torch.complex(imgs[..., 0], imgs[..., 1])
    mc = torch.complex(mu8[:, 0], mu8[:, 1])
    S = (mc.conj() * zc).sum(-1)                                                  # (n, 48): max_theta Re(e^{-i theta} S) = |S| at theta = arg S
    best = S.abs().argmax(1)
    idx = torch.arange(len(x))
    return rot(imgs[idx, best], torch.angle(S[idx, best]))                        # (n, 4, 2)


def fit_basin(x, mu8, eta=ETA):
    """ mu_c, Sigma (unshrunk), Sigma_shrunk from the samples x (n, 8) of one basin """
    mu_c = to_v(mu8)
    d = to_v(align(x, mu8)) - mu_c
    d = d - d.mean(0)
    Sigma = d.T @ d / (len(d) - 1)
    tang = torch.stack([-mu8[:, 1], mu8[:, 0]], -1)                              # J mu: rotation tangent
    t = to_v(tang)
    t = t / t.norm()
    Pperp = torch.eye(6) - torch.outer(t, t)
    Sp = Pperp @ Sigma @ Pperp
    vib = torch.linalg.eigvalsh(Sp)[1:]                                           # the 5 vibrational eigenvalues (the tangent direction has eigenvalue 0)
    Sigma = Sp + float(vib.median()) * torch.outer(t, t)
    return mu_c, Sigma, (1 - eta) * Sigma


def energy(x):
    return energy_grad_hess(x)[0]


def basin_labels(q):
    ids = state_ids(q)
    return ids


def fit_all(x, labels, types, mu8s):
    """ per type: (mu_c, Sigma, Sigma_shrunk, indices) -- None if fewer than MIN_N samples """
    fits = {}
    for a in types:
        idx = (labels == a).nonzero().flatten()
        fits[a] = fit_basin(x[idx], mu8s[a]) + (idx,) if len(idx) >= MIN_N else None
    return fits


def gd(x, E, fits, types):
    """ L_a (nan when the basin is not fitted), b_hat = softmax(-L) """
    L = torch.full((len(types),), float("nan"))
    for i, a in enumerate(types):
        f = fits[a]
        if f is None:
            continue
        idx = f[3]
        lN = log_qbar(x[idx], f[0], f[2]) + E[idx] / TAU
        L[i] = torch.logsumexp(lN, 0) - math.log(len(idx))                        # exp(L_a) = mean(N_bar e^{E/tau})
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=None, help="samples_final.pt of the AS run (model_samples)")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    ref = torch.tensor(np.load(root / "data" / "test_split_DW4.npy", allow_pickle=True)).double()
    qr, gr, _ = quench(ref)
    lab_r = state_ids(qr)
    freq = torch.bincount(lab_r, minlength=len(KEYS)).double() / len(lab_r)
    types = [k for k in range(len(KEYS)) if freq[k] > 0.001]
    types.sort(key=lambda k: -float(freq[k]))
    print(f"quenched reference ({len(ref)} samples, {100 * float((gr < 1e-4).double().mean()):.1f} % converged): 5 types expected, found {len(types)}")
    Er = energy(qr)
    mu8s = {}
    for k in types:                                                               # lowest-energy quenched minimum of the type, canonical frame (centred)
        idx = (lab_r == k).nonzero().flatten()
        mu8s[k] = centre(qr[idx[Er[idx].argmin()][None]])[0]
    refw = torch.stack([freq[k] for k in types])
    names = [NAMES[KEYS[k]] for k in types]
    # sanity of the angle quadrature on one basin
    E_ref = energy(ref)
    half = len(ref) // 2
    A_idx, B_idx = torch.arange(half), torch.arange(half, len(ref))
    f0 = fit_basin(ref[(lab_r == types[0]).nonzero().flatten()][:2000], mu8s[types[0]])
    xs = ref[:300]
    d_ang = float((log_qbar(xs, f0[0], f0[1], 256) - log_qbar(xs, f0[0], f0[1], 1024)).abs().max())
    # direct check of the trigonometric interpolation at random angles
    z = group_act(centre(xs[:5]))[:, 3]
    th = torch.rand(5, 7) * 2 * math.pi
    P = torch.linalg.inv(f0[1])
    direct = ((to_v(rot(z[:, None], th)) - f0[0]) @ P * (to_v(rot(z[:, None], th)) - f0[0])).sum(-1)
    th8 = torch.arange(8) * (2 * math.pi / 8)
    c = torch.fft.rfft(torch.einsum("naj,jk,nak->na", to_v(rot(z[:, None], th8)) - f0[0], P, to_v(rot(z[:, None], th8)) - f0[0]), dim=-1)
    pred = torch.stack([(c[:, 0].real + 2 * (c[:, 1:4] * torch.exp(1j * torch.arange(1, 4) * th[:, i:i + 1])).real.sum(-1) + (c[:, 4] * torch.cos(4 * th[:, i])).real) / 8 for i in range(7)], 1)
    print(f"quadrature: max |log q_bar(256 angles) - log q_bar(1024 angles)| = {d_ang:.2e}; trig-polynomial degree-2 interpolation vs direct evaluation at random angles: {float((direct - pred).abs().max()):.2e}")

    def report(title, x_fit, lab_fit, E_fit, ref_eval=True):
        fits = fit_all(x_fit, lab_fit, types, mu8s)
        L = gd(x_fit, E_fit, fits, types)
        bh = torch.softmax(-L, 0) if not torch.isnan(L).any() else torch.full_like(L, float("nan"))
        return fits, L, bh

    # ---- reference-fitted (fit AND Gelfand-Dey on half A of the reference; diagnostics on half B)
    fitsR, LR, bR = report("ref", ref[A_idx], lab_r[A_idx], E_ref[A_idx])
    # ---- AS-fitted
    out = {}
    if a.samples:
        model = torch.load(a.samples, map_location="cpu", weights_only=False)["model_samples"].double()
        qm, gm, _ = quench(model)
        lab_m = state_ids(qm)
        fm = torch.bincount(lab_m, minlength=len(KEYS)).double() / len(lab_m)
        asw = torch.stack([fm[k] for k in types])
        fitsA, LA, bA = report("AS", model, lab_m, energy(model))
        out = dict(asw=asw, bA=bA, n_as=[0 if fitsA[k] is None else len(fitsA[k][3]) for k in types], nm=len(model), conv=float((gm < 1e-4).double().mean()))
    print(f"\n{'basin':26}{'ref':>9}" + (f"{'AS':>9}{'b_hat(AS-fit)':>15}{'|log b/ref|':>13}{'|log AS/ref|':>14}" if out else "") + f"{'b_hat(ref-fit)':>16}{'|log b/ref|':>13}   n_ref_A")
    for i, k in enumerate(types):
        s = f"{names[i]:26}{float(refw[i]):9.4f}"
        if out:
            s += f"{float(out['asw'][i]):9.4f}{float(out['bA'][i]):15.4f}{abs(float(torch.log(out['bA'][i] / refw[i]))):13.3f}{abs(float(torch.log(out['asw'][i].clamp(min=1e-12) / refw[i]))):14.3f}"
        s += f"{float(bR[i]):16.4f}{abs(float(torch.log(bR[i] / refw[i]))):13.3f}   {len(fitsR[k][3]) if fitsR[k] else 0}"
        print(s)
    # ---- diagnostics on the held-out half B of the reference: SD of log q_bar + E, and coverage
    print("\nreference-fitted model, held-out half of the reference: SD within basin of log q_bar_a(x) + E(x)/tau (unshrunk q_bar):")
    xb, lb, Eb = ref[B_idx], lab_r[B_idx], E_ref[B_idx]
    lq = {}
    cov = torch.zeros(len(xb))
    for i, k in enumerate(types):
        if fitsR[k] is None:
            continue
        idx = (lb == k).nonzero().flatten()
        lqa = log_qbar(xb, fitsR[k][0], fitsR[k][1])
        lq[k] = lqa
        v = lqa[idx] + Eb[idx] / TAU
        print(f"  {names[i]:26} n = {len(idx):5d}   SD {float(v.std()):.3f}   mean {float(v.mean()):.3f}")
        cov = cov + torch.exp(-LR[i] + lqa + Eb / TAU)
    print(f"coverage sum_a exp(-L_a) q_bar_a(x) e^{{E/tau}} on held-out reference: mean {float(cov.mean()):.3f}  median {float(cov.median()):.3f}  p5 {float(cov.quantile(0.05)):.3f}  p95 {float(cov.quantile(0.95)):.3f}")
    if out:
        print(f"\nAS samples: {out['nm']} ({100 * out['conv']:.1f} % quench-converged); per-basin n: " + ", ".join(f"{nm} {n}" for nm, n in zip(names, out["n_as"])))
        d_hat = torch.log(out["bA"] / refw).abs()
        d_as = torch.log(out["asw"].clamp(min=1e-12) / refw).abs()
        big3 = refw.argsort(descending=True)[:3]
        undefined = bool(torch.isnan(out["bA"]).any())
        c1 = (not undefined) and bool((d_hat[big3] <= 0.1).all())
        c2 = (not undefined) and bool((d_hat < d_as).all())
        print(f"GO rule: (1) |log b_hat/ref| <= 0.1 for the three largest basins: {c1}  (2) smaller than |log AS/ref| for all five: {c2}" + ("  [b_hat undefined: a basin has < 30 AS samples]" if undefined else ""))
        print("VERDICT: " + ("GO" if (c1 and c2) else "NO-GO"))


if __name__ == "__main__":
    main()
