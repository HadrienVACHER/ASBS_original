""" DW-4, post-hoc IS / independence-MH on the ENDPOINTS with an ENCOMPASSING proposal (offline, CPU, no training, no matcher; report only).

    Motivation (user): the division CV needs mode models LESS extended than the target (N_bar / pi bounded), whereas a global IS / MH needs a proposal r that ENCOMPASSES the modes of the target (pi / r bounded).
    The post-hoc baseline of RESULTS.md used the unshrunk G-averaged Gaussian (eta = 0), not an inflated one: ESS 11-31 %, k_hat 0.63-0.72. Here the proposal is inflated on purpose.

      python scripts/dw4_is_inflate.py

    Proposal r = sum_a pi_a q_a, q_a = G-average (48 elements x continuous rotation) of a basin component fitted by EM on the AS samples ed8bb67 (mu_a, Sigma_a of scripts/dw4_division_trunc_params.json),
    mixture weights pi_a = the AS basin frequencies (quenched labels; ed8bb67: 0.4196, 0.3815, 0.1212, 0.0606, 0.0171; the reference is not used to build r).
    Families: Gaussian N(mu_a, s Sigma_a), s in {1, 1.5, 2, 3}; multivariate t with 5 degrees of freedom and scale matrix s Sigma_a, s in {1, 1.5, 2}.
    Density: trapezoid on the 256-angle grid in log space, as scripts/dw4_division_step0.log_qbar (exact for the Gaussian trig polynomial, same Mahalanobis form for the t); draws with continuous angle.
    For each setting: N = 20000 exact draws x ~ r (seed 7), weights w = e^{-E(x)} / r(x), and
      ESS = (sum w)^2 / sum w^2, Pareto k_hat of w (top 20 %, scripts/dw4_division_b.khat),
      self-normalised IS basin proportions after the quench (500-bootstrap SE over the draws),
      independence MH on the same proposals: 200 chains x 99 steps (the 20000 draws are used once, in order), accept min(1, w'/w), proportions of the states of the steps 50-99 (bootstrap over chains), mean acceptance.
    Reference proportions: ref_B (half B, odd indices of data/test_split_DW4.npy, quenched), compared with the same |log p / ref| as the AS (|log AS/ref| = 0.119, 0.073, 0.664, 0.246, 0.893; sum 1.995).

    PRE-REGISTERED (written before the first run; report only, no decision on the method): a setting is USABLE iff ESS > 50 % AND k_hat < 0.5. An encompassing proposal is usable on DW-4 iff at least one setting is.
    Run on the CLUSTER only (the EM coordinates in the params JSON depend on the machine-dependent eigh basis s0.B; the script aborts if E(mu_a) is not near the basin minima).
    Also reported (no rule): the sum over basins of |log p_IS / ref| and |log p_IMH / ref| of each setting.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dw4_division_b as b  # noqa: E402
import dw4_division_step0 as s0  # noqa: E402
from adjoint_samplers.utils.dw4_quench import quench  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, state_ids  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASW = torch.tensor([0.4196, 0.3815, 0.1212, 0.0606, 0.0171], dtype=torch.float64)
AS_ERR = torch.tensor([0.119, 0.073, 0.664, 0.246, 0.893], dtype=torch.float64)
NU = 5.0
N, N_BOOT, SEED = 20000, 500, 7
C, T, BURN = 200, 99, 50


def log_qbar_t(x, mu_c, Sigma, nu, n_ang=s0.N_ANG):
    """ log of the G-average of the multivariate t_nu(mu_c, Sigma) at x (n, 8): same trigonometric interpolation of the Mahalanobis form as s0.log_qbar """
    P = torch.linalg.inv(Sigma)
    logdet = torch.logdet(Sigma)
    imgs = s0.group_act(s0.centre(x))
    th8 = torch.arange(8) * (2 * math.pi / 8)
    w = s0.to_v(s0.rot(imgs[:, :, None], th8[None, None, :])) - mu_c
    m8 = torch.einsum("nhaj,jk,nhak->nha", w, P, w)
    c = torch.fft.rfft(m8, dim=-1)
    c = torch.nn.functional.pad(c, (0, n_ang // 2 + 1 - c.shape[-1]))
    m = torch.fft.irfft(c, n=n_ang, dim=-1) * (n_ang / 8)
    d = 6
    lc = math.lgamma((nu + d) / 2) - math.lgamma(nu / 2) - 0.5 * d * math.log(nu * math.pi) - 0.5 * logdet
    return torch.logsumexp(lc - 0.5 * (nu + d) * torch.log1p(m / nu), dim=(1, 2)) - math.log(48 * n_ang)


def draw_t(mu_c, Sigma, nu, n, gen):
    Lc = torch.linalg.cholesky(Sigma)
    chi = torch.distributions.Chi2(torch.tensor(nu, dtype=torch.float64)).sample((n,))
    g = torch.randn(n, 6, generator=gen, dtype=torch.float64) * torch.sqrt(nu / chi)[:, None]
    y = (mu_c + g @ Lc.T) @ s0.B.T
    th = torch.rand(n, generator=gen, dtype=torch.float64) * 2 * math.pi
    z = s0.rot(y.reshape(n, 4, 2), -th)
    h = torch.randint(48, (n,), generator=gen)
    return s0.group_act(z)[torch.arange(n), h].reshape(n, 8)


def build(fam, s, comps, gen):
    """ draws x (N, 8) and log r(x) for the mixture """
    comp = torch.multinomial(ASW, N, replacement=True, generator=gen)
    x = torch.zeros(N, 8, dtype=torch.float64)
    for i, (mu, Sg) in enumerate(comps):
        sel = (comp == i).nonzero().flatten()
        if len(sel):
            x[sel] = b.draw(mu, s * Sg, len(sel), gen) if fam == "gauss" else draw_t(mu, s * Sg, NU, len(sel), gen)
    lr = torch.stack([math.log(float(ASW[i])) + (s0.log_qbar(x, mu, s * Sg) if fam == "gauss" else log_qbar_t(x, mu, s * Sg, NU)) for i, (mu, Sg) in enumerate(comps)], 1)
    return x, torch.logsumexp(lr, 1)


def imh(lw, lab, gen):
    idx = torch.arange(C * (T + 1)).reshape(C, T + 1)
    state, acc, labs = idx[:, 0], [], []
    for t in range(1, T + 1):
        prop = idx[:, t]
        a = torch.exp((lw[prop] - lw[state]).clamp(max=0))
        ok = torch.rand(C, generator=gen, dtype=torch.float64) < a
        state = torch.where(ok, prop, state)
        acc.append(ok.double().mean())
        if t >= BURN:
            labs.append(lab[state])
    return float(torch.stack(acc).mean()), torch.stack(labs)                                   # (T - BURN + 1, C)


def props(counts_fn, K=5):
    return counts_fn


def main():
    ref, q, lab_r, types, mu8 = b.load_reference()
    names = [NAMES[KEYS[k]] for k in types]
    Bi = torch.arange(1, len(ref), 2)
    refB = torch.stack([(lab_r[Bi] == k).double().mean() for k in types])
    pj = json.loads((ROOT / "scripts" / "dw4_division_trunc_params.json").read_text())
    comps = [(torch.tensor(pj[n]["mu"], dtype=torch.float64), torch.tensor(pj[n]["Sigma"], dtype=torch.float64)) for n in names]
    # the V-coordinates (mu_a, Sigma_a) of the params file depend on the basis s0.B = eigh of a degenerate projector, which is machine dependent: they are only valid where they were produced (the cluster)
    e_mu = [float(s0.energy((mu @ s0.B.T)[None])[0]) for mu, _ in comps]
    if max(e_mu) > -5.0:
        sys.exit("basis mismatch: E(mu_a) = " + ", ".join(f"{e:.1f}" for e in e_mu) + " (expected near -25): the params JSON was produced with another eigh basis; run this script on the cluster")
    print(f"E(mu_a) (sanity, expected near the basin minima): " + ", ".join(f"{e:.1f}" for e in e_mu))
    print(f"reference half B: " + ", ".join(f"{n} {float(r):.4f}" for n, r in zip(names, refB)) + f"\nproposal weights (AS frequencies): " + ", ".join(f"{float(w):.4f}" for w in ASW))
    settings = [("gauss", s) for s in (1.0, 1.5, 2.0, 3.0)] + [("t5", s) for s in (1.0, 1.5, 2.0)]
    print(f"\n{'family':8}{'s':>5}{'ESS':>8}{'ESS %':>8}{'k_hat':>8}{'max w frac':>12}{'accept':>8}{'sum|log IS/ref|':>17}{'sum|log IMH/ref|':>18}   usable | |log IS/ref| per basin")
    usable = []
    for fam, s in settings:
        gen = torch.Generator().manual_seed(SEED)
        x, lr = build(fam, s, comps, gen)
        lw = -s0.energy(x) - lr
        w = torch.exp(lw - lw.max())
        ess = float(w.sum() ** 2 / (w ** 2).sum())
        kh = b.khat(w.numpy())
        qx, _, _ = quench(x)
        lab = state_ids(qx.double())
        oh = torch.stack([(lab == k).double() for k in types], 1)                                    # (N, 5)
        p_is = (w[:, None] * oh).sum(0) / w.sum()
        g2 = torch.Generator().manual_seed(SEED + 1)
        boots = torch.stack([(lambda i: (w[i][:, None] * oh[i]).sum(0) / w[i].sum())(torch.randint(N, (N,), generator=g2)) for _ in range(N_BOOT)])
        acc, labs = imh(lw, lab, torch.Generator().manual_seed(SEED + 2))
        p_mh = torch.stack([(labs == k).double().mean() for k in types])
        e_is = torch.log(p_is / refB).abs()
        e_mh = torch.log(p_mh.clamp(min=1e-12) / refB).abs()
        ok = ess / N > 0.5 and kh < 0.5
        usable.append(ok)
        print(f"{fam:8}{s:5.1f}{ess:8.0f}{100 * ess / N:8.1f}{kh:8.3f}{float(w.max() / w.sum()):12.4f}{acc:8.3f}{float(e_is.sum()):17.3f}{float(e_mh.sum()):18.3f}   {'YES' if ok else 'no':6} | "
              + " ".join(f"{float(e):.3f}" for e in e_is) + f"   (SE p_IS {' '.join(f'{float(v):.4f}' for v in boots.std(0))})", flush=True)
    print(f"\n(raw AS: sum|log AS/ref| = {float(AS_ERR.sum()):.3f}; per basin " + " ".join(f"{float(e):.3f}" for e in AS_ERR) + ")")
    print(f"PRE-REGISTERED (usable iff ESS > 50 % and k_hat < 0.5): usable settings {sum(usable)}/{len(usable)}; an encompassing proposal is usable on DW-4: {any(usable)}")


if __name__ == "__main__":
    main()
