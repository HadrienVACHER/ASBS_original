""" End-to-end EXACT zero-mean test of the DW-4 Barker term, under the law of x_t that TRAINING uses (no conditional estimation).

    x1 ~ reference (all 10k, 2 draws of the noise each); for 20 values of the schedule (kappa from 0.01 to 0.9995), x_t is drawn by the MATCHER'S OWN bridge sampler
    (sde.sample_base_posterior inside prepare_target) and the extra term is extracted from the matcher's own prepare_target (gamma = 1: term = target + adjoint1).
    For equivariant psi(x_t):  S_psi(t) = mean <term, psi(x_t)>  with its standard error. Theory: E[term | x_t] = 0, hence S_psi(t) = 0 for every t and psi.
      psi1   = x_t
      psi2_i = x_t,i |x_t,i|^2
      psi3_i = sum_j (x_t,i - x_t,j) exp(-|x_t,i - x_t,j|^2 / 8)
    PASS iff |S|/SE < 3 for every (t, psi) (all z reported). Positive control: v_t x 1.2 inside the Barker factor only -> |z| must become large.
    Then the same S_psi with the DIAGNOSTIC'S own (kappa, v = nu1 kappa (1 - kappa), nu1 = sigma_max^2, x_t = kappa x1 + sqrt(v) recentred noise) to locate its residual.

    python tests/test_dw4_end_to_end.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_dw4_barker_matcher import build  # noqa: E402
from adjoint_samplers.utils import dw4_reflections as dwr  # noqa: E402

KAPPAS = [0.01, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995]


def psis(xt):
    x = xt.reshape(-1, 4, 2)
    p1 = x
    p2 = x * (x ** 2).sum(-1, keepdim=True)
    d = x[:, :, None, :] - x[:, None, :, :]
    p3 = (d * torch.exp(-(d ** 2).sum(-1, keepdim=True) / 8)).sum(2)
    return [p1.reshape(len(xt), 8), p2.reshape(len(xt), 8), p3.reshape(len(xt), 8)]


def stat(term, xt):
    out = []
    for p in psis(xt):
        v = (term * p).sum(-1)
        out.append((float(v.mean()), float(v.std() / len(v) ** 0.5)))
    return out


def t_of_kappa(m, kappa):
    sde = m.sde.ref_sde
    nu = kappa * sde.total_var
    return float(-torch.log(1 - torch.tensor(nu / sde.sigma_max ** 2, dtype=torch.float64)) / (2 * torch.log(torch.tensor(sde.sigma_diff, dtype=torch.float64))))


def matcher_run(v_scale, reps=2, seed=0):
    m = build(1.0)
    m.v_scale = v_scale
    ref = torch.tensor(np.load(Path(__file__).resolve().parents[1] / "data" / "test_split_DW4.npy", allow_pickle=True)).float()
    x1 = ref.repeat(reps, 1)
    x0 = torch.zeros_like(x1)
    data = {"x0": x0, "x1": x1, "adjoint1": m.grad_term_cost(x1)}
    rows = []
    for kappa in KAPPAS:
        t = torch.full((len(x1), 1), t_of_kappa(m, kappa), dtype=torch.float32)
        m.sample_t = lambda x, _t=t: _t
        torch.manual_seed(seed)
        (tt, xt), target = m.prepare_target(data, "cpu")
        term = target + data["adjoint1"]
        kap_real = float((m._nu(tt) / m._nu(torch.ones_like(tt))).mean())
        rows.append((kap_real, stat(term.double(), xt.double())))
    return rows


def diagnostic_run(reps=2, seed=0):
    ref = dwr.recentre(torch.tensor(np.load(Path(__file__).resolve().parents[1] / "data" / "test_split_DW4.npy", allow_pickle=True)).double().reshape(-1, 4, 2)).reshape(-1, 8)
    x1 = ref.repeat(reps, 1)
    gen = torch.Generator().manual_seed(seed)
    nu1 = 4.0                                                           # sigma_max^2, as in scripts/dw4_barker_diagnostic.py
    rows = []
    for kappa in KAPPAS:
        v = nu1 * kappa * (1 - kappa)
        xt = kappa * x1 + v ** 0.5 * dwr.recentre(torch.randn(len(x1), 4, 2, generator=gen, dtype=torch.float64)).reshape(-1, 8)
        h, _ = dwr.barker_h(x1, xt, kappa, v, True)
        rows.append((kappa, stat(h / (nu1 * (1 - kappa)), xt)))
    return rows


def show(title, rows):
    print(f"\n{title}\n{'kappa':>8}" + "".join(f"{f'z psi{i}':>10}" for i in (1, 2, 3)) + f"{'S psi1':>14}{'SE psi1':>12}")
    zmax = 0.0
    for k, st in rows:
        zs = [s / se if se > 0 else 0.0 for s, se in st]
        zmax = max(zmax, max(abs(z) for z in zs))
        print(f"{k:8.4f}" + "".join(f"{z:+10.2f}" for z in zs) + f"{st[0][0]:14.3e}{st[0][1]:12.2e}")
    print(f"max |z| = {zmax:.2f}")
    return zmax


if __name__ == "__main__":
    z0 = show("MATCHER (own bridge sampler, own prepare_target), gamma = 1", matcher_run(1.0))
    z1 = show("POSITIVE CONTROL: v_t x 1.2 inside the Barker factor only", matcher_run(1.2))
    z2 = show("DIAGNOSTIC's own (kappa, v, x_t generator)", diagnostic_run())
    print(f"\nmatcher passes (max|z| < 3): {z0 < 3} ; positive control detected (max|z| >> 3): {z1 > 10} ; diagnostic's generator passes: {z2 < 3}")
