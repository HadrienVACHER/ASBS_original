""" DW-4 post-hoc baselines (REPORT ONLY, no rule, no training): what the variant-B basin model alone achieves.

      python scripts/dw4_division_b_baseline.py --samples <ed8bb67 AS samples>

    Same fits as scripts/dw4_division_b.py (ref-fitted on half A, AS-fitted; eta_a from scripts/dw4_division_b_eta.json; leakage-corrected Gelfand-Dey b_hat).
    1. Importance sampling with the step-4 proposal sum_a b_hat_a q_bar_a (unshrunk; N = 20000 exact draws, same seed as step 4), weights e^{-E} / proposal:
       Pareto k_hat of the weights (top 20 %; the ESS is unreliable if k_hat > 0.7), IS-reweighted basin proportions p_IS (the draws are quenched and labelled) against ref_a (half B), bootstrap SE (500),
       |log(p_IS/ref)| next to |log(AS/ref)|.
    2. Basin-level reweighting of the AS samples themselves: p_a = b_hat_a (no IS), |log(b_hat_a / ref_a)| with its bootstrap SE.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dw4_division_b as b  # noqa: E402
import dw4_division_step0 as s0  # noqa: E402
from adjoint_samplers.utils.dw4_quench import quench  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, state_ids  # noqa: E402

N, N_BOOT = 20000, 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    a = ap.parse_args()
    eta_j = json.loads(b.ETA_FILE.read_text())
    ref, q, lab, types, mu8 = b.load_reference()
    etas = {t: eta_j[NAMES[KEYS[t]]]["eta"] for t in types}
    names = [NAMES[KEYS[t]] for t in types]
    A, Bi = torch.arange(0, len(ref), 2), torch.arange(1, len(ref), 2)
    refB = torch.stack([(lab[Bi] == t).double().mean() for t in types])
    model = torch.load(a.samples, map_location="cpu", weights_only=False)["model_samples"].double()
    qm, _, _ = quench(model)
    lab_m = state_ids(qm.double())
    asw = torch.stack([(lab_m == t).double().mean() for t in types])
    d_as = torch.log(asw.clamp(min=1e-12) / refB).abs()
    runs = {"ref-fitted": b.mass_run("c", ref[A], lab[A], types, mu8, etas, refB, asw, 10),
            "AS-fitted": b.mass_run("d", model, lab_m, types, mu8, etas, refB, asw, 20)}
    gen = torch.Generator().manual_seed(7)                                       # same stream as step 4
    print(f"reference half B / AS ({len(model)} samples) basin weights: " + ", ".join(f"{n} {float(r):.4f}/{float(w):.4f}" for n, r, w in zip(names, refB, asw)))
    for title, r in runs.items():
        comp = torch.multinomial(r["b"], N, replacement=True, generator=gen)
        xs = torch.zeros(N, 8)
        for i, t in enumerate(types):
            sel = (comp == i).nonzero().flatten()
            if len(sel):
                xs[sel] = b.draw(*r["fits"][t], len(sel), gen)
        logq = torch.logsumexp(torch.stack([torch.log(r["b"][i]) + s0.log_qbar(xs, *r["fits"][t]) for i, t in enumerate(types)], 1), 1)
        lw = -s0.energy(xs) - logq
        w = torch.exp(lw - lw.max())
        ess = float(w.sum() ** 2 / (w ** 2).sum())
        kh = b.khat(w.numpy())
        qs, _, _ = quench(xs)
        labd = state_ids(qs.double())
        onehot = torch.stack([(labd == t).double() for t in types], 1)            # (N, 5)
        p = (w[:, None] * onehot).sum(0) / w.sum()
        g = torch.Generator().manual_seed(3)
        boots = []
        for _ in range(N_BOOT):
            idx = torch.randint(N, (N,), generator=g)
            boots.append((w[idx, None] * onehot[idx]).sum(0) / w[idx].sum())
        se = torch.stack(boots).std(0)
        print(f"\n[{title}] proposal sum_a b_hat_a q_bar_a: N = {N}, ESS = {ess:.0f} ({100 * ess / N:.1f} %), Pareto k_hat of the IS weights = {kh:.3f}" + ("  (> 0.7: ESS unreliable)" if not kh <= 0.7 else ""))
        print(f"{'basin':26}{'ref_B':>8}{'AS':>8}{'p_IS':>8}{'SE':>8}{'|log p_IS/ref|':>16}{'|log AS/ref|':>14}{'b_hat':>8}{'SE':>8}{'|log b_hat/ref|':>17}")
        for i, t in enumerate(types):
            print(f"{names[i]:26}{float(refB[i]):8.4f}{float(asw[i]):8.4f}{float(p[i]):8.4f}{float(se[i]):8.4f}{abs(float(torch.log(p[i].clamp(min=1e-12) / refB[i]))):16.3f}{float(d_as[i]):14.3f}"
                  f"{float(r['b'][i]):8.4f}{float(r['se'][i]):8.4f}{abs(float(torch.log(r['b'][i] / refB[i]))):17.3f}")
        print(f"  sum_a |log p_IS/ref| = {float((torch.log(p.clamp(min=1e-12) / refB).abs()).sum()):.3f}; sum_a |log b_hat/ref| = {float(torch.log(r['b'] / refB).abs().sum()):.3f}; sum_a |log AS/ref| = {float(d_as.sum()):.3f}")


if __name__ == "__main__":
    main()
