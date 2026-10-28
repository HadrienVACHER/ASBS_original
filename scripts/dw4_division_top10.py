""" DW-4 division, POST HOC diagnostic of the step-1 failure (declared post hoc, report only, no rule, no training): what are the samples that carry the heavy tail of u_a = N_bar_a e^{E/tau}?

      python scripts/dw4_division_top10.py --samples results/dw4_e1500_ed8bb67_samples.pt

    Same model as step 1 (EM-fitted mu_a, Sigma_a on the AS samples ed8bb67, eta_a of step 1, untruncated N_bar_a = G-average of N(mu_a, (1 - eta_a) Sigma_a) via log_qbar). For each of the three basins that failed
    (two disjoint pairs, K4 minus an edge, star): the 10 AS samples with the largest u_a, with the basin of the sample, E, log N_bar_a, log u_a and the max pair distance, plus the same summary for the top 100.
    User's prediction to check: the top samples are high-energy samples with a stretched pair distance, well beyond 5.5.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dw4_division_step0 as s0  # noqa: E402
import dw4_division_trunc as tr  # noqa: E402
from adjoint_samplers.utils.dw4_states import pair_distances  # noqa: E402

FAILING = ("2 disjoint pairs", "K4 minus an edge", "star (3 edges)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    a = ap.parse_args()
    d = tr.load_as_fit(a.samples)
    x, E, lab_m, names, types = d["model"], s0.energy(d["model"]), d["lab_m"], d["names"], d["types"]
    dmax = pair_distances(x).max(1).values
    own_d = lambda k: dmax[lab_m == k]
    print(f"AS samples: {len(x)}; max pair distance over all AS samples: median {float(dmax.median()):.2f}, p99 {float(dmax.quantile(0.99)):.2f}, max {float(dmax.max()):.2f}; E: median {float(E.median()):.2f}, p99 {float(E.quantile(0.99)):.2f}")
    for k, name in zip(types, names):
        if name not in FAILING:
            continue
        mu, Sg = d["fit"][k]
        ln = s0.log_qbar(x, mu, (1 - tr.ETA_EM[name]) * Sg)
        lu = ln + E
        top = lu.topk(100).indices
        print(f"\n== {name} (eta {tr.ETA_EM[name]}): top 10 AS samples by u_a   [own-basin samples: median max-pair-distance {float(own_d(k).median()):.2f}, median E {float(E[lab_m == k].median()):.2f}]")
        print(f"{'rank':>4}{'index':>7}{'basin of the sample':>26}{'E':>9}{'log N_bar':>11}{'log u':>9}{'max pair dist':>15}")
        for r, i in enumerate(top[:10]):
            print(f"{r + 1:4d}{int(i):7d}{names[types.index(int(lab_m[i]))]:>26}{float(E[i]):9.2f}{float(ln[i]):11.2f}{float(lu[i]):9.2f}{float(dmax[i]):15.2f}")
        own = (lab_m[top] == k).double().mean()
        print(f"   top 100: in own basin {100 * float(own):.0f} %, max pair distance > 5.5 for {int((dmax[top] > 5.5).sum())}, median E {float(E[top].median()):.2f}, median max pair distance {float(dmax[top].median()):.2f}; "
              f"top 10: max pair distance > 5.5 for {int((dmax[top[:10]] > 5.5).sum())}")
        # share of the total weight carried by the top samples
        w = torch.softmax(lu, 0)
        print(f"   share of sum_i u_a(x_i) carried by the top 1 / 10 / 100 samples: {float(w[top[0]]):.3f} / {float(w[top[:10]].sum()):.3f} / {float(w[top].sum()):.3f}")


if __name__ == "__main__":
    main()
