""" TEST 3 (décisif, après la fin du run dw4_as_states) : le noyau de réflexions à t = 0 appliqué aux échantillons de AS.

      python scripts/dw4_reflection_test3.py --samples results/dw4_as_states_s2/seed0/samples_final.pt [--line-only]

    Applique 1, 5 et 20 pas du noyau (r uniforme, T_r x accepté avec probabilité π(T_r x) / (π(x) + π(T_r x))) aux 5000 premiers échantillons de AS, et rapporte : fréquences
    d'états contre la référence (classes du graphe des paires courtes, comme analyse_dw4.py), TV des états après chaque nombre de pas, plancher (deux sous-échantillons
    disjoints de 5000 de la référence), et les comptes de transitions état -> état d'un pas (connectivité). Prédiction : le TV décroît vers le plancher (états rares en baisse,
    chemin à 3 particules en hausse). Sinon, les réflexions ne relient pas les bons états.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils import dw4_reflections as R  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, state_freq, state_ids, state_tv  # noqa: E402

N, REPEATS, STEPS = 5000, 25, (1, 5, 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--line-only", action="store_true", help="12 line reflections only (default: 24 maps, line + point reflections)")
    a_ = None
    a = ap.parse_args()
    a.variant = not a.line_only
    root = Path(__file__).resolve().parents[1]
    ref = torch.tensor(np.load(root / "data" / "test_split_DW4.npy", allow_pickle=True)).double()
    model = torch.load(a.samples, map_location="cpu", weights_only=False)["model_samples"].double()[:N]
    g = torch.Generator().manual_seed(0)
    fr = state_freq(state_ids(ref.float()))
    floors, tvs = [], {0: []}
    sub = [torch.randperm(len(ref), generator=g) for _ in range(REPEATS)]
    for p in sub:
        floors.append(state_tv(state_freq(state_ids(ref[p[:N]].float())), state_freq(state_ids(ref[p[N:2 * N]].float()))))
    p95 = float(np.percentile(floors, 95))
    freqs = {0: state_freq(state_ids(model.float()))}
    x, done = model.clone(), 0
    for s in STEPS:
        for _ in range(s - done):
            x, _r = R.kernel_step(x, a.variant, g)
        done = s
        freqs[s] = state_freq(state_ids(x.float()))
    print(f"kernel with R = {24 if a.variant else 12} maps ; n = {N} AS samples ; floor (two disjoint reference subsamples of {N}): mean {np.mean(floors):.4f}, p95 {p95:.4f}")
    print(f"\n{'state':28}{'reference':>10}" + "".join(f"{('AS' if s == 0 else f'{s} step'):>9}" for s in freqs) + f"{'ratio AS':>10}{'ratio 20':>10}")
    for k, key in enumerate(KEYS):
        if max(float(fr[k]), *[float(f[k]) for f in freqs.values()]) < 0.0005:
            continue
        print(f"{NAMES[key]:28}{float(fr[k]):10.4f}" + "".join(f"{float(f[k]):9.4f}" for f in freqs.values())
              + f"{(float(freqs[0][k]) / float(fr[k]) if fr[k] > 0 else float('nan')):10.2f}{(float(freqs[20][k]) / float(fr[k]) if fr[k] > 0 else float('nan')):10.2f}")
    print("\nstate TV to the reference (mean over 25 reference subsamples of 5000):")
    for s, f in freqs.items():
        tv = float(np.mean([state_tv(f, state_freq(state_ids(ref[p[:N]].float()))) for p in sub]))
        print(f"  {'AS (0 step)' if s == 0 else f'{s:2d} step(s)':12} TV {tv:.4f}   ({tv / p95:.2f} x floor p95)")
    new, _r = R.kernel_step(model.clone(), a.variant, g)
    A_, B_ = state_ids(model.float()), state_ids(new.float())
    moved = (new - R.recentre(model.reshape(-1, 4, 2)).reshape(-1, 8)).abs().sum(1) > 1e-9
    C = torch.zeros(len(KEYS), len(KEYS), dtype=torch.long)
    for i, j in zip(A_[moved].tolist(), B_[moved].tolist()):
        C[i, j] += 1
    used = [k for k in range(len(KEYS)) if C[k].sum() + C[:, k].sum() > 0]
    print(f"\none step on the AS samples: {int(moved.sum())} of {len(model)} moves accepted ({100 * float(moved.double().mean()):.1f} %); accepted-move counts (rows: before, columns: after):")
    print(" " * 26 + "".join(f"{NAMES[KEYS[k]][:9]:>10}" for k in used))
    for i in used:
        print(f"{NAMES[KEYS[i]][:25]:26}" + "".join(f"{int(C[i, j]):10d}" for j in used))


if __name__ == "__main__":
    main()
