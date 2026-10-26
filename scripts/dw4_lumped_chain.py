""" Chaîne LUMPED des états de DW-4 sous le noyau à 24 applications, construite sur la référence avec les conditionnelles exactes à l'intérieur de chaque état (rapport seulement, CPU).

      python scripts/dw4_lumped_chain.py

    P_ab = K_ab / p_a (a != b) avec le flux Rao-Blackwellisé K_ab = E_ref[1{s(x) = a} (1/R) sum_r w_r 1{s(T_r x) = b}], P_aa = 1 - sum_{b != a} P_ab, sur les 6 états de fréquence de
    référence >= 0.5 % (les classes rares sont traitées comme « rester »). Rapporte : réversibilité par rapport aux poids de référence, valeurs propres et trou spectral, puis le nombre
    de pas du noyau pour diviser par deux le déficit du triangle+pendant en partant des proportions d'états d'AS (run ed8bb67, états bruts), par itération exacte et par la décroissance spectrale.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils import dw4_reflections as R  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, state_freq, state_ids  # noqa: E402

AS_RAW = {"path of 2 (3 particles)": 0.3780, "2 disjoint pairs": 0.0842, "star (3 edges)": 0.0218, "path of 3 (4 particles)": 0.0450, "triangle + pendant": 0.4052, "K4 minus an edge": 0.0624}


def main():
    root = Path(__file__).resolve().parents[1]
    ref = torch.tensor(np.load(root / "data" / "test_split_DW4.npy", allow_pickle=True)).double()
    label = lambda z: state_ids(z.float())
    D, Fl, lab, K = R.rb_terms(ref, label, True, n_classes=len(KEYS))
    p = torch.bincount(lab, minlength=len(KEYS)).double() / len(lab)
    Kab = Fl.mean(0)
    main_ids = [k for k in range(len(KEYS)) if p[k] >= 0.005]
    names = [NAMES[KEYS[k]] for k in main_ids]
    n = len(main_ids)
    P = torch.zeros(n, n, dtype=torch.float64)
    for i, a in enumerate(main_ids):
        for j, b in enumerate(main_ids):
            if i != j:
                P[i, j] = Kab[a, b] / p[a]
        P[i, i] = 1 - P[i].sum()
    pi = p[main_ids] / p[main_ids].sum()
    print(f"états retenus ({n}), fréquence de référence renormalisée :")
    for nm, w in zip(names, pi):
        print(f"  {nm:28} {float(w):.4f}")
    print("\nmatrice de transition P (lignes : état de départ), un pas du noyau :")
    print(" " * 28 + "".join(f"{nm[:10]:>11}" for nm in names))
    for i, nm in enumerate(names):
        print(f"{nm:28}" + "".join(f"{float(P[i, j]):11.5f}" for j in range(n)))
    flux = pi[:, None] * P
    rev = float((flux - flux.T).abs().max() / flux.max())
    print(f"\nréversibilité par rapport aux poids de référence : max|pi_a P_ab - pi_b P_ba| / max flux = {rev:.3f} (les classes rares ignorées et le bruit d'échantillonnage l'expliquent)")
    S = torch.diag(pi.sqrt()) @ P @ torch.diag(1 / pi.sqrt())
    lam = torch.linalg.eigvals(0.5 * (S + S.T)).real.sort(descending=True).values
    gap = float(1 - lam[1])
    print(f"valeurs propres : {[round(float(l), 5) for l in lam]} ; trou spectral 1 - λ2 = {gap:.5f} ; temps de relaxation 1/gap = {1 / gap:.0f} pas")
    k_tp = names.index("triangle + pendant")
    p0 = torch.tensor([AS_RAW[nm] for nm in names], dtype=torch.float64)
    p0 = p0 / p0.sum()
    d0 = float(pi[k_tp] - p0[k_tp])
    pk, steps = p0.clone(), None
    for k in range(1, 20001):
        pk = pk @ P
        if float(pi[k_tp] - pk[k_tp]) <= d0 / 2:
            steps = k
            break
    tv0 = 0.5 * float((p0 - pi).abs().sum())
    print(f"\nproportions d'AS (ed8bb67, brutes, renormalisées sur ces états) : TV à la référence {tv0:.4f} ; déficit du triangle+pendant {d0:.4f}")
    print(f"pas du noyau pour diviser le déficit du triangle+pendant par deux : {steps} (itération exacte de la chaîne lumpée) ; estimation spectrale ln2 / gap = {np.log(2) / gap:.0f}")
    for k in (1, 5, 20, 100, 500):
        pk = p0 @ torch.linalg.matrix_power(P, k)
        print(f"  après {k:3d} pas : TV {0.5 * float((pk - pi).abs().sum()):.4f}, triangle+pendant {float(pk[k_tp]):.4f} (réf {float(pi[k_tp]):.4f})")


if __name__ == "__main__":
    main()
