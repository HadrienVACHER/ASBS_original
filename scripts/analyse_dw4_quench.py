""" DW-4 : analyse de ROBUSTESSE par trempe (quench), rapportée À CÔTÉ de la règle pré-enregistrée (scripts/analyse_dw4.py, inchangée).

      python scripts/analyse_dw4_quench.py --samples results/dw4_as_states/seed0/samples_final.pt

    Pourquoi : seuiller les distances brutes à d0 = 4 peut transformer une erreur de FORME (distributions de distances trop larges -> plus de configurations proches du seuil) en
    erreur de POIDS apparente. Chaque échantillon (modèle ET référence) est ramené à son minimum local d'énergie (Newton sans selle, ||grad E|| < 1e-4), puis le graphe des paires
    courtes (d < 4) est lu sur la configuration trempée : un bassin, donc la fréquence d'un état est un vrai poids de mode.
    Même protocole que la règle : n = 5000 échantillons du modèle contre 5000 de la référence (25 tirages), plancher = TV entre deux sous-échantillons disjoints de 5000 de la
    référence TREMPÉE (25 répétitions). Rapporte le tableau des états (brut et trempé), les rapports, le TV avec son plancher, et la distribution des distances trempées.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils.dw4_quench import quench  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, pair_distances, state_freq, state_ids, state_tv  # noqa: E402

N, REPEATS = 5000, 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    ref = torch.tensor(np.load(root / "data" / "test_split_DW4.npy", allow_pickle=True)).float()
    model = torch.load(a.samples, map_location="cpu", weights_only=False)["model_samples"].float()[:N]
    qm, gm, _ = quench(model)
    qr, gr, _ = quench(ref)
    print(f"trempe : modèle {100 * float((gm < 1e-4).float().mean()):.2f} % convergés (||grad E|| < 1e-4), référence {100 * float((gr < 1e-4).float().mean()):.2f} %")
    fm_raw, fr_raw = state_freq(state_ids(model)), state_freq(state_ids(ref))
    fm_q, fr_q = state_freq(state_ids(qm)), state_freq(state_ids(qr))
    print(f"\nÉTATS lus sur la configuration TREMPÉE (n = {N} modèle), à côté des états bruts :")
    print(f"{'état':28}{'réf brut':>10}{'mod brut':>10}{'réf trempé':>12}{'mod trempé':>12}{'rapport trempé':>16}")
    for k, key in enumerate(KEYS):
        if max(fr_raw[k], fm_raw[k], fr_q[k], fm_q[k]) < 0.0005:
            continue
        r = float(fm_q[k] / fr_q[k]) if fr_q[k] > 0 else float("nan")
        print(f"{NAMES[key]:28}{fr_raw[k]:10.4f}{fm_raw[k]:10.4f}{fr_q[k]:12.4f}{fm_q[k]:12.4f}{r:16.2f}")
    g = torch.Generator().manual_seed(0)
    tv_raw, tv_q, floors = [], [], []
    for _ in range(REPEATS):
        idx = torch.randperm(len(ref), generator=g)
        tv_raw.append(state_tv(fm_raw, state_freq(state_ids(ref[idx[:N]]))))
        tv_q.append(state_tv(fm_q, state_freq(state_ids(qr[idx[:N]]))))
        floors.append(state_tv(state_freq(state_ids(qr[idx[:N]])), state_freq(state_ids(qr[idx[N:2 * N]]))))
    fl95 = float(np.percentile(floors, 95))
    print(f"\nTV des états : brut {np.mean(tv_raw):.4f} ; TREMPÉ {np.mean(tv_q):.4f}")
    print(f"plancher trempé (2 sous-échantillons disjoints de {N} de la référence trempée) : moyenne {np.mean(floors):.4f}, p95 {fl95:.4f} ; 3 x p95 = {3 * fl95:.4f} -> TV trempé > 3 x p95 : {np.mean(tv_q) > 3 * fl95}")
    dm, dr = pair_distances(qm).flatten(), pair_distances(qr).flatten()
    bins = torch.tensor([0, 2.0, 2.3, 2.5, 2.7, 3.0, 3.5, 3.8, 4.2, 4.5, 5.0, 5.3, 5.5, 5.7, 6.0, 100.0])
    cm = torch.bucketize(dm, bins[1:-1]).bincount(minlength=len(bins) - 1).double() / len(dm)
    cr = torch.bucketize(dr, bins[1:-1]).bincount(minlength=len(bins) - 1).double() / len(dr)
    print("\nDISTANCES des configurations trempées (fraction) :")
    for k in range(len(bins) - 1):
        print(f"{bins[k]:>7.1f}-{bins[k + 1]:<6.1f}{cm[k]:9.4f}{cr[k]:11.4f}")
    print(f"part des distances trempées dans [3.8, 4.2] (autour du seuil) : modèle {float(((dm > 3.8) & (dm < 4.2)).float().mean()):.4f}, référence {float(((dr > 3.8) & (dr < 4.2)).float().mean()):.4f}")


if __name__ == "__main__":
    main()
