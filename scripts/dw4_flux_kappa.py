""" Flux d'états du noyau de Barker À kappa > 0 (celui que voit l'entraînement), sur les échantillons x1 d'un run (CPU, quelques dizaines de secondes).

      python scripts/dw4_flux_kappa.py --samples results/dw4_e1500_ed8bb67_samples.pt [--basins]

    Pour chaque kappa, x_t = kappa x1 + sqrt(v) eps (centre de masse nul), v = nu1 kappa (1 - kappa), nu1 = sigma_max^2, et pour chaque classe b
        Delta_b(kappa) = moyenne_n (1/R) sum_r omega_r(x_n; x_t) [ 1{s(T_r x_n) = b} - 1{s(x_n) = b} ],   omega_r = sigmoid(logit à kappa)  (24 applications).
    Au sens de l'espérance sous x1 ~ modèle, Delta_b est le changement de poids de la classe b que le terme de Barker induit en un « pas » ; son signe doit être celui de (réf_b - modèle_b)
    pour toute classe mal pondérée. Rapporte aussi la fraction du flux à kappa = 0 qui subsiste à chaque kappa et, sur la référence, la sanité |Delta_b| / SE < 3.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils import dw4_reflections as dwr  # noqa: E402
from adjoint_samplers.utils.dw4_quench import quench  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, state_freq, state_ids  # noqa: E402

KAPPAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7)


def flux(x, label, kappas, nu1, gen):
    imgs, dE = dwr.images(x)
    lab_x = label(x)
    lab_i = label(imgs.reshape(-1, 8)).reshape(len(x), -1)
    rows = {}
    for kappa in kappas:
        if kappa == 0.0:
            om = torch.sigmoid(-dE)
        else:
            v = nu1 * kappa * (1 - kappa)
            xt = kappa * x + v ** 0.5 * dwr.recentre(torch.randn(len(x), 4, 2, generator=gen, dtype=x.dtype)).reshape(-1, 8)
            om = torch.sigmoid(dwr.barker_logit(x, xt, kappa, v)[0])
        d = []
        for b in range(len(KEYS)):
            per = (om * ((lab_i == b).to(om.dtype) - (lab_x == b).to(om.dtype)[:, None])).mean(1)
            d.append((float(per.mean()), float(per.std() / len(per) ** 0.5)))
        rows[kappa] = d
    return rows, torch.bincount(lab_x, minlength=len(KEYS)).double() / len(lab_x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--sigma-max", type=float, default=2.0)
    ap.add_argument("--basins", action="store_true", help="also read the classes on quenched basins")
    ap.add_argument("--n", type=int, default=5000)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    gen = torch.Generator().manual_seed(0)
    prep = lambda z: dwr.recentre(z.double().reshape(-1, 4, 2)).reshape(-1, 8)
    ref = prep(torch.tensor(np.load(root / "data" / "test_split_DW4.npy", allow_pickle=True)))
    x = prep(torch.load(a.samples, map_location="cpu", weights_only=False)["model_samples"])[: a.n]
    for name, label in (("états BRUTS", lambda z: state_ids(z.float())),) + ((("BASSINS", lambda z: state_ids(quench(z.float())[0])),) if a.basins else ()):
        fr = torch.bincount(label(ref), minlength=len(KEYS)).double() / len(ref)
        rows, fa = flux(x, label, KAPPAS, a.sigma_max ** 2, gen)
        rrows, _ = flux(ref, label, KAPPAS, a.sigma_max ** 2, gen)
        main_ids = [k for k in range(len(KEYS)) if fr[k] >= 0.005]
        print(f"\n=== {name} : Delta_b(kappa) sur les échantillons du modèle (z = Delta/SE) ; signe attendu = signe(réf - modèle) ===")
        print(f"{'classe':26}{'réf':>7}{'modèle':>8}{'attendu':>9}" + "".join(f"{('k=' + str(k)):>14}" for k in KAPPAS))
        for b in main_ids:
            exp = "+" if fr[b] > fa[b] else "-"
            cells = [f"{rows[k][b][0]:+.5f}({rows[k][b][0] / rows[k][b][1] if rows[k][b][1] > 0 else 0:+.0f})" for k in KAPPAS]
            print(f"{NAMES[KEYS[b]]:26}{float(fr[b]):7.4f}{float(fa[b]):8.4f}{exp:>9}" + "".join(f"{c:>14}" for c in cells))
        print("fraction du flux à kappa = 0 qui subsiste (somme des |Delta_b| sur les classes principales) :",
              {k: round(sum(abs(rows[k][b][0]) for b in main_ids) / max(sum(abs(rows[0.0][b][0]) for b in main_ids), 1e-12), 2) for k in KAPPAS})
        zmax = {k: max(abs(rrows[k][b][0] / rrows[k][b][1]) for b in main_ids if rrows[k][b][1] > 0) for k in KAPPAS}
        print("sanité sur la référence, max |Delta_b| / SE par kappa :", {k: round(v, 1) for k, v in zmax.items()})


if __name__ == "__main__":
    main()
