""" Porte de validité de la phase 1 (pré-entraînement sur q = ManyWell symétrique, b = 0), PRÉ-ENREGISTRÉE, TROIS critères à satisfaire ensemble :

      python scripts/manywell_pretrain_gate.py [--samples results/manywell_pretrain_tf/seed0/samples_final.pt]

    (a) fraction de puits léger dans [0.4, 0.6] sur CHACUNE des 16 coordonnées double puits (10k échantillons du modèle) ;
    (b) part d'échantillons ayant AU MOINS UNE coordonnée DOUBLE PUITS à |x| > 3 : <= 2 %. Les 16 coordonnées GAUSSIENNES sont EXCLUES de ce critère
        (environ 4 % des échantillons exacts dépassent 3 sur au moins une d'elles, ce qui est normal) ; l'échantillonneur exact n'a aucune coordonnée double puits à |x| > 3 ;
    (c) energy_w2 contre 10k échantillons exacts de q <= 2 (même définition que l'évaluateur : W2 1D entre les énergies, racine carrée de emd2).
    Un seul échec = porte NON franchie. Écrit `gate_ok` à côté du fichier si les trois critères sont satisfaits (le job de la phase 2 le vérifie).
    Pré-entraînement v2 = DERNIÈRE tentative : en cas d'échec, documenter et s'arrêter, aucun autre réglage.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import ot as pot
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.energies.dist_energy import DistEnergy  # noqa: E402
from adjoint_samplers.utils.dist_utils import ManyWell  # noqa: E402

LO, HI, TAIL_MAX, W2_MAX, X_TAIL = 0.4, 0.6, 0.02, 2.0, 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="results/manywell_pretrain_tf/seed0/samples_final.pt")
    a = ap.parse_args()
    x = torch.load(a.samples, map_location="cpu", weights_only=False)["model_samples"]
    q = ManyWell(dim=x.shape[1], a=0.0, b=-6.0)
    energy = DistEnergy(q)
    torch.manual_seed(0)
    ref = q.sample([len(x)])

    f = (x[:, 0::2] < q.well_split).float().mean(0)
    ok_a = bool(((f >= LO) & (f <= HI)).all())
    print(f"{len(x)} échantillons du modèle. (a) fraction de puits léger par coordonnée (cible 0.5) :")
    print(" ", [round(v, 3) for v in f.tolist()])
    print(f"    min {f.min():.3f}, max {f.max():.3f}, moyenne {f.mean():.3f} ; dans [{LO}, {HI}] sur chaque coordonnée : {ok_a}")

    xd = x[:, 0::2]                                                    # coordonnées double puits uniquement (les gaussiennes sont exclues)
    tail = float((xd.abs() > X_TAIL).any(1).float().mean())
    tail_ref = float((ref[:, 0::2].abs() > X_TAIL).any(1).float().mean())
    tail_gauss = float((x[:, 1::2].abs() > X_TAIL).any(1).float().mean())
    ok_b = tail <= TAIL_MAX
    print(f"(b) part d'échantillons avec une coordonnée DOUBLE PUITS à |x| > {X_TAIL:g} : {100 * tail:.2f} % (exact : {100 * tail_ref:.2f} %) ; exigé <= {100 * TAIL_MAX:g} % : {ok_b}")
    print(f"    (coordonnées gaussiennes, exclues du critère : {100 * tail_gauss:.1f} % des échantillons du modèle ont une coordonnée gaussienne à |x| > {X_TAIL:g} ; ~4 % pour l'exact)")

    w2 = float(pot.emd2_1d(energy.eval(ref).numpy(), energy.eval(x).numpy())) ** 0.5
    ok_c = w2 <= W2_MAX
    e, er = energy.eval(x), energy.eval(ref)
    print(f"(c) energy_w2 contre des échantillons exacts de q : {w2:.3f} ; exigé <= {W2_MAX:g} : {ok_c}")
    print(f"    énergie, quantiles 50/99/99.9/max : modèle {[round(float(v), 1) for v in e.quantile(torch.tensor([.5, .99, .999])).tolist()] + [round(float(e.max()), 1)]}, "
          f"exact {[round(float(v), 1) for v in er.quantile(torch.tensor([.5, .99, .999])).tolist()] + [round(float(er.max()), 1)]}")

    ok = ok_a and ok_b and ok_c
    print(f"\nPORTE : {'FRANCHIE' if ok else 'NON FRANCHIE'} (a: {ok_a}, b: {ok_b}, c: {ok_c})")
    flag = Path(a.samples).parent / "gate_ok"
    if ok:
        flag.write_text("ok\n")
    elif flag.exists():
        flag.unlink()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
