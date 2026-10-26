""" Porte de validité de la phase 1 (pré-entraînement sur q = ManyWell symétrique, b = 0) :

      python scripts/manywell_pretrain_gate.py [--samples results/manywell_pretrain_tf/seed0/samples_final.pt]

    Les 10k échantillons du modèle doivent avoir les DEUX puits sur les 16 coordonnées avec une fraction de puits léger dans [0.4, 0.6] sur CHAQUE coordonnée.
    Sinon : corriger le pré-entraînement, ne PAS lancer la phase 2. Écrit `gate_ok` à côté du fichier si la porte est franchie (le job de la phase 2 le vérifie).
    (Le plateau du pré-entraînement se lit sur W&B : scripts/analyse_correction.py --gate.)
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils.dist_utils import ManyWell  # noqa: E402

LO, HI = 0.4, 0.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="results/manywell_pretrain_tf/seed0/samples_final.pt")
    a = ap.parse_args()
    x = torch.load(a.samples, map_location="cpu", weights_only=False)["model_samples"]
    q = ManyWell(dim=x.shape[1], a=0.0, b=-6.0)
    f = (x[:, 0::2] < q.well_split).float().mean(0)
    print(f"{len(x)} échantillons du modèle, fraction de puits léger par coordonnée (cible 0.5) :")
    print(" ", [round(v, 3) for v in f.tolist()])
    ok = bool(((f >= LO) & (f <= HI)).all())
    print(f"min {f.min():.3f}, max {f.max():.3f}, moyenne {f.mean():.3f} ; porte [{LO}, {HI}] sur chaque coordonnée : {'FRANCHIE' if ok else 'ÉCHEC'}")
    # information only (not part of the gate): tails of the model samples. energy_w2 of the pre-training run is 300-500 (≈1-4 in the AS runs): a handful of
    # far outliers (x^4 energy) would explain it
    g = ManyWell(dim=x.shape[1], a=0.0, b=-6.0)
    from adjoint_samplers.energies.dist_energy import DistEnergy
    e, ref = DistEnergy(g).eval(x), DistEnergy(g).eval(g.sample([len(x)]))
    xd = x[:, 0::2]
    print(f"queues : max |x_double| {xd.abs().max():.2f} (exact {g.sample([len(x)])[:, 0::2].abs().max():.2f}), échantillons avec |x_double| > 3 : {int((xd.abs() > 3).any(1).sum())}, "
          f"avec |x| > 4 (toutes coordonnées) : {int((x.abs() > 4).any(1).sum())}")
    print(f"énergie : quantiles 50/99/99.9/max du modèle {[round(float(v), 1) for v in e.quantile(torch.tensor([.5, .99, .999])).tolist()] + [round(float(e.max()), 1)]} ; "
          f"exact {[round(float(v), 1) for v in ref.quantile(torch.tensor([.5, .99, .999])).tolist()] + [round(float(ref.max()), 1)]}")
    flag = Path(a.samples).parent / "gate_ok"
    if ok:
        flag.write_text("ok\n")
    elif flag.exists():
        flag.unlink()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
