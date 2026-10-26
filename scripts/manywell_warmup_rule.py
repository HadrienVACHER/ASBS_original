""" V1 avec warm-up du lr (1e-5 -> 1e-4 sur 200 epochs), AS seul, graine 0 : MÊME règle pré-enregistrée que scripts/manywell_variants_rule.py
    (wells_visited_frac = 1.0 sur les 16 coordonnées ET mean |log R_i| > 0.3 ET dérive <= 0.05 sur les 500 dernières epochs ET loss finie).

      PYTHONPATH=. python scripts/manywell_warmup_rule.py --commit <hash du run warm-up> [--ref-commit 205efcd]

    Rapporte le run avec warm-up à côté du run V1 sans warm-up (205efcd) : R_i à l'epoch 50 et à la fin.
    PASS -> le même warm-up est appliqué à l'identique à Barker. FAIL -> ManyWell est fermé pour de bon.
"""
import argparse

import numpy as np

from manywell_variants_rule import evaluate

if __name__ == "__main__":
    import wandb
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--ref-commit", default="205efcd")
    a = ap.parse_args()
    runs = list(wandb.Api(timeout=120).runs("CCDS_NTU/adjoint_sampler_cv", per_page=200, order="-created_at"))
    ref = evaluate(runs, "V1 sans warm-up (référence)", "manywell_v1_as", 6.0, 0.3, a.ref_commit)
    new = evaluate(runs, "V1 avec warm-up", "manywell_v1w_as", 6.0, 0.3, a.commit)
    if new is None:
        raise SystemExit
    if ref is not None:
        print("\nR_i, sans warm-up -> avec warm-up")
        for k, lab in ((2, "epoch 50"), (1, "fin")):
            if ref[k] is not None and new[k] is not None:
                print(f"  {lab:9s} sans : {np.round(ref[k], 2)}\n  {'':9s} avec : {np.round(new[k], 2)}")
    print("\n=> PASS : appliquer le même warm-up à Barker" if new[0] else "\n=> FAIL : ManyWell est fermé pour de bon")
