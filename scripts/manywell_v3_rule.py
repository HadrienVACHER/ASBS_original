""" V3 (-x^4 + 6 x^2 + 0.2 x, puits léger 33.7 %), AS seul, graine 0 : MÊME règle pré-enregistrée que scripts/manywell_variants_rule.py.
    Variante ajoutée après coup (après l'échec de V1 et V2), la dernière.

      PYTHONPATH=. python scripts/manywell_v3_rule.py --commit <hash>
"""
import argparse

from manywell_variants_rule import evaluate

if __name__ == "__main__":
    import wandb
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    runs = list(wandb.Api(timeout=120).runs("CCDS_NTU/adjoint_sampler_cv", per_page=200, order="-created_at"))
    out = evaluate(runs, "V3", "manywell_v3_as", 6.0, 0.2, a.commit)
    if out is not None:
        print("\n=> PASS" if out[0] else "\n=> FAIL")
