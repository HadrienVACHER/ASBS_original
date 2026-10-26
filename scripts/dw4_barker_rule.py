""" CRITÈRES PRÉ-ENREGISTRÉS pour le run DW-4 AS + variable de contrôle de Barker (24 réflexions, gamma = 3) contre les deux runs AS de même config (e1500 ed8bb67 et e1500 94252/c166b5d).
    Engagés dans git AVANT la soumission du run Barker. Attente déclarée : atteindre le plancher du TV n'est PAS exigé.

      python scripts/dw4_barker_rule.py --cv results/dw4_as_barker_e1500/seed0/samples_final.pt \
          --as results/dw4_e1500_ed8bb67_samples.pt results/dw4_e1500_94252_samples.pt [--cv-commit <hash>]

    PRIMAIRE : TV des états lus sur les BASSINS TREMPÉS (5000 échantillons, contre la référence trempée), run CV < min(AS) - 2 SE, SE = sqrt(SE_cv^2 + SE_as^2), SE par bootstrap des
        échantillons (200 rééchantillonnages). Les runs AS n'ont pas enregistré le TV en bassins pendant l'entraînement : la comparaison primaire utilise donc, pour les TROIS runs, les
        échantillons finaux (samples_final.pt) ; pour le run CV, la moyenne des 6 dernières évaluations de `basin_tv` (5000 échantillons) est rapportée en plus.
    MÉCANISTE : sur les échantillons finaux du run CV, |Delta_b| / SE (statistique RB, bassins) pour les bassins entrants (2 paires disjointes, étoile, K4 moins une arête = sur-pondérés),
        le triangle+pendant et le chemin de 2 arêtes est INFÉRIEUR à la valeur AS (valeurs figées d'ed8bb67 : 6.6, 4.8, 4.9, 5.4, 11.1 ; et celles recalculées sur chaque run AS fourni, on
        prend le minimum) : l'équilibre des flux est le point fixe, Delta_b doit tendre vers 0.
    DIRECTION : les bassins sur-pondérés (2 paires disjointes, étoile, K4 moins une arête) DIMINUENT et les sous-pondérés (chemin de 2 arêtes, triangle+pendant) AUGMENTENT par rapport à la
        moyenne des runs AS (proportions en bassins, 5000 échantillons).
    SANITÉ : eq_w2 et energy_w2 du run CV à moins de 15 % de ceux des runs AS (moyenne des évaluations d'epoch >= 1000, W&B ; nécessite --cv-commit).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dw4_reflection_rb import label_basin, stats  # noqa: E402
from adjoint_samplers.utils.dw4_quench import quench  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, state_freq, state_ids, state_tv  # noqa: E402

N = 5000
OVER = ["2 disjoint pairs", "star (3 edges)", "K4 minus an edge"]            # over-weighted in AS (ed8bb67)
UNDER = ["path of 2 (3 particles)", "triangle + pendant"]                    # under-weighted
AS_Z_FROZEN = {"2 disjoint pairs": 6.6, "star (3 edges)": 4.8, "K4 minus an edge": 4.9, "triangle + pendant": 5.4, "path of 2 (3 particles)": 11.1}   # ed8bb67, bassins


def idx(name):
    return [NAMES[k] for k in KEYS].index(name)


def basin_freq(x):
    return state_freq(state_ids(quench(x.float())[0]))


def tv_with_se(x, ref_freq, boots=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = quench(x.float())[0]
    base = state_tv(state_freq(state_ids(q)), ref_freq)
    bs = [state_tv(state_freq(state_ids(q[torch.randint(len(q), (len(q),), generator=g)])), ref_freq) for _ in range(boots)]
    return base, float(np.std(bs))


def wandb_mean(project, name, commit, keys, from_epoch=1000):
    import wandb
    for r in wandb.Api(timeout=120).runs(project, per_page=200, order="-created_at"):
        if r.name == name and r.config.get("git_commit", "")[:7] == commit[:7] and r.state == "finished":
            h = r.history(keys=keys, samples=6000, pandas=True, x_axis="_step")
            return {k: float(h[k].dropna()[h["_step"][h[k].notna()] >= from_epoch].mean()) for k in keys}, h
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", required=True)
    ap.add_argument("--as", dest="as_", nargs="+", required=True)
    ap.add_argument("--cv-commit", default=None)
    ap.add_argument("--cv-name", default="dw4_as_barker_e1500")
    ap.add_argument("--as-name", default="dw4_as_states_e1500")
    ap.add_argument("--as-commits", nargs="*", default=["ed8bb67", "c166b5d"])
    ap.add_argument("--project", default="CCDS_NTU/as")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    ref = torch.tensor(np.load(root / "data" / "test_split_DW4.npy", allow_pickle=True)).double()
    ref_basin = basin_freq(ref)
    load = lambda p: torch.load(p, map_location="cpu", weights_only=False)["model_samples"].double()[:N]
    cv, ass = load(a.cv), [load(p) for p in a.as_]

    print("=== PRIMAIRE : TV des états en bassins (5000 échantillons finaux, bootstrap) ===")
    tv_cv, se_cv = tv_with_se(cv, ref_basin)
    tv_as = [tv_with_se(x, ref_basin) for x in ass]
    for p, (tv, se) in zip(a.as_, tv_as):
        print(f"  AS  {Path(p).name:40} TV {tv:.4f} ± {se:.4f}")
    print(f"  CV  {Path(a.cv).name:40} TV {tv_cv:.4f} ± {se_cv:.4f}")
    j = int(np.argmin([t for t, _ in tv_as]))
    tv_min, se_min = tv_as[j]
    thr = tv_min - 2 * np.sqrt(se_cv ** 2 + se_min ** 2)
    prim = tv_cv < thr
    print(f"  seuil min(AS) - 2 SE = {tv_min:.4f} - 2 x {np.sqrt(se_cv ** 2 + se_min ** 2):.4f} = {thr:.4f} ; TV_CV = {tv_cv:.4f} -> {'PASSE' if prim else 'ÉCHOUE'}")

    print("\n=== MÉCANISTE : |Delta_b| / SE (bassins, statistique RB) ===")
    s_cv = stats(cv, label_basin)
    s_as = [stats(x, label_basin) for x in ass]
    mech = True
    for name in OVER + UNDER:
        b = idx(name)
        z_cv = abs(float(s_cv["D"][b])) / float(s_cv["se"][b])
        z_as = min([AS_Z_FROZEN[name]] + [abs(float(s["D"][b])) / float(s["se"][b]) for s in s_as])
        ok = z_cv < z_as
        mech &= ok
        print(f"  {name:28} CV {z_cv:6.1f}   AS (min des valeurs figées ed8bb67 et des runs fournis) {z_as:6.1f}   {'ok' if ok else 'NON'}")
    print(f"  mécaniste : {'PASSE' if mech else 'ÉCHOUE'}")

    print("\n=== DIRECTION : proportions en bassins, CV contre la moyenne des runs AS ===")
    p_cv = s_cv["prop"]
    p_as = torch.stack([s["prop"] for s in s_as]).mean(0)
    direc = True
    for name in OVER + UNDER:
        b = idx(name)
        ok = (float(p_cv[b]) < float(p_as[b])) if name in OVER else (float(p_cv[b]) > float(p_as[b]))
        direc &= ok
        print(f"  {name:28} réf {float(ref_basin[b]):.4f}   AS {float(p_as[b]):.4f}   CV {float(p_cv[b]):.4f}   {'ok' if ok else 'NON'}")
    print(f"  direction : {'PASSE' if direc else 'ÉCHOUE'}")

    sanity = None
    if a.cv_commit:
        print("\n=== SANITÉ : eq_w2, energy_w2 (moyenne des évaluations d'epoch >= 1000) à moins de 15 % des runs AS ===")
        keys = ["eq_w2", "energy_w2"]
        cvm, hcv = wandb_mean(a.project, a.cv_name, a.cv_commit, keys)
        asm = [wandb_mean(a.project, a.as_name, c, keys)[0] for c in a.as_commits]
        asm = [m for m in asm if m]
        if cvm and asm:
            sanity = True
            for k in keys:
                base = float(np.mean([m[k] for m in asm]))
                rel = abs(cvm[k] - base) / base
                sanity &= rel <= 0.15
                print(f"  {k:10} CV {cvm[k]:.3f}   AS {[round(m[k], 3) for m in asm]} (moyenne {base:.3f})   écart relatif {100 * rel:.1f} %   {'ok' if rel <= 0.15 else 'NON'}")
            try:
                bt = hcv["basin_tv"].dropna().values[-6:]
                print(f"  (rapporté) TV en bassins du run CV, moyenne des 6 dernières évaluations : {bt.mean():.4f} ± {bt.std() / np.sqrt(len(bt)):.4f}")
            except Exception:
                pass
        else:
            print("  runs W&B introuvables ou non terminés : sanité non évaluée")
    print("\nRÉSUMÉ : primaire", prim, "| mécaniste", mech, "| direction", direc, "| sanité", sanity)
    print("Attente déclarée : atteindre le plancher du TV n'est PAS exigé.")


if __name__ == "__main__":
    main()
