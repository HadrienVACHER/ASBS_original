""" Rapport de l'A/B torch.compile (scripts/ab_compile_dw4.pbs) : A (fast_train par défaut), A2 (répétition de A, même graine = bruit de référence), B (reduce-overhead).

      python scripts/ab_compile_report.py [--logs logs] [--project CCDS_NTU/as]

    Imprime : secondes par epoch (médiane après 5 epochs de mise en route) et accélération de B sur A ; recompilations et ruptures de graphe comptées dans les logs (TORCH_LOGS) ;
    échec / succès du contrôle de compilation de train.py ; courbes de loss (moyenne sur les epochs 10-59) ; métriques de la dernière évaluation (W&B) ; et l'écart B-A rapporté
    au bruit A2-A. Le critère d'adoption (courbes et métriques égales au bruit près) est jugé par l'utilisateur sur ces chiffres.
"""
import argparse
import re
from pathlib import Path

import numpy as np

KEYS = ["energy_w2", "dist_w2", "state_tv", "short_frac"]


def parse(path):
    txt = Path(path).read_text(errors="ignore")
    rows = re.findall(r"ep=(\d+)\S*\s+\S*loss=([0-9.eE+-]+)\S*\s+t=([0-9.]+)s", re.sub(r"\x1b\[[0-9;]*m", "", txt))
    ep = np.array([int(a) for a, _, _ in rows])
    loss = np.array([float(b) for _, b, _ in rows])
    t = np.array([float(c) for _, _, c in rows])
    clean = re.sub(r"\x1b\[[0-9;]*m", "", txt)
    return dict(ep=ep, loss=loss, t=t, recompiles=len(re.findall(r"[Rr]ecompil", clean)), breaks=len(re.findall(r"[Gg]raph break", clean)),
                check=("check OK" in clean), check_failed=("check FAILED" in clean), fallback=("cache_size_limit" in clean or "hit config.cache_size_limit" in clean))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--project", default="CCDS_NTU/as")
    a = ap.parse_args()
    runs = {k: parse(Path(a.logs) / f"ab_compile_{k}.log") for k in ("A", "A2", "B")}
    print(f"{'run':4}{'s/epoch':>9}{'recompiles':>12}{'graph breaks':>14}{'check compile':>15}{'cache limit':>13}{'loss moy. (10-59)':>19}")
    for k, r in runs.items():
        if len(r["t"]) < 12:
            print(f"{k:4} log incomplet ({len(r['t'])} epochs)")
            return
        print(f"{k:4}{np.median(r['t'][5:]):9.3f}{r['recompiles']:12d}{r['breaks']:14d}{('OK' if r['check'] else 'ÉCHEC' if r['check_failed'] else '?'):>15}{str(r['fallback']):>13}{r['loss'][10:].mean():19.3f}")
    ta, tb = np.median(runs["A"]["t"][5:]), np.median(runs["B"]["t"][5:])
    print(f"\naccélération de B sur A : x{ta / tb:.2f} ({ta:.3f} -> {tb:.3f} s/epoch) ; bruit A2 vs A sur le temps : {np.median(runs['A2']['t'][5:]) / ta:.2f}")
    la, la2, lb = (runs[k]["loss"][10:].mean() for k in ("A", "A2", "B"))
    print(f"loss moyenne : |A2 - A| = {abs(la2 - la):.3f} (bruit), |B - A| = {abs(lb - la):.3f}")
    try:
        import wandb
        got = {}
        for r in wandb.Api(timeout=120).runs(a.project, per_page=200, order="-created_at"):
            for k in ("A", "A2", "B"):
                if r.name == f"ab_compile_{k}" and k not in got:
                    h = r.history(keys=KEYS, samples=100, pandas=True, x_axis="_step").dropna(subset=["state_tv"])
                    got[k] = h.iloc[-1]
        if len(got) == 3:
            print(f"\nmétriques de la dernière évaluation (W&B) : {'':4}" + "".join(f"{k:>12}" for k in KEYS))
            for k in ("A", "A2", "B"):
                print(f"  {k:4}" + " " * 36 + "".join(f"{got[k][m]:12.4f}" for m in KEYS))
            for m in KEYS:
                print(f"  {m:11} |A2-A| = {abs(got['A2'][m] - got['A'][m]):.4f}, |B-A| = {abs(got['B'][m] - got['A'][m]):.4f}")
    except Exception as e:
        print("W&B non lu:", e)


if __name__ == "__main__":
    main()
