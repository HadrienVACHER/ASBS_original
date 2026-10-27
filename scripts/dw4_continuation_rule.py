""" TEST DE CONTINUATION APPARIÉ, DW-4 : règle PRÉ-ENREGISTRÉE (engagée dans git avant la soumission).

    Point de départ commun : controller_final.pt et buffer (buffer_x1) du run AS 94252 (c166b5d : sigma_max 2, nfe 100, 1500 epochs). 500 epochs de continuation, évaluation toutes les 25 epochs à 5000 échantillons,
    proportions par bassin loggées (basin/<k>). Cinq bras, t uniforme partout, Adam repart de zéro dans tous les bras :
        C1 : AS graine 1     C2 : AS graine 2     G3 : Barker gamma = 3     G10 : Barker gamma = 10     G30 : Barker gamma = 30      (G* : graine 1, comme C1)

      PYTHONPATH=. python scripts/dw4_continuation_rule.py --commit <hash du run de continuation>

    Statistique : S = sum_b |pbar_b - ref_b|, pbar_b = proportion du bassin b moyennée sur les 10 DERNIÈRES évaluations (epochs 250 à 475), ref_b = proportions de la référence trempée (11 classes).
    Bruit  = |S_C1 - S_C2| + 2 x SE, SE = écart-type par bootstrap sur les évaluations (1000 rééchantillonnages des 10 évaluations) ; on prend le PLUS GRAND SE parmi les cinq bras.
    Effet à gamma  <=>  S_gamma < min(S_C1, S_C2) - bruit.
    La méthode est VIVANTE sur DW-4 <=>  un effet à gamma = 10 OU 30  ET  S décroissant en gamma (S_3 >= S_10 >= S_30 « à bruit près » : S_3 + bruit >= S_10 et S_10 + bruit >= S_30).
    Rapporte aussi le sens de chaque bassin par rapport à la moyenne des contrôles. Si G30 est instable (loss non finie, ou eq_w2 > 1.5 x celui des contrôles), on le rapporte au lieu de S.
    Attente (message de commit) : réponse FAIBLE, parce que l'information sur les poids se trouve à grand kappa, où les mouvements à deux points sont supprimés par le facteur gaussien.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils.dw4_quench import quench  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, state_freq, state_ids  # noqa: E402

ARMS = {"C1": "dw4_cont_C1", "C2": "dw4_cont_C2", "G3": "dw4_cont_G3", "G10": "dw4_cont_G10", "G30": "dw4_cont_G30"}
LAST = 10


def main():
    import wandb
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--project", default="CCDS_NTU/as")
    ap.add_argument("--arms", nargs="*", default=[], help="override run names, e.g. C1=my_run")
    a = ap.parse_args()
    names = dict(ARMS)
    names.update(dict(x.split("=") for x in a.arms))
    root = Path(__file__).resolve().parents[1]
    ref = torch.tensor(np.load(root / "data" / "test_split_DW4.npy", allow_pickle=True)).float()
    fr = state_freq(state_ids(quench(ref)[0])).numpy()
    runs = list(wandb.Api(timeout=120).runs(a.project, per_page=200, order="-created_at"))
    cols = [f"basin/{k}" for k in range(len(KEYS))]
    rng = np.random.default_rng(0)
    P, S, SE, ok, eq = {}, {}, {}, {}, {}
    for arm, nm in names.items():
        sel = [r for r in runs if r.name == nm and r.config.get("git_commit", "")[:7] == a.commit[:7]]
        if not sel or sel[0].state != "finished":
            print(f"{arm} ({nm}) : run introuvable ou non terminé")
            return
        r = sel[0]
        h = r.history(samples=6000, x_axis="_step", keys=["basin_tv"] + cols).dropna(subset=["basin_tv"]).set_index("_step").iloc[-LAST:]
        loss = r.history(samples=6000, x_axis="_step", keys=["adjoint_loss"])["adjoint_loss"].dropna().values
        e = r.history(samples=6000, x_axis="_step", keys=["eq_w2"])
        eq[arm] = float(e["eq_w2"].dropna().values[-2:].mean()) if "eq_w2" in e and e["eq_w2"].notna().any() else float("nan")
        ok[arm] = bool(np.isfinite(loss).all())
        X = h[cols].values
        P[arm] = X.mean(0)
        S[arm] = float(np.abs(P[arm] - fr).sum())
        bs = [float(np.abs(X[rng.integers(len(X), size=len(X))].mean(0) - fr).sum()) for _ in range(1000)]
        SE[arm] = float(np.std(bs))
        print(f"{arm:4} {nm:14} évaluations epochs {[int(i) for i in h.index][0]}..{[int(i) for i in h.index][-1]} : S = {S[arm]:.4f} ± {SE[arm]:.4f} ; eq_w2 (2 dernières) {eq[arm]:.3f} ; loss finie {ok[arm]}")
    noise = abs(S["C1"] - S["C2"]) + 2 * max(SE.values())
    base = min(S["C1"], S["C2"])
    print(f"\nbruit = |S_C1 - S_C2| + 2 SE_max = {abs(S['C1'] - S['C2']):.4f} + 2 x {max(SE.values()):.4f} = {noise:.4f} ; min(S_C1, S_C2) = {base:.4f} ; seuil d'effet = {base - noise:.4f}")
    eq_ctrl = np.nanmean([eq["C1"], eq["C2"]])
    effect = {}
    for g in ("G3", "G10", "G30"):
        unstable = (not ok[g]) or (eq[g] > 1.5 * eq_ctrl)
        effect[g] = (S[g] < base - noise) and not unstable
        print(f"  {g:4} S = {S[g]:.4f}  effet : {effect[g]}" + ("   INSTABLE (loss non finie ou eq_w2 > 1.5 x contrôles) : S non lu" if unstable else ""))
    mono = (S["G3"] + noise >= S["G10"]) and (S["G10"] + noise >= S["G30"])
    alive = (effect["G10"] or effect["G30"]) and mono
    print(f"\nS décroissant en gamma à bruit près : {mono} ; effet à gamma = 10 ou 30 : {effect['G10'] or effect['G30']}")
    print("=> MÉTHODE VIVANTE sur DW-4" if alive else "=> MÉTHODE NON démontrée vivante sur DW-4 dans ce test")
    ctrl = (P["C1"] + P["C2"]) / 2
    print("\nSens de chaque bassin (G - moyenne des contrôles ; attendu : sur-pondérés ↓, sous-pondérés ↑ ; réf, contrôles, G3, G10, G30)")
    for k in range(len(KEYS)):
        if fr[k] >= 0.005:
            over = ctrl[k] > fr[k]
            print(f"  {NAMES[KEYS[k]]:28} {'sur ' if over else 'sous'}-pondéré  réf {fr[k]:.4f}  ctrl {ctrl[k]:.4f}  " + "  ".join(f"{g} {P[g][k]:.4f}({'ok' if (P[g][k] < ctrl[k]) == over else 'NON'})" for g in ("G3", "G10", "G30")))


if __name__ == "__main__":
    main()
