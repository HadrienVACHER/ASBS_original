""" DW-4, AS seul, graine 0, config officielle : métriques d'ÉTATS et règle PRÉ-ENREGISTRÉE.

      python scripts/analyse_dw4.py --samples results/dw4_as_states/seed0/samples_final.pt [--commit <hash>]   # --commit : plateau et sanité depuis W&B

    État d'une configuration = graphe des paires COURTES (d < 4) sur les 4 particules, à isomorphisme près (11 classes).
    Tailles : n = 5000 échantillons du modèle (les 5000 premiers de samples_final.pt) contre 5000 échantillons de la référence MCMC (data/test_split_DW4.npy, 10000), 25 tirages ;
    plancher = TV entre deux sous-échantillons DISJOINTS de 5000 de la référence (25 répétitions).
    INFORMATIVE si (i) tout état de fréquence de référence >= 1 % a une fréquence du modèle >= 0.1 x référence, ET (ii) TV(modèle, référence) > 3 x percentile 95 du plancher,
    ET (iii) plateau : dérive de state_tv <= 0.01 sur les 20 % derniers de l'entraînement (moyenne de la 2e moitié moins moyenne de la 1re moitié de la fenêtre, évaluations
    toutes les 100 epochs). Sinon DW-4 n'est pas une cible pour la méthode de correction des poids.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, pair_distances, state_freq, state_ids, state_tv  # noqa: E402

N, REPEATS, D0 = 5000, 25, 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--commit", default=None)
    ap.add_argument("--project", default="CCDS_NTU/adjoint_sampler_cv")
    ap.add_argument("--name", default="dw4_as_states")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    ref = torch.tensor(np.load(root / "data" / "test_split_DW4.npy", allow_pickle=True)).float()
    model = torch.load(a.samples, map_location="cpu", weights_only=False)["model_samples"].float()[:N]
    f_ref_all = state_freq(state_ids(ref))
    f_model = state_freq(state_ids(model))
    g = torch.Generator().manual_seed(0)

    # --- sanity (W&B) and plateau
    drift = None
    if a.commit:
        import wandb
        sel = [r for r in wandb.Api(timeout=120).runs(a.project, per_page=200, order="-created_at")
               if r.name == a.name and r.config.get("seed") == 0 and r.config.get("git_commit", "")[:7] == a.commit[:7]]
        if sel and sel[0].state == "finished":
            h = sel[0].history(keys=["energy_w2", "eq_w2", "dist_w2", "state_tv", "short_frac"], samples=6000, pandas=True, x_axis="_step").dropna(subset=["state_tv"]).set_index("_step")
            last = h.iloc[-1]
            print(f"SANITÉ (dernière évaluation, epoch {int(h.index[-1])}) : eq_w2 = {last['eq_w2']:.3f}, energy_w2 = {last['energy_w2']:.3f}, dist_w2 = {last['dist_w2']:.4f}   "
                  f"[article : W2 ≈ 0.62, energy W2 ≈ 0.55 ; moyenne des 5 dernières évaluations : eq_w2 {h['eq_w2'].values[-5:].mean():.3f}, energy_w2 {h['energy_w2'].values[-5:].mean():.3f}]")
            end = h.index[-1]
            win = h["state_tv"][h.index >= end * 0.8].values
            half = len(win) // 2
            drift = float(win[half:].mean() - win[:half].mean())
            print(f"PLATEAU : state_tv sur les 20 % derniers ({len(win)} évaluations) : {np.round(win, 4)} ; dérive = {drift:+.4f}")
        else:
            print("run W&B introuvable ou non terminé : sanité et plateau non évalués")

    # --- distances
    dm, dr = pair_distances(model).flatten(), pair_distances(ref).flatten()
    bins = torch.tensor([0, 1.5, 2.0, 2.3, 2.5, 2.7, 3.0, 3.5, 4.0, 4.5, 5.0, 5.3, 5.5, 5.7, 6.0, 6.5, 7.0, 8.0, 100.0])
    cm = torch.bucketize(dm, bins[1:-1]).bincount(minlength=len(bins) - 1).double() / len(dm)
    cr = torch.bucketize(dr, bins[1:-1]).bincount(minlength=len(bins) - 1).double() / len(dr)
    print("\nHISTOGRAMME des distances par paires (fraction) :")
    print(f"{'intervalle':>14}{'modèle':>9}{'référence':>11}")
    for k in range(len(bins) - 1):
        print(f"{bins[k]:>7.1f}-{bins[k + 1]:<6.1f}{cm[k]:9.4f}{cr[k]:11.4f}")
    sf_m, sf_r = float((dm < D0).float().mean()), float((dr < D0).float().mean())
    print(f"fraction de distances courtes (d < {D0:g}) : modèle {sf_m:.4f}, référence {sf_r:.4f}")

    # --- states
    print(f"\nÉTATS (n = {N} modèle) :")
    print(f"{'état':32}{'référence':>11}{'modèle':>9}{'rapport':>9}")
    ok_i = True
    for k, key in enumerate(KEYS):
        fr, fm = float(f_ref_all[k]), float(f_model[k])
        ratio = fm / fr if fr > 0 else float("nan")
        flag = ""
        if fr >= 0.01:
            flag = "  <- état principal"
            if fm < 0.1 * fr:
                ok_i = False
                flag += " MANQUÉ"
        print(f"{NAMES[key]:32}{fr:11.4f}{fm:9.4f}{ratio:9.2f}{flag}")
    tvs, floors = [], []
    for _ in range(REPEATS):
        idx = torch.randperm(len(ref), generator=g)
        tvs.append(state_tv(f_model, state_freq(state_ids(ref[idx[:N]]))))
        floors.append(state_tv(state_freq(state_ids(ref[idx[:N]])), state_freq(state_ids(ref[idx[N:2 * N]]))))
    tv, fl95 = float(np.mean(tvs)), float(np.percentile(floors, 95))
    ok_ii = tv > 3 * fl95
    print(f"\nTV des états, modèle contre référence (moyenne sur {REPEATS} sous-échantillons) : {tv:.4f}")
    print(f"plancher (TV entre 2 sous-échantillons disjoints de {N} de la référence) : moyenne {np.mean(floors):.4f}, percentile 95 {fl95:.4f} ; 3 x p95 = {3 * fl95:.4f}")

    ok_iii = None if drift is None else abs(drift) <= 0.01
    print(f"\n(i) tous les états principaux visités (>= 0.1 x référence) : {ok_i}")
    print(f"(ii) TV {tv:.4f} > 3 x plancher p95 {3 * fl95:.4f} : {ok_ii}")
    print(f"(iii) plateau (dérive <= 0.01) : {ok_iii if ok_iii is not None else 'non évalué (--commit)'}")
    if ok_iii is None:
        print("=> verdict incomplet (plateau non évalué)")
    else:
        print("=> DW-4 INFORMATIVE" if (ok_i and ok_ii and ok_iii) else "=> DW-4 NON informative : pas une cible pour la méthode de correction des poids")


if __name__ == "__main__":
    main()
