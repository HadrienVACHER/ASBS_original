""" Expérience de CORRECTION des poids d'un échantillonneur qui couvre déjà tous les modes (ManyWell-32, graine 0) :
    phase 1 = pré-entraînement sur q (b = 0, puits léger 0.5), phase 2 = AS et Barker vers p (standard, puits léger 0.1557) depuis le même contrôleur et le même buffer.

      PYTHONPATH=. python scripts/analyse_correction.py --gate    --commit <hash>   # plateau du pré-entraînement
      PYTHONPATH=. python scripts/analyse_correction.py --commit <hash phase 2>      # comparaison AS / Barker

    Métriques pré-enregistrées (30 dernières évaluations, moyennées ; courbes imprimées toutes les 250 epochs) :
      primaire : moyenne sur les 16 coordonnées de |f_i - 0.156|   (light_frac_dev ; f_i = fraction de puits léger = R_i x 0.15569)
      puits conservés : nombre de coordonnées avec f_i > 0.01       (wells_kept)
      joint : TV entre la loi du nombre de puits légers par échantillon et Binomial(16, 0.156)   (light_count_tv)
      energy_w2 et les valeurs à l'epoch 50 ; valeurs de départ (epoch 0, f_i ~ 0.5).
    Pré-enregistré en plus (scripts/manywell_posthoc.py, scripts/manywell_model_quality.py) : les PROPORTIONS (métrique primaire) sont rapportées pour toutes les méthodes
    (AS, AS + reweight BAR, AS + MCMC orbite à budget apparié, Barker) ; l'ESS d'échantillonnage préférentiel de Girsanov et |log Z_hat - log Z| pour les trois modèles
    (AS, Barker, AS + post-hoc = le modèle AS).
    SUCCÈS = métrique primaire de Barker < celle d'AS ET Barker conserve au moins autant de puits qu'AS. Une graine ; si Barker échoue, on le documente et on s'arrête.
"""
import argparse

import numpy as np

KEYS = ["light_frac_dev", "wells_kept", "light_count_tv", "energy_w2"]
N = 16


def load(runs, name, commit):
    sel = [r for r in runs if r.name == name and r.config.get("seed") == 0 and r.config.get("git_commit", "")[:7] == commit[:7]]
    if not sel:
        return None, None
    r = sel[0]
    h = r.history(keys=KEYS + [f"well_ratio/{i}" for i in range(N)], samples=6000, pandas=True, x_axis="_step").dropna(subset=["energy_w2"]).set_index("_step")
    loss = r.history(keys=["adjoint_loss"], samples=6000, pandas=True, x_axis="_step")["adjoint_loss"].dropna()
    return (r, h, loss), r.state


def fmt_curve(h, key, every=250):
    idx = [s for s in h.index if s % every == 0 or s == h.index[-1]]
    return "  ".join(f"{int(s)}:{h.loc[s, key]:.3f}" for s in idx)


def main():
    import wandb
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    runs = list(wandb.Api(timeout=120).runs("CCDS_NTU/adjoint_sampler_cv", per_page=200, order="-created_at"))
    if a.gate:
        got, state = load(runs, "manywell_pretrain_tf", a.commit)
        if got is None or state != "finished":
            print("pré-entraînement introuvable ou non terminé")
            return
        r, h, loss = got
        f = h[[f"well_ratio/{i}" for i in range(N)]] * 0.5 / 0.5   # R_i par rapport à la masse exacte de q (0.5) : f_i = R_i x 0.5
        print(f"adjoint_loss : finie {bool(np.isfinite(loss.values).all())}, 50 dernières {loss.values[-50:].mean():.3f}, max après l'epoch 200 {loss[loss.index >= 200].max():.3g}")
        win = f.values[h.index.values > h.index.values[-1] - 500]
        half = len(win) // 2
        print(f"fraction moyenne de puits léger (cible 0.5) : 30 dernières {f.values[-30:].mean() * 0.5:.3f} ; dérive sur 500 epochs {(win[half:].mean() - win[:half].mean()) * 0.5:+.4f}")
        print("f_i (30 dernières) :", np.round(f.values[-30:].mean(0) * 0.5, 3))
        print("energy_w2 (30 dernières) :", round(float(h['energy_w2'].values[-30:].mean()), 3))
        return
    res = {}
    for k, name in (("AS", "manywell_corr_as"), ("Barker", "manywell_corr_barker")):
        got, state = load(runs, name, a.commit)
        if got is None or state != "finished":
            print(f"{k}: run introuvable ou non terminé")
            return
        res[k] = got
    print(f"{'':8}{'|f-0.156| (primaire)':>22}{'puits conservés':>17}{'TV joint':>10}{'energy_w2':>11}{'loss finie':>12}")
    last = {}
    for k, (r, h, loss) in res.items():
        last[k] = {key: float(h[key].values[-30:].mean()) for key in KEYS}
        print(f"{k:8}{last[k]['light_frac_dev']:22.4f}{last[k]['wells_kept']:17.2f}{last[k]['light_count_tv']:10.3f}{last[k]['energy_w2']:11.2f}{str(bool(np.isfinite(loss.values).all())):>12}")
    print("\nvaleurs de départ (premier point loggé) et epoch 50 :")
    for k, (r, h, loss) in res.items():
        s0 = h.index[0]
        for lab, s in (("départ", s0), ("epoch 50", 50)):
            if s in h.index:
                print(f"  {k:7}{lab:9}(epoch {int(s):4d}) |f-0.156| {h.loc[s, 'light_frac_dev']:.4f}, puits conservés {int(h.loc[s, 'wells_kept'])}, TV {h.loc[s, 'light_count_tv']:.3f}, energy_w2 {h.loc[s, 'energy_w2']:.2f}")
    print("\nf_i à la fin (30 dernières évaluations), AS puis Barker :")
    for k, (r, h, loss) in res.items():
        print(f"  {k:7}", np.round(h[[f'well_ratio/{i}' for i in range(N)]].values[-30:].mean(0) * 0.15569, 3))
    print("\ncourbes (|f-0.156| par epoch) :")
    for k, (r, h, loss) in res.items():
        print(f"  {k:7}", fmt_curve(h, "light_frac_dev"))
    for k, (r, h, loss) in res.items():
        print(f"  {k:7} puits conservés :", fmt_curve(h, "wells_kept"))
    better = last["Barker"]["light_frac_dev"] < last["AS"]["light_frac_dev"]
    kept = last["Barker"]["wells_kept"] >= last["AS"]["wells_kept"]
    print(f"\nBarker < AS sur la métrique primaire : {better} ; Barker conserve au moins autant de puits : {kept}")
    print("=> SUCCÈS" if better and kept else "=> ÉCHEC : documenter et s'arrêter (aucune variante)")


if __name__ == "__main__":
    main()
