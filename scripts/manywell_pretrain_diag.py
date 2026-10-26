""" Diagnostic du pré-entraînement : où les trajectoires du modèle quittent-elles la loi du processus optimal ? À chaque temps t, on compare l'état X_t du modèle (200 pas, comme à
    l'évaluation) à X_t de référence = pont de base posté sur des x1 EXACTS de q (même marginale que le processus optimal) :

      python scripts/manywell_pretrain_diag.py --controller results/manywell_pretrain/seed0/controller_final.pt [--experiment manywell_pretrain]

    Imprime par t : fraction d'échantillons avec une coordonnée double-puits à |x| > 3 (modèle / référence), quantile 99.9 % de |x_double|, ET l'amplitude moyenne du contrôle |u|.
    Une divergence précoce = le contrôleur n'a pas appris la bonne dérive ; tardive (t > 0.9) = instabilité de fin de trajectoire.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train  # noqa: F401,E402
from adjoint_samplers.components.sde import ControlledSDE, sdeint  # noqa: E402
from adjoint_samplers.utils import train_utils  # noqa: E402


def main():
    import hydra
    from hydra import compose, initialize_config_dir
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", required=True)
    ap.add_argument("--experiment", default="manywell_pretrain")
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=[f"experiment={a.experiment}"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy = hydra.utils.instantiate(cfg.energy, device=device)
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
    controller = hydra.utils.instantiate(cfg.controller).to(device)
    controller.load_state_dict(torch.load(a.controller, map_location=device, weights_only=False)["controller"])
    sde = ControlledSDE(ref_sde, controller).to(device)
    ts = train_utils.get_timesteps(**cfg.timesteps).to(device)
    torch.manual_seed(0)
    x0 = torch.zeros(a.n, energy.dim, device=device)
    with torch.no_grad():
        xs = torch.stack(sdeint(sde, x0, ts, only_boundary=False))                      # (T, N, D)
        x1 = energy.dist.sample([a.n]).to(device)
        print(f"{'t':>6}{'modèle: part |x|>3':>20}{'réf: part |x|>3':>17}{'modèle q99.9 |x|':>18}{'réf q99.9 |x|':>15}{'|u| moyen':>11}")
        for k in list(range(0, len(ts), max(1, len(ts) // 20))) + [len(ts) - 2, len(ts) - 1]:
            t = ts[k]
            tt = t.expand(a.n, 1)
            xr = ref_sde.sample_posterior(tt, x0, x1) if k < len(ts) - 1 else x1
            xm = xs[k]
            u = controller(t, xm).norm(dim=-1).mean() if k < len(ts) - 1 else torch.tensor(float("nan"))
            dm, dr = xm[:, 0::2].abs(), xr[:, 0::2].abs()
            print(f"{float(t):6.3f}{float((dm > 3).any(1).float().mean()):20.3f}{float((dr > 3).any(1).float().mean()):17.3f}"
                  f"{float(dm.flatten().quantile(0.999)):18.2f}{float(dr.flatten().quantile(0.999)):15.2f}{float(u):11.2f}")


if __name__ == "__main__":
    main()
