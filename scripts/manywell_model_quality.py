""" Qualité du MODÈLE (ce qu'un post-traitement ne change pas) : ESS d'échantillonnage préférentiel par poids de chemin de Girsanov et erreur sur log Z, avec chaque
    contrôleur final comme proposition. Log Z exact par quadrature 1D (ManyWell.log_Z). AS + post-traitement = le modèle AS : mêmes valeurs.

      python scripts/manywell_model_quality.py --controller results/manywell_corr_as/seed0/controller_final.pt --experiment manywell_corr_as --label AS

    Sur GPU (200 pas x 20k trajectoires). Imprime ESS, ESS/N, log Z_hat, |log Z_hat - log Z| (poolé) et l'écart-type de log Z_hat entre 4 lots indépendants.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train  # noqa: F401,E402  (registers the resolvers)
from adjoint_samplers.components.sde import ControlledSDE  # noqa: E402
from adjoint_samplers.utils import train_utils  # noqa: E402
from adjoint_samplers.utils.girsanov import ess_and_logz, log_path_weights, mode_weight_ess_frac  # noqa: E402


def main():
    import hydra
    from hydra import compose, initialize_config_dir
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=123)
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
    torch.manual_seed(a.seed)
    lws, xs, per_batch, B = [], [], [], a.n // 4
    for _ in range(4):
        lw, x1 = log_path_weights(sde, energy, torch.zeros(B, energy.dim, device=device), ts)
        lws.append(lw.cpu())
        xs.append(x1.cpu())
        per_batch.append(ess_and_logz(lw.cpu())["log_zhat"])
    log_z = energy.dist.log_Z()
    r = ess_and_logz(torch.cat(lws), log_z)
    sd = float(torch.tensor(per_batch).std())
    print(f"{a.label or a.experiment}: N = {a.n} trajectoires ; ESS = {r['ess']:.1f} ({100 * r['ess_frac']:.3f} %) ; log Z_hat = {r['log_zhat']:.3f} "
          f"(exact {log_z:.3f}) ; |log Z_hat - log Z| = {r['log_z_err']:.3f} ; écart-type de log Z_hat entre 4 lots = {sd:.3f}")
    # fallback if the path-weight ESS is ~0 for every model: mode-weight ESS from the per-coordinate light fractions of the model endpoints
    xe = torch.cat(xs)
    f = (xe[:, 0::2] < energy.dist.well_split).float().mean(0)
    print(f"  repli : ESS des poids de modes ESS_w/N = {100 * mode_weight_ess_frac(f, energy.dist.light_mass):.4f} % (f_i du modèle, cible {energy.dist.light_mass:.4f}) ; f_i = {[round(v, 3) for v in f.tolist()]}")


if __name__ == "__main__":
    main()
