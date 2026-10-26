""" Contrôle GPU AVANT un run DW-4 rapide : le sampling par graphe CUDA de l'EGNN (GraphVESDE, bruit tiré sur le GPU) doit (1) pouvoir être capturé, (2) donner la MÊME loi
    d'échantillons que le sampling pas à pas, (3) relire les poids en place après une mise à jour, (4) être plus rapide. Code de sortie 0 si tout est bon, 1 sinon (le job de
    production retombe alors sur le sampling classique) ; 2 si pas de GPU.

      python scripts/check_fast_dw4.py [--sigma-max 2]
    (la vérification de torch.compile est faite par train.py lui-même au démarrage : l'EGNN compilé doit reproduire le résultat eager, sinon train.py reste en eager.)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train  # noqa: F401,E402
from adjoint_samplers.components import sde as sde_mod  # noqa: E402
from adjoint_samplers.components.sde import ControlledSDE, sdeint  # noqa: E402
from adjoint_samplers.utils import train_utils  # noqa: E402
from adjoint_samplers.utils.dw4_states import pair_distances  # noqa: E402


def sample(sde, ts, n, device, graph):
    sde_mod.enable_cuda_graph_sampling(graph)
    x0 = torch.zeros(n, 8, device=device)
    return sdeint(sde, x0, ts, only_boundary=True)[1].cpu()


def main():
    import hydra
    from hydra import compose, initialize_config_dir
    from scipy.stats import ks_2samp
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-max", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--nfe", type=int, default=200)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("pas de GPU : contrôle non effectué")
        sys.exit(2)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=dw4_as_states", f"sigma_max={a.sigma_max}", f"nfe={a.nfe}"])
    device = "cuda"
    torch.manual_seed(0)
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
    controller = hydra.utils.instantiate(cfg.controller).to(device)
    sde = ControlledSDE(ref_sde, controller).to(device)
    ts = train_utils.get_timesteps(**cfg.timesteps).to(device)

    ok = True
    xe = sample(sde, ts, a.n, device, graph=False)
    xg = sample(sde, ts, a.n, device, graph=True)
    captured = any(v is not None for v in sde_mod._BOUNDARY_GRAPHS.values())
    print(f"(1) capture du graphe CUDA : {'OK' if captured else 'ÉCHEC (repli sur le sampling classique)'}")
    ok &= captured
    de, dg = pair_distances(xe).numpy(), pair_distances(xg).numpy()
    ps = [ks_2samp(de[:, k], dg[:, k]).pvalue for k in range(6)]
    mean_err = np.abs(xg.reshape(-1, 4, 2).mean(1)).max()
    same = min(ps) > 1e-3 and mean_err < 1e-4
    print(f"(2) même loi : p-values KS des 6 distances {np.round(ps, 3)}, centre de masse max {mean_err:.1e} -> {'OK' if same else 'ÉCHEC'}")
    ok &= bool(same)
    with torch.no_grad():
        for p_ in controller.parameters():
            p_.add_(0.05 * torch.randn_like(p_))
    xe2 = sample(sde, ts, a.n, device, graph=False)
    xg2 = sample(sde, ts, a.n, device, graph=True)
    ks = min(ks_2samp(pair_distances(xe2).numpy()[:, k], pair_distances(xg2).numpy()[:, k]).pvalue for k in range(6))
    moved = float((pair_distances(xg2).mean() - pair_distances(xg).mean()).abs())
    print(f"(3) poids relus après mise à jour : p-value KS min {ks:.3f}, les échantillons ont bougé de {moved:.3f} -> {'OK' if ks > 1e-3 else 'ÉCHEC'}")
    ok &= ks > 1e-3
    for graph, lab in ((False, "pas à pas"), (True, "graphe CUDA")):
        sample(sde, ts, a.n, device, graph)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            sample(sde, ts, a.n, device, graph)
        torch.cuda.synchronize()
        print(f"(4) {lab} : {(time.time() - t0) / 3:.3f} s pour {a.n} trajectoires de {len(ts) - 1} pas")
    print("=> contrôle", "RÉUSSI" if ok else "ÉCHOUÉ")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
