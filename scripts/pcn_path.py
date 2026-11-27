""" rules/rules_pcn_path.md (PHASE 2, PREPARED, NEVER RUN): pCN Metropolis-Hastings on the Brownian innovations of a trained AS controller.
    python scripts/pcn_path.py EPOCH RHO [--chains 512] [--steps 500] [--device cuda]     one configuration (RHO = 0 is independent MH with the same budget)
    NOT TESTED: run a short smoke test (--chains 4 --steps 2 --device cpu) and write the frozen rules before any real use. """
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import train, hydra  # noqa: registers resolvers
from omegaconf import OmegaConf
from adjoint_samplers.components.sde import ControlledSDE
from adjoint_samplers.utils import train_utils
ap = argparse.ArgumentParser(); ap.add_argument("epoch", type=int); ap.add_argument("rho", type=float); ap.add_argument("--chains", type=int, default=512); ap.add_argument("--steps", type=int, default=500); ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); dev = a.device; torch.manual_seed(a.seed); t0 = time.time()
cfg = OmegaConf.load(ROOT / "figures/gmmdk_step4/configs/gmmdk4_d128_sx0.5.config.yaml")
cfg.sigma_max = float(json.load(open(ROOT / "figures/gmmdk_step4/q_gmmdk4_d128_sx0.75.json"))["sigma_max"]); OmegaConf.resolve(cfg)
ts = train_utils.get_timesteps(**cfg.timesteps).to(dev); energy = hydra.utils.instantiate(cfg.energy, device=dev); D = int(cfg.dim); K = len(ts) - 1
ref = hydra.utils.instantiate(cfg.ref_sde).to(dev); ctrl = hydra.utils.instantiate(cfg.controller).to(dev)
ctrl.load_state_dict(torch.load(ROOT / f"figures/gmmdk_step4/controllers/gmmdk4_d128_sx0.75_ep{a.epoch}.pt", map_location=dev, weights_only=False)["controller"]); sde = ControlledSDE(ref, ctrl).to(dev); sde.train(False)


@torch.no_grad()
def run(xi):
    """ trajectories from innovations xi (K, N, D): returns (log w~, endpoints), same formula as girsanov.log_path_weights """
    N = xi.shape[1]; x = torch.zeros(N, D, device=dev); lr = torch.zeros(N, device=dev); V = 0.0
    for i in range(K):
        t, dt = ts[i], ts[i + 1] - ts[i]; v = sde.diff(t) ** 2 * dt; m = sde.drift(t, x) * dt; e = xi[i]
        lr += 0.5 * (m ** 2).sum(-1) / v + (e * m).sum(-1) / v.sqrt(); x = sde.propagate(x, m + v.sqrt() * e); V += float(v)
    lref = -0.5 * (x ** 2).sum(-1) / V - 0.5 * D * np.log(2 * np.pi * V)
    return -lr - energy.eval(x) - lref, x


xi = torch.randn(K, a.chains, D, device=dev); lw, x = run(xi); keep_x, keep_lw, acc = [], [], 0.0
for step in range(a.steps):
    prop = a.rho * xi + (1 - a.rho ** 2) ** 0.5 * torch.randn_like(xi); lwp, xp = run(prop)
    ok = torch.log(torch.rand(a.chains, device=dev)) < lwp - lw
    xi[:, ok] = prop[:, ok]; lw = torch.where(ok, lwp, lw); x = torch.where(ok[:, None], xp, x); acc += float(ok.float().mean())
    keep_x.append(x.cpu()); keep_lw.append(lw.cpu())
out = ROOT / "results" / "pcn_path"; out.mkdir(parents=True, exist_ok=True)
torch.save(dict(x=torch.stack(keep_x), lw=torch.stack(keep_lw), accept=acc / a.steps, rho=a.rho, epoch=a.epoch, time_s=time.time() - t0), out / f"pcn_ep{a.epoch}_rho{a.rho}.pt")
print(f"ep{a.epoch} rho {a.rho}: acceptance {acc / a.steps:.3f}, {time.time() - t0:.1f} s")
