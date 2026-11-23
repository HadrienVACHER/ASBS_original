""" rules_train_weightfree.md: start files of the three cells (same for the three arms): the start controller and 10^4 FRESH end points simulated from it on CPU WITH their path log weights
    (girsanov.log_path_weights, seed 7), so that the tempered-IS arm has the weights of its initial buffer.
    python scripts/wf_make_start.py CELL      CELL in {d8, d32, d128} """
import sys, time, json
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import train  # noqa: F401 (resolvers)
import hydra
from omegaconf import OmegaConf
from adjoint_samplers.components.sde import ControlledSDE
from adjoint_samplers.utils import girsanov, train_utils
T0 = time.time()
CELLS = {"d8": dict(cfg="results/gmm_dk_sweep/gmmdk_d8_K8_D8_s0_sx0.5/config.yaml", ckpt="results/gmm_dk_cont/gmmdk_d8_K8_D8_s0_sx0.5/controller_start.pt", sigma=None),
         "d32": dict(cfg="results/gmm_dk_sweep/gmmdk_d32_K8_D6_s0_sx0.5/config.yaml", ckpt="results/gmm_dk_cont/gmmdk_d32_K8_D6_s0_sx0.5/controller_start.pt", sigma=None),
         "d128": dict(cfg="figures/gmmdk_step4/configs/gmmdk4_d128_sx0.5.config.yaml", ckpt="figures/gmmdk_step4/controllers/gmmdk4_d128_sx0.75_ep1500.pt", sigma=1.602001),
         # rules_selftrain_dartbank.md: (32, 8, 8) x 0.5 (final of the window run, 1500 epochs; modes missing)
         "k8d32": dict(cfg="results/gmm_dk_window/w1/gmmdk_d32_K8_D8_s0_sx0.5/config.yaml", ckpt="results/gmm_dk_window/w1/gmmdk_d32_K8_D8_s0_sx0.5/controller_final.pt", sigma=None)}
c = sys.argv[1]; C = CELLS[c]
cfg = OmegaConf.load(ROOT / C["cfg"])
if C["sigma"]:
    cfg.sigma_max = C["sigma"]
OmegaConf.resolve(cfg)
ts = train_utils.get_timesteps(**cfg.timesteps); energy = hydra.utils.instantiate(cfg.energy, device="cpu")
ref = hydra.utils.instantiate(cfg.ref_sde); ctrl = hydra.utils.instantiate(cfg.controller)
ck = torch.load(ROOT / C["ckpt"], map_location="cpu", weights_only=False); ctrl.load_state_dict(ck["controller"]); sde = ControlledSDE(ref, ctrl)
torch.manual_seed(7)
with torch.no_grad():
    lw, x = girsanov.log_path_weights(sde, energy, torch.zeros(10000, int(cfg.dim)), ts)
out = ROOT / "results" / "wf_start" / c; out.mkdir(parents=True, exist_ok=True)
torch.save({"controller": ck["controller"]}, out / "controller_start.pt")
torch.save({"buffer_x1": x.float(), "model_samples": x.float(), "logw": lw.double()}, out / "start_samples.pt")
lwd = lw.double(); ess = float(torch.exp(2 * torch.logsumexp(lwd, 0) - torch.logsumexp(2 * lwd, 0)) / len(lwd)); ess5 = float(torch.exp(2 * torch.logsumexp(0.5 * lwd, 0) - torch.logsumexp(lwd, 0)) / len(lwd))
info = dict(cell=c, dim=int(cfg.dim), delta=float(cfg.delta), sigma_max=float(cfg.sigma_max), ess_w=ess, ess_w05=ess5, time_s=time.time() - T0)
json.dump(info, open(out / "info.json", "w"), indent=1); print(info)
