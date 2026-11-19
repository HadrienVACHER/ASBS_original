"""Local checks of gmm_dk.py (CPU, < 1 min). Run: python tests/test_gmm_dk.py"""
import math, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gmm_dk as G

t0 = time.time()
for d, K, D in [(8, 2, 8.0), (8, 8, 8.0), (32, 8, 4.0), (128, 32, 8.0)]:
    mu = G.means(d, K, D)
    dist = torch.cdist(mu, mu)
    off = dist[~torch.eye(K, dtype=bool)]
    w = G.weights(K)
    assert torch.allclose(off, torch.full_like(off, D), atol=1e-5), "pairwise distance != Delta"
    assert abs(w.sum().item() - 1) < 1e-6 and abs(w[0].item() / w[-1].item() - 10) < 1e-3, "weights"
    # Voronoi = nearest mean (cdist on the full d) on full-d draws
    x, _ = G.sample_exact(20000, d, K, D, torch.Generator().manual_seed(1), full=True)
    assert (G.voronoi(x, mu) == torch.cdist(x, mu).argmin(-1)).all(), "voronoi != cdist argmin"
    # the first-K-coordinates shortcut gives the same cell proportions as the full-d draw (same law)
    pk = G.true_proportions(d, K, D, n=400_000)
    ph = G.proportions(x, d, K, D)
    print(f"d={d:3d} K={K:2d} Delta={D}: sigma_max {G.sigma_max_rule(d, D):.3f}; w max/min {w[0]/w[-1]:.2f}; max|p_k - w_k| {(pk - w.double()).abs().max():.4f}; "
          f"TV(full-d 20k draws, p) {G.tv(ph, pk):.3f}")
# beta recovery on synthetic data: p_hat ∝ p^(1+b)
p = G.true_proportions(8, 8, 8.0, n=200_000)
for b in (0.0, -0.5, -1.0):
    ph = p ** (1 + b); ph = ph / ph.sum()
    # note: normalisation shifts the intercept only, not the slope
    cov, _, c = G.coverage(ph, p)
    print(f"beta synthetic {b:+.1f}: recovered {G.shrink_beta(ph, p, c)[0]:+.3f}  (coverage {cov}/8)")
# GMMDK object and Hydra config
from adjoint_samplers.utils.dist_utils import GMMDK
g = GMMDK(dim=8, K=4, delta=8.0)
x = g.sample((5000,))
assert g.assign(x).shape == (5000,) and torch.isfinite(g.log_prob(x)).all()
import train  # registers resolvers
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base="1.1"):
    cfg = compose(config_name="train.yaml", overrides=["experiment=gmm_dk_as", "dim=32", "K=8", "delta=4.0"])
    OmegaConf.resolve(cfg)
    print("hydra: sigma_max", cfg.sigma_max, "expected", G.sigma_max_rule(32, 4.0), "| epochs", cfg.num_epochs, "| exp_name", cfg.exp_name)
print(f"total {time.time() - t0:.1f}s")
