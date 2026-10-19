# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from typing import Dict
from pathlib import Path
import torch
import ot as pot
import numpy as np

from adjoint_samplers.energies import DoubleWellEnergy, LennardJonesEnergy
from adjoint_samplers.utils.graph_utils import remove_mean
from adjoint_samplers.utils.eval_utils import (
    dist_point_clouds,
    interatomic_dist,
    get_fig_axes,
    fig2img,
)


class DemoEvaluator:
    def __init__(self, energy) -> None:
        from adjoint_samplers.energies.dist_energy import DistEnergy
        assert isinstance(energy, DistEnergy)
        self.dist = energy.dist

        # Plot target samples
        self.fig, axes = get_fig_axes(ncol=6, nrow=10, ax_length_in=3)
        self.axes = axes.reshape(-1)
        self.subplot_idx = 0

    @property
    def ax(self):
        return self.axes[self.subplot_idx]

    def plot_hist(self, x, title=None) -> None:
        B, D = x.shape
        assert D == 1

        if title is None:
            title = f"Eval #{self.subplot_idx}"

        x = (x.reshape(-1)).detach().cpu()
        self.ax.hist(x, bins=50, density=True)
        self.ax.set_xlim(-3.5, 3.5)
        self.ax.set_ylim(0, 1.0)
        self.ax.grid(True)
        self.ax.set_title(title)

    def __call__(self, samples: torch.Tensor) -> Dict:
        # Plot target samples for reference
        if self.subplot_idx == 0:
            target_samples = self.dist.sample([10000,]).cpu()
            self.plot_hist(target_samples, title="Target")
            self.subplot_idx += 1

        # Plot model samples
        self.plot_hist(samples.cpu())
        self.subplot_idx += 1

        # Return figure
        self.fig.canvas.draw()
        PIL_img = fig2img(self.fig)
        return {"hist_img": PIL_img}


class SyntheticEenergyEvaluator:
    def __init__(self, ref_samples_path, energy) -> None:

        assert isinstance(energy, (DoubleWellEnergy, LennardJonesEnergy))
        self.energy = energy
        self.n_particles = energy.n_particles
        self.n_spatial_dim = energy.n_spatial_dim

        # Extract reference samples
        root = Path(os.path.abspath(__file__)).parent.parent.parent
        ref_samples_np = np.load(root / Path(ref_samples_path), allow_pickle=True)
        self.ref_samples = remove_mean(
            torch.tensor(ref_samples_np),
            energy.n_particles,
            energy.n_spatial_dim,
        )

    def __call__(self, samples: torch.Tensor) -> Dict:

        B, D = samples.shape
        assert D == self.energy.dim

        # Sample reference samples
        idxs = torch.randperm(len(self.ref_samples))[:B]
        ref_samples = self.ref_samples[idxs].to(samples.device)


        print("Computing energy W2...")
        gen_energy = self.energy.eval(samples)
        ref_energy = self.energy.eval(ref_samples)
        energy_w2 = pot.emd2_1d(ref_energy.cpu().numpy(), gen_energy.cpu().numpy())**0.5


        print("Computing interatomic W2...")
        gen_dist = interatomic_dist(samples, self.n_particles, self.n_spatial_dim)
        ref_dist = interatomic_dist(ref_samples, self.n_particles, self.n_spatial_dim)
        dist_w2 = pot.emd2_1d(
            gen_dist.cpu().numpy().reshape(-1),
            ref_dist.cpu().numpy().reshape(-1),
        )


        print("Computing particles W2...")
        M = dist_point_clouds(
            samples.reshape(-1, self.n_particles, self.n_spatial_dim).cpu(),
            ref_samples.reshape(-1, self.n_particles, self.n_spatial_dim).cpu(),
        )
        a = torch.ones(M.shape[0]) / M.shape[0]
        b = torch.ones(M.shape[0]) / M.shape[0]
        eq_w2 = pot.emd2(M=M**2, a=a, b=b)**0.5
        eq_w2 = eq_w2.item()


        return {
            "energy_w2": energy_w2,
            "eq_w2": eq_w2,
            "dist_w2": dist_w2,
        }


class GMMEvaluator:
    """ Compare generated samples against exact samples of a GMM target

        Each sample is assigned to its most likely mode k, and counts for k only if it
        lies inside it: ||x - μ_k||² <= χ²_d(in_mode_quantile). ŵ_k = share of samples
        inside mode k, ŵ_out = share inside no mode. Mode metrics:
          frac_in_modes   1 - ŵ_out
          mass_covered    Σ_k π_k 1[ŵ_k >= cover_ratio π_k]   (mode collapse, weighted by π)
          modes_covered   mean_k 1[ŵ_k >= cover_ratio π_k]    (mode collapse, unweighted)
          mode_tv         ½ (Σ_k |ŵ_k - π_k| + ŵ_out)         (mode weight mismatch)
          eff_modes_ratio exp(H(ŵ | inside)) / exp(H(π))      (1: right spread over modes)
          mode_spread     mean ||x - μ_k||² / d               (1: right width, <1 too narrow)
        Each also comes as `<name>_floor`: its value for exact samples of the same size,
        i.e. the best value reachable with this many samples.
        If K <= max_per_mode, also `mode_ratio/<k>` = ŵ_k / π_k for each mode
        (1: right proportion, 0: missed).

        energy_w2 and the mode metrics use all samples; sample_w2 (exact OT,
        roughly cubic in the number of samples) uses only the first `w2_samples`.
    """
    def __init__(
        self,
        energy,
        w2_samples: int = 2000,
        in_mode_quantile: float = 0.999,
        cover_ratio: float = 0.1,
        n_floor_draws: int = 5,
        max_per_mode: int = 64,
    ) -> None:
        from scipy.stats import chi2
        from adjoint_samplers.energies.dist_energy import DistEnergy
        from adjoint_samplers.utils.dist_utils import GMM
        assert isinstance(energy, DistEnergy) and isinstance(energy.dist, GMM)
        self.energy = energy
        self.dist = energy.dist
        self.w2_samples = w2_samples
        self.in_mode_r2 = float(chi2.ppf(in_mode_quantile, self.dist.dim))
        self.cover_ratio = cover_ratio
        self.n_floor_draws = n_floor_draws
        self.max_per_mode = max_per_mode
        self._floors = {} # number of samples -> floors

    def mode_metrics(self, x: torch.Tensor, per_mode: bool = False) -> Dict:
        B, D = x.shape
        pi = self.dist.weights.to(x.device)
        k = self.dist.assign(x)
        dist2 = (x - self.dist.means[k]).pow(2).sum(-1)
        inside = dist2 <= self.in_mode_r2

        counts = torch.bincount(k[inside], minlength=self.dist.K).float()
        w = counts / B
        frac_in = inside.float().mean()
        covered = w >= self.cover_ratio * pi

        q = counts / counts.sum().clamp(min=1)
        entropy = lambda p: -(p * p.clamp(min=1e-30).log()).sum()
        ratios = {}
        if per_mode and self.dist.K <= self.max_per_mode:
            ratios = {f"mode_ratio/{j}": r for j, r in enumerate((w / pi).tolist())}
        return {
            "frac_in_modes": frac_in.item(),
            "mass_covered": (pi * covered).sum().item(),
            "modes_covered": covered.float().mean().item(),
            "mode_tv": (0.5 * ((w - pi).abs().sum() + 1 - frac_in)).item(),
            "eff_modes_ratio": (entropy(q) - entropy(pi)).exp().item(),
            "mode_spread": (dist2.mean() / D).item(),
            **ratios,
        }

    def mode_floors(self, B: int, device) -> Dict:
        """ mode metrics of exact samples, averaged over a few draws; cached per B.
            Uses its own RNG state, so training randomness is unaffected.
        """
        if B not in self._floors:
            devices = [device] if torch.device(device).type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                draws = [self.mode_metrics(self.dist.sample([B,]).to(device))
                         for _ in range(self.n_floor_draws)]
            self._floors[B] = {f"{k}_floor": float(np.mean([d[k] for d in draws])) for k in draws[0]}
        return self._floors[B]

    def __call__(self, samples: torch.Tensor) -> Dict:
        B, D = samples.shape
        assert D == self.dist.dim

        ref_samples = self.dist.sample([B,]).to(samples.device)

        print("Computing energy W2...")
        gen_energy = self.energy.eval(samples)
        ref_energy = self.energy.eval(ref_samples)
        energy_w2 = pot.emd2_1d(
            ref_energy.cpu().numpy(), gen_energy.cpu().numpy()
        )**0.5

        print("Computing mode metrics...")
        mode_metrics = self.mode_metrics(samples, per_mode=True)
        mode_floors = self.mode_floors(B, samples.device)

        print("Computing sample W2...")
        n = min(B, self.w2_samples)
        M = torch.cdist(samples[:n].cpu().double(), ref_samples[:n].cpu().double())
        a = b = np.ones(n) / n
        sample_w2 = pot.emd2(a, b, M.numpy()**2)**0.5

        return {
            "energy_w2": energy_w2,
            "sample_w2": sample_w2,
            **mode_metrics,
            **mode_floors,
        }
