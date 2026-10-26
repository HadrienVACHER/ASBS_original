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

        # metrics against the target: left-mode mass and 1D W2 (quantile coupling)
        d = self.dist.distr
        self.locs = d.component_distribution.base_dist.loc.reshape(-1).cpu()
        self.weights = d.mixture_distribution.probs.cpu()
        self.n_ref = 100000
        self.ref = self.dist.sample([self.n_ref,]).reshape(-1).cpu().sort().values

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
        self.ax.set_xlim(-6, 6)
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
        return {"hist_img": PIL_img, **self.metrics(samples)}

    def metrics(self, samples: torch.Tensor) -> Dict:
        """ left_mass / right_mass: share of samples closer to the left / right mode mean
            (target: weights[0] / weights[1]); mode_tv = |left_mass - weights[0]| (total variation
            between the two mode weights); w2: 1D Wasserstein-2 to the target, by quantile coupling.
        """
        x = samples.reshape(-1).detach().cpu()
        left = (x - self.locs[0]).abs() < (x - self.locs[1]).abs()
        left_mass = left.float().mean().item()
        n = x.shape[0]
        q = ((torch.arange(n) + 0.5) / n * self.n_ref).long()
        w2 = (x.sort().values - self.ref[q]).pow(2).mean().sqrt().item()
        return {
            "left_mass": left_mass,
            "left_mass_ratio": left_mass / self.weights[0].item(), # 1: right weight
            "mode_tv": abs(left_mass - self.weights[0].item()),
            "w2": w2,
        }


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


        if getattr(self, "skip_eq_w2", False):
            return {"energy_w2": energy_w2, "dist_w2": dist_w2}
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


class DW4Evaluator(SyntheticEenergyEvaluator):
    """ SyntheticEenergyEvaluator (energy W2, equivariant W2, distance W2: unchanged) + the STATE metrics of DW-4: graph of short pairs (d < 4) up to isomorphism (11 classes).
        Logged: short_frac (fraction of pairwise distances < 4), state_tv (model vs the whole reference), state/<k> (model frequency of class k). """
    def __init__(self, ref_samples_path, energy, eq_w2_every: int = 0) -> None:
        super().__init__(ref_samples_path, energy)
        from adjoint_samplers.utils.dw4_states import state_freq, state_ids
        self._ref_freq = state_freq(state_ids(self.ref_samples))
        from adjoint_samplers.utils.dw4_quench import quench
        self._ref_basin_freq = state_freq(state_ids(quench(self.ref_samples)[0]))        # states read on the quenched basins (true mode weights)
        # the point-cloud (equivariant) W2 is the costly metric: computed every `eq_w2_every` epochs and at the last evaluation only (0 = at every evaluation);
        # energy W2, distance W2 and all the state metrics are computed at every evaluation. train.py sets `epoch` and `is_last` before the call.
        self.eq_w2_every, self.epoch, self.is_last = eq_w2_every, 0, True

    def __call__(self, samples: torch.Tensor) -> Dict:
        from adjoint_samplers.utils.dw4_states import pair_distances, state_freq, state_ids, state_tv
        self.skip_eq_w2 = bool(self.eq_w2_every) and self.epoch % self.eq_w2_every != 0 and not self.is_last
        out = super().__call__(samples)
        freq = state_freq(state_ids(samples.cpu()))
        out["short_frac"] = float((pair_distances(samples.cpu()) < 4.0).float().mean())
        out["state_tv"] = state_tv(freq, self._ref_freq)
        out.update({f"state/{k}": float(v) for k, v in enumerate(freq)})
        from adjoint_samplers.utils.dw4_quench import quench
        bfreq = state_freq(state_ids(quench(samples.cpu())[0]))
        out["basin_tv"] = state_tv(bfreq, self._ref_basin_freq)
        out.update({f"basin/{k}": float(v) for k, v in enumerate(bfreq)})
        return out


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
        dist2 = self.dist.sq_dist_to_mode(x, k) # Euclidean for GMM, Mahalanobis for HeteroGMM (χ²_d inside a mode)
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


from adjoint_samplers.utils.girsanov import mode_weight_ess_frac  # noqa: E402


class ManyWellEvaluator:
    """ ManyWell (FAB): per-coordinate mass of the LIGHT well. For each of the n_wells double-well coordinates x_{2i-1}, the share of samples
        in the left well (x < well_split) divided by the exact share p_left = 0.1557:

          well_ratio/<i>        R_i = left share / p_left       (1: right proportion, 0: the light well is never visited)
          well_ratio_mean       mean_i R_i
          well_logratio_mean    mean_i |log max(R_i, 1e-3)|
          wells_visited_frac    share of coordinates visiting BOTH wells (left share >= cover_ratio p_left and right share >= cover_ratio p_right)
          energy_w2             1D W2 between the energies of the samples and of exact samples
        and each aggregate also as `<name>_floor` for exact samples of the same size.
    """
    def __init__(self, energy, cover_ratio: float = 0.1, n_floor_draws: int = 5) -> None:
        from adjoint_samplers.energies.dist_energy import DistEnergy
        from adjoint_samplers.utils.dist_utils import ManyWell
        assert isinstance(energy, DistEnergy) and isinstance(energy.dist, ManyWell)
        self.energy, self.dist = energy, energy.dist
        self.cover_ratio, self.n_floor_draws = cover_ratio, n_floor_draws
        self.p_left = self.dist.light_mass
        self._floors = {}

    def well_metrics(self, x: torch.Tensor, per_coord: bool = False) -> Dict:
        left = (x[:, 0::2] < self.dist.well_split).float().mean(0)     # (n_wells,)
        R = left / self.p_left
        visited = (left >= self.cover_ratio * self.p_left) & ((1 - left) >= self.cover_ratio * (1 - self.p_left))
        # joint: TV between the law of the number of light wells per sample and Binomial(n_wells, p_left) (independent coordinates)
        n_light = (x[:, 0::2] < self.dist.well_split).sum(1).long()
        emp = torch.bincount(n_light, minlength=self.dist.n_wells + 1).double() / len(x)
        binom = torch.distributions.Binomial(self.dist.n_wells, torch.tensor(self.p_left, dtype=torch.float64)).log_prob(
            torch.arange(self.dist.n_wells + 1, dtype=torch.float64)).exp()
        out = {
            "light_count_tv": 0.5 * (emp.cpu() - binom).abs().sum().item(),
            "light_frac_dev": (left - self.p_left).abs().mean().item(),
            "wells_kept": int((left > 0.01).sum().item()),
            "mode_ess_frac": mode_weight_ess_frac(left.double(), self.p_left),
            "well_ratio_mean": R.mean().item(),
            "well_logratio_mean": R.clamp(min=1e-3).log().abs().mean().item(),
            "wells_visited_frac": visited.float().mean().item(),
        }
        if per_coord:
            out.update({f"well_ratio/{i}": r for i, r in enumerate(R.tolist())})
        return out

    def well_floors(self, B: int, device) -> Dict:
        if B not in self._floors:
            devices = [device] if torch.device(device).type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                draws = [self.well_metrics(self.dist.sample([B,]).to(device)) for _ in range(self.n_floor_draws)]
            self._floors[B] = {f"{k}_floor": float(np.mean([d[k] for d in draws])) for k in draws[0]}
        return self._floors[B]

    def __call__(self, samples: torch.Tensor) -> Dict:
        B, D = samples.shape
        assert D == self.dist.dim
        ref = self.dist.sample([B,]).to(samples.device)
        gen_energy, ref_energy = self.energy.eval(samples), self.energy.eval(ref)
        energy_w2 = pot.emd2_1d(ref_energy.cpu().numpy(), gen_energy.cpu().numpy())**0.5
        return {
            "energy_w2": energy_w2,
            **self.well_metrics(samples, per_coord=True),
            **self.well_floors(B, samples.device),
        }

