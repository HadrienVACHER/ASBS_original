# Copyright (c) Meta Platforms, Inc. and affiliates.

import warnings
from collections import defaultdict

import torch
from torch.func import grad
from torch.utils.data import DataLoader

from adjoint_samplers.components.buffer import BatchBuffer, DeviceBatchLoader
from adjoint_samplers.components.mode_correction import (
    EnergyCounter, TiedMixture, add_uncovered_centres, fit_gmm, mala_steps, merge_centres, refine_centres,
)
from adjoint_samplers.components.sde import BaseSDE, sdeint
from adjoint_samplers.components.state_cost import GradStateCost, ZeroGradStateCost
from adjoint_samplers.components.term_cost import GradEnergy, ScoreGradTermCost


class Matcher:
    def __init__(
        self,
        sde: BaseSDE | None = None,
        buffer: BatchBuffer | None = None,
        resample_size: int | None = None,
        duplicates: int | None = None,
        loss_scale: float = 1,
        **kwargs,
    ):
        self.sde = sde
        self.buffer = buffer
        self.resample_size = resample_size
        self.duplicates = duplicates
        self.loss_scale = loss_scale

    def build_dataloader(self, batch_size, collate_fn=None, device=None):
        dataset = self.buffer.build_dataset(self.duplicates)
        if device is not None:
            return DeviceBatchLoader(dataset, batch_size, device)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def populate_buffer(self):
        raise NotImplementedError()

    def prepare_target(self):
        raise NotImplementedError()


class AdjointMatcher(Matcher):
    def __init__(
        self,
        grad_term_cost: GradEnergy | None = None,
        grad_state_cost: GradStateCost | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.grad_term_cost = grad_term_cost
        self.grad_state_cost = grad_state_cost

    @torch.no_grad()
    def _backward_simulate(self, adjoint1, timesteps, xs):
        (T, B, D) = xs.shape
        assert len(timesteps) == T and T > 1
        assert adjoint1.shape == (B, D)

        adjoint = adjoint1.clone()
        adjoints = [adjoint]
        for j in range(T - 1, 0, -1):
            dt = timesteps[j] - timesteps[j - 1]
            assert dt > 0

            t = timesteps[j].repeat((B, 1))
            x = xs[j]
            assert t.shape == (B, 1) and x.shape == (B, D)

            # Compute a^T ∇ b(t, x)
            ref_sde = self.sde.ref_sde
            if not ref_sde.has_drift:
                f = torch.zeros_like(x)
            else:
                with torch.enable_grad():
                    f = grad(
                        lambda x: torch.sum(adjoint * ref_sde.drift(t, x))
                    )(x)

            # Compute gradient of state cost
            f = f + self.grad_state_cost(t, x)

            adjoint = adjoint + f * dt
            adjoints.append(adjoint)

        adjoints = torch.stack(adjoints[::-1])
        assert adjoints.shape == (T, B, D)
        return adjoints

    def _compute_adjoint1(self, x1, is_asbs_init_stage):
        if is_asbs_init_stage:
            # IPF init: First Adjoint Matching stage
            # of ASBS uses zero corrector
            adjoint1 = self.grad_term_cost.grad_E(x1)
        else:
            adjoint1 = self.grad_term_cost(x1)
        return adjoint1

    def populate_buffer(
            self,
            x0: torch.Tensor,
            timesteps: torch.Tensor,
            is_asbs_init_stage: bool,
    ):
        (B, D), T = x0.shape, len(timesteps)
        assert x0.device == timesteps.device

        ts = timesteps.unsqueeze(1).repeat((1, B))[..., None]
        assert ts.shape == (T, B, 1)

        xs = sdeint(
            self.sde,
            x0,
            timesteps,
            only_boundary=False,
        )
        xs = torch.stack(xs)
        assert xs.shape == (T, B, D)

        adjoint1 = self._compute_adjoint1(xs[-1], is_asbs_init_stage).clone()
        adjoints = self._backward_simulate(adjoint1, timesteps, xs)
        assert adjoints.shape == (T, B, D)

        # note: use entire traj as one smaple. this improves training.
        ts = ts.transpose(0, 1)
        xs = xs.transpose(0, 1)
        adjoints = adjoints.transpose(0, 1)
        assert ts.shape == (B, T, 1)
        assert adjoints.shape == xs.shape == (B, T, D)

        self.buffer.add({
            "t": ts.reshape(B, T).detach().cpu(),
            "xt": xs.reshape(B, T * D).detach().cpu(),
            "adjointt": adjoints.reshape(B, T * D).detach().cpu(),
        })

    def prepare_target(self, data, device):
        t = data["t"].to(device)
        xt = data["xt"].to(device)
        adjointt = data["adjointt"].to(device)

        (B, T), D = t.shape, xt.shape[1] // t.shape[1]
        assert xt.shape == adjointt.shape == (B, T * D)

        # randomly select B index
        # TODO(ghliu) not used for AS / ASBS. Refac this logic.
        idx = torch.randint(high=T, size=(B,))
        idx = [i*T+id for i, id in enumerate(idx)]

        t = t.reshape(B * T, 1)[idx]
        xt = xt.reshape(B * T, D)[idx]
        adjointt = adjointt.reshape(B * T, D)[idx]
        return (t, xt), - adjointt


class AdjointVEMatcher(AdjointMatcher):
    """ Efficient computation of AM when the base SDE has no drift (e.g., VE)
        and the SOC problem has no state cost.
    """
    def __init__(self, t_focus: float = 0.0, t_focus_max: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.t_focus = t_focus
        self.t_focus_max = t_focus_max
        self._check_soc_problem()

    def _check_soc_problem(self):
        assert not self.sde.ref_sde.has_drift
        assert isinstance(self.grad_state_cost, ZeroGradStateCost)

    def _check_buffer_sample_shape(self, x0, x1, adjoint1):
        (B, D) = x0.shape
        assert x1.shape == adjoint1.shape == (B, D)

    def populate_buffer(
            self,
            x0: torch.Tensor,
            timesteps: torch.Tensor,
            is_asbs_init_stage: bool,
    ):
        (x0, x1) = sdeint(
            self.sde,
            x0,
            timesteps,
            only_boundary=True,
        )
        adjoint1 = self._compute_adjoint1(x1, is_asbs_init_stage).clone()

        self._check_buffer_sample_shape(x0, x1, adjoint1)
        self.buffer.add({
            "x0": x0.to("cpu"),
            "x1": x1.to("cpu"),
            "adjoint1": adjoint1.to("cpu"),
        })

    def sample_t(self, x):
        (B, D) = x.shape
        t = torch.rand(B, 1, device=x.device)
        if self.t_focus > 0:
            # a fraction t_focus of the times in [0, t_focus_max], where modes are chosen. The pointwise
            # minimizer of the regression, hence the fixed point, does not depend on the law of t.
            focus = torch.rand(B, 1, device=x.device) < self.t_focus
            t = torch.where(focus, t * self.t_focus_max, t)
        return t

    def _check_target_shape(self, t, xt, adjoint):
        (B, D) = xt.shape
        assert t.shape == (B, 1) and adjoint.shape == (B, D)

    def prepare_target(self, data, device):
        x0 = data["x0"].to(device)
        x1 = data["x1"].to(device)
        adjoint1 = data["adjoint1"].to(device)

        t = self.sample_t(x0).to(device)
        xt = self.sde.sample_base_posterior(t, x0, x1)
        adjoint = adjoint1 # const w.r.t. time in this case

        self._check_target_shape(t, xt, adjoint)
        return (t, xt), - adjoint


class AdjointVPMatcher(AdjointVEMatcher):
    """ Efficient computation of AM when the base SDE has linear drift (e.g., VP)
        and the SOC problem has no state cost.
    """
    def _check_soc_problem(self):
        assert self.sde.ref_sde.has_drift
        assert isinstance(self.grad_state_cost, ZeroGradStateCost)

    def prepare_target(self, data, device):
        x0 = data["x0"].to(device)
        x1 = data["x1"].to(device)
        adjoint1 = data["adjoint1"].to(device)

        t = self.sample_t(x0).to(device)
        xt = self.sde.sample_base_posterior(t, x0, x1)
        adjoint = adjoint1 # const w.r.t. time in this case
        adjoint = adjoint * torch.exp(self.sde.ref_sde.coeff2(t))

        self._check_target_shape(t, xt, adjoint)
        return (t, xt), - adjoint


class AdjointVEDiracMatcher(AdjointVEMatcher):
    """ AdjointVEMatcher for AS from a Dirac X0, where with ν_t = ∫_0^t σ², κ_t = ν_t / ν_1:

            p^base(xt | x1) = N(xt; (1 - κ_t) X0 + κ_t x1, ν_1 κ_t (1 - κ_t))
    """
    def _check_soc_problem(self):
        super()._check_soc_problem()
        # AS from a Dirac X0: p^base_1 = N(X0, ν_1)
        assert isinstance(self.grad_term_cost, ScoreGradTermCost)
        var1 = self.grad_term_cost.var1
        assert torch.allclose(var1, self._nu(torch.ones_like(var1)))

    def _nu(self, t):
        # ν_t = ∫_0^t σ(s)² ds
        return self.sde.ref_sde._pt_gauss_param(t, torch.zeros_like(t))[1]


class AdjointVETargetBufferMatcher(AdjointVEDiracMatcher):
    """ Sanity check: X1 of the buffer is drawn from the target p1 (exact sampler of `energy.dist`)
        instead of from the model. The regression is then the one of u*, so the model should converge
        whatever the weights it currently has; X0 is the Dirac source.
    """
    def populate_buffer(
            self,
            x0: torch.Tensor,
            timesteps: torch.Tensor,
            is_asbs_init_stage: bool,
    ):
        dist = self.grad_term_cost.energy.dist
        x1 = dist.sample([x0.shape[0],]).reshape(x0.shape).to(x0)
        adjoint1 = self._compute_adjoint1(x1, is_asbs_init_stage).clone()

        self._check_buffer_sample_shape(x0, x1, adjoint1)
        self.buffer.add({
            "x0": x0.to("cpu"),
            "x1": x1.to("cpu"),
            "adjoint1": adjoint1.to("cpu"),
        })


class AdjointVEFixedBufferMatcher(AdjointVETargetBufferMatcher):
    """ Pre-training buffer: X1 are EXACT samples of the target (`energy.dist`), drawn ONCE until `fixed_size` samples are stored, then never refreshed
        (populate_buffer is a no-op). The regression is that of u* for this fixed set of endpoints (target ∇g of the energy it is built with). """
    def __init__(self, fixed_size: int = 10000, **kwargs):
        super().__init__(**kwargs)
        self.fixed_size = fixed_size
        self._filled = 0

    def populate_buffer(self, x0, timesteps, is_asbs_init_stage):
        n = min(x0.shape[0], self.fixed_size - self._filled)
        if n <= 0:
            return
        super().populate_buffer(x0[:n], timesteps, is_asbs_init_stage)
        self._filled += n


class AdjointVEIMHBufferMatcher(AdjointVEDiracMatcher):
    """ X1 of the buffer comes from an independent Metropolis-Hastings chain on the target p1 = e^{-E} / Z,
        not from the model. The proposal r is the target with equal mode weights (`dist.uniform_weights()`):

            x' ~ r,   α(x, x') = min(1, e^{-[E(x') - E(x)]} r(x) / r(x'))

        B chains (one per sample of the batch) persist across epochs and make `n_steps` MH steps each time
        the buffer is filled; they start from r and burn in for `burn_in` steps. Like the exact-sample
        buffer (AdjointVETargetBufferMatcher), the regression is then the one of u* up to the chain's
        distance to p1; X0 is the Dirac source.
    """
    def __init__(self, n_steps: int = 1, burn_in: int = 100, **kwargs):
        super().__init__(**kwargs)
        assert n_steps >= 1 and burn_in >= 0
        self.n_steps = n_steps
        self.burn_in = burn_in
        self.r = self.grad_term_cost.energy.dist.uniform_weights()
        self.chain = None # (x, E(x), log r(x))
        self._stats = defaultdict(float)

    def _step(self, x, E, logr):
        energy = self.grad_term_cost.energy
        y = self.r.sample([x.shape[0],]).reshape(x.shape).to(x)
        Ey = energy.eval(y).reshape(-1)
        logry = self.r.log_prob(y).reshape(-1)
        log_alpha = - (Ey - E) + logr - logry
        accept = torch.rand_like(E).log() < log_alpha
        self._stats["accepted"] += accept.sum().item()
        self._stats["proposals"] += accept.numel()
        return (
            torch.where(accept[:, None], y, x),
            torch.where(accept, Ey, E),
            torch.where(accept, logry, logr),
        )

    @torch.no_grad()
    def _advance(self, like, n_steps):
        energy = self.grad_term_cost.energy
        if self.chain is None:
            x = self.r.sample([like.shape[0],]).reshape(like.shape).to(like)
            self.chain = (x, energy.eval(x).reshape(-1), self.r.log_prob(x).reshape(-1))
            n_steps += self.burn_in
        for _ in range(n_steps):
            self.chain = self._step(*self.chain)
        return self.chain[0]

    def populate_buffer(
            self,
            x0: torch.Tensor,
            timesteps: torch.Tensor,
            is_asbs_init_stage: bool,
    ):
        x1 = self._advance(x0, self.n_steps).clone()
        adjoint1 = self._compute_adjoint1(x1, is_asbs_init_stage).clone()

        self._check_buffer_sample_shape(x0, x1, adjoint1)
        self.buffer.add({
            "x0": x0.to("cpu"),
            "x1": x1.to("cpu"),
            "adjoint1": adjoint1.to("cpu"),
        })

    def pop_stats(self):
        st, self._stats = self._stats, defaultdict(float)
        return {"imh/accept_rate": st["accepted"] / st["proposals"]} if st["proposals"] else {}


class AdjointVECorrectedBufferMatcher(AdjointVEDiracMatcher):
    """ The buffer is kept as close as possible to the target p1 = e^{-E} / Z by Metropolis-Hastings steps on persistent chains.

        First `warmup` epochs: plain AS (X1 simulated from the model). Then:
          - every `refit_every` epochs, a proposal r is rebuilt from `n_fit` fresh MODEL samples (not chain states, so the
            adaptation does not depend on the chains): EM (`n_components`, tied covariance), centres refined by descent on
            E, the modes no centre explains added (coverage step), and the minima merged into a memory that only grows.
            r = equal-weight mixture of the memory centres with the shared covariance.
          - C = batch-size chains, started from model samples and burnt in for `burn_in` steps, make `steps_per_epoch`
            independent-MH steps per epoch (accept min(1, e^{-[E(y) - E(x)]} r(x) / r(y))), optionally followed by
            `mala_steps` MALA steps (local moves: within-mode diversity). Every kernel leaves p1 invariant.
          - the chain states are the X1 of the buffer, with the adjoint computed at them; X0 is the Dirac source.
        A mode with no centre has no proposals: the chains already there keep the proportion they had at the start.
        The model only enters through the fits of r (it sets which modes are found).

        Alternation with plain AS (on-policy fills): `model_frac` = share of every batch simulated from the model, `stop_epoch`
        = plain AS again after that many fills, `alternate_every` = blocks of that many fills, MH buffer then plain AS.
    """
    def __init__(
        self,
        n_components: int,
        warmup: int = 100,
        refit_every: int = 50,
        n_fit: int = 10000,
        burn_in: int = 30,
        steps_per_epoch: int = 2,
        mala_steps: int = 0,
        mala_h0: float = 0.5,
        inflate: float = 1.0,
        merge_tol_mult: float = 1.0,
        cover_quantile: float = 0.99,
        max_outliers: int = 1000,
        model_frac: float = 0.0,
        stop_epoch: int | None = None,
        alternate_every: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert warmup >= 0 and refit_every >= 1 and steps_per_epoch >= 1 and 0.0 <= model_frac < 1.0
        self.model_frac, self.stop_epoch, self.alternate_every = model_frac, stop_epoch, alternate_every
        self.last_refit_fill = None
        self.n_components, self.warmup, self.refit_every, self.n_fit = n_components, warmup, refit_every, n_fit
        self.burn_in, self.steps_per_epoch, self.n_mala, self.mala_h0 = burn_in, steps_per_epoch, mala_steps, mala_h0
        self.inflate, self.merge_tol_mult = inflate, merge_tol_mult
        self.cover_quantile, self.max_outliers = cover_quantile, max_outliers
        self.fills = 0
        self.r, self.memory, self.chain = None, None, None
        self.ec = EnergyCounter(self.grad_term_cost.energy)
        self._stats = defaultdict(float)

    def _model_samples(self, x0, timesteps, n):
        out = []
        for _ in range(-(-n // x0.shape[0])):
            out.append(sdeint(self.sde, x0, timesteps, only_boundary=True)[1])
        return torch.cat(out)[:n]

    def _refit(self, x0, timesteps):
        samples = self._model_samples(x0, timesteps, self.n_fit)
        d = samples.shape[1]
        _, mu, cov = fit_gmm(samples, self.n_components, cov_type="tied")
        cov_t = cov[0]
        tol = self.merge_tol_mult * (cov_t.trace() / d).sqrt().item()
        centres, _ = refine_centres(self.ec, mu, merge_tol=tol)
        centres, n_added, _, _ = add_uncovered_centres(
            self.ec, samples, centres, cov_t, quantile=self.cover_quantile, max_outliers=self.max_outliers, merge_tol=tol)
        empty = centres.new_zeros(0, d)
        self.memory, n_new, _ = merge_centres(self.ec, empty if self.memory is None else self.memory, centres, tol)
        self.r = TiedMixture(self.memory, cov_t * self.inflate)
        self.cov_t = cov_t
        if self.chain is not None: # the proposal changed: refresh log r of the current states
            x, E, _ = self.chain
            self.chain = (x, E, self.r.logpdf(x))
        self._stats["refits"] += 1
        self._stats["centres_new_in_memory"] += n_new
        self.last_refit_fill = self.fills

    @torch.no_grad()
    def _advance(self, n_steps):
        x, E, logr = self.chain
        C = x.shape[0]
        for _ in range(n_steps):
            y = self.r.sample(C)
            Ey, logry = self.ec.E(y), self.r.logpdf(y)
            a = torch.rand_like(E).log() < - (Ey - E) + logr - logry
            x, E, logr = torch.where(a[:, None], y, x), torch.where(a, Ey, E), torch.where(a, logry, logr)
            self._stats["accepted"] += a.sum().item()
            self._stats["proposals"] += C
        if self.n_mala > 0:
            h = self.mala_h0 * x.shape[1] ** (-1 / 3)
            x, _ = mala_steps(self.ec, x, self.n_mala, h)
            E, logr = self.ec.E(x), self.r.logpdf(x)
        self.chain = (x, E, logr)
        return x

    def _mh_epoch(self):
        """ does this fill take its X1 from the chains? plain AS during the warm-up, after `stop_epoch`, and in the odd
            blocks of `alternate_every` fills """
        if self.fills <= self.warmup:
            return False
        if self.stop_epoch is not None and self.fills > self.stop_epoch:
            return False
        if self.alternate_every is not None and ((self.fills - self.warmup - 1) // self.alternate_every) % 2 == 1:
            return False
        return True

    def populate_buffer(
            self,
            x0: torch.Tensor,
            timesteps: torch.Tensor,
            is_asbs_init_stage: bool,
    ):
        self.fills += 1
        use_mh = self._mh_epoch()
        self._stats["mh_epoch"] += float(use_mh)
        if not use_mh:
            return super().populate_buffer(x0, timesteps, is_asbs_init_stage)
        if self.r is None or self.fills - self.last_refit_fill >= self.refit_every:
            self._refit(x0, timesteps)
        if self.chain is None:
            x = sdeint(self.sde, x0, timesteps, only_boundary=True)[1].clone()
            self.chain = (x, self.ec.E(x), self.r.logpdf(x))
            self._advance(self.burn_in)
        x1 = self._advance(self.steps_per_epoch).clone()
        n_model = int(round(self.model_frac * x1.shape[0]))
        if n_model > 0: # a fraction of the batch is still simulated from the model (on-policy)
            keep = torch.randperm(x1.shape[0], device=x1.device)[: x1.shape[0] - n_model]
            x1 = torch.cat([x1[keep], sdeint(self.sde, x0, timesteps, only_boundary=True)[1][:n_model]])
        adjoint1 = self._compute_adjoint1(x1, is_asbs_init_stage).clone()

        self._check_buffer_sample_shape(x0, x1, adjoint1)
        self.buffer.add({
            "x0": x0.to("cpu"),
            "x1": x1.to("cpu"),
            "adjoint1": adjoint1.to("cpu"),
        })

    def pop_stats(self):
        st, self._stats = self._stats, defaultdict(float)
        out = {"cb/n_E": float(self.ec.n_E), "cb/n_grad": float(self.ec.n_grad), "cb/mh_fraction_of_epochs": st["mh_epoch"]}
        if self.memory is not None:
            out["cb/M_memory"] = float(len(self.memory))
        if st["proposals"]:
            out["cb/accept"] = st["accepted"] / st["proposals"]
        if st["refits"]:
            out["cb/refits"] = st["refits"]
            out["cb/centres_new_in_memory"] = st["centres_new_in_memory"]
        return out


def _nearest(x, centres):
    # index of the nearest centre: (N, D) x (M, D) -> (N,)
    return torch.cdist(x, centres).argmin(-1)


def _kmeans(x, M, init=None, iters=20):
    """ Lloyd's k-means on x (N, D), from `init` or k-means++. An empty cluster restarts at the
        point farthest from its centre.
    """
    if init is None:
        centres = x[torch.randint(len(x), (1,), device=x.device)]
        for _ in range(M - 1):
            d2 = torch.cdist(x, centres).min(-1).values ** 2
            centres = torch.cat([centres, x[torch.multinomial(d2 / d2.sum(), 1)]])
    else:
        centres = init.to(x).clone()

    for _ in range(iters):
        a = _nearest(x, centres)
        counts = torch.bincount(a, minlength=M)
        sums = torch.zeros_like(centres).index_add_(0, a, x)
        centres = torch.where(counts[:, None] > 0, sums / counts.clamp(min=1)[:, None], centres)
        for j in torch.nonzero(counts == 0).flatten().tolist():
            centres[j] = x[(x - centres[a]).norm(dim=-1).argmax()]
    return centres


def _threshold_centres(x, radius, min_count=1, refine_iters=3, generator=None):
    """ Leader (threshold) clustering of x (N, D): no number of clusters, no local minima, no energy evaluation.
        Take an unassigned point (random order), give every unassigned point within `radius` of it to its
        cluster, repeat; then refine: centre = mean of the points closest to it (within `radius`). Clusters with
        fewer than `min_count` points are dropped (noise). One pass per cluster, O(M N D).
        `radius` must lie between the spread of a blob and half the minimum separation of the blobs.
        Returns the centres (M, D) and their point counts (M,).
    """
    N = len(x)
    unassigned = torch.ones(N, dtype=torch.bool, device=x.device)
    order = torch.randperm(N, generator=generator).to(x.device)
    leaders = []
    for i in order.tolist():
        if not unassigned[i]:
            continue
        leaders.append(x[i])
        unassigned &= (x - x[i]).norm(dim=-1) > radius
    centres = torch.stack(leaders)
    for _ in range(refine_iters):
        d = torch.cdist(x, centres)
        near = d.argmin(-1)
        ok = d.min(-1).values <= radius
        counts = torch.bincount(near[ok], minlength=len(centres))
        sums = torch.zeros_like(centres).index_add_(0, near[ok], x[ok])
        centres = torch.where(counts[:, None] > 0, sums / counts.clamp(min=1)[:, None], centres)
    d = torch.cdist(x, centres)
    near, ok = d.argmin(-1), d.min(-1).values <= radius
    counts = torch.bincount(near[ok], minlength=len(centres))
    keep = counts >= min_count
    return centres[keep], counts[keep]


@torch.no_grad()
def _bb_denoise(energy, x, E0, s0, max_steps=8, tol=0.05, back=6):
    """ Per-point descent on E from x (N, D) to the minimum of its basin, with no global state: the limit of a
        converged point depends only on x and E, so a cached result never goes stale.
        Step 1: Armijo backtracking from eta = 1 (one E per try; eta = 1 is the exact step E[m | x] - x on unit-
        covariance Gaussian modes). Next steps: Barzilai-Borwein eta = |dx|^2 / <dx, ds> from the last two
        iterates (exact 1 / curvature on an isotropic quadratic basin), clipped to [eta_1 / 100, 100 eta_1].
        A point has converged when its last step is < `tol` mode widths, width = 1 / sqrt(secant curvature)
        = sqrt(|dx|^2 / <dx, ds>). E0 = E(x) (N,), s0 = ∇E(x) (N, D) are given (stored with the sample).
        Returns the final iterates, converged (N,), the secant curvature (N,), steps taken (N,) and the number
        of gradient / energy evaluations (excluding the stored ∇E(x)).
    """
    N = len(x)
    eta = torch.ones(N, device=x.device, dtype=x.dtype)
    g2 = (s0 ** 2).sum(-1)
    n_E, pending = 0, torch.ones(N, dtype=torch.bool, device=x.device)
    for _ in range(back + 1):
        p = pending.nonzero().flatten()
        if len(p) == 0:
            break
        En = energy.eval(x[p] - eta[p, None] * s0[p]).reshape(-1)
        n_E += len(p)
        ok = En <= E0[p] - 1e-4 * eta[p] * g2[p] + 1e-7 * (1 + E0[p].abs())
        pending[p[ok]] = False
        eta[p[~ok]] *= 0.5
    eta1 = eta.clone()

    xp, sp = x.clone(), s0.clone()
    xc = x - eta[:, None] * s0
    sc = energy.grad_E(xc)
    n_grad, steps = N, torch.ones(N, device=x.device)
    active = torch.ones(N, dtype=torch.bool, device=x.device)
    conv = torch.zeros_like(active)
    lam_out = torch.ones(N, device=x.device, dtype=x.dtype)
    while True:
        dx, ds = xc - xp, sc - sp
        dxx, dxs = (dx * dx).sum(-1), (dx * ds).sum(-1)
        lam = dxs / dxx.clamp(min=1e-30)
        pos = lam > 1e-12
        width_steps = dxx.sqrt() * lam.clamp(min=0).sqrt()
        newly = active & ((pos & (width_steps < tol)) | (dxx < 1e-20))
        lam_out = torch.where(active & pos, lam, lam_out)
        conv |= newly
        active &= ~newly
        if not active.any() or steps.max() >= max_steps:
            break
        a = active.nonzero().flatten()
        eta_a = torch.where(pos[a], 1.0 / lam[a].clamp(min=1e-12), eta[a])
        eta_a = torch.minimum(torch.maximum(eta_a, eta1[a] * 1e-2), eta1[a] * 1e2)
        eta[a] = eta_a
        xn = xc[a] - eta_a[:, None] * sc[a]
        sn = energy.grad_E(xn)
        n_grad += len(a)
        xp[a], sp[a], xc[a], sc[a] = xc[a], sc[a], xn, sn
        steps[a] += 1
    return xc, conv, lam_out, steps, n_grad, n_E


@torch.no_grad()
def _denoise_separable(energy, x, max_steps=12, tol=0.05, h=1e-2, cap=1.0, fallback=0.3):
    """ Per-coordinate Newton descent for a PRODUCT-structured energy E(x) = Σ_j e_j(x_j): the coordinates decouple, so each is denoised on its own
        in 1D (a 32-d Barzilai-Borwein step with quartic curvature ~24 next to Gaussian curvature 1 is ill-conditioned). ∂_j E depends on x_j only, hence
        one gradient at x + h (all coordinates shifted) gives every H_jj = (∂_jE(x + h) - ∂_jE(x)) / h. Newton step -∂_jE / H_jj when H_jj > 0 (capped at
        `cap` per coordinate), else a downhill step of size `fallback` (between the wells H < 0). A point has converged when every |dx_j| sqrt(H_jj) < tol.
        Returns the iterates, converged (N,), the mean curvature (N,) as the BB version does, steps taken, gradient evaluations and 0 (no E evaluation).
    """
    N, D = x.shape
    xc = x.clone()
    sc = energy.grad_E(xc)
    active = torch.ones(N, dtype=torch.bool, device=x.device)
    conv = torch.zeros_like(active)
    lam_out = torch.ones(N, device=x.device, dtype=x.dtype)
    steps = torch.zeros(N, device=x.device)
    n_grad = N
    for _ in range(max_steps):
        a = active.nonzero().flatten()
        if len(a) == 0:
            break
        H = (energy.grad_E(xc[a] + h) - sc[a]) / h
        ok = H > 1e-2
        dx = torch.where(ok, -sc[a] / H.clamp(min=1e-2), -fallback * sc[a].sign()).clamp(-cap, cap)
        xc[a] = xc[a] + dx
        sc[a] = energy.grad_E(xc[a])
        n_grad += 2 * len(a)
        steps[a] += 1
        lam_out[a] = torch.where(ok, H, torch.ones_like(H)).mean(-1)
        done = (ok & (dx.abs() * H.clamp(min=0).sqrt() < tol)).all(-1)
        conv[a[done]] = True
        active[a[done]] = False
    return xc, conv, lam_out, steps, n_grad, 0


@torch.no_grad()
def _barrier_merge(energy, centres, counts, radius, factor=8.0, n_seg=9, tol=0.1):
    """ Candidate centres that are not separated by an energy barrier belong to the same basin: along the segment
        between two centres closer than factor x radius, a barrier is max E(segment) - max(E(a), E(b)) > tol. If
        there is none, the higher-energy centre is a stalled point near a saddle: drop it (its count goes to the
        other). Returns the kept centres, counts and the number merged.
    """
    M = len(centres)
    if M < 2:
        return centres, counts, 0
    d = torch.cdist(centres, centres)
    iu = torch.triu_indices(M, M, 1, device=centres.device)
    near = d[iu[0], iu[1]] <= factor * radius
    i, j = iu[0][near], iu[1][near]
    if len(i) == 0:
        return centres, counts, 0
    t = torch.linspace(0, 1, n_seg + 2, device=centres.device, dtype=centres.dtype)[1:-1]
    seg = centres[i].unsqueeze(1) + t[None, :, None] * (centres[j] - centres[i]).unsqueeze(1)
    E_seg = energy.eval(seg.reshape(-1, centres.shape[1])).reshape(len(i), n_seg)
    E_c = energy.eval(centres).reshape(-1)
    barrier = E_seg.max(1).values - torch.maximum(E_c[i], E_c[j])
    counts, keep, merged = counts.clone(), torch.ones(M, dtype=torch.bool, device=centres.device), 0
    for k in torch.argsort(d[i, j]).tolist():
        a, b = int(i[k]), int(j[k])
        if barrier[k] <= tol and keep[a] and keep[b]:
            drop, stay = (a, b) if E_c[a] > E_c[b] else (b, a)
            keep[drop] = False
            counts[stay] += counts[drop]
            merged += 1
    return centres[keep], counts[keep], merged


@torch.no_grad()
def _converged_centres(energy, pts, conv, lam, radius_scale=1.0, min_count=1, generator=None):
    """ Centres from cached denoised points: only the converged ones are clustered (leader clustering with
        radius = radius_scale x median width, width = 1 / sqrt(secant curvature)), then centres without an
        energy barrier between them are merged. Returns the centres, counts and a dict of diagnostics.
    """
    n = len(pts)
    info = {"conv_frac": float(conv.float().mean()) if n else 0.0}
    if conv.sum() == 0:
        return pts[:0], torch.zeros(0, dtype=torch.long, device=pts.device), info
    pc = pts[conv]
    width = (1.0 / lam[conv].clamp(min=1e-12).sqrt()).median()
    radius = radius_scale * float(width)
    centres, counts = _threshold_centres(pc, radius, min_count=min_count, generator=generator)
    n_raw = len(centres)
    centres, counts, merged = _barrier_merge(energy, centres, counts, radius)
    near = torch.cdist(pc, centres).min(-1).values <= radius
    info.update({"n_raw": n_raw, "n_merged": merged, "radius": radius, "width": float(width),
                 "explained": float(near.float().sum() / n)})
    return centres, counts, info


@torch.no_grad()
def _laplace_factors(energy, centres, h=1e-2, eps=1e-3):
    """ Laplace covariances of the basins at `centres` (K, D): Σ_k = H_k^{-1}, H_k = ∇²E(μ_k) by central differences of
        ∇E (2 D gradients per centre), factor L_k = V diag(λ^{-1/2}) with H = V diag(λ) V^T (so Σ_k = L_k L_k^T; any
        factor of Σ_k defines a valid affine map between modes). Eigenvalues are clipped at `eps`.
        Returns L (K, D, D), L^{-1}, log|det L| (K,) and the condition numbers of the Σ_k.
    """
    K, D = centres.shape
    eye = torch.eye(D, device=centres.device, dtype=centres.dtype)
    pts = torch.stack([centres[:, None, :] + h * eye, centres[:, None, :] - h * eye], dim=2)  # (K, D, 2, D)
    g = energy.grad_E(pts.reshape(-1, D)).reshape(K, D, 2, D)
    H = (g[:, :, 0, :] - g[:, :, 1, :]) / (2 * h)                                              # (K, D, D): H[k, j, i] = ∂_j ∂_i E
    H = 0.5 * (H + H.transpose(-1, -2))
    lam, V = torch.linalg.eigh(H)
    lam = lam.clamp(min=eps)
    L = V * lam.rsqrt().unsqueeze(-2)                                                          # V diag(λ^{-1/2})
    return L, torch.linalg.inv(L), torch.linalg.slogdet(L)[1], lam.max(-1).values / lam.min(-1).values


@torch.no_grad()
def _product_wells(energy, xd, conv, lam, radius_scale=1.0, h=1e-2, n_seg=33, tol=0.1, eps=1e-3):
    """ Wells of a PRODUCT-structured target, one coordinate at a time. From the converged basin minima `xd` (N, D) of the buffer samples, each
        coordinate j is clustered in 1D (leader clustering, radius = radius_scale x median width); a coordinate with exactly two clusters
        separated by an energy barrier along j (all the other coordinates at a converged point, the target being a product) is a double well:
        centres μ_L < μ_R, split = the top of the barrier, Laplace widths σ = H_jj^{-1/2} at the centres (central differences of ∂_j E).
        Returns idx (J,), mu (J, 2), sigma (J, 2), split (J,) and the number of coordinates with one cluster / with none separated.
    """
    N, D = xd.shape
    out = dict(idx=xd.new_zeros(0, dtype=torch.long), mu=xd.new_zeros(0, 2), sigma=xd.new_zeros(0, 2), split=xd.new_zeros(0))
    if conv.sum() == 0:
        return out, 0
    pc = xd[conv]
    width = (1.0 / lam[conv].clamp(min=1e-12).sqrt()).median()
    radius = float(radius_scale * width)
    base = pc[0].clone()
    cand, cents = [], []
    for j in range(D):
        c, _ = _threshold_centres(pc[:, j:j + 1], radius, min_count=1)
        if len(c) > 2:
            # converged points can sit on a saddle (zero gradient, barrier top): keep the centres with positive curvature along j
            q = base.expand(len(c), 2, D).clone()
            q[:, 0, j] = c.reshape(-1) + h
            q[:, 1, j] = c.reshape(-1) - h
            gq = energy.grad_E(q.reshape(-1, D)).reshape(len(c), 2, D)[:, :, j]
            c = c[((gq[:, 0] - gq[:, 1]) / (2 * h)) > eps]
        if len(c) == 2:
            cand.append(j)
            cents.append(c.reshape(-1).sort().values)
    if not cand:
        return out, 0
    cents = torch.stack(cents)                                            # (C, 2)
    C = len(cand)
    t = torch.linspace(0, 1, n_seg, device=xd.device, dtype=xd.dtype)
    path = cents[:, :1] + t * (cents[:, 1:] - cents[:, :1])               # (C, n_seg) coordinate values between the two centres
    pts = base.expand(C, n_seg, D).clone()
    jj = torch.tensor(cand, device=xd.device)
    pts[torch.arange(C), :, jj] = path
    E = energy.eval(pts.reshape(-1, D)).reshape(C, n_seg)
    barrier = E.max(1).values - torch.maximum(E[:, 0], E[:, -1])
    keep = barrier > tol
    if not keep.any():
        return out, 0
    split = path[torch.arange(C), E.argmax(1)]
    pts_h = base.expand(C, 2, 2, D).clone()                                # (C, well, +-, D)
    for sgn, k in ((1.0, 0), (-1.0, 1)):
        pts_h[torch.arange(C), :, k, jj] = cents + sgn * h
    g = energy.grad_E(pts_h.reshape(-1, D)).reshape(C, 2, 2, D)
    gj = g[torch.arange(C), :, :, jj]                                      # (C, well, +-)
    H = (gj[:, :, 0] - gj[:, :, 1]) / (2 * h)
    sigma = H.clamp(min=eps).rsqrt()
    sel = keep.nonzero().flatten()
    out = dict(idx=jj[sel], mu=cents[sel], sigma=sigma[sel], split=split[sel])
    return out, int(sel.numel())


def _merge_wells(bank, new):
    """ PERSISTENT well bank for product-structured targets: coordinates fitted in `new` replace their entry, coordinates absent from `new` (e.g. a well
        that emptied, so the buffer no longer shows two clusters) keep the entry of `bank`: a discovered well is never deleted, so the translates into
        it stay available and the coordinate cannot become absorbing. The bank depends on past samples only: the zero mean stays exact. """
    if bank is None or len(bank["idx"]) == 0:
        return new
    keep = torch.tensor([int(j) not in set(new["idx"].tolist()) for j in bank["idx"]], dtype=torch.bool, device=bank["idx"].device)
    cat = {k: torch.cat([new[k].to(bank[k]), bank[k][keep]]) for k in ("idx", "mu", "sigma", "split")}
    order = cat["idx"].argsort()
    return {k: v[order] for k, v in cat.items()}


class AdjointVEMHMatcher(AdjointVEDiracMatcher):
    """ AdjointVEMatcher with a Metropolis-Hastings control variate added to the target:

            target = - ∇g(X1) + γ h / ν_{1|t},   h = Σ_{k ≠ c} 1/(M - 1) α_t(X1, Y_k) (Y_k - X1)

        Y_k = X1 + m_k - m_c moves X1 from its cluster c to cluster k (centres m_1..m_M) and is kept
        only if Y_k falls in cluster k, so that the proposal is symmetric. α_t = min(1, π(Y | Xt) / π(X1 | Xt))
        with π(x1 | xt) ∝ p1(x1) p^base(xt | x1) the posterior of the target:

            log π(y | xt) / π(x | xt) = - (E(y) - E(x)) - (κ_t |y - x|² - 2 <r, y - x>) / (2 ν_{1|t}),
            r = xt - (1 - κ_t) X0 - κ_t x,   ν_{1|t} = ν_1 - ν_t.

        By detailed balance E_π[h | Xt] = 0 for any centres (Dellaportas & Kontoyiannis, 2012): u* stays
        the fixed point. α uses energy values, so under the model posterior h is the net flux of an MH
        chain between clusters, from over- to under-weighted modes: with equal-shape modes and t → 0,
        E[h] = Σ_k c_k m_k, c = - G^T (w - ŵ), G the generator with rates min(1, w_k / w_j) / (M - 1).
        As t → 1 every jump is rejected, so the term acts at small t. The controller here is u / σ(t).

        Centres come from k-means on the buffer before each new batch, and each sample's proposals and
        energies are drawn and frozen when it enters the buffer, so its proposal kernel does not depend
        on it. k-means runs on x1 + s ∇log p1(x1), one gradient step toward the mode (the mode centre
        for unit-variance GMM modes, s = 1), with ∇log p1 = ∇log p^base_1 - ∇g read off the buffer: on
        raw points it misses the light modes once d is comparable to the squared mode separation. `rao_blackwell` keeps all M - 1 jumps (M - 1 energy values per sample), else one at random.

        γ: the mode-level flux is - G^T e, e = w - ŵ, with -G^T in [0, 2]; but jumps are only accepted at
        small t, and one outer iteration solved exactly moves the masses by about γ/5 e for d = K = 12
        (γ/1.7 e for d = 6, K = 4), against 0.08 e for AS itself (tests/test_mh_matcher.py): γ = 2
        corrects 40% of the weight error per iteration without overshoot.
    """
    T_EDGES = (0.0, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 1.0)

    def __init__(
        self,
        gamma: float = 2.0,
        n_clusters: int = 2,
        rao_blackwell: bool = False,
        kmeans_iters: int = 20,
        denoise_step: float = 1.0,
        stats_every: int = 10,
        oracle_centres: bool = False,
        centre_method: str = "kmeans",
        denoise_steps: int = 12,
        denoise_tol: float = 0.05,
        radius_scale: float = 1.0,
        max_centres: int = 128,
        orbit: str = "translation",
        orbit_size: int | None = None,
        gamma_anneal_frac: float = 0.0,
        product_subset: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert gamma >= 0 and n_clusters >= 2
        assert 0.0 <= gamma_anneal_frac <= 1.0
        self.gamma_anneal_frac = gamma_anneal_frac # γ decays linearly to 0 over the last `frac` of the epochs (the term has zero mean at the fixed point)
        self.gamma_scale = 1.0
        assert orbit in ("translation", "affine", "product")
        self.product_subset = product_subset # product orbit: number of double-well coordinates toggled per sample (2^s - 1 translates)
        self.wells = None # product orbit: dict(idx (J,), mu (J, 2), sigma (J, 2), split (J,)), refit from the cached basin minima
        assert orbit_size is None or orbit_size >= 2
        self.orbit = orbit # translation: Y_k = x + m_k - m_c; affine: Y_k = μ_k + L_k L_c^{-1}(x - μ_c) (needs covariances)
        self.orbit_size = orbit_size # None: every other cluster; s: a uniform random subset of s - 1 of them, per sample
        self.factors = self.prec = self.logdet = None # (K, D, D), (K, D, D), (K,): L_k, L_k^{-1}, log|det L_k|
        if oracle_centres:
            centre_method = "oracle"
        if orbit == "affine":
            assert centre_method in ("threshold", "oracle"), "the affine orbit needs covariances: threshold or oracle centres"
        assert centre_method in ("kmeans", "threshold", "oracle", "wells")
        assert (orbit == "product") == (centre_method == "wells"), "the product orbit goes with centre_method = wells (and only it)"
        self.centre_method = centre_method
        self.denoise_steps = denoise_steps
        self.max_centres = max_centres # threshold clustering: capacity (padded translates are invalid: no energy call)
        self.denoise_tol = denoise_tol
        self.radius_scale = radius_scale
        self._centre_info = {}
        self.gamma = gamma
        self.n_clusters = n_clusters
        self.rao_blackwell = rao_blackwell
        self.kmeans_iters = kmeans_iters
        self.denoise_step = denoise_step
        self.oracle_centres = centre_method == "oracle" # GMM only: centres = the true means (upper bound on the clustering)
        self._mode_batch = None
        self.stats_every = stats_every # diagnostics are many small kernels: not every step
        self.centres = None
        self._stats = defaultdict(float)
        self._step = 0

    def set_progress(self, epoch: int, num_epochs: int) -> None:
        """ called by train.py at every epoch: γ_eff = γ x clip((1 - epoch / num_epochs) / gamma_anneal_frac, 0, 1) """
        f = self.gamma_anneal_frac
        self.gamma_scale = 1.0 if f <= 0 else float(min(max((1.0 - epoch / num_epochs) / f, 0.0), 1.0))

    @property
    def gamma_eff(self) -> float:
        return self.gamma * self.gamma_scale

    @property
    def capacity(self):
        """ number of clusters the stored translates are padded to """
        return self.max_centres if self.centre_method == "threshold" else self.n_clusters

    def _assign(self, x):
        """ cluster of each point (N, D) -> (N,), the SAME rule for x1 and for every translate: nearest centre (translation
            orbit), or the Gaussian component with the largest likelihood (affine orbit)
        """
        m = self.centres.to(x)
        if self.orbit == "translation":
            return _nearest(x, m)
        z = torch.einsum("kde,nke->nkd", self.prec.to(x), x.unsqueeze(1) - m)
        return (0.5 * z.pow(2).sum(-1) + self.logdet.to(x)).argmin(-1)

    def _orbit(self, x1):
        """ translates Y_k of x1 (N, D) toward other clusters: jumps Y_k - x1 (N, J, D), valid = 1[Y_k in cluster k] (N, J) and,
            for the affine orbit, log|det L_k| - log|det L_c| (N, J). J = M - 1 (all other clusters, padded to the capacity),
            1 (one random cluster) or orbit_size - 1 (a uniform random subset, drawn independently of the position of x1,
            so that P(S | x) = P(S | Y_k) for k in S: the sub-orbit is exact too). No centres yet: no valid jump.
        """
        if self.orbit == "product":
            return self._orbit_product(x1)
        (N, D), Mmax = x1.shape, self.capacity
        J = self.orbit_size - 1 if self.orbit_size else (Mmax - 1 if self.rao_blackwell else 1)
        affine = self.orbit == "affine"
        if self.centres is None or len(self.centres) < 2:
            return x1.new_zeros(N, J, D), x1.new_zeros(N, J), (x1.new_zeros(N, J) if affine else None)

        m = self.centres.to(x1)
        M = len(m) # = n_clusters for k-means / oracle; <= the capacity for threshold clustering
        c = self._assign(x1)
        others = torch.arange(M, device=x1.device).expand(N, M)
        others = others[others != c[:, None]].view(N, M - 1)
        if self.orbit_size:
            keep = min(self.orbit_size - 1, M - 1)
            others = others.gather(1, torch.rand(N, M - 1, device=x1.device).argsort(1)[:, :keep])
        elif not self.rao_blackwell:
            others = others.gather(1, torch.randint(M - 1, (N, 1), device=x1.device))

        if affine:
            u = torch.einsum("nde,ne->nd", self.prec.to(x1)[c], x1 - m[c]) # whitened coordinate, the same in every cluster
            Y = m[others] + torch.einsum("njde,ne->njd", self.factors.to(x1)[others], u)
            jumps = Y - x1.unsqueeze(1)
            logdet = self.logdet.to(x1)[others] - self.logdet.to(x1)[c].unsqueeze(1)
        else:
            jumps, logdet = m[others] - m[c].unsqueeze(1), None
        valid = (self._assign((x1.unsqueeze(1) + jumps).reshape(-1, D)).view(N, -1) == others).to(x1)
        jumps = jumps * valid.unsqueeze(-1)
        if affine:
            logdet = logdet * valid
        pad = J - jumps.shape[1]
        if pad > 0: # fewer clusters than the capacity: pad with invalid translates
            jumps = torch.cat([jumps, x1.new_zeros(N, pad, D)], 1)
            valid = torch.cat([valid, x1.new_zeros(N, pad)], 1)
            if affine:
                logdet = torch.cat([logdet, x1.new_zeros(N, pad)], 1)
        return jumps, valid, logdet

    def _orbit_product(self, x1, subset=None):
        """ ORBIT OF THE SUBGROUP Z_2^S for a product-structured target. Per sample, a subset S of s double-well coordinates is drawn uniformly
            and INDEPENDENTLY of the position of x1; the orbit is the 2^s well combinations on S. Coordinate j in S has two wells (L, R), each with
            a Laplace centre μ and width σ; with u = (x_j - μ_w) / σ_w the whitened coordinate, the toggle sends x_j to μ_w' + σ_w' u (w' the other
            well), with Jacobian σ_w' / σ_w. These maps form a group Z_2 per coordinate: the orbit is the same set from every member, whatever
            the centres, widths and split, so E_π[h | Xt] = 0 stays exact; the Barker weights are π(Y) Π_{j in S} σ_{w'_j} (logdet stored).
            Returns the 2^s - 1 translates as jumps (N, 2^s - 1, D), valid (N, 2^s - 1) and logdet (N, 2^s - 1). `subset` (N, s) fixes S (tests).
        """
        N, D = x1.shape
        s = self.product_subset
        ncomb = 2 ** s - 1
        w = self.wells
        if w is None or len(w["idx"]) == 0:
            return x1.new_zeros(N, ncomb, D), x1.new_zeros(N, ncomb), x1.new_zeros(N, ncomb)
        J = len(w["idx"])
        se = min(s, J)
        if subset is None:
            subset = torch.rand(N, J, device=x1.device).argsort(1)[:, :se]       # indices into the J double-well coordinates, uniform
        se = subset.shape[1]
        idx = w["idx"].to(x1.device)[subset]                                    # (N, se) coordinates
        mu, sigma, split = w["mu"].to(x1)[subset], w["sigma"].to(x1)[subset], w["split"].to(x1)[subset]   # (N, se, 2), (N, se, 2), (N, se)
        xs = x1.gather(1, idx)
        side = (xs >= split).long()                                             # current well of every coordinate of S
        mu_own, sig_own = mu.gather(2, side[..., None]).squeeze(-1), sigma.gather(2, side[..., None]).squeeze(-1)
        u = (xs - mu_own) / sig_own
        bits = ((torch.arange(1, 2 ** se, device=x1.device)[:, None] >> torch.arange(se, device=x1.device)) & 1).bool()   # (2^se - 1, se) toggled?
        side_new = side[:, None, :] ^ bits[None].long()                         # (N, 2^se - 1, se)
        mu_new = torch.where(side_new.bool(), mu[:, None, :, 1], mu[:, None, :, 0])
        sig_new = torch.where(side_new.bool(), sigma[:, None, :, 1], sigma[:, None, :, 0])
        xnew = mu_new + sig_new * u[:, None, :]
        valid = ((xnew >= split[:, None, :]) == side_new.bool()).all(-1)
        Y = x1[:, None, :].repeat(1, 2 ** se - 1, 1)
        Y.scatter_(2, idx[:, None, :].expand(N, 2 ** se - 1, se), xnew)
        logdet = (torch.log(sig_new) - torch.log(sig_own)[:, None, :]).sum(-1)
        valid = valid.to(x1)
        jumps, logdet = (Y - x1[:, None, :]) * valid[..., None], logdet * valid
        pad = ncomb - jumps.shape[1]
        if pad > 0: # fewer double-well coordinates than s
            jumps = torch.cat([jumps, x1.new_zeros(N, pad, D)], 1)
            valid = torch.cat([valid, x1.new_zeros(N, pad)], 1)
            logdet = torch.cat([logdet, x1.new_zeros(N, pad)], 1)
        return jumps, valid, logdet

    def propose(self, x1):
        """ jumps (N, J, D) and valid (N, J) of the orbit of x1 """
        jumps, valid, _ = self._orbit(x1)
        return jumps, valid

    def _propose_for_buffer(self, x1):
        jumps, valid, logdet = self._orbit(x1)
        return jumps, valid, ({} if logdet is None else {"logdet": logdet})

    def populate_buffer(
            self,
            x0: torch.Tensor,
            timesteps: torch.Tensor,
            is_asbs_init_stage: bool,
    ):
        (x0, x1) = sdeint(
            self.sde,
            x0,
            timesteps,
            only_boundary=True,
        )
        adjoint1 = self._compute_adjoint1(x1, is_asbs_init_stage).clone()
        self._check_buffer_sample_shape(x0, x1, adjoint1)

        if self.oracle_centres and self.centres is None:
            means = self.grad_term_cost.energy.dist.means
            assert len(means) == self.n_clusters, "oracle centres need n_clusters = K"
            self.centres = means.to(x1).clone()
            if self.orbit == "affine": # the true covariances of the mixture
                dist = self.grad_term_cost.energy.dist
                self.factors, self.prec, self.logdet = dist.L.to(x1), dist.P.to(x1), dist.logdet.to(x1)

        # centres from the buffer before this batch, so they do not depend on its samples
        if len(self.buffer) > 0 and self.centre_method == "threshold":
            self._refit_threshold_centres(x1)
        if len(self.buffer) > 0 and self.centre_method == "wells":
            self._refit_wells(x1)
        if len(self.buffer) > 0 and self.centre_method == "kmeans":
            n = self.buffer.buffer_size
            buffer_x1 = torch.cat(self.buffer.batches["x1"])[-n:].to(x1)
            buffer_adjoint1 = torch.cat(self.buffer.batches["adjoint1"])[-n:].to(x1)
            score1 = self.grad_term_cost.base_score(buffer_x1) - buffer_adjoint1 # ∇log p1
            points = buffer_x1 + self.denoise_step * score1
            self.centres = _kmeans(points, self.n_clusters, self.centres, self.kmeans_iters)

        jumps, valid, extra = self._propose_for_buffer(x1)
        energy = self.grad_term_cost.energy
        with torch.no_grad():
            E1 = energy.eval(x1).reshape(-1)
            EY = torch.zeros_like(valid)
            mask = valid.bool()
            if mask.any():
                EY[mask] = energy.eval((x1.unsqueeze(1) + jumps)[mask]).reshape(-1) # GMM1D returns (N, 1)
        self._stats["evals"] += mask.sum()
        self._stats["samples"] += x1.shape[0]

        data = {
            "x0": x0.to("cpu"),
            "x1": x1.to("cpu"),
            "adjoint1": adjoint1.to("cpu"),
            "E1": E1.reshape(-1, 1).to("cpu"),
            "jumps": jumps.to("cpu"),
            "EY": EY.to("cpu"),
            "valid": valid.to("cpu"),
            **{k: v.to("cpu") for k, v in extra.items()},
        }
        # GMM diagnostic: accepted jumps that stay in the same true mode do nothing for the weights
        dist = getattr(energy, "dist", None)
        if hasattr(dist, "assign"):
            m1 = dist.assign(x1)
            Je = 0 if self.centres is None else (len(self.centres) - 1 if self.rao_blackwell and not self.orbit_size else jumps.shape[1])
            mY = m1.unsqueeze(1).repeat(1, jumps.shape[1]) # padded / invalid translates: same mode, weight 0
            if Je > 0:
                mY[:, :Je] = dist.assign(x1.unsqueeze(1) + jumps[:, :Je])
            same = mY == m1.unsqueeze(1)
            data["same_mode"] = (same.to(x1) * valid).to("cpu")
            # true mode of x1 and of every translate, for the per-mode flux diagnostics of the Barker matcher
            data["mode1"] = m1.to("cpu")
            data["modeY"] = mY.to("cpu")
        if self.centre_method in ("threshold", "wells"):
            # basin minimum of each new sample, computed once: it depends only on x and E, never on the model.
            # ∇E(x1) is stored with the sample (adjoint1 = ∇E + ∇log p^base_1), so the first gradient is free
            assert not is_asbs_init_stage
            s0 = adjoint1 - self.grad_term_cost.base_score(x1)
            if self.centre_method == "wells":
                xd, conv, lam, steps, n_grad, n_E = _denoise_separable(energy, x1, self.denoise_steps, self.denoise_tol)
            else:
                xd, conv, lam, steps, n_grad, n_E = _bb_denoise(energy, x1, E1, s0, self.denoise_steps, self.denoise_tol)
            data.update({"xd": xd.to("cpu"), "conv": conv.to("cpu"), "lam": lam.to("cpu")})
            st = self._stats
            st["dn_n"] += x1.shape[0]
            st["dn_grad"] += n_grad
            st["dn_E"] += n_E
            st["dn_steps"] += steps.sum()
            st["dn_conv"] += conv.sum()
        self.buffer.add(data)

    def init_wells(self, x1):
        """ well bank fitted on a given set of endpoints (the initial buffer of a correction run), so every coordinate starts with its wells """
        assert self.centre_method == "wells"
        energy = self.grad_term_cost.energy
        xd, conv, lam, *_ = _denoise_separable(energy, x1.to(self.grad_term_cost.mu1), self.denoise_steps, self.denoise_tol)
        wells, n_active = _product_wells(energy, xd, conv, lam, self.radius_scale)
        if n_active > 0:
            self.wells = _merge_wells(self.wells, wells)
        return n_active

    def _refit_wells(self, like):
        """ per-coordinate double wells from the cached converged basin minima of the buffer (product-structured target) """
        n = self.buffer.buffer_size
        xd = torch.cat(self.buffer.batches["xd"])[-n:].to(like)
        conv = torch.cat(self.buffer.batches["conv"])[-n:].to(like.device)
        lam = torch.cat(self.buffer.batches["lam"])[-n:].to(like)
        wells, n_active = _product_wells(self.grad_term_cost.energy, xd, conv, lam, self.radius_scale)
        self._centre_info = {"n_active": n_active, "conv_frac": float(conv.float().mean())}
        if n_active > 0:
            self.wells = _merge_wells(self.wells, wells)
            self._centre_info["n_fit"] = n_active
            self._centre_info["n_active"] = len(self.wells["idx"])           # the persistent bank
            self._centre_info["sigma_min"] = float(self.wells["sigma"].min())
            self._centre_info["sigma_max"] = float(self.wells["sigma"].max())

    def _refit_threshold_centres(self, like):
        """ centres from the cached converged points of the buffer, before the new batch is inserted """
        n = self.buffer.buffer_size
        xd = torch.cat(self.buffer.batches["xd"])[-n:].to(like)
        conv = torch.cat(self.buffer.batches["conv"])[-n:].to(like.device)
        lam = torch.cat(self.buffer.batches["lam"])[-n:].to(like)
        centres, counts, info = _converged_centres(self.grad_term_cost.energy, xd, conv, lam, self.radius_scale)
        info["n_found"] = len(centres)
        info["dropped"] = 0
        if len(centres) > self.max_centres:
            # over capacity: the most populated are kept, i.e. the LIGHT modes are the ones dropped
            info["dropped"] = len(centres) - self.max_centres
            warnings.warn(f"threshold centres: {len(centres)} found > max_centres = {self.max_centres}: "
                          f"{info['dropped']} least populated dropped (light modes); raise max_centres")
            centres = centres[torch.topk(counts, self.max_centres).indices]
        info["n"] = len(centres)
        if len(centres) >= 2:
            self.centres = centres
            if self.orbit == "affine": # Laplace covariances at the basin minima (uses 2 D gradients per centre)
                self.factors, self.prec, self.logdet, cond = _laplace_factors(self.grad_term_cost.energy, centres)
                info["cond_median"] = float(cond.median())
                info["cond_max"] = float(cond.max())
        self._centre_info = info

    def mh_term(self, t, x0, xt, x1, E1, jumps, EY, valid):
        """ γ h / ν_{1|t} (N, D) and the acceptance α (N, J) """
        nut, nu1 = self._nu(t), self._nu(torch.ones_like(t))
        kappa, nu1t = nut / nu1, (nu1 - nut).clamp(min=1e-12)

        r = xt - (1 - kappa) * x0 - kappa * x1
        gauss = (kappa * jumps.pow(2).sum(-1) - 2 * (jumps * r.unsqueeze(1)).sum(-1)) / (2 * nu1t)
        alpha = (- (EY - E1) - gauss).clamp(max=0).exp() * valid

        h = (alpha.unsqueeze(-1) * jumps).mean(1)
        return self.gamma_eff * h / nu1t, alpha

    def prepare_target(self, data, device):
        (t, xt), target = super().prepare_target(data, device)

        x0, x1 = data["x0"].to(device), data["x1"].to(device)
        E1, EY = data["E1"].to(device), data["EY"].to(device)
        jumps, valid = data["jumps"].to(device), data["valid"].to(device)

        extra = {"logdet": data["logdet"].to(device)} if "logdet" in data else {}
        term, alpha = self.mh_term(t, x0, xt, x1, E1, jumps, EY, valid, **extra)
        self._step += 1
        if self._step % self.stats_every == 0:
            same = data["same_mode"].to(device) if "same_mode" in data else None
            self._mode_batch = (data["mode1"].to(device), data["modeY"].to(device)) if "mode1" in data else None
            self._update_stats(t, jumps, valid, alpha, term, - target, same)

        self._check_target_shape(t, xt, term)
        return (t, xt), target + term

    @torch.no_grad()
    def _update_stats(self, t, jumps, valid, alpha, term, adjoint1, same):
        st = self._stats
        t = t.squeeze(-1)
        nb = len(self.T_EDGES) - 1
        b = torch.bucketize(t, t.new_tensor(self.T_EDGES[1:-1]), right=True)
        b = b.unsqueeze(1).expand_as(alpha).reshape(-1)

        # acceptance of valid jumps per t bucket, and 2Φ(-√s/2), s = κ|Δ|² / ν_{1|t}: the acceptance
        # between equal-energy points when r ~ N(0, ν_1 κ (1 - κ)), i.e. without the energy term
        nut, nu1 = self._nu(t), self._nu(torch.ones_like(t))
        s = (nut / nu1).unsqueeze(1) * jumps.pow(2).sum(-1) / (nu1 - nut).clamp(min=1e-12).unsqueeze(1)
        pred = 2 * torch.special.ndtr(- s.sqrt() / 2) * valid
        for name, v in (("n_valid", valid), ("acc", alpha), ("acc_pred", pred)):
            st[name] = st[name] + alpha.new_zeros(nb).index_add_(0, b, v.reshape(-1))

        # proposals among the real translates only (the columns beyond the centres are padding)
        n_cols = alpha.shape[1] if (self.centres is None or not self.rao_blackwell or self.orbit_size) else min(len(self.centres) - 1, alpha.shape[1])
        st["proposals"] += alpha.shape[0] * n_cols
        st["valid"] += valid.sum()
        if same is not None:
            st["acc_same_mode"] += (alpha * same).sum()
            st["acc_total"] += alpha.sum()
        active = t < 0.1
        st["term_norm"] += term.norm(dim=-1)[active].sum()
        st["grad_norm"] += adjoint1.norm(dim=-1)[active].sum()

    def pop_stats(self):
        """ epoch diagnostics, reset after each call """
        st, self._stats = self._stats, defaultdict(float)
        out = {}
        if st["samples"]:
            out["mh/energy_evals_per_sample"] = float(st["evals"] / st["samples"])
        if st["proposals"]:
            out["mh/valid_frac"] = float(st["valid"] / st["proposals"])
        if st["acc_total"]:
            out["mh/wasted_frac"] = float(st["acc_same_mode"] / st["acc_total"])
        if st["grad_norm"]:
            out["mh/term_vs_grad_t<0.1"] = float(st["term_norm"] / st["grad_norm"])
        if torch.is_tensor(st["n_valid"]):
            n, acc, pred = (st[k].tolist() for k in ("n_valid", "acc", "acc_pred"))
            for i in range(len(n)):
                if n[i] > 0:
                    bucket = f"t{self.T_EDGES[i]:g}-{self.T_EDGES[i + 1]:g}"
                    out[f"mh/acc/{bucket}"] = acc[i] / n[i]
                    out[f"mh/acc_pred/{bucket}"] = pred[i] / n[i]
        if st["dn_n"]: # threshold centres: cost of the denoising (per sample, beyond the stored first gradient) and quality
            out["denoise/grad_evals_per_sample"] = float(st["dn_grad"] / st["dn_n"])
            out["denoise/E_evals_per_sample"] = float(st["dn_E"] / st["dn_n"])
            out["denoise/steps_mean"] = float(st["dn_steps"] / st["dn_n"])
            out["denoise/converged_frac_new"] = float(st["dn_conv"] / st["dn_n"])
        for k, v in self._centre_info.items():
            out[f"centres/{k}"] = v
        out["barker/gamma_eff"] = self.gamma_eff
        return out


class AdjointVEBarkerMatcher(AdjointVEMHMatcher):
    """ AdjointVEMHMatcher where the M - 1 translates of X1 are aggregated by Barker resampling over the
        whole translation orbit instead of M - 1 Metropolis acceptances:

            target = - ∇g(X1) + γ h / ν_{1|t},   h = Σ_{k ≠ c} ω_k (Y_k - X1),   ω = softmax over the orbit

        {X1, Y_1, .., Y_{M-1}} of log π(. | Xt), so ω_own = π(X1 | Xt) / Σ_orbit π, ω_k = π(Y_k | Xt) / Σ_orbit π
        (same log-ratio as the MH term, with the own point at log-weight 0, invalid translates at -inf).

        Zero mean: the orbit O = {X1 + m_j - m_c : Y_j in cluster j} is the same set seen from any of its
        members (same centres, same nearest-centre rule on raw points), so E_π[h | O] = Σ_x p_O(x) Σ_y p_O(y)
        (y - x) = 0 exactly, for any centres (also M = 2K or random centres): u* stays the fixed point.

        Mode level (small t, equal-shape separated modes, M = K, centres = means): ω_k = w_k, the chain moves
        X1 to an independent draw of the mode, transition matrix 1 w^T, and E_{p1^u}[h] = Σ_k (w_k - ŵ_k) m_k:
        the flux is the weight error e, with a flat spectrum (MH: - G^T e, eigenvalues spread, wider with K).
        γ = 1 is the flux scale, not the scale of the final weights (the term only acts at small t).

        All M - 1 translates and their energies are stored per sample (rao_blackwell is always true): M - 1 energy
        evaluations per sample, against < 1 for MH with one jump.
    """
    def __init__(self, gamma: float = 1.0, rao_blackwell: bool = True, **kwargs):
        assert rao_blackwell, "the orbit needs all M - 1 translates"
        super().__init__(gamma=gamma, rao_blackwell=True, **kwargs)

    def mh_term(self, t, x0, xt, x1, E1, jumps, EY, valid, logdet=None):
        """ γ h / ν_{1|t} (N, D) and the Barker weights ω_k of the translates (N, J). `logdet` (affine orbit) = log|det L_k| -
            log|det L_c|: the Jacobian of the map x -> Y_k, so that the orbit is weighted by π(Y_k) |det L_k| (the density of
            the member k of the orbit in the common whitened coordinate)
        """
        nut, nu1 = self._nu(t), self._nu(torch.ones_like(t))
        kappa, nu1t = nut / nu1, (nu1 - nut).clamp(min=1e-12)

        r = xt - (1 - kappa) * x0 - kappa * x1
        gauss = (kappa * jumps.pow(2).sum(-1) - 2 * (jumps * r.unsqueeze(1)).sum(-1)) / (2 * nu1t)
        logr = - (EY - E1) - gauss # log π(Y_k | xt) / π(x1 | xt)
        if logdet is not None:
            logr = logr + logdet
        logr = logr.masked_fill(valid < 0.5, float("-inf"))
        logits = torch.cat([torch.zeros_like(logr[:, :1]), logr], 1) # own point: log-weight 0
        w = logits.softmax(-1)[:, 1:]

        h = (w.unsqueeze(-1) * jumps).sum(1)
        return self.gamma_eff * h / nu1t, w

    @torch.no_grad()
    def _update_stats(self, t, jumps, valid, alpha, term, adjoint1, same):
        super()._update_stats(t, jumps, valid, alpha, term, adjoint1, same) # alpha = ω here
        st = self._stats
        nb = len(self.T_EDGES) - 1
        b = torch.bucketize(t.squeeze(-1), t.new_tensor(self.T_EDGES[1:-1]), right=True)
        own = 1 - alpha.sum(-1)
        ess = 1 / (own ** 2 + alpha.pow(2).sum(-1)) # over the whole orbit, own point included
        for name, v in (("n", torch.ones_like(own)), ("own", own), ("ess", ess)):
            st[f"barker_{name}"] = st[f"barker_{name}"] + alpha.new_zeros(nb).index_add_(0, b, v)

        if self._mode_batch is not None: # GMM: mass moved between true modes by the translates, t < 0.1
            (m1, mY), K = self._mode_batch, len(self.grad_term_cost.energy.dist.weights)
            active = t.squeeze(-1) < 0.1
            moved = alpha * (mY != m1[:, None]) * active[:, None]
            zeros = alpha.new_zeros(K)
            st["flux_out"] = st["flux_out"] + zeros.index_add(0, m1, moved.sum(1))
            st["flux_in"] = st["flux_in"] + zeros.index_add(0, mY.reshape(-1), moved.reshape(-1))
            st["flux_n"] = st["flux_n"] + active.sum()
            st["mode_cnt"] = st["mode_cnt"] + zeros.index_add(0, m1, torch.ones_like(m1, dtype=zeros.dtype))

    def pop_stats(self):
        st = self._stats
        out = super().pop_stats()
        # the MH "acceptance" keys hold ω here: rename, and drop the equal-energy prediction
        out = {k.replace("mh/acc/", "barker/translate_weight/"): v
               for k, v in out.items() if not k.startswith("mh/acc_pred/")}
        if torch.is_tensor(st["barker_n"]):
            n, own, ess = (st[f"barker_{k}"].tolist() for k in ("n", "own", "ess"))
            for i in range(len(n)):
                if n[i] > 0:
                    bucket = f"t{self.T_EDGES[i]:g}-{self.T_EDGES[i + 1]:g}"
                    out[f"barker/own_weight/{bucket}"] = own[i] / n[i]
                    out[f"barker/orbit_ess/{bucket}"] = ess[i] / n[i]

        dist = getattr(self.grad_term_cost.energy, "dist", None)
        if hasattr(dist, "means") and self.centres is not None:
            # clustering quality against the true modes: a mode without a centre can neither send nor receive mass
            d = torch.cdist(dist.means.to(self.centres), self.centres)
            out["barker/centre_coverage"] = float((d.min(1).values < 1.0).float().mean())
            out["barker/centres_spurious"] = float((d.min(0).values >= 1.0).float().mean())
        if torch.is_tensor(st["flux_n"]) and st["flux_n"] > 0:
            # net mass flux into each true mode (t < 0.1, per sample) against the weight error e = w - ŵ of the buffer
            net = (st["flux_in"] - st["flux_out"]) / st["flux_n"]
            w_hat = st["mode_cnt"] / st["mode_cnt"].sum()
            e = dist.weights.to(w_hat) - w_hat
            if net.std() > 0 and e.std() > 0:
                out["barker/flux_err_corr"] = float(torch.corrcoef(torch.stack([net, e]))[0, 1])
            for j in range(len(net)):
                out[f"barker/flux_net/{j}"] = float(net[j])
                out[f"barker/err/{j}"] = float(e[j])
        return out


class CorrectorMatcher(Matcher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _check_buffer_sample_shape(self, x0, x1):
        (B, D) = x0.shape
        assert x1.shape == (B, D)

    def populate_buffer(
            self,
            x0: torch.Tensor,
            timesteps: torch.Tensor,
            is_asbs_init_stage: bool,
    ):
        # IPF init: First Corrector Matching stage
        # of ASBS uses zero controller (i.e., ref_sde)
        sde = self.sde.ref_sde if is_asbs_init_stage else self.sde

        (x0, x1) = sdeint(
            sde,
            x0,
            timesteps,
            only_boundary=True,
        )

        self._check_buffer_sample_shape(x0, x1)
        self.buffer.add({
            "x0": x0.to("cpu"),
            "x1": x1.to("cpu"),
        })

    def _check_target_shape(self, t1, x1, score):
        (B, D) = x1.shape
        assert t1.shape == (B, 1) and score.shape == (B, D)

    def prepare_target(self, data, device):
        x0 = data["x0"].to(device)
        x1 = data["x1"].to(device)

        t1 = torch.ones(x0.shape[0], 1).to(device)
        score = self.sde.ref_sde.cond_score(x0, t1, x1)

        self._check_target_shape(t1, x1, score)
        return (t1, x1,), score
