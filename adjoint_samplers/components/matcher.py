# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections import defaultdict

import torch
from torch.func import grad, jvp, vmap
from torch.utils.data import DataLoader

from adjoint_samplers.components.buffer import BatchBuffer, DeviceBatchLoader
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        return torch.rand(B, 1, device=x.device)

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


def _bmv(M, v):
    # batched matrix-vector product: (B, d, d) x (B, d) -> (B, d)
    return (M @ v.unsqueeze(-1)).squeeze(-1)


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


class AdjointVESteinMatcher(AdjointVEDiracMatcher):
    """ AdjointVEMatcher with the Stein term of ram_stein.tex added to the target:

            target = - (∇g(X1) + T F),   (T F)_i = Σ_j F_ij ∂_j log π(X1 | Xt) + ∂_j F_ij

        π(x1 | xt) ∝ p1(x1) p^base(xt | x1) is the posterior of the target p1, not of the model,
        F = c ν_{1|t} A(t, Xt, X1, E(X1)), c < 0, A = B B^T ⪰ 0. For dX = σ(t) dW from the Dirac X0,
        with ν_t = ∫_0^t σ², κ_t = ν_t / ν_1 and ν_{1|t} = ν_1 - ν_t,

            T F = c [ A (Xt - (1 - κ_t) X0 - κ_t X1) + ν_{1|t} (A ∇log p1(X1) + ∇_{X1}·A) ]

        (constant σ, X0 = 0: κ_t = t, ν_{1|t} = σ²(1 - t)). The controller here is u / σ(t), so the
        target has no σ(t). T F has zero mean under π, not under the model posterior: the regressed
        drift moves by - c ν_{1|t} E[A ∇log(p1 / p1^u)(X1) | Xt], so the step of AS along
        ∇log(p1 / p1^u) is preconditioned by I + |c| ν_{1|t} A ⪰ I (up to 1 + |c| ν_1 at t = 0 for A = I),
        toward target mass the model under-samples; u* stays the fixed point. ∇log p1 = ∇log p^base_1 - ∇g
        is read off the buffer, so p1 is the target encoded by g (with max_grad_E_norm, its clipped score).

        Without `stein_net`, A = I and ∇·A = 0. Otherwise B = stein_net(t, Xt, X1, E(X1)) is trained
        with the controller on the same loss (for it, the conditional variance of the target) and
        ∇_{X1}·A is the total derivative, through E(X1) with the stored ∇E(X1): exact (d JVPs) or
        Hutchinson (1 JVP, unbiased).
    """
    def __init__(
        self,
        c: float = -0.05,
        stein_net: torch.nn.Module | None = None,
        div_estimator: str = "exact",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert c <= 0, "c < 0 pushes toward under-sampled target mass, c > 0 away from it"
        assert div_estimator in ("exact", "hutchinson")
        self.c = c
        self.stein_net = stein_net
        self.div_estimator = div_estimator

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
        data = {
            "x0": x0.to("cpu"),
            "x1": x1.to("cpu"),
            "adjoint1": adjoint1.to("cpu"),
        }
        if self.stein_net is not None:
            # A is a function of E(X1)
            with torch.no_grad():
                E1 = self.grad_term_cost.energy.eval(x1)
            data["E1"] = E1.reshape(-1, 1).to("cpu")
        self.buffer.add(data)

    def _div_A(self, B, Bx, x1, E1, grad_E1):
        """ ∇_{x1}·A for A = B B^T: Σ_j (D_j A) e_j, D_v A = (D_v B) B^T + B (D_v B)^T the derivative
            along v, through E1 by D_v E = <∇E, v>. Hutchinson: (D_v A) v, Rademacher v.
        """
        def DA_v(v):
            dB = jvp(B, (x1, E1), (v, (grad_E1 * v).sum(-1, keepdim=True)))[1]
            return _bmv(dB, _bmv(Bx.mT, v)) + _bmv(Bx, _bmv(dB.mT, v))

        (N, D) = x1.shape
        if self.div_estimator == "exact":
            eye = torch.eye(D, device=x1.device, dtype=x1.dtype)
            return vmap(DA_v)(eye[:, None, :].expand(D, N, D)).sum(0)
        return DA_v(2 * torch.randint_like(x1, 2) - 1)

    def stein_term(self, t, x0, xt, x1, adjoint1, E1=None):
        nut, nu1 = self._nu(t), self._nu(torch.ones_like(t))
        kappa, nu1t = nut / nu1, nu1 - nut

        score1 = self.grad_term_cost.base_score(x1) - adjoint1 # ∇log p1 = ∇log p^base_1 - ∇g
        resid = xt - (1 - kappa) * x0 - kappa * x1 # = ν_{1|t} ∇_{x1} log p^base(xt | x1)

        if self.stein_net is None: # A = I
            return self.c * (resid + nu1t * score1)

        B = lambda x1, E1: self.stein_net(t, xt, x1, E1)
        Bx = B(x1, E1)
        div = self._div_A(B, Bx, x1, E1, - score1) # ∇E = - ∇log p1
        A = lambda v: _bmv(Bx, _bmv(Bx.mT, v))
        return self.c * (A(resid) + nu1t * (A(score1) + div))

    def prepare_target(self, data, device):
        (t, xt), target = super().prepare_target(data, device)
        adjoint1 = - target

        x0 = data["x0"].to(device)
        x1 = data["x1"].to(device)
        E1 = data["E1"].to(device) if self.stein_net is not None else None

        stein = self.stein_term(t, x0, xt, x1, adjoint1, E1)
        self._check_target_shape(t, xt, stein)
        return (t, xt), - (adjoint1 + stein)


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
        on it. `rao_blackwell` keeps all M - 1 jumps (M - 1 energy values per sample), else one at random.
    """
    T_EDGES = (0.0, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 1.0)

    def __init__(
        self,
        gamma: float = 0.5,
        n_clusters: int = 2,
        rao_blackwell: bool = True,
        kmeans_iters: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert gamma >= 0 and n_clusters >= 2
        self.gamma = gamma
        self.n_clusters = n_clusters
        self.rao_blackwell = rao_blackwell
        self.kmeans_iters = kmeans_iters
        self.centres = None
        self._stats = defaultdict(float)

    def propose(self, x1):
        """ jumps Y - X1 = m_k - m_c (N, J, D) from the cluster c of X1 to every other cluster k
            (J = M - 1) or to one at random (J = 1), and valid = 1[Y in cluster k] (N, J).
            No centres yet: no valid jump.
        """
        (N, D), M = x1.shape, self.n_clusters
        J = M - 1 if self.rao_blackwell else 1
        if self.centres is None:
            return x1.new_zeros(N, J, D), x1.new_zeros(N, J)

        m = self.centres.to(x1)
        c = _nearest(x1, m)
        others = torch.arange(M, device=x1.device).expand(N, M)
        others = others[others != c[:, None]].view(N, M - 1)
        if not self.rao_blackwell:
            others = others.gather(1, torch.randint(M - 1, (N, 1), device=x1.device))

        jumps = m[others] - m[c].unsqueeze(1)
        valid = _nearest((x1.unsqueeze(1) + jumps).reshape(-1, D), m).view(N, J) == others
        valid = valid.to(x1)
        return jumps * valid.unsqueeze(-1), valid

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

        # centres from the buffer before this batch, so they do not depend on its samples
        if len(self.buffer) > 0:
            buffer_x1 = torch.cat(self.buffer.batches["x1"])[-self.buffer.buffer_size:]
            self.centres = _kmeans(buffer_x1.to(x1), self.n_clusters, self.centres, self.kmeans_iters)

        jumps, valid = self.propose(x1)
        energy = self.grad_term_cost.energy
        with torch.no_grad():
            E1 = energy.eval(x1)
            EY = torch.zeros_like(valid)
            mask = valid.bool()
            if mask.any():
                EY[mask] = energy.eval((x1.unsqueeze(1) + jumps)[mask])
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
        }
        # GMM diagnostic: accepted jumps that stay in the same true mode do nothing for the weights
        dist = getattr(energy, "dist", None)
        if hasattr(dist, "assign"):
            same = dist.assign(x1.unsqueeze(1) + jumps) == dist.assign(x1).unsqueeze(1)
            data["same_mode"] = (same.to(x1) * valid).to("cpu")
        self.buffer.add(data)

    def mh_term(self, t, x0, xt, x1, E1, jumps, EY, valid):
        """ γ h / ν_{1|t} (N, D) and the acceptance α (N, J) """
        nut, nu1 = self._nu(t), self._nu(torch.ones_like(t))
        kappa, nu1t = nut / nu1, (nu1 - nut).clamp(min=1e-12)

        r = xt - (1 - kappa) * x0 - kappa * x1
        gauss = (kappa * jumps.pow(2).sum(-1) - 2 * (jumps * r.unsqueeze(1)).sum(-1)) / (2 * nu1t)
        alpha = (- (EY - E1) - gauss).clamp(max=0).exp() * valid

        h = (alpha.unsqueeze(-1) * jumps).mean(1)
        return self.gamma * h / nu1t, alpha

    def prepare_target(self, data, device):
        (t, xt), target = super().prepare_target(data, device)

        x0, x1 = data["x0"].to(device), data["x1"].to(device)
        E1, EY = data["E1"].to(device), data["EY"].to(device)
        jumps, valid = data["jumps"].to(device), data["valid"].to(device)

        term, alpha = self.mh_term(t, x0, xt, x1, E1, jumps, EY, valid)
        same = data["same_mode"].to(device) if "same_mode" in data else None
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

        st["proposals"] += alpha.numel()
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
