# Copyright (c) Meta Platforms, Inc. and affiliates.

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


class AdjointVESteinMatcher(AdjointVEMatcher):
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

    def _check_soc_problem(self):
        super()._check_soc_problem()
        # AS from a Dirac X0: p^base_1 = N(X0, ν_1)
        assert isinstance(self.grad_term_cost, ScoreGradTermCost)
        var1 = self.grad_term_cost.var1
        assert torch.allclose(var1, self._nu(torch.ones_like(var1)))

    def _nu(self, t):
        # ν_t = ∫_0^t σ(s)² ds
        return self.sde.ref_sde._pt_gauss_param(t, torch.zeros_like(t))[1]

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


class AdjointVEGelfandDeyMatcher(AdjointVEMatcher):
    """ AdjointVEMatcher with a Gelfand-Dey control variate added to the target (1D, K = 2 modes):

            target = - (∇g(X1) + c · cv(Xt, X1)),
            cv = h(Xt) Σ_k λ_k N_k(X1) e^{E(X1)} / C_k(Xt),   λ = (1, -1) (modes sorted by position)

        N_k = N(μ_k, Σ_k) is the Laplace approximation of mode k of the target e^{-E}: μ_k a local
        minimum of E, Σ_k = sigma_scale / E''(μ_k), fitted once from the first buffer batch.
        With X0 = 0 (Dirac), κ_t = ν_t / ν_1, v_t = ν_1 κ_t (1 - κ_t):

            C_k(xt) = ∫ N_k(x1) p^base(xt | x1) dx1 = N(xt; κ_t μ_k, v_t + κ_t² Σ_k),
            h(xt) = 1 / Σ_k C_k(xt)^-1  (any function of xt keeps the zero mean; this one makes
                                         h / C_k ≤ 1, so cv is bounded by Σ_k sup(N_k e^E))

        E_{π(x1|xt)}[cv | xt] = 0 for π ∝ p1 p^base(xt | x1), p1 = e^{-E}: exact, and h = 1 is NOT
        enough (E[cv²] = ∞ for every Σ_k). Under the model posterior it is not zero but

            E_{π^u}[cv | xt] ∝ h(xt) Σ_k λ_k E_{ψ_k(.|xt)}[p1^u / p1],

        a signed detector of the mode weights; u* stays the fixed point. The carrier is w = 1
        (a function of xt only, as needed): with λ = (1, -1) and c < 0, the drift is pushed toward
        the left mode when it is under-sampled and toward the right one when it is over-sampled.
        Assumes Z = 1 (as for DistEnergy of a normalized density); otherwise cv scales with 1/Z.

        Which N_k are used (`mode_source`, tracked by `log_stats`):
            "manual":   `modes=[{mu, var}, {mu, var}]` given in the config, no fit
            "laplace":  `modes=None`, fitted as above
            "disabled": `modes=None` and the fit did not find exactly 2 modes: cv = 0, i.e. demo_as
    """
    SOURCE_CODE = {"disabled": 0, "laplace": 1, "manual": 2}

    def __init__(
        self,
        c: float = -0.05,
        sigma_scale: float = 0.8,
        modes: list | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert c <= 0
        assert 0 < sigma_scale < 2, "N_k must be thinner than 2 x the mode variance"
        self.c = c
        self.sigma_scale = sigma_scale
        self.modes = None # (mu, var), each (K,) double
        self.mode_source = None
        self._reset_stats()

        if modes is not None:
            assert len(modes) == 2 and all(m["var"] > 0 for m in modes), "need 2 modes with var > 0"
            modes = sorted(modes, key=lambda m: m["mu"])
            self.modes = (
                torch.tensor([float(m["mu"]) for m in modes], dtype=torch.float64),
                torch.tensor([float(m["var"]) for m in modes], dtype=torch.float64),
            )
            self.mode_source = "manual"
            print(f"[GelfandDey] manual modes mu={self.modes[0].tolist()} var={self.modes[1].tolist()}")

    def _check_soc_problem(self):
        super()._check_soc_problem()
        # AS from a Dirac X0: p^base_1 = N(X0, ν_1)
        assert isinstance(self.grad_term_cost, ScoreGradTermCost)
        var1 = self.grad_term_cost.var1
        assert torch.allclose(var1, self._nu(torch.ones_like(var1)))

    def _nu(self, t):
        # ν_t = ∫_0^t σ(s)² ds
        return self.sde.ref_sde._pt_gauss_param(t, torch.zeros_like(t))[1]

    def fit_modes(self, x1, n_seeds=64, n_iter=200, h=1e-3):
        """ Laplace approximation of the modes of e^{-E}: damped Newton descent on E from
            quantiles of the buffer samples x1, then μ_k = minimum, Σ_k = sigma_scale / E''(μ_k).
        """
        energy = self.grad_term_cost.energy
        assert x1.shape[1] == 1, "Gelfand-Dey matcher is 1D"

        def grad_E(x):
            return energy.grad_E(x)

        def hess_E(x):
            return (grad_E(x + h) - grad_E(x - h)) / (2 * h)

        q = torch.linspace(0, 1, n_seeds, device=x1.device, dtype=torch.float64)
        x = torch.quantile(x1[:, 0].double(), q).reshape(-1, 1)
        for _ in range(n_iter):
            x = x - grad_E(x) / hess_E(x).clamp(min=1.0)
        g, H = grad_E(x), hess_E(x)

        ok = (g.abs() < 1e-4) & (H > 1e-3)
        mus = x[ok].sort().values
        keep = [mus[0]] if len(mus) else []
        for m in mus[1:]:
            if m - keep[-1] > 1e-2:
                keep.append(m)
        if len(keep) == 2:
            mu = torch.stack(keep)
            var = self.sigma_scale / hess_E(mu.reshape(-1, 1)).reshape(-1)
            self.modes = (mu, var)
            self.mode_source = "laplace"
            print(f"[GelfandDey] laplace modes mu={mu.tolist()} var={var.tolist()}")
        else:
            self.modes = (None, None)
            self.mode_source = "disabled"
            print(f"[GelfandDey] found {len(keep)} modes, need 2: control variate disabled")

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
        if self.modes is None:
            self.fit_modes(x1)

        self._check_buffer_sample_shape(x0, x1, adjoint1)
        with torch.no_grad():
            E1 = self.grad_term_cost.energy.eval(x1)
        self.buffer.add({
            "x0": x0.to("cpu"),
            "x1": x1.to("cpu"),
            "adjoint1": adjoint1.to("cpu"),
            "E1": E1.reshape(-1, 1).to("cpu"),
        })

    def control_variate(self, t, x0, xt, x1, E1):
        """ cv(xt, x1), (B, 1). Computed in log space, in double.
        """
        (mu, var) = self.modes
        if mu is None:
            return torch.zeros_like(x1)
        (t, x0, xt, x1, E1) = (a.double() for a in (t, x0, xt, x1, E1))
        mu, var = mu.to(x1.device), var.to(x1.device)

        nut, nu1 = self._nu(t), self._nu(torch.ones_like(t))
        kappa = nut / nu1
        v = nu1 * kappa * (1 - kappa)

        s2 = v + kappa ** 2 * var # (B, K)
        m = (1 - kappa) * x0 + kappa * mu
        logC = - 0.5 * (xt - m) ** 2 / s2 - 0.5 * torch.log(2 * torch.pi * s2)
        logN = - 0.5 * (x1 - mu) ** 2 / var - 0.5 * torch.log(2 * torch.pi * var)

        # log (h / C_k) + log N_k(x1) e^{E(x1)}
        l = logN + E1 - logC - torch.logsumexp(- logC, dim=-1, keepdim=True)
        lam = torch.tensor([1.0, -1.0], dtype=l.dtype, device=l.device)
        lmax = l.max(dim=-1, keepdim=True).values
        return torch.exp(lmax) * (lam * torch.exp(l - lmax)).sum(-1, keepdim=True)

    def prepare_target(self, data, device):
        (t, xt), target = super().prepare_target(data, device)
        x0 = data["x0"].to(device)
        x1 = data["x1"].to(device)
        E1 = data["E1"].to(device)

        cv = self.control_variate(t, x0, xt, x1, E1).to(target)
        self._check_target_shape(t, xt, cv)
        self._accumulate_stats(cv, target)
        return (t, xt), target - self.c * cv

    def _reset_stats(self):
        self._stats = None

    @torch.no_grad()
    def _accumulate_stats(self, cv, target):
        # tensors only (no host sync): summed over the train steps since the last log_stats()
        s = torch.stack([
            cv.sum(), (cv ** 2).sum(), cv.abs().max(), (self.c * cv).abs().sum(),
            target.abs().sum(), torch.tensor(float(len(cv)), device=cv.device),
        ])
        if self._stats is None:
            self._stats = s
        else:
            self._stats = torch.cat([
                (self._stats[:2] + s[:2]),
                torch.maximum(self._stats[2:3], s[2:3]),
                (self._stats[3:] + s[3:]),
            ])

    def log_stats(self):
        """ Scalars for the writer: which N_k option is active, its parameters, and the
            statistics of cv over the train steps since the last call.
        """
        out = {"gd/source": self.SOURCE_CODE.get(self.mode_source, -1)} # -1: not fitted yet
        mu, var = self.modes if self.modes is not None else (None, None)
        out["gd/active"] = float(mu is not None)
        if mu is not None:
            for k in range(len(mu)):
                out[f"gd/mu_{k}"], out[f"gd/var_{k}"] = mu[k].item(), var[k].item()
        if self._stats is not None:
            (sum1, sum2, amax, shift, tgt, n) = self._stats.tolist()
            mean = sum1 / n
            out.update({
                "gd/cv_mean": mean,
                "gd/cv_std": max(sum2 / n - mean ** 2, 0.0) ** 0.5,
                "gd/cv_absmax": amax,
                "gd/shift_over_target": shift / max(tgt, 1e-12), # mean |c cv| / mean |target|
            })
        self._reset_stats()
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
