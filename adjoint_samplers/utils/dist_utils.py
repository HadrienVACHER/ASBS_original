# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy
import math
import torch
from torch import distributions


########################################
######### Target Distributions #########
########################################

class GMM1D(distributions.Distribution):
    """ A simple bi-modal Gaussian mixtures in 1D for demo purposes
    """
    def __init__(self, device="cpu") -> None:
        super().__init__()

        self.dim = 1
        self.name = "gmm1d"
        self.weights = torch.tensor([.1, .9], dtype=torch.float)
        self._initialize_distr(device)

    def _initialize_distr(self, device) -> None:
        loc = torch.tensor([-3, 3], device=device, dtype=torch.float).reshape(2, 1)
        scale = torch.tensor([.5, .5], device=device, dtype=torch.float).reshape(2, 1)
        weights = self.weights.to(device)

        modes = distributions.Independent(
            distributions.Normal(loc, scale), 1
        )
        mix = distributions.Categorical(weights)
        self.distr = distributions.MixtureSameFamily(mix, modes)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_prob = self.distr.log_prob(x).unsqueeze(-1)
        assert log_prob.shape == (*x.shape[:-1], 1)
        return log_prob

    def sample(self, shape: tuple) -> torch.Tensor:
        return self.distr.sample(torch.Size(shape))

    def to(self, device) -> distributions.Distribution:
        self._initialize_distr(device)
        return self

    def uniform_weights(self) -> "GMM1D":
        """ same modes with equal weights: r in the independent MH of AdjointVEIMHBufferMatcher """
        other = copy.copy(self)
        other.weights = torch.full_like(self.weights, 1 / len(self.weights))
        other._initialize_distr(self.weights.device)
        return other

class GMM(distributions.Distribution):
    """ Isotropic Gaussian mixture in `dim` dimensions with K modes

        p(x) = Σ_k π_k N(x | μ_k, I_d)

        Means (`layout`):
          cube    μ_k ~ Uniform([-a, a]^d),  a = 8√(3/(2d)) = 4√(6/d)  (E‖μ_k‖² = 32)
          sphere  μ_k = r u_k,  u_k = v_k / ‖v_k‖,  v_k ~ N(0, I_d): every mode at distance r
                  from the origin, in random directions
          spread  same, with the directions pushed apart: for K <= d + 1 a regular simplex,
                  every pair of modes at distance r √(2K/(K-1)) (≈ r√2 for large K)
          r = `radius`, default √32 (E‖μ_k‖² as in the cube layout)
        Weights (`weight_law`), π_k = U_k / Σ_j U_j:
          uniform      U_k ~ Uniform(0.1, 0.9)                          (ratio max/min <= 9)
          loguniform   U_k = exp(-E_k),  E_k ~ Uniform(0, log weight_ratio) (ratio <= weight_ratio)
          exponential  U_k = exp(-E_k),  E_k ~ Exponential(mean energy_scale): Boltzmann
                       weights of random mode energies; the ratio grows with K (~ K^energy_scale)

        The target is fully determined by `seed`, independently of the training seed.
    """
    def __init__(
        self,
        dim: int,
        K: int = 5,
        seed: int = 0,
        layout: str = "cube",
        weight_law: str = "uniform",
        weight_ratio: float = 20.0,
        energy_scale: float = 1.0,
        radius: float | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.dim = dim
        self.K = K
        self.seed = seed
        self.name = f"gmm{dim}d"

        g = torch.Generator().manual_seed(seed)
        if layout == "cube":
            a = GMM.half_width(dim)
            self.means = (2 * torch.rand(K, dim, generator=g) - 1) * a # (K, d)
        elif layout in ("sphere", "spread"):
            r = radius if radius is not None else GMM.default_radius()
            v = torch.randn(K, dim, generator=g)
            u = v / v.norm(dim=1, keepdim=True)
            if layout == "spread":
                u = GMM.spread_directions(u)
            self.means = u * r # (K, d)
        else:
            raise ValueError(f"unknown layout {layout}")

        if weight_law == "uniform":
            U = 0.1 + 0.8 * torch.rand(K, generator=g) # (K,)
        elif weight_law == "loguniform":
            U = torch.exp(-math.log(weight_ratio) * torch.rand(K, generator=g))
        elif weight_law == "exponential":
            E = -energy_scale * torch.log1p(-torch.rand(K, generator=g)) # Exponential, mean energy_scale
            U = torch.exp(-E)
        else:
            raise ValueError(f"unknown weight_law {weight_law}")
        self.weights = U / U.sum()
        self.to(device)

    @staticmethod
    def default_radius() -> float:
        """ √(E‖μ_k‖²) of the cube layout = √32, for any dim
        """
        return math.sqrt(32)

    @staticmethod
    def spread_directions(u: torch.Tensor, steps: int = 1000) -> torch.Tensor:
        """ Unit vectors pushed apart from the start `u` by minimising Σ_{i≠j} 1/‖u_i - u_j‖⁴
            on the sphere. Deterministic given `u`.
        """
        K = u.shape[0]
        if K < 2:
            return u
        eye = torch.eye(K, dtype=torch.float64)
        with torch.enable_grad():
            v = u.double().clone().requires_grad_(True)
            opt = torch.optim.Adam([v], lr=0.01)
            for _ in range(steps):
                w = v / v.norm(dim=1, keepdim=True)
                D = torch.cdist(w, w) + eye
                loss = ((D ** -4) * (1 - eye)).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
        v = v.detach()
        return (v / v.norm(dim=1, keepdim=True)).to(u.dtype)

    @staticmethod
    def half_width(dim: int) -> float:
        """ a, chosen so that E‖μ_k‖² = 32 for every dim (typical distance between two means ≈ 8)
        """
        return 8 * math.sqrt(3 / (2 * dim))

    @staticmethod
    def coord_std(dim: int, radius: float | None = None) -> float:
        """ std of one coordinate of x ~ p (expected over the means): √(1 + E‖μ_k‖²/d)
            = √(1 + r²/d); r = √32 for the cube layout and the default radius
        """
        r = radius if radius is not None else GMM.default_radius()
        return math.sqrt(1 + r**2 / dim)

    def to(self, device) -> distributions.Distribution:
        self.means = self.means.to(device)
        self.weights = self.weights.to(device)
        return self

    def uniform_weights(self) -> "GMM":
        """ same modes with equal weights: r in the independent MH of AdjointVEIMHBufferMatcher """
        other = copy.copy(self)
        other.weights = torch.full_like(self.weights, 1 / self.weights.numel())
        return other

    def _component_logits(self, x: torch.Tensor) -> torch.Tensor:
        """ log π_k + log N(x | μ_k, I), shape (..., K)
        """
        sq_dist = (x.unsqueeze(-2) - self.means).pow(2).sum(-1)
        return (
            self.weights.log()
            - 0.5 * sq_dist
            - 0.5 * self.dim * math.log(2 * math.pi)
        )

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(self._component_logits(x), dim=-1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """ ∇log p(x) = Σ_k r_k(x) μ_k - x, where r = softmax over components
        """
        resp = torch.softmax(self._component_logits(x), dim=-1)
        return resp @ self.means - x

    def sq_dist_to_mode(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """ ||x - μ_k||², squared Mahalanobis distance of x to its mode k (χ²_d for samples of mode k) """
        return (x - self.means[k]).pow(2).sum(-1)

    def assign(self, x: torch.Tensor) -> torch.Tensor:
        """ most likely mode of each sample
        """
        return self._component_logits(x).argmax(dim=-1)

    def sample(self, shape: tuple) -> torch.Tensor:
        shape = torch.Size(shape)
        modes = torch.multinomial(
            self.weights, shape.numel(), replacement=True
        ).reshape(shape)
        noise = torch.randn(*shape, self.dim, device=self.means.device)
        return self.means[modes] + noise


class ManyWell(distributions.Distribution):
    """ ManyWell of FAB (Midgley et al. 2023, fab-torch `ManyWellEnergy` / `DoubleWellEnergy`), dim = 2 n_wells, product of n_wells pairs:

            log p~(x) = Σ_i [ - (a x_{2i-1} + b x_{2i-1}^2 + c x_{2i-1}^4) - x_{2i}^2 / 2 ],   a = -0.5, b = -6, c = 1
                      = Σ_i [ - x_{2i-1}^4 + 6 x_{2i-1}^2 + 0.5 x_{2i-1} - 0.5 x_{2i}^2 ]

        The first coordinate of each pair is a double well (wells near -1.7 (light, mass 0.1557) and +1.7), the second a standard Gaussian.
        Exact sampling: the first coordinate by inverse CDF on a fine grid (FAB uses rejection sampling), the second N(0, 1).
        The energy is unnormalised, as in FAB (normalised = False).
    """
    def __init__(self, dim: int = 32, a: float = -0.5, b: float = -6.0, c: float = 1.0, device: str = "cpu") -> None:
        super().__init__(validate_args=False)
        assert dim % 2 == 0
        self.dim, self.n_wells, self.a, self.b, self.c = dim, dim // 2, a, b, c
        self.name = f"manywell{dim}d"
        x = torch.linspace(-6.0, 6.0, 200001, dtype=torch.float64)
        p = torch.exp(-(a * x + b * x ** 2 + c * x ** 4) + (a * x + b * x ** 2 + c * x ** 4).min())
        self._grid, self._cdf = x, torch.cumsum(p, 0) / p.sum()
        inner = (x > -1.0) & (x < 1.0)
        self.well_split = float(x[inner][torch.argmin(p[inner])]) # the barrier top, ≈ -0.0417
        self.light_mass = float(p[x < self.well_split].sum() / p.sum()) # ≈ 0.15569
        dx = float(x[1] - x[0])
        self._m2_first = float((x ** 2 * p).sum() / p.sum())     # E[x_1^2] about the origin
        self.to(device)

    def log_Z(self) -> float:
        """ log of the normalising constant of the unnormalised density exp(log_prob): n_wells x log(Z_1D sqrt(2 pi)), Z_1D = ∫ exp(-(a x + b x^2 + c x^4)) dx by
            the trapezoid rule on a fine grid (FAB's TARGET_Z for the standard well: 11784.509) """
        x = self._grid
        e = self.a * x + self.b * x ** 2 + self.c * x ** 4
        z1 = torch.trapezoid(torch.exp(-(e - e.min())), x).item() * math.exp(-e.min().item())
        return self.n_wells * (math.log(z1) + 0.5 * math.log(2 * math.pi))

    def to(self, device) -> distributions.Distribution:
        self._dev = device
        self._grid_d, self._cdf_d = self._grid.to(device), self._cdf.to(device)
        return self

    def coord_std(self) -> float:
        """ √(mean over the coordinates of E[x_j^2]): the scale of the VE noise from a Dirac at the origin """
        return math.sqrt((self._m2_first + 1.0) / 2)

    def _pairs(self, x):
        z = x.reshape(*x.shape[:-1], self.n_wells, 2)
        return z[..., 0], z[..., 1]

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self._pairs(x)
        return (- (self.a * x1 + self.b * x1 ** 2 + self.c * x1 ** 4) - 0.5 * x2 ** 2).sum(-1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self._pairs(x)
        g1 = - (self.a + 2 * self.b * x1 + 4 * self.c * x1 ** 3)
        return torch.stack([g1, - x2], dim=-1).reshape(x.shape)

    def sample(self, shape: tuple) -> torch.Tensor:
        shape = torch.Size(shape)
        n = shape.numel()
        u = torch.rand(n, self.n_wells, device=self._dev, dtype=torch.float64)
        idx = torch.searchsorted(self._cdf_d, u).clamp(1, len(self._cdf_d) - 1)
        c0, c1 = self._cdf_d[idx - 1], self._cdf_d[idx]
        t = ((u - c0) / (c1 - c0).clamp(min=1e-300)).clamp(0, 1)
        x1 = self._grid_d[idx - 1] + t * (self._grid_d[idx] - self._grid_d[idx - 1])
        x2 = torch.randn(n, self.n_wells, device=self._dev, dtype=torch.float64)
        return torch.stack([x1, x2], dim=-1).reshape(*shape, self.dim).float()


class HeteroGMM(GMM):
    """ Gaussian mixture with HETEROGENEOUS geometry: p(x) = Σ_k π_k N(x | μ_k, Σ_k), Σ_k = L_k L_k^T

        Means: μ_k = r_k u_k, u_k a random direction, r_k ~ Uniform(radius_min, radius_max): modes at unequal distances.
        Covariances: L_k = Q_k diag(s_k), Q_k a random rotation, s_k log-uniform in [scale_min, scale_max]: unequal widths,
        non-axis-aligned. Weights as in GMM (loguniform, max/min <= weight_ratio).
        Components are redrawn until every pair is separated: |μ_i - μ_j| >= min_sep (σ_i(u) + σ_j(u)), u the unit vector from
        μ_i to μ_j and σ_k(u) = √(u^T Σ_k u), so that the modes can be told apart.
        The target is fully determined by `seed`.
    """
    def __init__(
        self,
        dim: int,
        K: int = 5,
        seed: int = 0,
        radius_min: float = 7.0,
        radius_max: float = 11.0,
        scale_min: float = 0.6,
        scale_max: float = 1.6,
        weight_ratio: float = 20.0,
        min_sep: float = 3.5,
        layout: str = "random",
        simplex_radius: float = 4.25,
        device: str = "cpu",
    ) -> None:
        distributions.Distribution.__init__(self, validate_args=False)
        self.dim, self.K, self.seed = dim, K, seed
        self.name = f"hetero{dim}d"
        assert layout in ("random", "simplex")

        g = torch.Generator().manual_seed(seed)
        means = torch.zeros(K, dim)
        L = torch.zeros(K, dim, dim)
        if layout == "simplex":
            # the mode POSITIONS and WEIGHTS of the isotropic GMM (layout="spread", radius = simplex_radius, loguniform weights, same seed:
            # same generator draws), only the covariances are heterogeneous. No separation check (the normalised separation
            # 6.1 / (σ_i + σ_j) is <= 3.05 for widths >= 1: the modes overlap more than in the isotropic benchmark).
            v = torch.randn(K, dim, generator=g)
            u = GMM.spread_directions(v / v.norm(dim=1, keepdim=True))
            self.means = u * simplex_radius
            U = torch.exp(-math.log(weight_ratio) * torch.rand(K, generator=g))
            self.weights = U / U.sum()
            for k in range(K):
                Q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g))
                s = torch.exp(math.log(scale_min) + (math.log(scale_max) - math.log(scale_min)) * torch.rand(dim, generator=g))
                L[k] = Q * s
            self.L = L
            self.to(device)
            return

        def draw(k):
            v = torch.randn(dim, generator=g)
            means[k] = v / v.norm() * (radius_min + (radius_max - radius_min) * torch.rand(1, generator=g))
            Q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g))
            s = torch.exp(math.log(scale_min) + (math.log(scale_max) - math.log(scale_min)) * torch.rand(dim, generator=g))
            L[k] = Q * s # Q diag(s)

        def sigma_along(k, u):
            return (L[k].T @ u).norm()

        def separated(i, j):
            d = means[j] - means[i]
            u = d / d.norm()
            return d.norm() >= min_sep * (sigma_along(i, u) + sigma_along(j, u))

        for k in range(K):
            for _ in range(10000):
                draw(k)
                if all(separated(i, k) for i in range(k)):
                    break
            else:
                raise RuntimeError(f"cannot separate {K} modes in d = {dim} with these radii / scales")
        self.means, self.L = means, L
        U = torch.exp(-math.log(weight_ratio) * torch.rand(K, generator=g))
        self.weights = U / U.sum()
        self.to(device)

    def to(self, device) -> distributions.Distribution:
        self.means, self.weights, self.L = self.means.to(device), self.weights.to(device), self.L.to(device)
        self.P = torch.linalg.inv(self.L)                                    # L^{-1}, (K, d, d)
        self.logdet = torch.linalg.slogdet(self.L)[1]                        # log |det L_k|, (K,)
        return self

    def _whiten(self, x):
        # z_k = L_k^{-1} (x - μ_k): (..., K, d)
        return torch.einsum("kde,...ke->...kd", self.P, x.unsqueeze(-2) - self.means)

    def _component_logits(self, x: torch.Tensor) -> torch.Tensor:
        z = self._whiten(x)
        return self.weights.log() - 0.5 * z.pow(2).sum(-1) - self.logdet - 0.5 * self.dim * math.log(2 * math.pi)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """ ∇log p(x) = - Σ_k r_k(x) Σ_k^{-1} (x - μ_k) """
        r = torch.softmax(self._component_logits(x), dim=-1)
        z = self._whiten(x)
        return - torch.einsum("...k,kde,...kd->...e", r, self.P, z)

    def sq_dist_to_mode(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nde,ne->nd", self.P[k], x - self.means[k]).pow(2).sum(-1)

    def sample(self, shape: tuple) -> torch.Tensor:
        shape = torch.Size(shape)
        modes = torch.multinomial(self.weights, shape.numel(), replacement=True).reshape(shape)
        z = torch.randn(*shape, self.dim, device=self.means.device, dtype=self.means.dtype)
        return self.means[modes] + torch.einsum("...de,...e->...d", self.L[modes], z)

    def coord_std(self) -> float:
        """ √ (mean coordinate variance of the mixture), the scale of the VE noise """
        mbar = (self.weights[:, None] * self.means).sum(0)
        var = (self.weights * ((self.means - mbar).pow(2).sum(-1) + (self.L ** 2).sum((-1, -2)))).sum() / self.dim
        return float(var.sqrt())


########################################
######### Source Distributions #########
########################################

class Gauss(distributions.Distribution):
    def __init__(
        self,
        dim,
        loc: float = 0.0,
        scale: float = 1.0,
        device: str ="cpu"
    ) -> None:
        super().__init__()

        self.dim = dim
        self.loc = torch.tensor(loc, device=device, dtype=torch.float)
        self.scale = torch.tensor(scale, device=device, dtype=torch.float)
        self.name = "gauss"

    def sample(self, shape: tuple) -> torch.Tensor:
        z = torch.randn(*shape, self.dim, device=self.loc.device)
        return z * self.scale + self.loc


class Delta(distributions.Distribution):
    def __init__(self, dim, loc: float = 0.0, device="cpu") -> None:
        super().__init__()

        self.name = "delta"
        self.dim = dim
        self.loc = torch.tensor(loc, device=device, dtype=torch.float)
        self.scale = torch.tensor(0.0, device=device, dtype=torch.float)

    def sample(self, shape: tuple) -> torch.Tensor:
        return self.loc.repeat(*shape, self.dim)


class CenteredParticlesGauss(distributions.Distribution):
    """ Sample particles with zero center of mass
    """
    def __init__(
        self,
        n_particles,
        spatial_dim,
        scale: float = 1.0,
        device="cpu",
    ):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.dim = n_particles * spatial_dim
        # Centered distribution always has zero mean
        self.loc = torch.tensor(0.0, device=device, dtype=torch.float)
        self.scale = torch.tensor(scale, device=device, dtype=torch.float)
        self.device = device
        self.name = "meanfree"

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        samples = torch.randn(*shape, self.dim, device=self.device) * self.scale
        samples = samples.reshape(-1, self.n_particles, self.spatial_dim)
        samples = samples - samples.mean(-2, keepdims=True)
        return samples.reshape(*shape, self.n_particles * self.spatial_dim)


class CenteredParticlesHarmonic(distributions.Distribution):
    """ Sample particles with zero center of mass from
        non-isotropic Gaussian based on a harmonic prior
        https://arxiv.org/pdf/2304.02198
    """
    def __init__(
        self,
        n_particles,
        spatial_dim,
        scale: float = 1.0,
        device="cpu",
    ):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.dim = n_particles * spatial_dim
        # Centered distribution always has zero mean
        self.loc = torch.tensor(0.0, device=device, dtype=torch.float)
        self.scale = torch.tensor(scale, device=device, dtype=torch.float)
        self.device = device
        self.name = "harmonic"

        cov = self._compute_cov(n_particles, spatial_dim)
        self.rank, self.A = self._decompose_svd(cov)

    def _compute_cov(self, n_particles, spatial_dim):
        """ e.g., n_particles = 2, spatial_dim = 3 would generate

            R = tensor([[  1,   0,   0, -0.5,   0,   0 ],
                        [  0,   1,   0,   0, -0.5,   0 ],
                        [  0,   0,   1,   0,    0, -0.5],
                        [-0.5,  0,   0,   1,    0,   0 ],
                        [  0, -0.5,   0,  0,    1,   0 ],
                        [  0,   0, -0.5,  0,    0,   1 ]])

            Denote x = [a1, a2, a3, b1, b2, b3]. This yields

            0.5 * x^T R x = a**2 + b**2 - ab = (a - b)**2
        """
        # TODO(ghliu) assume all particles are connected; otherwise changes A
        A = - 0.5 * torch.ones(n_particles, n_particles)
        A[torch.arange(n_particles), torch.arange(n_particles)] = 1.
        B = torch.eye(spatial_dim)
        return torch.kron(A, B).inverse()

    def _decompose_svd(self, cov):
        """ return the rank of cov and A where `cov = UΣV = AA^T`
        """
        U, S, Vt = torch.svd(cov)

        rank = (S > 1e-8).sum()
        A = U[:, :rank] @ torch.diag(S[:rank]).sqrt()
        return rank, A

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        """ generate samples by
            1. z ~ N(0,I) in the subspace with dim=rank
            2. x = Az, hence x ~ N(0, AA^T)
            3. make x zero COM
        """
        if shape is None:
            shape = tuple()

        B = math.prod(shape) # batch
        z = torch.randn(B, self.rank, device=self.device)
        samples = z @ self.A.to(z).T # (B, R) x (R, D) = (B, D)
        assert samples.shape == (B, self.dim)

        samples = samples * self.scale # note: scale = sqrt(alpha)

        samples = samples.reshape(*shape, self.n_particles, self.spatial_dim)
        samples = samples - samples.mean(-2, keepdims=True)
        return samples.reshape(*shape, self.dim)
