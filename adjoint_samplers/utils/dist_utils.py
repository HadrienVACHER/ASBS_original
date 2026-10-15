# Copyright (c) Meta Platforms, Inc. and affiliates.

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
        self._initialize_distr(device)

    def _initialize_distr(self, device) -> None:
        loc = torch.tensor([-1, 2], device=device, dtype=torch.float).reshape(2, 1)
        scale = torch.tensor([.7, .4], device=device, dtype=torch.float).reshape(2, 1)
        weights = torch.tensor([.5, .5], device=device, dtype=torch.float).reshape(2)

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
