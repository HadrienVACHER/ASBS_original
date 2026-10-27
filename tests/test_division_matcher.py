""" AdjointVEDivisionMatcher wiring (no training): warm-up, versioning of the frozen parameters, and the END-TO-END exact zero-mean test with the matcher's own bridge sampler and
    prepare_target (as tests/test_dw4_end_to_end.py). Endpoints x1 are injected through a patched sdeint.

    Thresholds (fixed in advance): warm-up samples carry version -1 and a zero term; every later batch carries the version current at its insertion (non-decreasing, never an older one);
    prepare_target equals the AS target for version -1; with exact pi endpoints inserted after the first version, max |z| of S_psi = mean <term, psi(x_t)> < 3.5 over 8 values of kappa
    (psi1 = x_t, psi2 = x_t |x_t|^2); positive control (endpoints from a wrong-weight sampler): max |z| > 5.
    python tests/test_division_matcher.py
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adjoint_samplers.components.matcher as matcher_module  # noqa: E402
from adjoint_samplers.components.buffer import BatchBuffer  # noqa: E402
from adjoint_samplers.components.matcher import AdjointVEDivisionMatcher  # noqa: E402
from adjoint_samplers.components.sde import ControlledSDE, VESDE  # noqa: E402
from adjoint_samplers.components.state_cost import ZeroGradStateCost  # noqa: E402
from adjoint_samplers.components.term_cost import ScoreGradTermCost  # noqa: E402
from adjoint_samplers.energies.dist_energy import DistEnergy  # noqa: E402
from adjoint_samplers.utils.dist_utils import GMM, Delta  # noqa: E402

D = K = 12
gmm = GMM(dim=D, K=K, seed=0, layout="spread", weight_law="loguniform", weight_ratio=20.0, radius=4.25)
SIGMA_MAX = 2 * GMM.coord_std(D, 4.25)


def build(gamma=1.0, min_buffer=2000):
    energy = DistEnergy(gmm)
    source = Delta(dim=D)
    ref = VESDE(sigma_min=0.001, sigma_max=SIGMA_MAX)
    term = ScoreGradTermCost(source=source, ref_sde=ref, energy=energy)
    sde = ControlledSDE(ref, torch.nn.Identity())
    return AdjointVEDivisionMatcher(gamma=gamma, min_buffer=min_buffer, grad_state_cost=ZeroGradStateCost(), grad_term_cost=term, sde=sde, buffer=BatchBuffer(100000),
                                    resample_size=1000, duplicates=1, stats_every=1)


def sample(n, weights, gen):
    k = torch.multinomial(weights, n, replacement=True, generator=gen)
    return gmm.means[k] + torch.randn(n, D, generator=gen)


def insert(m, x1):
    """ populate_buffer with prescribed endpoints (patched sdeint) """
    real = matcher_module.sdeint
    matcher_module.sdeint = lambda sde, x0, ts, only_boundary=True: (x0, x1)
    try:
        m.populate_buffer(torch.zeros(len(x1), D), torch.linspace(0, 1, 10), False)
    finally:
        matcher_module.sdeint = real


def data_of(m, last):
    b = m.buffer.batches
    return {k: torch.cat(b[k][-last:]) for k in ("x0", "x1", "adjoint1", "E1", "version")}


def test_versioning_and_warmup():
    gen = torch.Generator().manual_seed(0)
    m = build(min_buffer=2000)
    for _ in range(2):                                                      # buffer < 2000 at the start of each call: warm-up
        insert(m, sample(1000, gmm.weights, gen))
    assert m.current == -1 and all((v == -1).all() for v in m.buffer.batches["version"])
    # warm-up: the target is exactly the AS target
    data = data_of(m, 2)
    (t, xt), tgt = m.prepare_target(data, "cpu")
    assert torch.allclose(tgt, -data["adjoint1"])
    insert(m, sample(1000, gmm.weights, gen))                              # len(buffer) = 2000 >= min_buffer: refit BEFORE this batch is inserted
    insert(m, sample(1000, gmm.weights, gen))
    insert(m, sample(1000, gmm.weights, gen))
    vers = [int(v[0]) for v in m.buffer.batches["version"]]
    assert vers[:2] == [-1, -1] and vers[2] == 0 and vers[3] == 1 and vers[4] == 2 and m.current == 2, vers
    assert len(m.versions[0]["mu"]) == K, "all modes found"
    data = data_of(m, 5)
    (t, xt), tgt = m.prepare_target(data, "cpu")
    extra = tgt + data["adjoint1"]
    assert torch.isfinite(tgt).all() and float(extra[data["version"] < 0].abs().max()) == 0.0 and float(extra[data["version"] >= 0].abs().max()) > 0
    st = m.pop_stats()
    need = ["div/K_clusters", "div/K_matched_distinct", "div/w_relerr_max_ge1pct", "div/xi_mean", "div/version", "div/warmup_frac"] + [f"div/kappa[{lo},{hi}]/{k}" for lo, hi in (("0", "0.1"), ("0.1", "0.7"), ("0.7", "1"))
                                                                                                                                for k in ("frac_t", "term_over_gradg", "ess_frac", "max_weight_frac")]
    missing = [k for k in need if k not in st and not k.endswith(("term_over_gradg", "ess_frac", "max_weight_frac"))]   # per-bin ratios only exist if the bin holds >= 2 active samples
    assert not missing, missing
    assert st["div/K_clusters"] == K and st["div/K_matched_distinct"] == K and st["div/w_relerr_max_ge1pct"] < 1e-3, st
    print(f"ok  warm-up and versioning: batch versions {vers}; target = AS target during warm-up; K found {len(m.versions[0]['mu'])}; xi mean {m._fit_info['xi_mean']:.3f}")


def end_to_end(weights_buf, seed=1, n_batches=12):
    gen = torch.Generator().manual_seed(seed)
    m = build()
    for _ in range(3):
        insert(m, sample(1000, gmm.weights, gen))                         # first versions from exact-weight endpoints
    for _ in range(n_batches):
        insert(m, sample(1000, weights_buf, gen))                         # inserted AFTER a version exists; exact pi or wrong weights
    data = data_of(m, n_batches)
    assert (data["version"] >= 0).all()
    out = []
    ref = m.sde.ref_sde
    for kappa in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.97):
        nu = kappa * ref.total_var
        t = float(-torch.log(1 - torch.tensor(nu / ref.sigma_max ** 2, dtype=torch.float64)) / (2 * torch.log(torch.tensor(ref.sigma_diff, dtype=torch.float64))))
        tt = torch.full((len(data["x1"]), 1), t, dtype=torch.float32)
        m.sample_t = lambda x, _t=tt: _t
        torch.manual_seed(5)
        (tv, xt), tgt = m.prepare_target(data, "cpu")
        term = (tgt + data["adjoint1"]).double()
        zs = []
        for p in (xt.double(), xt.double() * (xt.double() ** 2).sum(-1, keepdim=True)):
            v = (term * p).sum(-1)
            zs.append(float(v.mean() / (v.std() / len(v) ** 0.5)))
        out.append((kappa, max(abs(z) for z in zs)))
    return out


def test_end_to_end_zero_mean():
    ok = end_to_end(gmm.weights)
    zmax = max(z for _, z in ok)
    wrong = gmm.weights ** 0.3
    wrong = wrong / wrong.sum()
    bad = end_to_end(wrong)
    zctl = max(z for _, z in bad)
    print("    exact pi endpoints: max |z| per kappa " + ", ".join(f"{k:g}:{z:.1f}" for k, z in ok))
    print("    wrong-weight endpoints (positive control): " + ", ".join(f"{k:g}:{z:.1f}" for k, z in bad))
    assert zmax < 3.5 and zctl > 5, (zmax, zctl)
    print(f"ok  end-to-end through the matcher's own bridge sampler and prepare_target: max |z| = {zmax:.2f} (exact endpoints), {zctl:.1f} (wrong-weight endpoints)")


if __name__ == "__main__":
    test_versioning_and_warmup()
    test_end_to_end_zero_mean()
    print("all passed")
