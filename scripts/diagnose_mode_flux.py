""" Why does a given mode stay under-weighted? On a saved dump (GMM target, oracle centres = true means):
      python scripts/diagnose_mode_flux.py --samples K36_g1_oracle.pt --dim 36 --mode 22 --sigma-max 2.4509

    1. geometry of the mode (distance to the other modes, to the target mean) and its weight in the buffer / the model
    2. cos(m_mode - m_bar, Σ_j e_j m_j), e = true weights - buffer masses: does the mean correction point toward the mode?
    3. mean Barker orbit weight ω_mode of the samples of the OTHER modes at small t, against w_mode
    4. net flux into the mode per t-bucket (inflow - outflow, mass per sample), against e_mode = w_mode - ŵ_mode
    Computed with the real AdjointVEBarkerMatcher (mh_term), on a subsample of the buffer.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from test_mh_matcher import make_matcher  # noqa: E402
from analyse_buffer_centres import make_target  # noqa: E402
from adjoint_samplers.components.matcher import AdjointVEBarkerMatcher, _nearest  # noqa: E402


def t_of_kappa(matcher, kappa):
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        t = torch.tensor([mid])
        k = (matcher._nu(t) / matcher._nu(torch.ones_like(t))).item()
        lo, hi = (mid, hi) if k < kappa else (lo, mid)
    return (lo + hi) / 2


@torch.no_grad()
def orbit_weights(matcher, energy, x1, t_vec, chunk=300):
    """ ω (N, M-1), the column of every sample's cluster-to-target translate is returned by the caller; also c (N,) """
    out, valid_all = [], []
    for i in range(0, len(x1), chunk):
        x = x1[i:i + chunk]
        t = t_vec[i:i + chunk].reshape(-1, 1)
        x0 = torch.zeros_like(x)
        xt = matcher.sde.sample_base_posterior(t, x0, x)
        jumps, valid = matcher.propose(x)
        EY = torch.zeros_like(valid)
        mask = valid.bool()
        EY[mask] = energy.eval((x.unsqueeze(1) + jumps)[mask]).reshape(-1)
        E1 = energy.eval(x).reshape(-1, 1)
        _, w = matcher.mh_term(t, x0, xt, x, E1, jumps, EY, valid)
        out.append(w); valid_all.append(valid)
    return torch.cat(out), torch.cat(valid_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--mode", type=int, required=True)
    ap.add_argument("--sigma-max", type=float, required=True)
    ap.add_argument("--n", type=int, default=1500)
    args = ap.parse_args()
    D, j0 = args.dim, args.mode
    g = make_target(D, D, 4.25, 20.0, 0)
    g.K = D
    from adjoint_samplers.energies.dist_energy import DistEnergy
    energy = DistEnergy(g)
    d = torch.load(args.samples, map_location="cpu")
    buf, mod = d["buffer_x1"].float(), d["model_samples"].float()
    w = g.weights.clone()
    w_buf = torch.bincount(g.assign(buf), minlength=D).float() / len(buf)
    w_mod = torch.bincount(g.assign(mod), minlength=D).float() / len(mod)
    print(f"{d.get('exp_name', '?')}: mode {j0} (rang de poids {int((w < w[j0]).sum()) + 1}/{D})")

    print("\n[1] géométrie et poids")
    dist = torch.cdist(g.means, g.means).fill_diagonal_(1e9)
    mbar_t = (w[:, None] * g.means).sum(0)
    mbar_u = g.means.mean(0)
    print(f"    distance au mode le plus proche {dist[j0].min():.2f} (médiane des autres modes: {dist.min(1).values.median():.2f}), "
          f"|m - moyenne cible| {(g.means[j0] - mbar_t).norm():.2f} (médiane des modes {(g.means - mbar_t).norm(dim=1).median():.2f})")
    print(f"    poids cible {w[j0]:.5f} | masse buffer {w_buf[j0]:.5f} (ratio {w_buf[j0] / w[j0]:.2f}) | masse modèle {w_mod[j0]:.5f} (ratio {w_mod[j0] / w[j0]:.2f})")

    print("\n[2] cos(m_j - m_bar, Σ_k e_k m_k), e = poids vrais - masses du buffer")
    for name, ww in (("buffer", w_buf), ("modèle", w_mod)):
        e = w - ww
        F = e @ g.means
        for mb_name, mb in (("moyenne cible", mbar_t), ("moyenne des modes", mbar_u)):
            u = g.means[j0] - mb
            print(f"    e = w - ŵ_{name}, m_bar = {mb_name}: cos = {float(F @ u / (F.norm() * u.norm())):+.3f} (e_j = {e[j0]:+.5f}, |e|_2 = {e.norm():.4f}; "
                  f"cos si seul e_j comptait ≈ {float(e[j0] / e.norm()):+.3f})")

    matcher = make_matcher(g, AdjointVEBarkerMatcher, sigma_max=args.sigma_max, n_clusters=D, gamma=1.0)
    matcher.centres = g.means.clone()
    torch.manual_seed(0)
    idx = torch.randperm(len(buf))[:args.n]
    x1 = buf[idx]
    m1 = g.assign(x1)
    c = _nearest(x1, matcher.centres)
    other = m1 != j0
    col = torch.where(torch.tensor(j0) < c, torch.tensor(j0), torch.tensor(j0 - 1))

    print(f"\n[3] poids d'orbite moyen ω_{j0} des échantillons des AUTRES modes ({int(other.sum())} échantillons), contre w_{j0} = {w[j0]:.5f}")
    for kappa in (0.001, 0.02, 0.05, 0.1, 0.2):
        t = t_of_kappa(matcher, kappa)
        om, valid = orbit_weights(matcher, energy, x1, torch.full((len(x1),), t))
        o22 = om[torch.arange(len(x1)), col.clamp(0, om.shape[1] - 1)]
        sel = other
        print(f"    κ = {kappa:5.3f} (t = {t:.4f}): moyenne ω_{j0} = {float(o22[sel].mean()):.5f}  (ω/w = {float(o22[sel].mean() / w[j0]):.2f}), "
              f"traduction valide pour {float(valid[torch.arange(len(x1)), col.clamp(0, om.shape[1] - 1)][sel].mean()):.2f} des échantillons")

    print(f"\n[4] flux net vers le mode {j0} par tranche de t (masse par échantillon) contre e_j = w_j - ŵ_j = {w[j0] - w_buf[j0]:+.5f}")
    edges = AdjointVEBarkerMatcher.T_EDGES
    e_j = float(w[j0] - w_buf[j0])
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > 0.5:
            continue
        t_vec = lo + (hi - lo) * torch.rand(len(x1))
        om, valid = orbit_weights(matcher, energy, x1, t_vec)
        o22 = om[torch.arange(len(x1)), col.clamp(0, om.shape[1] - 1)]
        inflow = float(o22[other].sum() / len(x1))
        outflow = float(om[~other].sum() / len(x1))
        # mass moved between true modes only (translates in the same mode do nothing)
        print(f"    t in [{lo:g}, {hi:g}]: inflow {inflow:.5f}, outflow {outflow:.5f}, net {inflow - outflow:+.5f}  (net / e_j = {(inflow - outflow) / e_j:.2f})")


if __name__ == "__main__":
    main()
