""" Offline analysis of the centres of the Barker / MH control variate on a saved buffer (GMM target).

    python scripts/analyse_buffer_centres.py --samples results/.../samples_final.pt --dim 36 [--focus 22 12]
    python scripts/analyse_buffer_centres.py --simulate-run <wandb run id> --dim 36    # buffer/model resampled from the
                                                                                         # GMM with the logged mode ratios

    1. reachable coverage: true modes with >= 1 buffer sample (what no data-driven clustering can beat)
    2. k-means on the denoised points (fresh inits and a warm-started chain over buffer refreshes): coverage, parasite
       centres, split / merged modes, where the centre of a focus mode goes
    3. leader (threshold) clustering after n_denoise denoising steps x <- x + ∇log p1(x): coverage and parasites
    4. energy W2: per mode, and after reweighting the samples to the true mode weights
    Light: seconds on 10000 points. Never needs the cluster.
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adjoint_samplers.components.matcher import _kmeans, _threshold_centres  # noqa: E402
from adjoint_samplers.energies.dist_energy import DistEnergy  # noqa: E402
from adjoint_samplers.utils.dist_utils import GMM  # noqa: E402


def make_target(dim, K, radius, weight_ratio, seed):
    return GMM(dim=dim, K=K, seed=seed, layout="spread", weight_law="loguniform", weight_ratio=weight_ratio, radius=radius)


def with_weights(gmm, weights):
    other = copy.copy(gmm)
    other.weights = weights / weights.sum()
    return other


def denoise(g, x, n):
    for _ in range(n):
        x = x + g.score(x) # = E[m | x] for unit-covariance modes
    return x


def coverage(g, c, tol=1.0):
    d = torch.cdist(g.means, c)                      # (modes, centres)
    covered = d.min(1).values < tol
    spurious = torch.cdist(c, g.means).min(1).values >= tol
    dup = ((d < tol).sum(1) >= 2)
    return covered, spurious, dup


def fmt(g, c):
    cov, spur, dup = coverage(g, c)
    return (f"{len(c):3d} centres | modes couverts {cov.float().mean():.2f} ({int((~cov).sum())} sans centre)"
            f" | parasites {int(spur.sum())} | modes à >=2 centres {int(dup.sum())}")


def section1(g, x, label):
    a = g.assign(x)
    counts = torch.bincount(a, minlength=g.K)
    exp = g.weights * len(x)
    print(f"\n[1] Couverture atteignable ({label}, {len(x)} échantillons)")
    for k in (1, 3, 10):
        print(f"    modes avec >= {k:2d} échantillon(s): {int((counts >= k).sum())}/{g.K}")
    miss = torch.nonzero(counts < 3).flatten().tolist()
    rank = torch.argsort(torch.argsort(g.weights)).tolist()
    print("    modes avec < 3 échantillons (rang de poids, poids, attendus, observés): "
          + (", ".join(f"{j} (#{rank[j] + 1}, {g.weights[j]:.4f}, {exp[j]:.1f}, {int(counts[j])})" for j in miss) or "aucun"))
    return counts


def section2(g, x, focus, n_denoise_kmeans=1, n_fresh=5, chain=8, window=5000):
    pts = denoise(g, x, n_denoise_kmeans)
    print(f"\n[2] k-means (M = K = {g.K}) sur les points débruités ({n_denoise_kmeans} pas)")
    for s in range(n_fresh):
        torch.manual_seed(s)
        print(f"    init aléatoire {s}: " + fmt(g, _kmeans(pts, g.K, None, 20)))
    # chaîne à chaud : rafraîchissements successifs sur des fenêtres glissantes du buffer, comme pendant l'entraînement
    torch.manual_seed(0)
    centres, step = None, max((len(pts) - window) // max(chain - 1, 1), 1)
    for r in range(chain):
        chunk = pts[r * step: r * step + window]
        centres = _kmeans(chunk, g.K, centres, 20)
    print(f"    chaîne à chaud ({chain} rafraîchissements, fenêtres de {window}): " + fmt(g, centres))
    # typologie des défauts
    cov, spur, dup = coverage(g, centres)
    near = torch.cdist(pts, centres).argmin(-1)
    assign = g.assign(x)
    print("    centres parasites (membres par mode vrai, 2 plus gros):")
    for i in torch.nonzero(spur).flatten().tolist():
        m = torch.bincount(assign[near == i], minlength=g.K)
        top = torch.topk(m, 2)
        print(f"      centre {i}: {int((near == i).sum())} points; modes {top.indices.tolist()} ({top.values.tolist()})")
    for j in focus:
        d = torch.cdist(g.means[j:j + 1], centres)[0]
        i = int(d.argmin())
        m = torch.bincount(assign[near == i], minlength=g.K)
        print(f"    mode {j}: centre le plus proche {i} à {d[i]:.2f}; ce centre capte {int(m.sum())} points "
              f"(mode {j}: {int(m[j])}, autres modes dans le centre: {[(int(k), int(m[k])) for k in torch.nonzero(m).flatten() if int(k) != j][:5]})")
    return centres


def section3(g, x, n_denoise_list=(1, 2, 3, 4, 5), mincounts=(1, 3)):
    sep = torch.cdist(g.means, g.means).fill_diagonal_(1e9).min().item()
    print(f"\n[3] Clustering par seuil (leader) après n pas de débruitage; séparation minimale des vraies moyennes = {sep:.2f}")
    for n in n_denoise_list:
        pts = denoise(g, x, n)
        dm = torch.cdist(pts, g.means).min(1).values
        row = []
        for frac in (0.3, 0.4):
            for mc in mincounts:
                torch.manual_seed(0)
                c, _ = _threshold_centres(pts, frac * sep, min_count=mc)
                row.append(f"r={frac}sep mc={mc}: {fmt(g, c)}")
        print(f"    n_denoise={n} (dist max point-mode {dm.max():.2f}, q99 {dm.quantile(0.99):.2f})")
        for r in row:
            print(f"        {r}")


def section4(g, model_x, n_ref=20000):
    from scipy.stats import wasserstein_distance
    energy = DistEnergy(g)
    with torch.no_grad():
        torch.manual_seed(1)
        ref = g.sample([n_ref])
        e_ref, e_mod = energy.eval(ref).reshape(-1).numpy(), energy.eval(model_x).reshape(-1).numpy()
    a_ref, a_mod = g.assign(ref).numpy(), g.assign(model_x).numpy()
    w = g.weights.numpy()
    w_hat = np.bincount(a_mod, minlength=g.K) / len(a_mod)
    print(f"\n[4] energy W2 ({len(model_x)} échantillons du modèle contre {n_ref} exacts)")
    print(f"    global: {wasserstein_distance(e_ref, e_mod):.3f} (W1 des énergies ; la métrique du run est le W2 1D, voir plus bas)")

    def w2_1d(a, b, wa=None, wb=None, n=2000):
        q = (np.arange(n) + 0.5) / n
        def quant(v, wv):
            o = np.argsort(v); v = v[o]; cw = np.cumsum(wv[o] if wv is not None else np.ones_like(v)); cw = cw / cw[-1]
            return np.interp(q, cw, v)
        return float(np.sqrt(np.mean((quant(a, wa) - quant(b, wb)) ** 2)))
    print(f"    W2 global (quantiles): {w2_1d(e_ref, e_mod):.3f}")
    rw = (w / np.maximum(w_hat, 1e-6))[a_mod]
    print(f"    W2 global après repondération des échantillons aux poids vrais: {w2_1d(e_ref, e_mod, None, rw):.3f}")
    per = [(j, w2_1d(e_ref[a_ref == j], e_mod[a_mod == j])) for j in range(g.K) if (a_ref == j).sum() > 20 and (a_mod == j).sum() > 20]
    vals = np.array([v for _, v in per])
    print(f"    W2 par mode (énergie, {len(per)} modes): moyenne {vals.mean():.3f}, médiane {np.median(vals):.3f}, max {vals.max():.3f} (mode {per[int(vals.argmax())][0]})")
    order = np.argsort(w)
    light = [v for j, v in per if j in set(order[:g.K // 4].tolist())]
    heavy = [v for j, v in per if j in set(order[-(g.K // 4):].tolist())]
    print(f"    W2 par mode: quart léger {np.mean(light) if light else float('nan'):.3f} ({len(light)} modes), quart lourd {np.mean(heavy) if heavy else float('nan'):.3f} ({len(heavy)} modes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=None, help="samples_final.pt (buffer_x1, model_samples)")
    ap.add_argument("--simulate-run", default=None, help="W&B run id: resample buffer / model from the GMM with the logged ratios")
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--radius", type=float, default=4.25)
    ap.add_argument("--weight-ratio", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0, help="target seed")
    ap.add_argument("--focus", type=int, nargs="*", default=[])
    ap.add_argument("--n-denoise-kmeans", type=int, default=1)
    args = ap.parse_args()
    K = args.K or args.dim
    g = make_target(args.dim, K, args.radius, args.weight_ratio, args.seed)
    g.K = K

    if args.samples:
        d = torch.load(args.samples, map_location="cpu")
        buf, mod = d["buffer_x1"].float(), d["model_samples"].float()
        label = f"buffer final de {d.get('exp_name', '?')}"
    else:
        import wandb
        run = wandb.Api(timeout=60).run(f"CCDS_NTU/adjoint_sampler_cv/{args.simulate_run}")
        rb = torch.tensor([run.summary[f"buffer/mode_ratio/{j}"] for j in range(K)], dtype=torch.float32)
        rm = torch.tensor([run.summary[f"mode_ratio/{j}"] for j in range(K)], dtype=torch.float32)
        torch.manual_seed(0)
        buf, mod = with_weights(g, g.weights * rb).sample([10000]), with_weights(g, g.weights * rm).sample([10000])
        label = f"SIMULATION d'après les rapports de {run.name}"
    print(f"GMM d={args.dim} K={K} rayon {args.radius} rapport de poids {args.weight_ratio}; {label}")
    section1(g, buf, "buffer")
    section2(g, buf, args.focus, args.n_denoise_kmeans)
    section3(g, buf)
    section4(g, mod)


if __name__ == "__main__":
    main()
