""" CPU diagnostic (rules_mos_step1.md, amendment 1) on the MoS d = 50 controller of step 1 (sigma_max 23.2). Descriptive, no rule.
    python scripts/mos_diag.py --part 1|2|3a|3b|4    (each < 1 min) """
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import dart_lib as L

T0 = time.perf_counter()
ap = argparse.ArgumentParser(); ap.add_argument("--part", required=True); ap.add_argument("--n", type=int, default=10000); ap.add_argument("--steps", type=int, default=150); ap.add_argument("--tag", default="")
args = ap.parse_args()
K = 10
TR = json.load(open(ROOT / "figures/mog_mos_truth_mos_d50.json")); ptrue = np.array(TR["p"]); floor = TR["floor_n"]
MU = np.load(ROOT / "data/mog_mos_means.npz")["mos_K10_d50"]
pi = L.Mix("S", MU, np.full(K, 1.0 / K), 1.0)
HS = json.load(open(ROOT / "figures/mos_step1_a_n10000.json"))["mala_step"]               # MALA step tuned in step 1 (same samples, same protocol)
Xas = torch.load(ROOT / "results/mos_pilot/r/mospilot_s23.2/samples_final.pt", weights_only=False)["model_samples"].double().numpy()
rng = np.random.default_rng(2027)


def exact(n):
    comp = rng.choice(K, size=n)
    return MU[comp] + rng.standard_t(2.0, size=(n, 50))


def minima(X):
    y = X[:500].copy()
    for it in range(4000):
        _, g = pi.logp_grad(y); y = y + 0.7 * g
        if it >= 5 and it % 5 == 0 and (np.linalg.norm(g, axis=1) < 1e-3).mean() >= 0.95: break
    mins, cnt = [], []
    for p in y:
        if mins:
            dist = np.linalg.norm(np.array(mins) - p, axis=1); i = int(dist.argmin())
            if dist[i] < 0.5: cnt[i] += 1; continue
        mins.append(p.copy()); cnt.append(1)
    M = np.array([m for m, c in zip(mins, cnt) if c >= 5])
    for _ in range(50):
        _, g = pi.logp_grad(M); M = M + 0.7 * g
    return M


def chain(X, M, S, record=False, seed=99):
    """ protocol chain (1 jump + 5 MALA per sweep, translation); optionally records, per jump proposal: validity, MH acceptance, log pi(y) - log pi(x), distance of x to its minimum. """
    n = len(X); Mk = len(M); M2 = (M ** 2).sum(1); vor = lambda Z: (-2 * Z @ M.T + M2[None, :]).argmin(1)
    rd = np.random.default_rng(seed); Xc = X.copy(); lp, g = pi.logp_grad(Xc); rec = []
    for sweep in range(S):
        a = vor(Xc); b = (a + 1 + rd.integers(Mk - 1, size=n)) % Mk
        Y = Xc - M[a] + M[b]; valid = vor(Y) == b; lpy = pi.logp(Y)
        dx = np.linalg.norm(Xc - M[a], axis=1)
        acc = valid & (np.log(rd.random(n)) < lpy - lp)
        if record: rec.append(dict(valid=valid.copy(), acc=acc.copy(), logr=(lpy - lp).copy(), dist=dx.copy()))
        if acc.any():
            _, gy = pi.logp_grad(Y[acc]); Xc[acc] = Y[acc]; lp = np.where(acc, lpy, lp); g[acc] = gy
        for _s in range(5):
            Xp = Xc + HS * g + math.sqrt(2 * HS) * rd.standard_normal(Xc.shape); lpp, gp = pi.logp_grad(Xp)
            al = lpp - lp - ((Xc - Xp - HS * gp) ** 2).sum(1) / (4 * HS) + ((Xp - Xc - HS * g) ** 2).sum(1) / (4 * HS); am = np.log(rd.random(n)) < al
            Xc = np.where(am[:, None], Xp, Xc); lp = np.where(am, lpp, lp); g = np.where(am[:, None], gp, g)
    return Xc, rec


def basin_assign(X, M, steps=150):
    y = X.copy()
    for _ in range(steps):
        _, g = pi.logp_grad(y); y = y + 0.7 * g
    d = ((y[:, None, :] - M[None]) ** 2).sum(-1)
    return d.argmin(1), float(np.mean(d.min(1) < 1.0))


if args.part == "1":
    M = minima(Xas); Xc, rec = chain(Xas, M, 10, record=True)
    valid = np.concatenate([r["valid"] for r in rec]); acc = np.concatenate([r["acc"] for r in rec]); logr = np.concatenate([r["logr"] for r in rec])
    tot = len(valid); rej_mh = valid & ~acc; inval = ~valid
    print(f"part 1: minima {len(M)}; {tot} jump proposals over 10 sweeps: accepted {acc.mean():.3f}; invalid (image outside cell j) {inval.mean():.3f}; valid but refused by MH {rej_mh.mean():.3f}")
    q = np.quantile(logr[rej_mh], [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]) if rej_mh.any() else []
    qa = np.quantile(logr[acc], [0.01, 0.1, 0.5, 0.9, 0.99]) if acc.any() else []
    print("   log pi(y)/pi(x) of MH-refused jumps, quantiles 1/10/25/50/75/90/99 %:", np.round(q, 2).tolist())
    print("   log pi(y)/pi(x) of accepted jumps, quantiles 1/10/50/90/99 %:", np.round(qa, 2).tolist())
    print(f"   by sweep: accepted " + " ".join(f"{r['acc'].mean():.2f}" for r in rec) + " | invalid " + " ".join(f"{(~r['valid']).mean():.2f}" for r in rec))
    json.dump(dict(accepted=float(acc.mean()), invalid=float(inval.mean()), mh_refused=float(rej_mh.mean()), refused_logr_q=np.asarray(q).tolist(), accepted_logr_q=np.asarray(qa).tolist(), by_sweep_acc=[float(r["acc"].mean()) for r in rec]), open(ROOT / "figures/mos_diag_1.json", "w"), indent=1)
elif args.part == "2":
    M = minima(Xas); Xex = exact(args.n)
    res = {}
    for name, X in (("AS", Xas), ("exact", Xex)):
        _, rec = chain(X, M, 1, record=True); res[name] = rec[0]
    edges = np.quantile(res["exact"]["dist"], np.linspace(0, 1, 11)); edges[0] = 0; edges[-1] = np.inf
    print(f"part 2: distance of x to its minimum: median AS {np.median(res['AS']['dist']):.2f}, exact {np.median(res['exact']['dist']):.2f}; first-jump acceptance (no MALA before) overall AS {res['AS']['acc'].mean():.3f}, exact {res['exact']['acc'].mean():.3f}; invalid AS {(~res['AS']['valid']).mean():.3f}, exact {(~res['exact']['valid']).mean():.3f}")
    print("   decile of distance (edges from the exact samples) | share of AS samples | acceptance AS | acceptance exact")
    out = []
    for i in range(10):
        m_a = (res["AS"]["dist"] >= edges[i]) & (res["AS"]["dist"] < edges[i + 1]); m_e = (res["exact"]["dist"] >= edges[i]) & (res["exact"]["dist"] < edges[i + 1])
        row = (i + 1, float(m_a.mean()), float(res["AS"]["acc"][m_a].mean()) if m_a.any() else float("nan"), float(res["exact"]["acc"][m_e].mean()) if m_e.any() else float("nan")); out.append(row)
        print(f"   {row[0]:2d} [{edges[i]:6.1f}, {edges[i + 1]:6.1f}) | {row[1]:.3f} | {row[2]:.3f} | {row[3]:.3f}")
    json.dump(dict(edges=edges.tolist(), rows=out), open(ROOT / "figures/mos_diag_2.json", "w"), indent=1)
elif args.part in ("3a", "3b"):
    M = minima(Xas); n = args.n
    X = chain(Xas, M, 40)[0][:n] if args.part == "3a" else exact(n)
    vor_c = L.cells(X, MU); basin, frac_conv = basin_assign(X, M, args.steps)
    # map each discovered minimum to the nearest true mean (label) so that basins and Voronoi cells share labels
    lab = np.array([int(np.argmin(((MU - m[None]) ** 2).sum(1))) for m in M]); basin_lab = lab[basin]
    json.dump(dict(vor=L.props(vor_c, K).tolist(), basin=L.props(basin_lab, K).tolist(), spill=float((vor_c != basin_lab).mean()), frac_converged=frac_conv, n=n, steps=args.steps), open(ROOT / f"figures/mos_diag_{args.part}{args.tag}.json", "w"), indent=1)
    print(f"part {args.part}: n={n}; share of samples whose Voronoi cell differs from the quench basin {np.mean(vor_c != basin_lab):.3f}; quench converged within 1.0 of a minimum for {frac_conv:.3f}")
elif args.part == "4":
    Xex = exact(args.n); r = L.darting(pi, Xex, MU, ptrue, SW=(10, 40))
    print(f"part 4: darting from EXACT samples: raw TV {L.tv(L.props(L.cells(Xex, MU), K), ptrue):.4f}; floor {floor:.4f}; minima {r['minima_found']}; " + " ".join(f"S={S}: TV {v['tv']:.4f} jump acc {v['jump_acc']:.3f}" for S, v in r["by_S"].items()))
    json.dump(dict(raw=L.tv(L.props(L.cells(Xex, MU), K), ptrue), floor=floor, by_S=r["by_S"], minima=r["minima_found"]), open(ROOT / "figures/mos_diag_4.json", "w"), indent=1, default=float)
print(f"time {time.perf_counter() - T0:.1f} s")
