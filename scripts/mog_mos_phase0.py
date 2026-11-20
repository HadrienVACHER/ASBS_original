""" Phase 0 of rules_mog_mos_phase0.md: darting post hoc (jump + 5 MALA steps per sweep) versus exact-weight IS at equal cost, on the Beyond ELBOs MoG (K = 40, N(mu_k, I)) and MoS (K = 10, product of 1D t_2, scale 1),
    synthetic samplers (no network). CPU only, one (target, d, case) per execution (< 1 min).
      python scripts/mog_mos_phase0.py --target mog|mos|mogextra --d D --part truth     # 1e6-draw Voronoi truth + floor(n = 1e4), saved to figures/mog_mos_truth_<target>_d<D>.json
      python scripts/mog_mos_phase0.py --target mog|mos|mogextra --d D --part run [--discover 500]
    Means: data/mog_mos_means.npz (jax 0.4.28, the repository's generator). """
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import genpareto

T0 = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True); ap.add_argument("--d", type=int, required=True); ap.add_argument("--part", default="run")
ap.add_argument("--discover", type=int, default=500); ap.add_argument("--n", type=int, default=10_000); ap.add_argument("--is_cap", type=int, default=0)
args = ap.parse_args()
MEANS = np.load(ROOT / "data/mog_mos_means.npz")
NU = 2.0
CT = gammaln((NU + 1) / 2) - gammaln(NU / 2) - 0.5 * math.log(NU * math.pi)          # constant of the t_2 density, scale 1
DIL = 0.1
SW = [1, 5, 10]
CHUNK = 20_000


class Mix:
    """ isotropic-Gaussian mixture ('G', std) or mixture of products of 1D t_2 with scale 1 ('S'). """
    def __init__(self, fam, mu, w, std=1.0):
        self.fam, self.mu, self.w, self.std = fam, mu, np.asarray(w, float), std
        self.K, self.d = mu.shape
        self.logw = np.log(self.w); self.mu2 = (mu ** 2).sum(1)

    def comp(self, X, f32=False):
        if self.fam == "G":
            sq = (X ** 2).sum(1)[:, None] - 2 * X @ self.mu.T + self.mu2[None, :]
            return -0.5 * sq / self.std ** 2 - self.d * math.log(self.std) - 0.5 * self.d * math.log(2 * math.pi)
        out = np.empty((len(X), self.K))
        Xf = X.astype(np.float32) if f32 else X
        for k in range(self.K):
            u = Xf - (self.mu[k].astype(np.float32) if f32 else self.mu[k])
            out[:, k] = self.d * CT - (NU + 1) / 2 * np.log1p(u * u / NU).sum(1, dtype=np.float64)
        return out

    def logp(self, X, f32=False):
        return logsumexp(self.logw + self.comp(X, f32), axis=1)

    def logp_grad(self, X):
        lc = self.logw + self.comp(X)
        lp = logsumexp(lc, axis=1)
        r = np.exp(lc - lp[:, None])
        if self.fam == "G":
            return lp, (r @ self.mu - X * r.sum(1)[:, None]) / self.std ** 2
        g = np.zeros_like(X)
        for k in range(self.K):
            u = X - self.mu[k]
            g += r[:, k:k + 1] * (-(NU + 1) * u / (NU + u * u))
        return lp, g

    def sample(self, rng, n):
        comp = rng.choice(self.K, size=n, p=self.w)
        z = rng.standard_normal((n, self.d)) if self.fam == "G" else rng.standard_t(NU, size=(n, self.d))
        return self.mu[comp] + self.std * z


def cells(X, mu):
    return (-2 * X @ mu.T + (mu ** 2).sum(1)[None, :]).argmin(1)


def props(c, K, w=None):
    return np.bincount(c, weights=w, minlength=K) / (len(c) if w is None else w.sum())


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def khat(lw):
    w = np.exp(lw - lw.max()); w = np.sort(w)
    M = int(min(0.2 * len(w), 3 * math.sqrt(len(w))))
    tail = w[-M:] - w[-M - 1]; tail = tail[tail > 0]
    return float(genpareto.fit(tail, floc=0)[0]) if len(tail) >= 10 else float("nan")


def is_stats(lw, c, K, ptrue):
    w = np.exp(lw - lw.max())
    return dict(tv=tv(props(c, K, w), ptrue), ess=float(w.sum() ** 2 / (len(w) * (w ** 2).sum())), khat=khat(lw))


def build(name, d):
    if name == "mog":
        mu = MEANS[f"mog_K40_d{d}"]; w = np.full(40, 1 / 40); return Mix("G", mu, w, 1.0), mu
    if name == "mos":
        mu = MEANS[f"mos_K10_d{d}"]; w = np.full(10, 1 / 10); return Mix("S", mu, w), mu
    if name == "mogextra":
        mu = MEANS[f"mogextra_K5_d{d}"]; w = np.full(5, 1 / 5); return Mix("G", mu, w, 1.0), mu
    raise ValueError(name)


def build_q(name, pi, mu):
    K = len(mu)
    if name == "mogextra":                                                      # one mode (index 0) present but very under-represented
        w = np.full(K, (1 - 1e-3) / (K - 1)); w[0] = 1e-3
        return Mix("G", mu, w, 1.0 + DIL)
    perm = np.random.default_rng(12345).permutation(K)                          # random geometric weights, ratio 10 (target weights are equal)
    w = 10 ** (-perm / (K - 1)); w = w / w.sum()
    if name == "mog":
        return Mix("G", mu, w, 1.0 + DIL)
    return Mix("G", mu, w, math.sqrt(1.0 / ((NU + 1) / NU)))                     # same Hessian at the minimum as t_2 (H = 3/2): std^2 = 2/3


pi, mu = build(args.target, args.d)
K, d, n = pi.K, pi.d, args.n
tag = f"{args.target}_d{d}"
truth_path = ROOT / f"figures/mog_mos_truth_{tag}.json"

if args.part == "truth":
    rng = np.random.default_rng(1)
    cnt = np.zeros(K); N6 = 1_000_000
    agree, n_agree = 0.0, 0
    for i in range(0, N6, 50_000):
        X = pi.sample(rng, 50_000); c = cells(X, mu); cnt += np.bincount(c, minlength=K)
        if i == 0:                                                              # exploratory: Voronoi versus argmax of the component density (Beyond ELBOs mode descriptors)
            agree = float((pi.comp(X[:5000]).argmax(1) == c[:5000]).mean())
    ptrue = cnt / N6
    fl = [tv(props(cells(pi.sample(rng, n), mu), K), ptrue) for _ in range(100)]
    json.dump(dict(target=args.target, d=d, K=K, p=ptrue.tolist(), floor_n=float(np.median(fl)), floor_p5=float(np.quantile(fl, 0.05)), floor_p95=float(np.quantile(fl, 0.95)), voronoi_vs_argmax_agreement=agree,
                   min_p=float(ptrue.min()), max_p=float(ptrue.max()), time_s=time.perf_counter() - T0), open(truth_path, "w"), indent=1)
    print(f"truth {tag}: floor(1e4) {np.median(fl):.4f}; p in [{ptrue.min():.4f}, {ptrue.max():.4f}]; Voronoi/argmax agreement {agree:.3f}; {time.perf_counter() - T0:.1f} s")
    sys.exit(0)

TR = json.load(open(truth_path)); ptrue = np.array(TR["p"]); floor = TR["floor_n"]
q = build_q(args.target, pi, mu)
rng = np.random.default_rng(777 + d)
X = q.sample(rng, n)
c0 = cells(X, mu)
res = dict(target=args.target, d=d, K=K, n=n, floor=floor, discover=args.discover, q_weights_ratio=float(q.w.max() / q.w.min()))
lw_n = pi.logp(X) - q.logp(X)
res["IS_n"] = is_stats(lw_n, c0, K, ptrue)
res["tv_q"] = tv(props(c0, K), ptrue)

# ---- minima and Hessians by quench of discover q samples (no oracle)
cost = dict(E=0, G=0)
Xd = X[:args.discover] if args.discover <= n else q.sample(rng, args.discover)
y = Xd.copy(); steps = 0
for it in range(4000):
    _, g = pi.logp_grad(y); cost["G"] += len(y); steps += 1
    y = y + 0.7 * g                                                             # gradient ascent on log pi = descent on the energy
    if it >= 5 and it % 5 == 0 and (np.linalg.norm(g, axis=1) < 1e-3).mean() >= 0.95:
        break
mins, cnt = [], []
for p in y:
    if mins:
        dist = np.linalg.norm(np.array(mins) - p, axis=1); i = int(dist.argmin())
        if dist[i] < 0.5: cnt[i] += 1; continue
    mins.append(p.copy()); cnt.append(1)
M = np.array([m for m, c in zip(mins, cnt) if c >= 5]); res["minima_found"] = int(len(M))
for _ in range(50):
    if len(M) == 0: break
    _, g = pi.logp_grad(M); cost["G"] += len(M); M = M + 0.7 * g
res["time_quench_s"] = time.perf_counter() - T0
if len(M) < 2:
    res["darting"] = None; res["error"] = f"{len(M)} minima discovered: darting cannot jump"
else:
    h = 1e-4; Hs = []
    for m in M:
        E = np.eye(d) * h
        _, gp = pi.logp_grad(m[None, :] + E); _, gm = pi.logp_grad(m[None, :] - E); cost["G"] += 2 * d
        H = -(gp - gm).T / (2 * h); Hs.append(0.5 * (H + H.T))
    Hs = np.array(Hs)
    same = float(np.abs(Hs - Hs[0][None]).max())
    res["hessian_spread_max"] = same
    Mk = len(M)
    translation = same < 1e-6
    res["map"] = "translation (equal Hessians to 1e-6)" if translation else "general Hessian map"
    if not translation:
        assert Mk * Mk * d * d * 8 < 6e8, "general maps too large for this execution"
        ev, V = np.linalg.eigh(Hs)
        Hh = [(V[k] * np.sqrt(ev[k])) @ V[k].T for k in range(Mk)]; Hmh = [(V[k] / np.sqrt(ev[k])) @ V[k].T for k in range(Mk)]
        logdet = np.log(ev).sum(1); Amap = {(a, b): Hmh[b] @ Hh[a] for a in range(Mk) for b in range(Mk) if a != b}
    M2 = (M ** 2).sum(1)

    def vor(Z):
        return (-2 * Z @ M.T + M2[None, :]).argmin(1)

    def mala(Xc, lp, g, hstep, rng_):
        z = rng_.standard_normal(Xc.shape)
        Xp = Xc + hstep * g + math.sqrt(2 * hstep) * z
        lpp, gp = pi.logp_grad(Xp)
        a = lpp - lp - ((Xc - Xp - hstep * gp) ** 2).sum(1) / (4 * hstep) + ((Xp - Xc - hstep * g) ** 2).sum(1) / (4 * hstep)
        acc = np.log(rng_.random(len(Xc))) < a
        return np.where(acc[:, None], Xp, Xc), np.where(acc, lpp, lp), np.where(acc[:, None], gp, g), acc

    # MALA step tuned ONCE (bisection on log step, 8 iterations, 500 q samples, no jumps) for an acceptance of 0.6
    pil = X[:500].copy(); lp0, g0 = pi.logp_grad(pil); cost["E"] += 500; cost["G"] += 500
    lo, hi = math.log(1e-8), math.log(10.0); rt = np.random.default_rng(5)
    for _ in range(8):
        mid = 0.5 * (lo + hi); Xc, lp, g = pil.copy(), lp0.copy(), g0.copy(); accs = []
        for _s in range(5):
            Xc, lp, g, a_ = mala(Xc, lp, g, math.exp(mid), rt); accs.append(a_.mean()); cost["E"] += 500; cost["G"] += 500
        if np.mean(accs) > 0.6: lo = mid
        else: hi = mid
    hstep = math.exp(0.5 * (lo + hi)); res["mala_step"] = hstep
    res["mala_tuning_acceptance"] = float(np.mean(accs))
    cost_setup = dict(cost)

    # ---- darting chains: 1 jump + 5 MALA steps per sweep, snapshots at S = 1, 5, 10
    rd = np.random.default_rng(99)
    Xc = X.copy(); lp, g = pi.logp_grad(Xc); cost["E"] += n; cost["G"] += n
    jump_acc = jump_prop = mala_acc = mala_prop = 0
    snaps = {}
    for sweep in range(1, max(SW) + 1):
        a = vor(Xc); b = (a + 1 + rd.integers(Mk - 1, size=n)) % Mk
        if translation:
            Y = Xc - M[a] + M[b]; logJ = 0.0
        else:
            Y = np.empty_like(Xc)
            for a0 in range(Mk):
                for b0 in range(Mk):
                    if a0 == b0: continue
                    idx = np.nonzero((a == a0) & (b == b0))[0]
                    if len(idx): Y[idx] = M[b0] + (Xc[idx] - M[a0]) @ Amap[(a0, b0)].T
            logJ = 0.5 * (logdet[a] - logdet[b])
        valid = vor(Y) == b
        lpy = pi.logp(Y); cost["E"] += n
        acc = valid & (np.log(rd.random(n)) < lpy - lp + logJ)
        jump_acc += int(acc.sum()); jump_prop += n
        if acc.any():
            _, gy = pi.logp_grad(Y[acc]); cost["G"] += int(acc.sum())            # gradient at an accepted jump (cached state for MALA)
            Xc[acc] = Y[acc]; lp = np.where(acc, lpy, lp); g[acc] = gy
        for _s in range(5):
            Xc, lp, g, am = mala(Xc, lp, g, hstep, rd); mala_acc += int(am.sum()); mala_prop += n; cost["E"] += n; cost["G"] += n
        if sweep in SW:
            pk = props(cells(Xc, mu), K)
            snaps[sweep] = dict(tv=tv(pk, ptrue), jump_acc=jump_acc / jump_prop, mala_acc=mala_acc / mala_prop, cost=cost["E"] + cost["G"], cost_dart_only=cost["E"] + cost["G"] - sum(cost_setup.values()) + sum(cost_setup.values()))
    res["darting"] = snaps
    res["cost_setup"] = cost_setup

# ---- equal-cost IS comparator (nested prefixes of one q sample)
if res.get("darting"):
    need = {S: int(res["darting"][S]["cost"]) for S in SW}
    nmax = max(need.values())
    if args.is_cap: nmax = min(nmax, args.is_cap)
    lws, cs = [], []
    t_is = time.perf_counter(); done = 0; truncated = False
    while done < nmax:
        m = min(CHUNK, nmax - done)
        Xi = q.sample(rng, m)
        lws.append(pi.logp(Xi, f32=True) - q.logp(Xi, f32=True)); cs.append(cells(Xi, mu)); done += m
        if len(lws) == 1:
            proj = (time.perf_counter() - t_is) * nmax / m
            if time.perf_counter() - T0 + proj > 52:                           # time guard: cap the comparator and say so
                nmax = int(max(CHUNK, (52 - (time.perf_counter() - T0)) / (time.perf_counter() - t_is) * m * 0.9)); truncated = True
    lw_all, c_all = np.concatenate(lws), np.concatenate(cs)
    res["IS_equal_cost"] = {}
    for S in SW:
        nn = need[S]
        if nn <= len(lw_all):
            res["IS_equal_cost"][S] = dict(n_IS=nn, **is_stats(lw_all[:nn], c_all[:nn], K, ptrue))
        else:
            res["IS_equal_cost"][S] = dict(n_IS=nn, truncated=True, n_available=len(lw_all), **is_stats(lw_all, c_all, K, ptrue))
    G = {}
    for S in SW:
        td, ti = res["darting"][S]["tv"], res["IS_equal_cost"][S]["tv"]
        G[S] = bool(td <= 0.5 * ti and td <= 2 * floor and not res["IS_equal_cost"][S].get("truncated", False))
    res["G"] = dict(any=bool(any(G.values())), by_S=G, S10=G[10])
res["time_s"] = time.perf_counter() - T0
out = ROOT / f"figures/mog_mos_phase0_{tag}_disc{args.discover}.json"
json.dump(res, open(out, "w"), indent=1, default=float)
dd = res.get("darting")
print(f"{tag} [disc {args.discover}]: floor {floor:.4f} | TV q {res['tv_q']:.3f} | IS(n) TV {res['IS_n']['tv']:.4f} ESS {res['IS_n']['ess']:.4f} khat {res['IS_n']['khat']:.2f} | minima {res['minima_found']}/{K}")
if dd:
    print("  darting " + " | ".join(f"S{S}: TV {dd[S]['tv']:.4f} jump acc {dd[S]['jump_acc']:.2f} mala acc {dd[S]['mala_acc']:.2f} cost {dd[S]['cost']:.2e} -> IS({res['IS_equal_cost'][S]['n_IS']:.2e}) TV {res['IS_equal_cost'][S]['tv']:.4f} ESS {res['IS_equal_cost'][S]['ess']:.4f}" + ("*" if res['IS_equal_cost'][S].get('truncated') else "") for S in SW))
    print(f"  map: {res['map']}; mala step {res['mala_step']:.3g} (tuning acc {res['mala_tuning_acceptance']:.2f}); G by S {res['G']['by_S']}; G any {res['G']['any']}")
else:
    print("  darting n/a:", res.get("error"))
print(f"  time {res['time_s']:.1f} s")
