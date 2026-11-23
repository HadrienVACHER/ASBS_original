""" rules_bank_incomplete.md: toy with a far, differently shaped mode absent from the initial bank. CPU, every call < 1 min.
    python scripts/toy_bank_incomplete.py init                          initial bank composition of the 20 seeds, p, floor
    python scripts/toy_bank_incomplete.py disc EPS                      discovery by enrichment, 40 refreshes, seeds 1..20 (kept seeds)
    python scripts/toy_bank_incomplete.py dart EPS ARM WINDOW [ctrl] [S0 S1]   ARM in {trans, affine}; WINDOW in {end, post}; ctrl = bank restricted to the 8 near minima; seeds S0..S1 (inclusive)
    python scripts/toy_bank_incomplete.py verdict """
import json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import dart_lib as L, basin_lib as B, gmm_dk
T0 = time.perf_counter()
OUT = ROOT / "figures" / "bank_incomplete"; OUT.mkdir(parents=True, exist_ok=True)
D, KN, DELTA, RFAR, WFAR, NQ, NR, NB, S = 8, 8, 8.0, 40.0, 0.05, 1000, 40, 512, 10
EPS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]; SEEDS = list(range(1, 21)); NU1 = 8.0


class AnisoMix(L.Mix):
    """ Gaussian mixture with a diagonal covariance per component (var [K, d]); fam 'G' so that basin_lib / dart_lib use the plain-gradient quench. """
    def __init__(self, mu, w, var):
        super().__init__("G", mu, w, 1.0)
        self.var = np.asarray(var, float); self.prec = 1.0 / self.var; self.logdet = np.log(self.var).sum(1)

    def comp(self, X, f32=False):
        out = np.empty((len(X), self.K))
        for k in range(self.K):
            out[:, k] = -0.5 * (((X - self.mu[k]) ** 2) * self.prec[k]).sum(1) - 0.5 * self.logdet[k] - 0.5 * self.d * math.log(2 * math.pi)
        return out

    def logp_grad(self, X):
        lc = self.logw + self.comp(X); lp = logsumexp(lc, axis=1); r = np.exp(lc - lp[:, None]); g = np.zeros_like(X)
        for k in range(self.K):
            g += r[:, k:k + 1] * (-(X - self.mu[k]) * self.prec[k])
        return lp, g


def _gl(pi, X):
    lp, g = pi.logp_grad(X); return lp, g, (pi.logw + pi.comp(X)).argmax(1)


B._grad_and_logp = _gl                                                                 # quench / basins for the anisotropic mixture (the only change to basin_lib behaviour is in this process)
mu_near = gmm_dk.means(D, KN, DELTA).double().numpy(); wn = gmm_dk.weights(KN).double().numpy(); wn = wn / wn.sum() * (1 - WFAR)
mu_far = RFAR * np.ones(D) / math.sqrt(D); var_far = np.array([4.0] * 4 + [0.5] * 4)
MU = np.vstack([mu_near, mu_far]); W = np.append(wn, WFAR); VAR = np.vstack([np.ones((KN, D)), var_far[None]])
PI = AnisoMix(MU, W, VAR); K = KN + 1
assert B.quench(PI, MU.copy(), maxit=5000, tol=1e-6)[1].all()


def exact(n, rng, w=W):
    comp = rng.choice(K, size=n, p=w / w.sum()); return MU[comp] + np.sqrt(VAR[comp]) * rng.standard_normal((n, D)), comp


def sampler(s, ei, r, eps):
    """ 512 raw points of the fixed synthetic sampler: masses (1 - eps)/8 on the near modes, eps on the far mode; deterministic in (seed, eps index, refresh) """
    rng = np.random.default_rng([s, ei, r]); pv = np.append(np.full(KN, (1 - eps) / KN), eps)
    comp = rng.choice(K, size=NB, p=pv); X = MU[comp] + np.sqrt(VAR[comp]) * rng.standard_normal((NB, D)); return X


def label(Y):
    """ matching to the true minima (index 0..8) or -1 (spurious: farther than MERGE from every true minimum) """
    d = np.linalg.norm(Y[:, None, :] - MU[None], axis=2); j = d.argmin(1); return np.where(d[np.arange(len(Y)), j] < B.MERGE, j, -1)


def add_min(bank, Y, conv):
    new = []
    for y in Y[conv]:
        if len(bank) < 64 and (not bank or np.linalg.norm(np.array(bank) - y, axis=1).min() >= B.MERGE):
            bank.append(y.copy()); new.append(y.copy())
    return new


def init_bank(s):
    X0 = np.random.default_rng(s).standard_normal((NQ, D)) * math.sqrt(NU1); Y, conv, its = B.quench(PI, X0)
    bank = []; add_min(bank, Y, conv); return np.array(bank), float(conv.mean()), int(its)


def jload(p): return json.load(open(p))
def jsave(p, r): r["time_s"] = time.perf_counter() - T0; json.dump(r, open(p, "w"), indent=1, default=float)


cmd = sys.argv[1]
if cmd == "init":
    comp, kept, banks = {}, [], {}
    for s in SEEDS:
        b, fr, its = init_bank(s); lab = label(b); banks[s] = b
        comp[str(s)] = dict(n_minima=len(b), near_found=sorted(set(int(x) for x in lab if 0 <= x < KN)), far_found=bool((lab == KN).any()), spurious=int((lab < 0).sum()), frac_converged=fr, iterations=its)
        if not comp[str(s)]["far_found"]:
            kept.append(s)
    np.savez(OUT / "init_banks.npz", **{str(s): banks[s] for s in SEEDS})
    rg = np.random.default_rng(0); Xf = MU[KN] + np.sqrt(VAR[KN]) * rg.standard_normal((5000, D)); Yf, cf, _ = B.quench(PI, Xf)
    p = float(((np.linalg.norm(Yf - MU[KN], axis=1) < B.MERGE) & cf).mean())
    fl = []
    for i in range(100):
        Xe, ce = exact(5120, np.random.default_rng(1000 + i)); fl.append(0.5 * np.abs(np.bincount(ce, minlength=K) / 5120 - W).sum())
    out = dict(composition=comp, kept=kept, excluded=[s for s in SEEDS if s not in kept], n_excluded=len(SEEDS) - len(kept), p=p, floor_5120=float(np.median(fl)), w=W.tolist())
    jsave(OUT / "init.json", out)
    print(f"init: kept {len(kept)}/20 seeds (excluded {out['n_excluded']}), p = {p:.4f}, floor(n=5120) = {out['floor_5120']:.4f}, bank sizes {[c['n_minima'] for c in comp.values()]}, spurious {[c['spurious'] for c in comp.values()]}, near found "
          f"{sorted(set(len(c['near_found']) for c in comp.values()))}, far found in {sum(c['far_found'] for c in comp.values())} seeds ({out['time_s']:.1f} s)")

elif cmd == "disc":
    eps = float(sys.argv[2]); ei = EPS.index(eps); I = jload(OUT / "init.json"); IB = np.load(OUT / "init_banks.npz"); res = {}
    for s in I["kept"]:
        bank = [m for m in IB[str(s)]]; rf = None; added = []; spur = 0
        for r in range(1, NR + 1):
            Y, conv, _ = B.quench(PI, sampler(s, ei, r, eps)); new = add_min(bank, Y, conv)
            for y in new:
                lb = int(label(y[None])[0]); added.append(dict(r=r, minimum=y.tolist(), label=lb)); spur += lb < 0
                if lb == KN and rf is None:
                    rf = r
        res[str(s)] = dict(r_found=rf, added=added, spurious_added=spur, bank_final=len(bank))
    jsave(OUT / f"disc_{eps}.json", dict(eps=eps, seeds=res))
    rfs = [v["r_found"] for v in res.values()]; found = [x for x in rfs if x is not None]
    print(f"disc eps={eps}: found in {len(found)}/{len(rfs)} seeds, median r_found {np.median(found) if found else None}, spurious {sum(v['spurious_added'] for v in res.values())}, max bank {max(v['bank_final'] for v in res.values())} ({time.perf_counter() - T0:.1f} s)")

elif cmd == "dart":
    eps, arm, win = float(sys.argv[2]), sys.argv[3], sys.argv[4]; rest = sys.argv[5:]; ctrl = "ctrl" in rest; rest = [x for x in rest if x != "ctrl"]
    s0, s1 = (int(rest[0]), int(rest[1])) if len(rest) == 2 else (1, 20)
    ei = EPS.index(eps); I = jload(OUT / "init.json"); IB = np.load(OUT / "init_banks.npz"); DJ = jload(OUT / f"disc_{eps}.json")["seeds"]; res = {}
    for s in [x for x in I["kept"] if s0 <= x <= s1]:
        rf = DJ[str(s)]["r_found"]
        if win == "end":
            rs = list(range(NR - 9, NR + 1))
        else:
            if rf is None or rf > NR - 9:
                continue
            rs = list(range(rf, rf + 10))
        Xs, jacc = [], []
        for r in rs:
            bank = [m for m in IB[str(s)]] + [np.array(a["minimum"]) for a in DJ[str(s)]["added"] if a["r"] <= r]
            bank = np.array(bank)
            if ctrl:
                bank = bank[label(bank) >= 0]; bank = bank[label(bank) < KN]
            dr = L.darting(PI, sampler(s, ei, r, eps), MU, W, SW=(S,), seed=1000 + r, keep_final=True, minima=bank, force_translation=(arm == "trans"))
            if not dr.get("by_S"):
                continue
            Xs.append(dr["by_S"][S].pop("X_final")); jacc.append(dr["by_S"][S]["jump_acc"])
        X = np.vstack(Xs); b, fc, fa = B.basins(PI, X, MU); pb = B.basin_props(b, K); pv = L.props(L.cells(X, MU), K)
        near = pb[:KN] / pb[:KN].sum(); wnear = W[:KN] / W[:KN].sum()
        res[str(s)] = dict(r_found=rf, n=len(X), p_basin=pb.tolist(), far_mass=float(pb[KN]), tv_basin=L.tv(pb, W), tv_vor=L.tv(pv, W), tv_near_basin=L.tv(near, wnear), frac_assigned=fa, jump_acc=float(np.mean(jacc)))
    jsave(OUT / f"dart_{eps}_{arm}_{win}{'_ctrl' if ctrl else ''}_{s0}-{s1}.json", dict(eps=eps, arm=arm, window=win, ctrl=ctrl, seeds=res))
    if res:
        v = list(res.values()); print(f"dart eps={eps} {arm} {win}{' ctrl' if ctrl else ''} seeds {s0}-{s1}: n {len(v)} median far mass {np.median([x['far_mass'] for x in v]):.4f}, TV basin {np.median([x['tv_basin'] for x in v]):.4f}, "
                                      f"near TV {np.median([x['tv_near_basin'] for x in v]):.4f}, jump acc {np.mean([x['jump_acc'] for x in v]):.2f} ({time.perf_counter() - T0:.1f} s)")
    else:
        print(f"dart eps={eps} {arm} {win}: no seed in window ({time.perf_counter() - T0:.1f} s)")

elif cmd == "verdict":
    I = jload(OUT / "init.json"); kept = I["kept"]; n = len(kept); V = dict(n_kept=n, n_excluded=I["n_excluded"], p=I["p"], floor_5120=I["floor_5120"])
    D_ = {e: jload(OUT / f"disc_{e}.json")["seeds"] for e in EPS}
    ld = lambda e, a, w, c="": jload(OUT / f"dart_{e}_{a}_{w}{c}_1-20.json")["seeds"]
    rf = lambda e: [D_[e][str(s)]["r_found"] for s in kept]
    # B3: discovery curve against 1 - exp(-512 r eps p)
    B3 = {}
    for e in EPS[1:]:
        r_ = rf(e); curve = [float(np.mean([x is not None and x <= r for x in r_])) for r in range(1, NR + 1)]
        pred = [1 - math.exp(-NB * r * e * I["p"]) for r in range(1, NR + 1)]
        f = [x for x in r_ if x is not None]
        B3[str(e)] = dict(found_by_40=float(np.mean([x is not None for x in r_])), median_r_found=float(np.median(f)) if f else None, max_abs_dev=float(np.max(np.abs(np.array(curve) - np.array(pred)))), curve=curve, pred=pred,
                          tv_basin_end={a: float(np.median([v["tv_basin"] for v in ld(e, a, "end").values()])) for a in ("trans", "affine")})
    V["B3"] = B3
    B1 = {}
    for a in ("trans", "affine"):
        disc5 = float(np.mean([x is not None and x <= 5 for x in rf(0.01)])); post = ld(0.01, a, "post"); fm = [v["far_mass"] for v in post.values()]
        B1[a] = dict(disc_le5=disc5, disc_ok=bool(disc5 >= 0.95), median_far_mass_post=float(np.median(fm)), frac_in_band=float(np.mean([0.04 <= m <= 0.06 for m in fm])), n_post=len(fm), mass_ok=bool(0.04 <= np.median(fm) <= 0.06))
        B1[a]["value"] = B1[a]["disc_ok"] and B1[a]["mass_ok"]
    V["B1"] = B1
    B2 = {}
    nofound = all(x is None for x in rf(0.0))
    for a in ("trans", "affine"):
        tv = float(np.median([v["tv_basin"] for v in ld(0.0, a, "end").values()])); B2[a] = dict(never_found=nofound, tv_basin_median=tv, tv_ok=bool(tv >= 0.8 * WFAR))
    B2["value"] = bool(nofound and all(B2[a]["tv_ok"] for a in ("trans", "affine"))); V["B2"] = B2
    B4 = {}
    spur = sum(D_[e][str(s)]["spurious_added"] for e in EPS for s in kept) + sum(c["spurious"] for c in I["composition"].values()); mx = max(D_[e][str(s)]["bank_final"] for e in EPS for s in kept)
    for a in ("trans", "affine"):
        act, ctl = ld(0.01, a, "end"), ld(0.01, a, "end", "_ctrl"); diff = [act[s]["tv_near_basin"] - ctl[s]["tv_near_basin"] for s in act]
        B4[a] = dict(median_diff_near_tv=float(np.median(diff)), max_diff=float(np.max(diff)), ok_ii=bool(np.median(diff) <= 0.01))
    B4["spurious_total"] = int(spur); B4["max_bank"] = int(mx); B4["ok_i"] = bool(spur == 0 and mx <= 9); B4["value"] = bool(B4["ok_i"] and all(B4[a]["ok_ii"] for a in ("trans", "affine"))); V["B4"] = B4
    jsave(OUT / "verdict.json", V)
    print(json.dumps({k: v for k, v in V.items() if k != "B3"}, indent=1))
    for e, v in B3.items():
        print(f"B3 eps={e}: found by 40 {v['found_by_40']:.2f} (pred {1 - math.exp(-NB * NR * float(e) * I['p']):.2f}), median r_found {v['median_r_found']}, max |curve - pred| {v['max_abs_dev']:.3f}, TV basin end trans {v['tv_basin_end']['trans']:.4f} affine {v['tv_basin_end']['affine']:.4f}")
