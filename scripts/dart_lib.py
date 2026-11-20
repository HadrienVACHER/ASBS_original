""" Library of the phase-0 darting protocol (rules_mog_mos_phase0.md) for a mixture target `Mix` and arbitrary samples X (used by scripts/mos_pilot_eval.py on real AS samples). """
import math, time
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import genpareto

NU = 2.0
CT = gammaln((NU + 1) / 2) - gammaln(NU / 2) - 0.5 * math.log(NU * math.pi)


class Mix:
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
        lc = self.logw + self.comp(X); lp = logsumexp(lc, axis=1); r = np.exp(lc - lp[:, None])
        if self.fam == "G":
            return lp, (r @ self.mu - X * r.sum(1)[:, None]) / self.std ** 2
        g = np.zeros_like(X)
        for k in range(self.K):
            u = X - self.mu[k]; g += r[:, k:k + 1] * (-(NU + 1) * u / (NU + u * u))
        return lp, g


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
    n = len(lw)
    wn = w / w.mean()
    return dict(tv=tv(props(c, K, w), ptrue), ess=float(w.sum() ** 2 / (n * (w ** 2).sum())), khat=khat(lw),
                logz=float(logsumexp(lw) - math.log(n)), logz_se=float(wn.std() / math.sqrt(n)), p_hat=props(c, K, w).tolist())


def darting(pi, X, mu, ptrue, discover=500, SW=(1, 5, 10), seed=99):
    """ protocol of the phase 0 (500 discovery samples, quench, Hessians by finite differences, 1 jump + 5 MALA steps per sweep, MALA step tuned once for 0.6): returns a dict (None-safe) with TV, acceptances and cost per S. """
    K, d, n = pi.K, pi.d, len(X)
    cost = dict(E=0, G=0); res = {}
    y = X[:discover].copy()
    for it in range(4000):
        _, g = pi.logp_grad(y); cost["G"] += len(y); y = y + 0.7 * g
        if it >= 5 and it % 5 == 0 and (np.linalg.norm(g, axis=1) < 1e-3).mean() >= 0.95: break
    mins, cnt = [], []
    for p in y:
        if mins:
            dist = np.linalg.norm(np.array(mins) - p, axis=1); i = int(dist.argmin())
            if dist[i] < 0.5: cnt[i] += 1; continue
        mins.append(p.copy()); cnt.append(1)
    M = np.array([m for m, c in zip(mins, cnt) if c >= 5]); res["minima_found"] = int(len(M))
    if len(M) < 2:
        res["error"] = f"{len(M)} minima discovered"; return res
    for _ in range(50):
        _, g = pi.logp_grad(M); cost["G"] += len(M); M = M + 0.7 * g
    h = 1e-4; Hs = []
    for m in M:
        E = np.eye(d) * h
        _, gp = pi.logp_grad(m[None, :] + E); _, gm = pi.logp_grad(m[None, :] - E); cost["G"] += 2 * d
        H = -(gp - gm).T / (2 * h); Hs.append(0.5 * (H + H.T))
    Hs = np.array(Hs); Mk = len(M)
    translation = float(np.abs(Hs - Hs[0][None]).max()) < 1e-6
    res["map"] = "translation" if translation else "general Hessian map"; res["hessian_spread"] = float(np.abs(Hs - Hs[0][None]).max())
    if not translation:
        assert Mk * Mk * d * d * 8 < 6e8
        ev, V = np.linalg.eigh(Hs)
        Hh = [(V[k] * np.sqrt(ev[k])) @ V[k].T for k in range(Mk)]; Hmh = [(V[k] / np.sqrt(ev[k])) @ V[k].T for k in range(Mk)]
        logdet = np.log(ev).sum(1); Amap = {(a, b): Hmh[b] @ Hh[a] for a in range(Mk) for b in range(Mk) if a != b}
    M2 = (M ** 2).sum(1)
    vor = lambda Z: (-2 * Z @ M.T + M2[None, :]).argmin(1)

    def mala(Xc, lp, g, hs, rg):
        Xp = Xc + hs * g + math.sqrt(2 * hs) * rg.standard_normal(Xc.shape)
        lpp, gp = pi.logp_grad(Xp)
        a = lpp - lp - ((Xc - Xp - hs * gp) ** 2).sum(1) / (4 * hs) + ((Xp - Xc - hs * g) ** 2).sum(1) / (4 * hs)
        acc = np.log(rg.random(len(Xc))) < a
        return np.where(acc[:, None], Xp, Xc), np.where(acc, lpp, lp), np.where(acc[:, None], gp, g), acc

    pil = X[:500].copy(); lp0, g0 = pi.logp_grad(pil); cost["E"] += 500; cost["G"] += 500
    lo, hi, rt = math.log(1e-8), math.log(10.0), np.random.default_rng(5)
    for _ in range(8):
        mid = 0.5 * (lo + hi); Xc, lp, g = pil.copy(), lp0.copy(), g0.copy(); accs = []
        for _s in range(5):
            Xc, lp, g, a_ = mala(Xc, lp, g, math.exp(mid), rt); accs.append(a_.mean()); cost["E"] += 500; cost["G"] += 500
        lo, hi = (mid, hi) if np.mean(accs) > 0.6 else (lo, mid)
    hs = math.exp(0.5 * (lo + hi)); res["mala_step"] = hs
    rd = np.random.default_rng(seed)
    Xc = X.copy(); lp, g = pi.logp_grad(Xc); cost["E"] += n; cost["G"] += n
    ja = jp = ma = mp = 0; res["by_S"] = {}
    for sweep in range(1, max(SW) + 1):
        a = vor(Xc); b = (a + 1 + rd.integers(Mk - 1, size=n)) % Mk
        if translation:
            Y = Xc - M[a] + M[b]; logJ = 0.0
        else:
            Y = np.empty_like(Xc)
            for a0 in range(Mk):
                for b0 in range(Mk):
                    if a0 != b0:
                        idx = np.nonzero((a == a0) & (b == b0))[0]
                        if len(idx): Y[idx] = M[b0] + (Xc[idx] - M[a0]) @ Amap[(a0, b0)].T
            logJ = 0.5 * (logdet[a] - logdet[b])
        valid = vor(Y) == b
        lpy = pi.logp(Y); cost["E"] += n
        acc = valid & (np.log(rd.random(n)) < lpy - lp + logJ); ja += int(acc.sum()); jp += n
        if acc.any():
            _, gy = pi.logp_grad(Y[acc]); cost["G"] += int(acc.sum()); Xc[acc] = Y[acc]; lp = np.where(acc, lpy, lp); g[acc] = gy
        for _s in range(5):
            Xc, lp, g, am = mala(Xc, lp, g, hs, rd); ma += int(am.sum()); mp += n; cost["E"] += n; cost["G"] += n
        if sweep in SW:
            pk = props(cells(Xc, mu), K)
            res["by_S"][sweep] = dict(tv=tv(pk, ptrue), jump_acc=ja / jp, mala_acc=ma / mp, cost=cost["E"] + cost["G"], coverage=int((pk >= 0.25 * ptrue).sum()))
    return res
