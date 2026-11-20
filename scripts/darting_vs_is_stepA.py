""" Step A of rules_darting_vs_is.md: synthetic sampler (no network). Darting post hoc versus exact-weight IS, K = 8 modes, d in {32, ..., 512}, Delta = 8.
    A1: pi_k = N(m_k, I).  A2: pi_k = product of 1D Student t (nu = 3, scale^2 = 4/3, Hessian I at the minimum) for even k (1-indexed), N(m_k, I) for odd k.
    q = sum_k (1/K) q_k, q_k = pi_k dilated by (1 + delta) around m_k, delta = 0.1 (control delta = 0).  CPU only, one budget of < 1 min.  python scripts/darting_vs_is_stepA.py [--quick]
    All densities are evaluated from sufficient statistics (|x|^2, sum_j f_t(x_j), the K first coordinates), O(n d + n K); the darting map is the translation m_j - m_k once the quenched Hessians are checked equal to I. """
import json, math, sys, time
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import genpareto
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0 = time.perf_counter()
QUICK = "--quick" in sys.argv
QUENCH_MAX = 4000
K, DELTA_M, NU, SC2, DIL = 8, 8.0, 3.0, 4.0 / 3.0, 0.1
MM = DELTA_M / math.sqrt(2)
SC = math.sqrt(SC2)
DS = [32, 64] if QUICK else [32, 64, 128, 256, 512]
N, N10 = (2000, 20000) if QUICK else (10_000, 100_000)
SWEEPS = [1, 2, 5, 10]
RHO = 10 ** (-1.0 / (K - 1))
WTS = RHO ** np.arange(K); WTS = WTS / WTS.sum()
LOGW = np.log(WTS)
EVEN0 = np.array([(k + 1) % 2 == 0 for k in range(K)])      # 1-indexed even k are Student in A2
CT = gammaln((NU + 1) / 2) - gammaln(NU / 2) - 0.5 * math.log(NU * math.pi)


def ft(r, s):
    return CT - math.log(s) - (NU + 1) / 2 * np.log1p(r ** 2 / (NU * s ** 2))


def stats(x, fam, s_list):
    """ sufficient statistics of x [n, d]: sq, {s: sum_j f_t(x_j; s)} (A2 only), first K coordinates. """
    sq = (x ** 2).sum(1)
    Ft = {s: ft(x, s).sum(1) for s in s_list} if fam == "A2" else {}
    return sq, Ft, x[:, :K].copy()


def comp_logpi(sq, Ft, xK, d, fam, dil=0.0):
    """ log of the K components pi_k (dil = 0) or q_k (dilated by 1 + dil), [n, K]. """
    c = 1.0 + dil
    gauss = -0.5 * (sq[:, None] - 2 * MM * xK + MM ** 2) / c ** 2 - d * math.log(c) - 0.5 * d * math.log(2 * math.pi)
    if fam == "A1":
        return gauss
    s = SC * c
    stud = Ft[s][:, None] - ft(xK, s) + ft(xK - MM, s)
    return np.where(EVEN0[None, :], stud, gauss)


def logpi(sq, Ft, xK, d, fam):
    return logsumexp(LOGW[None, :] + comp_logpi(sq, Ft, xK, d, fam, 0.0), axis=1)


def logq(sq, Ft, xK, d, fam, dil):
    return logsumexp(-math.log(K) + comp_logpi(sq, Ft, xK, d, fam, dil), axis=1)


def draw(rng, comp, d, fam, dil):
    """ draws from the components `comp` [n] (pi_k dilated by 1 + dil): full vectors [n, d]. """
    n = len(comp)
    z = rng.standard_normal((n, d))
    if fam == "A2":
        st = EVEN0[comp]
        if st.any():
            z[st] = SC * rng.standard_t(NU, size=(int(st.sum()), d))
    x = (1.0 + dil) * z
    x[np.arange(n), comp] += MM
    return x


def draw_pi_K(rng, n, fam):
    comp = rng.choice(K, size=n, p=WTS)
    z = rng.standard_normal((n, K))
    if fam == "A2":
        st = EVEN0[comp]
        z[st] = SC * rng.standard_t(NU, size=(int(st.sum()), K))
    z[np.arange(n), comp] += MM
    return z


def cells(xK):
    return xK.argmax(1)


def props(c, w=None):
    return np.bincount(c, weights=w, minlength=K) / (len(c) if w is None else w.sum())


def tv(p, ptrue):
    return 0.5 * float(np.abs(p - ptrue).sum())


def khat(logw):
    w = np.exp(logw - logw.max()); w = np.sort(w)
    M = int(min(0.2 * len(w), 3 * math.sqrt(len(w))))
    tail = w[-M:] - w[-M - 1]
    tail = tail[tail > 0]
    return float(genpareto.fit(tail, floc=0)[0]) if len(tail) >= 10 else float("nan")


def is_stats(logw, c, ptrue):
    w = np.exp(logw - logw.max())
    return dict(tv=tv(props(c, w), ptrue), ess=float(w.sum() ** 2 / (len(w) * (w ** 2).sum())), khat=khat(logw))


def grad_neglogpi(X, fam):
    """ analytic gradient of -log pi at the points X [n, d]. """
    d = X.shape[1]
    sq, Ft, xK = stats(X, fam, [SC] if fam == "A2" else [])
    lc = LOGW[None, :] + comp_logpi(sq, Ft, xK, d, fam, 0.0)
    r = np.exp(lc - logsumexp(lc, axis=1, keepdims=True))                      # responsibilities [n, K]
    gauss = EVEN0 if fam == "A2" else np.zeros(K, bool)
    rg = r[:, ~gauss].sum(1) if fam == "A2" else r.sum(1)
    G = -(X * rg[:, None])                                                      # sum_k r_k grad log pi_k (before the sign), Gaussian part: -(x - m_k)
    gm = np.zeros_like(X)
    for k in np.nonzero(~gauss)[0]:
        gm[:, k] += r[:, k] * MM
    G = G + gm
    if fam == "A2":
        g0 = -(NU + 1) * X / (NU * SC2 + X ** 2)
        rs = r[:, gauss].sum(1)
        G = G + g0 * rs[:, None]
        for k in np.nonzero(gauss)[0]:
            u = X[:, k] - MM
            G[:, k] += r[:, k] * (-(NU + 1) * u / (NU * SC2 + u ** 2) - g0[:, k])
    return -G


def discover_minima(x500, fam, steps=100, lr=0.7, merge_tol=0.5, min_count=5):
    """ quench the q samples by gradient descent on -log pi, greedy merge, keep minima reached by >= min_count samples; Hessians by central differences of the analytic gradient. """
    y = x500.copy()
    for it in range(QUENCH_MAX):                                                # quench until >= 95% of the samples have |grad| < 1e-3 (amendment 2: 100 steps left heavy-tailed starts unconverged)
        g = grad_neglogpi(y, fam)
        y -= lr * g
        if it >= steps - 1 and it % 25 == 0 and (np.linalg.norm(g, axis=1) < 1e-3).mean() >= 0.95:
            break
    mins, cnt = [], []
    for p in y:
        if mins:
            dist = np.linalg.norm(np.array(mins) - p, axis=1); i = int(dist.argmin())
            if dist[i] < merge_tol:
                cnt[i] += 1; continue
        mins.append(p.copy()); cnt.append(1)
    mins = np.array([m for m, c in zip(mins, cnt) if c >= min_count])
    for _ in range(100):                                                        # refinement of the kept minima (the 100-step quench leaves Student minima ~0.05 from the mode)
        mins = mins - lr * grad_neglogpi(mins, fam)
    d = x500.shape[1]; h = 1e-4; Hs = []
    for m in mins:
        E = np.eye(d) * h
        Gp = grad_neglogpi(m[None, :] + E, fam); Gm = grad_neglogpi(m[None, :] - E, fam)
        H = (Gp - Gm).T / (2 * h)                                              # H[:, i] = (g(m + h e_i) - g(m - h e_i)) / (2h)
        Hs.append(0.5 * (H + H.T))
    return mins, np.array(Hs)


def darting_chain(xK0, sq0, Ft0, fam, d, rng):
    """ MH chains with the translation map y = x - m_k + m_j (Jacobian 1); only the coordinates k and j change, so the state is (xK, sq, Ft, logp). """
    n = len(xK0)
    xK, sq = xK0.copy(), sq0.copy()
    Ft = {SC: Ft0[SC].copy()} if fam == "A2" else {}
    lp = logpi(sq, Ft, xK, d, fam)
    idx = np.arange(n)
    acc_tot, prop_tot = 0, 0
    snaps = {}
    for sweep in range(1, max(SWEEPS) + 1):
        k = xK.argmax(1)
        j = (k + 1 + rng.integers(K - 1, size=n)) % K
        xk, xj = xK[idx, k], xK[idx, j]
        yk, yj = xk - MM, xj + MM
        sq_y = sq - xk ** 2 - xj ** 2 + yk ** 2 + yj ** 2
        yK = xK.copy(); yK[idx, k] = yk; yK[idx, j] = yj
        Ft_y = {SC: Ft[SC] - ft(xk, SC) - ft(xj, SC) + ft(yk, SC) + ft(yj, SC)} if fam == "A2" else {}
        valid = yK.argmax(1) == j
        lp_y = logpi(sq_y, Ft_y, yK, d, fam)
        accept = valid & (np.log(rng.random(n)) < lp_y - lp)
        xK = np.where(accept[:, None], yK, xK); sq = np.where(accept, sq_y, sq); lp = np.where(accept, lp_y, lp)
        if fam == "A2":
            Ft[SC] = np.where(accept, Ft_y[SC], Ft[SC])
        acc_tot += int(accept.sum()); prop_tot += n
        if sweep in SWEEPS:
            snaps[sweep] = (xK.argmax(1).copy(), acc_tot / prop_tot)
    return snaps


def darting_chain_general(x0, fam, d, mins, Hs, rng):
    """ MH chains with the full map y = m_b + H_b^(-1/2) H_a^(1/2) (x - m_a), |J| = sqrt(det H_a / det H_b), partner b != a uniform, validity = y in the cell of the minimum b (nearest minimum). Used when the Hessians are not I. """
    n = len(x0); Kn = len(mins)
    ev, V = np.linalg.eigh(Hs)                                                   # [K, d], [K, d, d]
    assert (ev > 0).all(), "Hessian not positive definite"
    Hh = [(V[k] * np.sqrt(ev[k])) @ V[k].T for k in range(Kn)]
    Hmh = [(V[k] / np.sqrt(ev[k])) @ V[k].T for k in range(Kn)]
    logdet = np.log(ev).sum(1)
    A = {(a, b): Hmh[b] @ Hh[a] for a in range(Kn) for b in range(Kn) if a != b}   # y - m_b = A_ab (x - m_a)
    x = x0.copy()
    s_list = [SC] if fam == "A2" else []
    sq, Ft, xK = stats(x, fam, s_list)
    lp = logpi(sq, Ft, xK, d, fam)
    M2 = (mins ** 2).sum(1)
    acc_tot, prop_tot = 0, 0
    snaps = {}
    for sweep in range(1, max(SWEEPS) + 1):
        a = np.argmin(sq[:, None] - 2 * x @ mins.T + M2[None, :], axis=1)
        b = (a + 1 + rng.integers(Kn - 1, size=n)) % Kn
        Y = np.empty_like(x)
        for a0 in range(Kn):
            for b0 in range(Kn):
                if a0 == b0: continue
                idx = np.nonzero((a == a0) & (b == b0))[0]
                if len(idx): Y[idx] = mins[b0] + (x[idx] - mins[a0]) @ A[(a0, b0)].T
        sqY, FtY, yK = stats(Y, fam, s_list)
        valid = np.argmin(sqY[:, None] - 2 * Y @ mins.T + M2[None, :], axis=1) == b
        lpY = logpi(sqY, FtY, yK, d, fam)
        logJ = 0.5 * (logdet[a] - logdet[b])
        accept = valid & (np.log(rng.random(n)) < lpY + logJ - lp)
        x[accept] = Y[accept]; sq = np.where(accept, sqY, sq); lp = np.where(accept, lpY, lp)
        if fam == "A2":
            Ft = {SC: np.where(accept, FtY[SC], Ft[SC])}
        acc_tot += int(accept.sum()); prop_tot += n
        if sweep in SWEEPS:
            snaps[sweep] = (cells(x[:, :K]), acc_tot / prop_tot)
    return snaps


def exact_truth(fam, rng):
    c = []
    n = 2_000_000
    cnt = np.zeros(K)
    for _ in range(10):
        cnt += np.bincount(cells(draw_pi_K(rng, n // 10, fam)), minlength=K)
    ptrue = cnt / n
    f4, f5 = [], []
    for _ in range(100 if not QUICK else 30):
        f4.append(tv(props(cells(draw_pi_K(rng, N, fam))), ptrue))
    for _ in range(40 if not QUICK else 10):
        f5.append(tv(props(cells(draw_pi_K(rng, N10, fam))), ptrue))
    return ptrue, float(np.median(f4)), float(np.median(f5))


def run_config(fam, d, dil, rng, ptrue, do10, tlimit):
    t0 = time.perf_counter()
    comp = rng.integers(K, size=N)
    x = draw(rng, comp, d, fam, dil)
    s_list = [SC, SC * (1 + dil)] if fam == "A2" else []
    sq, Ft, xK = stats(x, fam, s_list)
    lw = logpi(sq, Ft, xK, d, fam) - logq(sq, Ft, xK, d, fam, dil)
    res = dict(d=d, dil=dil, IS_n=is_stats(lw, cells(xK), ptrue))
    out10 = None; t10 = 0.0
    if do10:
        t_a = time.perf_counter()
        lws, cs = [], []
        for _ in range(N10 // 20000 if N10 >= 20000 else 1):
            m = min(20000, N10)
            cp = rng.integers(K, size=m); xx = draw(rng, cp, d, fam, dil)
            s2, F2, x2 = stats(xx, fam, s_list)
            lws.append(logpi(s2, F2, x2, d, fam) - logq(s2, F2, x2, d, fam, dil)); cs.append(cells(x2))
        out10 = is_stats(np.concatenate(lws), np.concatenate(cs), ptrue)
        t10 = time.perf_counter() - t_a
    res["IS_10n"] = out10; res["t10"] = t10
    t_min = time.perf_counter()
    mins, Hs = discover_minima(x[:500], fam)
    herr = float(np.abs(Hs - np.eye(d)[None]).max()) if len(Hs) else float("nan")
    res["minima"] = dict(found=int(len(mins)), max_hessian_dev_from_I=herr)
    if len(mins) != K:
        res["darting"] = None
        res["error"] = f"minima found {len(mins)} instead of {K}"
    else:
        res["map"] = "translation (H = I to 1e-6)" if herr < 1e-6 else "general Hessian map"
        snaps = darting_chain(xK, sq, Ft, fam, d, rng) if herr < 1e-6 else darting_chain_general(x, fam, d, mins, Hs, rng)
        res["darting"] = {S: dict(tv=tv(props(snaps[S][0]), ptrue), acc=snaps[S][1], cost=N * (1 + S)) for S in SWEEPS}
    if fam == "A2" and dil > 0:                                                   # exploratory (amendment 3): oracle centres, translation map m_j - m_k (the "linear map" of the statement)
        sn = darting_chain(xK, sq, Ft, fam, d, np.random.default_rng(777 + d))
        res["darting_oracle"] = {S: dict(tv=tv(props(sn[S][0]), ptrue), acc=sn[S][1]) for S in SWEEPS}
    res["time_s"] = time.perf_counter() - t0
    return res


def main():
    out = {}
    for fam in ("A1", "A2"):
        rng = np.random.default_rng(2026)
        ptrue, floor_n, floor_10n = exact_truth(fam, rng)
        out[fam] = dict(ptrue=ptrue.tolist(), floor_n=floor_n, floor_10n=floor_10n, rows=[], control=[])
        skip10 = False; t10_rate = None
        for d in DS:
            rng = np.random.default_rng(1000 + d)
            elapsed = time.perf_counter() - T0
            est10 = 1.3 * t10_rate * d if t10_rate else 0.0                   # time guard: skip IS(10n) when its projected cost would put the minute at risk
            do10 = (not skip10) and (elapsed + est10 < 40)
            if not do10: skip10 = True
            r = run_config(fam, d, DIL, rng, ptrue, do10, None)
            if do10 and r["t10"] > 0: t10_rate = max(t10_rate or 0.0, r["t10"] / d)
            if time.perf_counter() - T0 > 56: print("*** hard stop: time budget reached ***"); break
            out[fam]["rows"].append(r)
            dd = r["darting"]
            print(f"{fam} d={d:3d}: IS(n) TV {r['IS_n']['tv']:.4f} ESS {r['IS_n']['ess']:.4f} khat {r['IS_n']['khat']:.2f}" + (f" | IS(10n) TV {r['IS_10n']['tv']:.4f} ESS {r['IS_10n']['ess']:.4f}" if r['IS_10n'] else " | IS(10n) skipped")
                  + (" | darting TV " + " ".join(f"S{S}:{dd[S]['tv']:.4f}" for S in SWEEPS) + f" | acc S10 {dd[10]['acc']:.3f}" if dd else " | " + r.get("error", "")) + f" | floor(n) {floor_n:.4f} | minima {r['minima']} | {r['time_s']:.1f} s | total {time.perf_counter() - T0:.1f} s")
        for r in out[fam]["rows"]:
            if r.get("darting_oracle"):
                do = r["darting_oracle"]; print(f"{fam} d={r['d']:3d} EXPLORATORY oracle translation: darting TV " + " ".join(f"S{S}:{do[S]['tv']:.4f}" for S in SWEEPS) + f" | acc S10 {do[10]['acc']:.3f}")
        for d in ([DS[0], DS[-1]] if not QUICK else DS[:1]):                  # exploratory control delta = 0
            rng = np.random.default_rng(5000 + d)
            r = run_config(fam, d, 0.0, rng, ptrue, False, None)
            out[fam]["control"].append(r)
            dd = r["darting"]
            print(f"{fam} control delta=0 d={d:3d}: IS(n) TV {r['IS_n']['tv']:.4f} ESS {r['IS_n']['ess']:.4f} | darting TV " + (" ".join(f"S{S}:{dd[S]['tv']:.4f}" for S in SWEEPS) if dd else "none") + f" | total {time.perf_counter() - T0:.1f} s")
    # verdicts
    A1, A2 = out["A1"], out["A2"]
    r1_hits, r1_hits10 = [], []
    for r in A1["rows"]:
        if r["IS_10n"] is None or r["darting"] is None: continue
        for S in SWEEPS:
            if r["IS_n"]["ess"] < 0.05 and r["darting"][S]["tv"] <= 0.5 * r["IS_10n"]["tv"] and r["darting"][S]["tv"] <= 2 * A1["floor_n"]:
                r1_hits.append((r["d"], S))
                if S == 10: r1_hits10.append(r["d"])
    R1 = bool(r1_hits)
    a32 = next(r for r in A2["rows"] if r["d"] == DS[0])["darting"]; a512 = next(r for r in A2["rows"] if r["d"] == DS[-1])["darting"]
    cond_a = bool(a32 and a512 and a512[10]["acc"] < 0.5 * a32[10]["acc"])
    b_list = []
    for r in A2["rows"]:
        if r["darting"] is None: continue
        ref = r["IS_10n"]["tv"] if r["IS_10n"] else r["IS_n"]["tv"]
        b_list.append((r["d"], r["darting"][10]["tv"] > ref, r["darting"][10]["tv"] > r["IS_n"]["tv"]))
    cond_b = any(b[1] for b in b_list)
    R2 = cond_a and cond_b
    total = time.perf_counter() - T0
    print(f"\nR1 (A1: ESS_IS<5% and TV_darting(S) <= 0.5 TV_IS(10n) and <= 2 floor): {'TRUE' if R1 else 'FALSE'}  hits (d, S): {r1_hits}  (S=10 only: {r1_hits10})")
    fa = (f"{a512[10]['acc']:.3f} vs {a32[10]['acc']:.3f}" if (a32 and a512) else "n/a")
    fa = (f"{a512[10]['acc']:.3f} vs {a32[10]['acc']:.3f}" if (a32 and a512) else "n/a: darting could not run at d = 512 (minima not discoverable without an oracle)")
    verdict2 = "TRUE" if R2 else ("(a) NOT EVALUABLE; (b) " + str(cond_b) if not (a32 and a512) else "FALSE (A2 does not break the map as specified)")
    print(f"R2 (A2: acceptance(d={DS[-1]},S=10) < 0.5 acceptance(d={DS[0]},S=10) [{cond_a}: {fa}] and exists d with TV_darting(S=10) > TV_IS [{cond_b}]): {verdict2}  (d, > IS(ref), > IS(n)): {b_list}")
    acc_or = [(r["d"], round(r["darting_oracle"][10]["acc"], 3)) for r in A2["rows"] if r.get("darting_oracle")]
    print(f"EXPLORATORY (oracle translation, A2): acceptance S=10 by d: {acc_or}")
    print(f"TOTAL TIME {total:.1f} s (budget 60 s){'  *** OVER BUDGET ***' if total > 60 else ''}")
    json.dump(dict(out=out, R1=R1, R1_hits=r1_hits, R2=R2, R2_cond_a=cond_a, R2_cond_b=cond_b, R2_b=b_list, total_time_s=total, n=N, n10=N10, ds=DS), open("/tmp/stepA_quick.json" if QUICK else "figures/darting_vs_is_stepA.json", "w"), indent=1, default=float)
    if not QUICK:
        fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.3), facecolor="#fcfcfb", sharey=True)
        for ax, fam, title in zip(axs, ("A1", "A2"), ("A1: all components Gaussian (translation map exact)", "A2: Student for even k (translation map wrong)")):
            rows = out[fam]["rows"]; ds = [r["d"] for r in rows]
            ax.plot(ds, [r["IS_n"]["tv"] for r in rows], color="#2a78d6", marker="o", ms=7, lw=2, label="IS, n = 10⁴")
            r10 = [(r["d"], r["IS_10n"]["tv"]) for r in rows if r["IS_10n"]]
            if r10: ax.plot([a for a, _ in r10], [b for _, b in r10], color="#1baf7a", marker="s", ms=7, lw=2, label="IS, n = 10⁵")
            for S, mk, ls in zip(SWEEPS, ("^", "v", "D", "s"), (":", "-.", "--", "-")):
                rr = [(r["d"], r["darting"][S]["tv"]) for r in rows if r["darting"]]
                ax.plot([a for a, _ in rr], [b for _, b in rr], color="#eb6834", marker=mk, ms=6, lw=1.6, ls=ls, label=f"darting, S = {S}")
            if fam == "A2":
                ro = [(r["d"], r["darting_oracle"][10]["tv"]) for r in rows if r.get("darting_oracle")]
                if ro: ax.plot([a for a, _ in ro], [b for _, b in ro], color="#52514e", marker="x", ms=7, lw=1.4, ls="--", label="darting S = 10, ORACLE centres (exploratory)")
            ax.axhline(out[fam]["floor_n"], color="#52514e", lw=1, ls=":"); ax.text(ds[0], out[fam]["floor_n"] * 1.08, "floor (exact sampler, n = 10⁴)", fontsize=7.5, color="#52514e")
            ax.set(xscale="log", yscale="log", xlabel="dimension d", title=title); ax.set_xticks(ds); ax.set_xticklabels([str(v) for v in ds])
            ax.grid(color="#e4e3de", lw=0.8); ax.spines[["top", "right"]].set_visible(False); ax.set_facecolor("#fcfcfb")
        axs[1].text(0.03, 0.04, "x: darting S = 10 with ORACLE centres (exploratory).\nNo-oracle darting stops at d = 64: the Student minima\nare not found for d >= 128 (4 of 8 minima).", transform=axs[1].transAxes, fontsize=7.5, color="#52514e")
        axs[0].set_ylabel("TV of the mode proportions"); axs[0].legend(frameon=False, fontsize=8, loc="center left")
        fig.suptitle("Darting post hoc versus exact-weight IS, K = 8, Δ = 8, δ = 0.1", x=0.01, ha="left", fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig("figures/darting_vs_is_stepA.png", dpi=150, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
