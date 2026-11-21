""" Numerical checks of memo_cv_failure_modes.md. Rules: rules_cv_failure_modes.md (commit eea3ce1, written before this script).
    python scripts/cv_failure_modes_checks.py PART      PART in {1, 2, 3, 4, 5}      (one CPU execution per part, each timed, < 1 min)
    Units of the Failure Modes paper after its space-time scaling (their appendix A): base = standard Brownian motion in the clock nu in [0, 1], X_0 = 0,
    p_base = N(0, 1), kappa = nu, v = nu (1 - nu), nu_{1|t} = 1 - nu.  Fokker-Planck in nu: d_nu p = -d_x(D p) + 1/2 d_xx p, the SDE drift is sigma_t u = sigma_t^2 D.
    Solver (conservative finite volumes, Crank-Nicolson, zero flux at the ends, geometric nu grid) taken from scripts/quadrature_1d_failure_modes.py (commit bed2cf3). """
import json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.linalg import solve_banded
from scipy.special import expit, logsumexp, ndtr

T0 = time.perf_counter()
SQ2, SQPI = math.sqrt(2.0), math.sqrt(math.pi)
OUT = Path("figures/cv_failure_modes_checks.json")


class Mix:
    """ 1D Gaussian mixture sum_k w_k N(m_k, s2_k). """
    def __init__(self, w, m, s2):
        w = np.asarray(w, float)
        self.w, self.m, self.s2 = w / w.sum(), np.asarray(m, float), np.asarray(s2, float)
        self.lw = np.log(self.w)

    def _lc(self, z):
        z = np.asarray(z, float)[..., None]
        return self.lw - 0.5 * (z - self.m) ** 2 / self.s2 - 0.5 * np.log(2 * np.pi * self.s2), z

    def logpdf(self, z):
        lc, _ = self._lc(z)
        return logsumexp(lc, axis=-1)

    def score(self, z):
        lc, zz = self._lc(z)
        r = np.exp(lc - logsumexp(lc, axis=-1, keepdims=True))
        return (r * (self.m - zz) / self.s2).sum(-1)

    def cells(self, xf):
        return np.diff(sum(w * ndtr((xf - m) / math.sqrt(s2)) for w, m, s2 in zip(self.w, self.m, self.s2)))

    def left_mass(self):
        return float(sum(w * ndtr(-m / math.sqrt(s2)) for w, m, s2 in zip(self.w, self.m, self.s2)))


def hermite(ng):
    xi, wg = np.polynomial.hermite.hermgauss(ng)
    return xi, wg / SQPI


def posterior(mix, x, nu, xi, wg):
    """ posterior of X1 given X_nu = x under the reciprocal measure with terminal law mix: component weights a [nx,K], component means mk [nx,K],
        Gauss-Hermite nodes z [nx,K,ng] and weights W [nx,K,ng] (normalised), exact posterior mean E [nx]. """
    P = 1 / mix.s2 + nu / (1 - nu)                                                          # 1/s2 + kappa^2/v
    mk = (mix.m / mix.s2 + x[:, None] / (1 - nu)) / P                                       # (m/s2 + kappa x / v)/P
    sk = P ** -0.5
    vv = nu * (1 - nu) + nu ** 2 * mix.s2                                                   # marginal variance of X_nu under component k
    la = mix.lw - 0.5 * (x[:, None] - nu * mix.m) ** 2 / vv - 0.5 * np.log(vv)
    a = np.exp(la - logsumexp(la, axis=1, keepdims=True))
    z = mk[:, :, None] + SQ2 * sk[None, :, None] * xi[None, None, :]
    W = a[:, :, None] * wg[None, None, :]
    return dict(a=a, mk=mk, z=z, W=W, E=(a * mk).sum(1))


def log_post(pi, z, x, nu):
    """ log pi~(z | x) up to a constant: log pi(z) + log N(x; kappa z, v). x broadcast on the leading axis of z. """
    xs = x.reshape((-1,) + (1,) * (z.ndim - 1))
    return pi.logpdf(z) - (xs - nu * z) ** 2 / (2 * nu * (1 - nu))


def drifts(x, nu, pi, q, spec, xi, wg):
    """ spec: list of method names. 'q', 'star', 'AS', 'NS:<H>', 'NSHBpost:<H>', 'NSHBtarget:<H>', 'P:<p>', 'IDEM'. Returns {name: drift on x}. """
    out = {}
    pq = posterior(q, x, nu, xi, wg)
    out["q"] = (pq["E"] - x) / (1 - nu)
    if any(s == "star" or s.startswith("P:") for s in spec):
        pp = posterior(pi, x, nu, xi, wg)
        out["star"] = (pp["E"] - x) / (1 - nu)
    if any(s == "AS" or s.startswith("NS") or s == "IDEM" for s in spec):
        Es = (pi.score(pq["z"]) * pq["W"]).sum((1, 2))                                       # E_q~[s_pi(X1)]
        out["AS"] = Es + pq["E"]                                                             # E_q~[s_pi(X1) + X1]
        EcS = Es + (x - nu * pq["E"]) / (1 - nu)                                             # E_q~[c_S]
    for s in spec:
        if s.startswith("NS"):
            H = float(s.split(":")[1])
            Lam = 1 / ((1 - nu) * H + nu)
            dns = out["AS"] + (Lam - 1) * EcS
            if s.startswith("NS:"):
                out[s] = dns
            else:                                                                            # heat-bath darting between the two modes of pi
                muL, muR = float(pi.m.min()), float(pi.m.max()); s2 = float(pi.s2[0])
                if s.startswith("NSHBpost"):
                    Pp = 1 / s2 + nu / (1 - nu)
                    aL = (muL / s2 + x / (1 - nu)) / Pp; aR = (muR / s2 + x / (1 - nu)) / Pp
                    Delta = (muR - muL) / (s2 * Pp); c = (0.5 * (aL + aR))[:, None, None]
                else:
                    Delta = muR - muL; c = 0.5 * (muL + muR)
                z = pq["z"]
                inR = z >= c
                y = np.where(inR, z - Delta, z + Delta)
                valid = np.where(inR, y < c, y >= c)
                pw = valid * expit(log_post(pi, y, x, nu) - log_post(pi, z, x, nu))
                out[s] = dns + ((pw * (y - z)) * pq["W"]).sum((1, 2)) / (1 - nu)
        elif s.startswith("P:"):
            p = float(s.split(":")[1])
            out[s] = out["q"] + p * (out["star"] - out["q"])
        elif s == "IDEM":                                                                    # one independent proposal y ~ N(x, 1 - nu), Barker on phi = pi/p_base
            sd = math.sqrt(1 - nu); sig = math.sqrt(float(pi.s2.min()))
            du = min(0.05, sig / (6 * sd))
            u = np.arange(-7.0, 7.0 + du / 2, du); wu = np.exp(-0.5 * u ** 2); wu /= wu.sum()
            y = x[:, None] + sd * u[None, :]                                                 # [nx, ny]
            lpy = pi.logpdf(y) + 0.5 * y ** 2                                                # log phi(y) + const
            z = pq["z"]; lpz = pi.logpdf(z) + 0.5 * z ** 2                                   # [nx, K, ng]
            al = expit(lpy[:, None, None, :] - lpz[..., None])                               # [nx, K, ng, ny]
            disp = (al * (wu * y[:, None, None, :])).sum(-1) - z * (al * wu).sum(-1)
            out[s] = out["AS"] + (disp * pq["W"]).sum((1, 2)) / (1 - nu)
    return out


def fp_step(p, Df, h, dx, theta):
    """ one conservative finite-volume step of d_nu p = -d_x(D p) + 1/2 d_xx p (zero flux at both ends); theta = 1: implicit Euler, 0.5: Crank-Nicolson (bed2cf3). """
    n = len(p)
    aL = Df / 2 + 1 / (2 * dx)
    aR = Df / 2 - 1 / (2 * dx)
    aL[0] = aR[0] = aL[-1] = aR[-1] = 0.0
    diag = -(aL[1:] - aR[:-1]) / dx
    upper = -aR[1:n] / dx
    lower = aL[1:n] / dx
    Ap = diag * p
    Ap[:-1] += upper * p[1:]
    Ap[1:] += lower * p[:-1]
    rhs = p + h * (1 - theta) * Ap
    ab = np.zeros((3, n))
    ab[0, 1:] = -h * theta * upper
    ab[1] = 1 - h * theta * diag
    ab[2, :-1] = -h * theta * lower
    return solve_banded((1, 1), ab, rhs)


def run(pi, q, sig, spec, ng=48, dxdiv=20, nt=400):
    """ integrate all methods of spec on the same grid; returns terminal cell masses per method and the grid. """
    dx = sig / dxdiv
    half = int(math.ceil(6.0 / dx)); nx = 2 * half
    xf = (np.arange(nx + 1) - half) * dx; xc = 0.5 * (xf[1:] + xf[:-1])
    nu0 = (5 * dx) ** 2; wend = (dx / 10) ** 2
    n1 = int(round(0.3 * nt)); n2 = nt - n1
    nus = np.concatenate([nu0 * (0.5 / nu0) ** (np.arange(n1 + 1) / n1), (1 - 0.5 * (wend / 0.5) ** (np.arange(n2 + 1) / n2))[1:]])
    p0 = np.diff(ndtr(xf / math.sqrt(nu0))) / dx
    pm = {s: p0.copy() for s in spec}
    xi, wg = hermite(ng)
    mmax = max(np.abs(pi.m).max(), np.abs(q.m).max()); s2max = max(pi.s2.max(), q.s2.max())
    for n in range(nt):
        h = nus[n + 1] - nus[n]; nu_m = 0.5 * (nus[n] + nus[n + 1])
        X = nu_m * mmax + 9 * math.sqrt(nu_m * (1 - nu_m) + nu_m ** 2 * s2max)
        j0 = max(1, int(math.floor((-X - xf[0]) / dx))); j1 = min(nx - 1, int(math.ceil((X - xf[0]) / dx)))
        D = drifts(xf[j0:j1 + 1], nu_m, pi, q, spec, xi, wg)
        theta = 1.0 if n < 4 else 0.5
        for s in spec:
            d = D[s]
            Df = np.empty(nx + 1); Df[j0:j1 + 1] = d; Df[:j0] = d[0]; Df[j1 + 1:] = d[-1]; Df[0] = Df[-1] = 0.0
            pm[s] = fp_step(pm[s], Df, h, dx, theta)
    return {s: pm[s] * dx for s in spec}, dict(dx=dx, xf=xf, xc=xc, nx=nx)


def stats(m, g, target=None):
    xc = g["xc"]; tot = m.sum(); m = m / tot
    mean = float((m * xc).sum()); var = float((m * (xc - mean) ** 2).sum())
    left = float(m[xc < 0].sum())
    mr = float((m * xc)[xc > 0].sum() / max(m[xc > 0].sum(), 1e-300)); ml = float((m * xc)[xc < 0].sum() / max(m[xc < 0].sum(), 1e-300))
    out = dict(mass=float(tot), mean=mean, var=var, left=left, half_sep=0.5 * (mr - ml))
    if target is not None:
        out["tv"] = float(0.5 * np.abs(m - target.cells(g["xf"])).sum())
    return out


def save(part, res):
    allr = json.load(open(OUT)) if OUT.exists() else {}
    allr[str(part)] = res
    json.dump(allr, open(OUT, "w"), indent=1, default=float)


def F(r):
    return r * math.log(r) / (r - 1)


def part1():
    pi = Mix([1], [1.0], [0.05]); q = Mix([1], [1.1], [0.05]); sig = math.sqrt(0.05)
    rhos = [0.5, 2.0, 4.0, 6.0]
    spec = ["q", "star", "AS", "NS:20"] + [f"NS:{1 / (r * 0.05)}" for r in rhos]
    M, g = run(pi, q, sig, spec)
    S = {s: stats(M[s], g, pi) for s in spec}
    mq, ms = S["q"]["mean"], S["star"]["mean"]
    A = {s: (S[s]["mean"] - ms) / (mq - ms) for s in spec}
    res = dict(floor=dict(mean_q=mq, mean_star=ms, var_q=S["q"]["var"], var_star=S["star"]["var"], tv_star=S["star"]["tv"]))
    res["A_AS"] = A["AS"]; res["A_NS_exact"] = A["NS:20"]; res["var_ratio_NS"] = S["NS:20"]["var"] / S["star"]["var"]
    res["A_rho"] = {r: A[f"NS:{1 / (r * 0.05)}"] for r in rhos}; res["A_rho_theory"] = {r: 1 - F(r) for r in rhos}
    res["var_ratio_rho"] = {r: S[f"NS:{1 / (r * 0.05)}"]["var"] / S["star"]["var"] for r in rhos}
    P1a = bool(-2.203 <= A["AS"] <= -2.103)
    P1b = bool(abs(A["NS:20"]) <= 0.02 and abs(res["var_ratio_NS"] - 1) <= 0.02)
    P1c = bool(all(abs(res["A_rho"][r] - res["A_rho_theory"][r]) <= 0.05 for r in rhos))
    print(f"[1] floors: mean q {mq:.5f} (target 1.1), mean star {ms:.5f} (target 1.0), var q {S['q']['var']:.5f}, var star {S['star']['var']:.5f} (target 0.05), TV(star, pi) {S['star']['tv']:.2e}")
    print(f"[1] P1a A_AS = {A['AS']:.4f} (theory -2.1534) -> {P1a}")
    print(f"[1] P1b A_NS = {A['NS:20']:.4f}, var ratio {res['var_ratio_NS']:.4f} -> {P1b}")
    for r in rhos:
        print(f"[1] P1c rho {r}: A = {res['A_rho'][r]:.4f}, theory 1 - F = {res['A_rho_theory'][r]:.4f}, var ratio {res['var_ratio_rho'][r]:.4f}")
    print(f"[1] P1c -> {P1c}   (time {time.perf_counter() - T0:.1f} s)")
    # P1d: non-Gaussian current law
    q2 = Mix([0.5, 0.5], [0.8, 1.3], [0.02, 0.02])
    M2, g2 = run(pi, q2, sig, ["q", "star", "AS", "NS:20"])
    S2 = {s: stats(M2[s], g2, pi) for s in M2}
    P1d = bool(S2["NS:20"]["tv"] <= S2["star"]["tv"] + 0.005)
    res["P1d"] = dict(tv_q=S2["q"]["tv"], tv_star=S2["star"]["tv"], tv_AS=S2["AS"]["tv"], tv_NS=S2["NS:20"]["tv"], mean_NS=S2["NS:20"]["mean"], var_NS=S2["NS:20"]["var"],
                      mean_AS=S2["AS"]["mean"], var_AS=S2["AS"]["var"], tv_q_vs_q0=float(0.5 * np.abs(M2['q'] / M2['q'].sum() - q2.cells(g2['xf'])).sum()))
    print(f"[1] P1d q = 1/2 N(0.8,.02) + 1/2 N(1.3,.02): TV(next, pi): D_q {S2['q']['tv']:.4f}, D_* {S2['star']['tv']:.2e}, AS {S2['AS']['tv']:.4f}, NS {S2['NS:20']['tv']:.2e}; "
          f"NS mean {S2['NS:20']['mean']:.5f} var {S2['NS:20']['var']:.5f}; AS mean {S2['AS']['mean']:.5f} var {S2['AS']['var']:.5f} -> {P1d}")
    verdict = "Théorème A confirmé" if (P1a and P1b and P1c and P1d) else ("non évaluable (solveur non validé)" if not P1a else "Théorème A non confirmé")
    res.update(P1a=P1a, P1b=P1b, P1c=P1c, P1d_pass=P1d, verdict=verdict, time_s=time.perf_counter() - T0)
    print(f"[1] verdict: {verdict}  (total {res['time_s']:.1f} s)")
    save(1, res)


def part2():
    sig = math.sqrt(0.05); res = {}
    sym = Mix([0.5, 0.5], [-1, 1], [0.05, 0.05])
    # (b) symmetric displacement
    qb = Mix([0.5, 0.5], [-1.05, 1.05], [0.05, 0.05])
    spec = ["q", "star", "AS", "NS:20", "NSHBpost:20", "NSHBtarget:20"]
    M, g = run(sym, qb, sig, spec); S = {s: stats(M[s], g, sym) for s in spec}
    hq, hs = S["q"]["half_sep"], S["star"]["half_sep"]
    R = {s: (S[s]["half_sep"] - hs) / (hq - hs) for s in spec}
    valid_b = bool(abs(S["star"]["left"] - sym.left_mass()) <= 0.005 and abs(S["q"]["left"] - qb.left_mass()) <= 0.005)
    P2b = bool(valid_b and abs(R["NSHBpost:20"]) <= 0.1)
    res["b"] = dict(half_sep_q=hq, half_sep_star=hs, R=R, valid=valid_b, P2b=P2b)
    print(f"[2b] half-separation q {hq:.5f} (1.05), star {hs:.5f} (1.0); R: AS {R['AS']:.3f}, NS {R['NS:20']:.3f}, NS+HB(post) {R['NSHBpost:20']:.3f}, NS+HB(target) {R['NSHBtarget:20']:.3f}; valid {valid_b} -> P2b {P2b}  ({time.perf_counter() - T0:.1f} s)")
    # (c) missing mode
    qc = Mix([1], [1.0], [0.05])
    M, g = run(sym, qc, sig, spec); S = {s: stats(M[s], g, sym) for s in spec}
    wq, ws = S["q"]["left"], S["star"]["left"]
    valid_c = bool(abs(ws - sym.left_mass()) <= 0.005 and abs(wq - qc.left_mass()) <= 0.005)
    P2c1 = bool(valid_c and abs(S["AS"]["left"] - wq) <= 1e-4 and abs(S["NS:20"]["left"] - wq) <= 1e-4)
    P2c2 = bool(valid_c and 0.47 <= S["NSHBpost:20"]["left"] <= 0.53)
    res["c"] = dict(left={s: S[s]["left"] for s in spec}, left_q0=qc.left_mass(), valid=valid_c, P2c1=P2c1, P2c2=P2c2, tv={s: S[s]["tv"] for s in spec})
    print(f"[2c] left mass: q {wq:.3e} (exact {qc.left_mass():.3e}), star {ws:.5f}; AS {S['AS']['left']:.3e}, NS {S['NS:20']['left']:.3e}, NS+HB(post) {S['NSHBpost:20']['left']:.5f}, NS+HB(target) {S['NSHBtarget:20']['left']:.5f}; "
          f"valid {valid_c} -> P2c1 {P2c1}, P2c2 {P2c2}  ({time.perf_counter() - T0:.1f} s)")
    # (d) wrong weights
    pid = Mix([10 / 11, 1 / 11], [-1, 1], [0.05, 0.05])
    M, g = run(pid, sym, sig, spec); S = {s: stats(M[s], g, pid) for s in spec}
    wq, ws = S["q"]["left"], S["star"]["left"]
    valid_d = bool(abs(ws - pid.left_mass()) <= 0.005 and abs(wq - sym.left_mass()) <= 0.005)
    P2d1 = bool(valid_d and abs(S["AS"]["left"] - wq) <= 0.01 and abs(S["NS:20"]["left"] - wq) <= 0.01)
    P2d2 = bool(valid_d and 0.889 <= S["NSHBpost:20"]["left"] <= 0.929)
    res["d"] = dict(left={s: S[s]["left"] for s in spec}, valid=valid_d, P2d1=P2d1, P2d2=P2d2, tv={s: S[s]["tv"] for s in spec})
    print(f"[2d] left mass: q {wq:.5f} (0.5), star {ws:.5f} (0.9091); AS {S['AS']['left']:.5f}, NS {S['NS:20']['left']:.5f}, NS+HB(post) {S['NSHBpost:20']['left']:.5f}, NS+HB(target) {S['NSHBtarget:20']['left']:.5f}; "
          f"valid {valid_d} -> P2d1 {P2d1}, P2d2 {P2d2}")
    vC = "Théorème C (localisation, cas Stein) confirmé" if (P2c1 and P2d1) else "Théorème C (cas Stein) non confirmé"
    vB = "Proposition B confirmée" if (P2b and P2c2 and P2d2) else "Proposition B non confirmée"
    res.update(verdict_C=vC, verdict_B=vB, time_s=time.perf_counter() - T0)
    print(f"[2] verdicts: {vC}; {vB}  (total {res['time_s']:.1f} s)")
    save(2, res)


def w_geo(s2, p, xx=np.linspace(-8, 8, 200001)):
    q = np.exp(-0.5 * (xx - 1) ** 2 / s2); pi = 0.5 * np.exp(-0.5 * (xx + 1) ** 2 / s2) + 0.5 * np.exp(-0.5 * (xx - 1) ** 2 / s2)
    lg = (1 - p) * np.log(q + 1e-300) + p * np.log(pi + 1e-300)
    gg = np.exp(lg - lg.max())
    return float(gg[xx < 0].sum() / gg.sum())


def part3():
    s2 = 0.16; sig = math.sqrt(s2)
    pi = Mix([0.5, 0.5], [-1, 1], [s2, s2]); q = Mix([1], [1.0], [s2])
    ps = [0.0, 0.5, 0.75, 0.9, 1.0]
    spec = ["q", "star"] + [f"P:{p}" for p in ps]
    M, g = run(pi, q, sig, spec); S = {s: stats(M[s], g, pi) for s in spec}
    pred = {p: w_geo(s2, p) for p in ps}
    meas = {p: S[f"P:{p}"]["left"] for p in ps}
    lr = {p: math.log(meas[p] / pred[p]) for p in ps}
    val = bool(abs(meas[1.0] - 0.5) <= 0.01 and abs(meas[0.0] - 0.00621) <= 0.002)
    P3 = bool(val and all(abs(lr[p]) <= 1.5 for p in (0.5, 0.75, 0.9)))
    for p in ps:
        print(f"[3] p = {p}: w(p) = {meas[p]:.5f}, w_geo(p) = {pred[p]:.5f}, ln ratio {lr[p]:+.3f}")
    verdict = "Proposition C2 confirmée" if P3 else "Proposition C2 non confirmée"
    res = dict(measured=meas, predicted=pred, ln_ratio=lr, validation=val, P3=P3, verdict=verdict, left_q=S["q"]["left"], left_star=S["star"]["left"], time_s=time.perf_counter() - T0)
    print(f"[3] validation {val}; P3 {P3}; verdict: {verdict}  ({res['time_s']:.1f} s)")
    save(3, res)


def part4():
    from scipy.special import ndtr as Phi
    rng = np.random.default_rng(0)
    d, n, prec, step, iters = 50, 100_000, 16.0, 0.05, 200
    muf = 0.75 * np.ones(d); D = 0.75 * math.sqrt(d)
    X0 = rng.standard_normal((n, d)); X = X0.copy()
    for _ in range(iters):
        l0 = -0.5 * prec * (X ** 2).sum(1); lf = -0.5 * prec * ((X - muf) ** 2).sum(1)
        rf = expit(lf - l0)
        X = X + step * prec * ((1 - rf)[:, None] * (-X) + rf[:, None] * (muf - X))
    far = ((X - muf) ** 2).sum(1) < (X ** 2).sum(1)
    dist_end = np.where(far, np.sqrt(((X - muf) ** 2).sum(1)), np.sqrt((X ** 2).sum(1)))
    half = X0.sum(1) / math.sqrt(d) > D / 2
    p_hat = float(far.mean()); agree = float((far == half).mean())
    blocks = far.reshape(100, 1000).any(1).mean()
    theory = float(Phi(-D / 2))
    P4 = bool(0.0034 <= p_hat <= 0.0046 and agree >= 0.999 and 0.93 <= blocks <= 1.0)
    res = dict(p_hat=p_hat, p_theory=theory, agree=agree, coverage_1000=float(blocks), coverage_theory=1 - (1 - theory) ** 1000,
               max_dist_to_center_end=float(dist_end.max()), P4=P4, verdict="prédiction de couverture par recherche externe confirmée" if P4 else "non confirmée",
               grad_evals=n * iters, time_s=time.perf_counter() - T0)
    print(f"[4] p_hat {p_hat:.5f} (theory {theory:.5f}); agreement with half-space {agree:.5f}; blocks of 1000 with >= 1 far hit {blocks:.2f} (theory {res['coverage_theory']:.3f}); "
          f"max distance to the arrival centre {res['max_dist_to_center_end']:.2e}; gradient evaluations {n * iters:.1e} -> P4 {P4}  ({res['time_s']:.1f} s)")
    save(4, res)


def part5():
    res = {}
    for s2 in (0.05, 0.09):
        t = time.perf_counter()
        sig = math.sqrt(s2)
        pi = Mix([0.5, 0.5], [-1, 1], [s2, s2]); q = Mix([1], [1.0], [s2])
        M, g = run(pi, q, sig, ["q", "star", "AS", "IDEM"], ng=16)
        S = {s: stats(M[s], g, pi) for s in M}
        w = S["IDEM"]["left"]
        lo, hi = 0.0, 1.0
        if w <= w_geo(s2, 0.0):
            peff = 0.0
        else:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if w_geo(s2, mid) < w: lo = mid
                else: hi = mid
            peff = 0.5 * (lo + hi)
        res[s2] = dict(left_q=S["q"]["left"], left_star=S["star"]["left"], left_AS=S["AS"]["left"], left_IDEM=w, p_eff=peff, time_s=time.perf_counter() - t)
        print(f"[5] sigma^2 {s2}: left mass q {S['q']['left']:.3e}, star {S['star']['left']:.4f}, AS {S['AS']['left']:.3e}, AS + iDEM-like CV {w:.3e}; p_eff {peff:.3f}  ({time.perf_counter() - t:.1f} s)")
    res["time_s"] = time.perf_counter() - T0
    save(5, res)


def part2diag():
    """ POST HOC, descriptive (written after part 2): pointwise decomposition of the exact gap D_* - D_q in panel (b) into the within-mode part
        sum_k wq_k (a'_k - E_qk[X1])/(1 - nu) and the label part sum_k (W_k - wq_k) a'_k/(1 - nu), compared with NS - D_q and with the heat-bath term. """
    xi, wg = hermite(48)
    sym = Mix([0.5, 0.5], [-1, 1], [0.05, 0.05]); qb = Mix([0.5, 0.5], [-1.05, 1.05], [0.05, 0.05])
    rows = []
    for nu in (0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 0.99):
        x = np.array([-0.6, 0.0, 0.3, 0.6, 1.0]) * math.sqrt(nu) * 1.5
        D = drifts(x, nu, sym, qb, ["q", "star", "NS:20", "NSHBpost:20"], xi, wg)
        pq = posterior(qb, x, nu, xi, wg); pp = posterior(sym, x, nu, xi, wg)
        within = (pq["a"] * (pp["mk"] - pq["mk"])).sum(1) / (1 - nu)
        label = ((pp["a"] - pq["a"]) * pp["mk"]).sum(1) / (1 - nu)
        P = 1 / 0.05 + nu / (1 - nu)
        for i in range(len(x)):
            rows.append(dict(nu=nu, x=float(x[i]), gap=float(D["star"][i] - D["q"][i]), NS_minus_q=float(D["NS:20"][i] - D["q"][i]),
                             HB=float(D["NSHBpost:20"][i] - D["NS:20"][i]), within=float(within[i]), label=float(label[i]), Delta_over_std=2 / (0.05 * math.sqrt(P))))
            r = rows[-1]
            print(f"nu {nu:4.2f} x {r['x']:+.2f} | gap {r['gap']:+.4f} NS-Dq {r['NS_minus_q']:+.4f} HB {r['HB']:+.4f} | within {r['within']:+.4f} label {r['label']:+.4f} | Delta/std {r['Delta_over_std']:.1f}")
    save("2diag", dict(note="post hoc, descriptive", rows=rows))


def part3b():
    """ POST HOC, descriptive (written after part 3): constant-fraction drift interpolation in the paper's setting (sigma^2 = 0.05), mass transferred to
        the missing mode in one iteration, against the geometric mixture and the arithmetic mixture (1 - p) q + p pi. """
    s2 = 0.05; sig = math.sqrt(s2)
    pi = Mix([0.5, 0.5], [-1, 1], [s2, s2]); q = Mix([1], [1.0], [s2])
    ps = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    M, g = run(pi, q, sig, [f"P:{p}" for p in ps]); S = {p: stats(M[f"P:{p}"], g)["left"] for p in ps}
    out = {}
    for p in ps:
        out[p] = dict(measured=S[p], geometric=w_geo(s2, p), arithmetic=(1 - p) * q.left_mass() + p * 0.5)
        print(f"[3b] p = {p}: left mass {S[p]:.3e}; geometric {out[p]['geometric']:.3e}; arithmetic {out[p]['arithmetic']:.3e}")
    save("3b", dict(note="post hoc, descriptive", rows=out))


if __name__ == "__main__":
    part = sys.argv[1]
    {"1": part1, "2": part2, "3": part3, "4": part4, "5": part5, "2diag": part2diag, "3b": part3b}[part]()
    print(f"TOTAL {time.perf_counter() - T0:.1f} s (budget 60 s){'  *** OVER BUDGET ***' if time.perf_counter() - T0 > 60 else ''}")
