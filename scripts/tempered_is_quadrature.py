""" rules_tempered_is_quadrature.md: AS regression weighted by r^alpha (r = pi/q, self-normalised) = AS update evaluated on q_alpha ~ q^(1-alpha) pi^alpha, 1D quadrature in the units of Failure Modes (nu_1 = 1, sigma^2 = 0.05).
    python scripts/tempered_is_quadrature.py PANEL [--alphas 0,0.25,0.5,0.9,1]      PANEL in {g, w, m, ess}
    Reuses the Fokker-Planck solver, the grid and the closed-form posteriors (reference runs D_q, D_*) of scripts/cv_failure_modes_checks.py. """
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp, ndtr, logit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_failure_modes_checks as C

T0 = time.perf_counter()
OUT = Path("figures/tempered_is_quadrature.json")
S2 = 0.05; SIG = math.sqrt(S2)
GH_X, GH_W = np.polynomial.hermite.hermgauss(64); GH_W = GH_W / math.sqrt(math.pi)
DY = SIG / 50


def log_qa(q, pi, a, y):
    return (1 - a) * q.logpdf(y) + a * pi.logpdf(y)


def drift_alpha(x, nu, pi, q, a, ygrid):
    """ D_alpha(x) = E_{q~_alpha}[s_pi(X1) + X1 | X_nu = x] """
    tau = math.sqrt((1 - nu) / nu)
    if tau >= 4 * DY:
        y = ygrid; lq = log_qa(q, pi, a, y)
        keep = lq >= lq.max() - 60; y, lq = y[keep], lq[keep]
        lw = lq[None, :] - (x[:, None] - nu * y[None, :]) ** 2 / (2 * nu * (1 - nu))
        w = np.exp(lw - lw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        g = pi.score(y) + y
        return w @ g
    y = x[:, None] / nu + math.sqrt(2) * tau * GH_X[None, :]                   # likelihood N(y; x/nu, tau^2) as the GH weight
    lw = np.log(GH_W)[None, :] + log_qa(q, pi, a, y)
    w = np.exp(lw - lw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    return (w * (pi.score(y) + y)).sum(1)


def run_alpha(pi, q, alphas, dxdiv=20, nt=400):
    """ same grid and time steps as cv_failure_modes_checks.run; methods: 'q', 'star' (closed form) and 'a:<alpha>' """
    dx = SIG / dxdiv
    half = int(math.ceil(6.0 / dx)); nx = 2 * half
    xf = (np.arange(nx + 1) - half) * dx; xc = 0.5 * (xf[1:] + xf[:-1])
    nu0 = (5 * dx) ** 2; wend = (dx / 10) ** 2
    n1 = int(round(0.3 * nt)); n2 = nt - n1
    nus = np.concatenate([nu0 * (0.5 / nu0) ** (np.arange(n1 + 1) / n1), (1 - 0.5 * (wend / 0.5) ** (np.arange(n2 + 1) / n2))[1:]])
    p0 = np.diff(ndtr(xf / math.sqrt(nu0))) / dx
    spec = ["q", "star"] + [f"a:{a}" for a in alphas]
    pm = {s: p0.copy() for s in spec}
    xi, wg = C.hermite(48)
    ygrid = np.arange(-6.0, 6.0 + DY / 2, DY)
    mmax = max(np.abs(pi.m).max(), np.abs(q.m).max()); s2max = max(pi.s2.max(), q.s2.max())
    for n in range(nt):
        h = nus[n + 1] - nus[n]; nu_m = 0.5 * (nus[n] + nus[n + 1])
        X = nu_m * mmax + 9 * math.sqrt(nu_m * (1 - nu_m) + nu_m ** 2 * s2max)
        j0 = max(1, int(math.floor((-X - xf[0]) / dx))); j1 = min(nx - 1, int(math.ceil((X - xf[0]) / dx)))
        xs = xf[j0:j1 + 1]
        D = C.drifts(xs, nu_m, pi, q, ["star"], xi, wg)
        for a in alphas:
            D[f"a:{a}"] = drift_alpha(xs, nu_m, pi, q, a, ygrid)
        theta = 1.0 if n < 4 else 0.5
        for s in spec:
            d = D[s]
            Df = np.empty(nx + 1); Df[j0:j1 + 1] = d; Df[:j0] = d[0]; Df[j1 + 1:] = d[-1]; Df[0] = Df[-1] = 0.0
            pm[s] = C.fp_step(pm[s], Df, h, dx, theta)
    return {s: C.stats(pm[s] * dx, dict(xc=xc, xf=xf), pi) for s in spec}


def qa_left(q, pi, a):
    y = np.arange(-8.0, 8.0, DY / 4) + DY / 8; lq = log_qa(q, pi, a, y)
    return float(np.exp(logsumexp(lq[y < 0]) - logsumexp(lq)))


def ess(q, pi, a):
    y = np.arange(-8.0, 8.0, DY / 4) + DY / 8; lq = q.logpdf(y); lr = pi.logpdf(y) - lq; ldy = math.log(DY / 4)
    l1 = logsumexp(lq + a * lr) + ldy; l2 = logsumexp(lq + 2 * a * lr) + ldy
    return float(math.exp(2 * l1 - l2)), float(2 * l1 - l2)


def save(key, res):
    allr = json.load(open(OUT)) if OUT.exists() else {}
    allr.setdefault(key, {}).update(res)
    json.dump(allr, open(OUT, "w"), indent=1, default=float)


ap = argparse.ArgumentParser(); ap.add_argument("panel"); ap.add_argument("--alphas", default="0,0.25,0.5,0.9,1")
args = ap.parse_args()
AL = [float(a) for a in args.alphas.split(",")]
MEMO = json.load(open("figures/cv_failure_modes_checks.json"))
sym = C.Mix([0.5, 0.5], [-1, 1], [S2, S2])

if args.panel == "g":
    pi, q = C.Mix([1], [1.0], [S2]), C.Mix([1], [1.1], [S2]); Am = -2.153
    S = run_alpha(pi, q, AL); mq, ms = S["q"]["mean"], S["star"]["mean"]
    res = dict(mean_q=mq, mean_star=ms, var_star=S["star"]["var"], alphas={})
    for a in AL:
        A = (S[f"a:{a}"]["mean"] - ms) / (mq - ms); pred = (1 - a) * Am
        res["alphas"][str(a)] = dict(A=A, pred=pred, ok=bool(abs(A - pred) <= 0.05), var_ratio=S[f"a:{a}"]["var"] / S["star"]["var"])
        print(f"[g] alpha {a}: A = {A:+.4f}, predicted {pred:+.4f}, var ratio {res['alphas'][str(a)]['var_ratio']:.4f} -> {abs(A - pred) <= 0.05}")
    if 0.0 in AL:
        res["control"] = dict(A0=res["alphas"]["0.0"]["A"], memo=MEMO["1"]["A_AS"], ok=bool(abs(res["alphas"]["0.0"]["A"] - MEMO["1"]["A_AS"]) <= 1e-3))
        print(f"[g] control alpha 0: {res['control']['A0']:.5f} vs memo {res['control']['memo']:.5f} -> {res['control']['ok']}")
elif args.panel == "w":
    pi, q = C.Mix([10 / 11, 1 / 11], [-1, 1], [S2, S2]), sym
    S = run_alpha(pi, q, AL); res = dict(left_q=S["q"]["left"], left_star=S["star"]["left"], alphas={})
    for a in AL:
        v = S[f"a:{a}"]["left"]; pred = float(1 / (1 + math.exp(-(a * math.log(10)))))
        res["alphas"][str(a)] = dict(left=v, pred=pred, dlogit=float(logit(v) - logit(pred)), ok=bool(abs(logit(v) - logit(pred)) <= 0.05))
        print(f"[w] alpha {a}: left weight {v:.5f}, predicted {pred:.4f}, d logit {logit(v) - logit(pred):+.4f} -> {abs(logit(v) - logit(pred)) <= 0.05}")
    if 0.0 in AL:
        m0 = MEMO["2"]["d"]["left"]["AS"]; res["control"] = dict(left0=res["alphas"]["0.0"]["left"], memo=m0, ok=bool(abs(res["alphas"]["0.0"]["left"] - m0) <= 1e-3))
        print(f"[w] control alpha 0: {res['control']['left0']:.5f} vs memo {m0:.5f} -> {res['control']['ok']}")
elif args.panel == "m":
    pi, q = sym, C.Mix([1], [1.0], [S2])
    S = run_alpha(pi, q, AL); res = dict(left_q=S["q"]["left"], left_star=S["star"]["left"], alphas={})
    for a in AL:
        v = S[f"a:{a}"]["left"]; qa = qa_left(q, pi, a); ratio = v / qa if qa > 0 else float("inf")
        res["alphas"][str(a)] = dict(left=v, qa_left=qa, ratio=ratio, factor3=bool(1 / 3 <= ratio <= 3), asym=math.exp(-2 * a * (1 - a) / S2))
        print(f"[m] alpha {a}: left mass {v:.4e}, q_alpha(x<0) {qa:.4e}, ratio {ratio:.3f}, e^(-2a(1-a)/s2) {math.exp(-2 * a * (1 - a) / S2):.3e}")
    if 0.0 in AL:
        m0 = MEMO["2"]["c"]["left"]["AS"]; res["control"] = dict(left0=res["alphas"]["0.0"]["left"], memo=m0, ok=bool(abs(res["alphas"]["0.0"]["left"] - m0) <= 1e-3))
        print(f"[m] control alpha 0: {res['control']['left0']:.4e} vs memo {m0:.4e} -> {res['control']['ok']}")
elif args.panel == "ess":
    res = {}
    for nm, (pi, q) in dict(g=(C.Mix([1], [1.0], [S2]), C.Mix([1], [1.1], [S2])), w=(C.Mix([10 / 11, 1 / 11], [-1, 1], [S2, S2]), sym), m=(sym, C.Mix([1], [1.0], [S2]))).items():
        res[nm] = {str(a): dict(zip(("ess", "log_ess"), ess(q, pi, a))) for a in AL}
        print(f"[ess] {nm}: " + "; ".join(f"a={a}: {res[nm][str(a)]['ess']:.3e} (log {res[nm][str(a)]['log_ess']:.1f})" for a in AL))
res["time_s"] = time.perf_counter() - T0
save(args.panel, res)
print(f"[{args.panel}] time {res['time_s']:.1f} s")
