""" 1D quadrature (Failure Modes): speed of weight correction of AS alone versus AS + darting control variate. Rules: rules_quadrature_1d.md (written before this script).
    CPU only, deterministic, no Monte Carlo (Gauss-Hermite), no network, one script.  python scripts/quadrature_1d_failure_modes.py [--quick]
    In the variable nu (nu_t = int sigma^2): d_nu p = -d_x(D p) + 1/2 d_xx p, D = E[d log phi(X1) | x] (+ E[c | x] for the control variate); the SDE drift is sigma_t u = sigma_t^2 D. """
import json, math, sys, time
import numpy as np
from scipy.linalg import solve_banded
from scipy.special import erf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T_START = time.perf_counter()
QUICK = "--quick" in sys.argv
W, WP = 0.8, 0.5
SMAX, SMIN = 3.0, 0.001
RR = SMAX / SMIN
C2 = 2 * math.log(RR)
NU1 = SMAX ** 2 * (1 - RR ** -2)
EPS_LIST = [0.6, 0.5, 0.4, 0.35, 0.3, 0.25]
ETAS = [0.0, 0.1, 0.3]
LADDER = {"A": dict(dxdiv=20, nt=400, ng=48), "B": dict(dxdiv=10, nt=300, ng=48), "C": dict(dxdiv=8, nt=240, ng=32)}
SQ2, SQPI = math.sqrt(2.0), math.sqrt(math.pi)
MU = np.array([1.0, -1.0])


def t_of_nu(nu):
    return -np.log1p(-np.asarray(nu) / SMAX ** 2) / C2


class Quad:
    def __init__(self, ng):
        xi, wg = np.polynomial.hermite.hermgauss(ng)
        self.xi, self.wg = xi, wg / SQPI


def logpi(z, w, eps):
    return np.logaddexp(math.log(w) - (z - 1) ** 2 / (2 * eps ** 2), math.log(1 - w) - (z + 1) ** 2 / (2 * eps ** 2)) - math.log(math.sqrt(2 * math.pi) * eps)


def dlogpi(z, w, eps):
    return (np.tanh(0.5 * (math.log(w / (1 - w)) + 2 * z / eps ** 2)) - z) / eps ** 2


def posterior(Q, x, kappa, v, w, eps):
    """ nodes z [nx, 2, ng] and weights [nx, 2, ng] of the posterior of X1 given X_t = x under the terminal mixture (w, 1-w; +-1; eps^2). """
    prec = 1 / eps ** 2 + kappa ** 2 / v
    s = prec ** -0.5
    m = (MU[None, :] / eps ** 2 + kappa * x[:, None] / v) / prec                     # [nx, 2]
    vv = v + kappa ** 2 * eps ** 2
    lw = np.log([w, 1 - w])[None, :] - 0.5 * (x[:, None] - kappa * MU[None, :]) ** 2 / vv
    lw -= lw.max(1, keepdims=True)
    a = np.exp(lw)
    a /= a.sum(1, keepdims=True)
    z = m[:, :, None] + SQ2 * s * Q.xi[None, None, :]
    return z, a[:, :, None] * Q.wg[None, None, :]


def drifts(Q, x, nu, eps, wq, need, etas=None):
    """ drifts D on the points x at the variance clock nu, and E_q[omega | x] for each eta. need: set of {'q', 'star', 'AS', 'CV'}. """
    kappa = nu / NU1
    v = nu * (1 - kappa)
    wc = NU1 - nu
    out, om_out = {}, {}
    zq, Wq = posterior(Q, x, kappa, v, wq, eps)
    if "AS" in need or "CV" in need:
        D_as = ((dlogpi(zq, W, eps) + zq / NU1) * Wq).sum((1, 2))
        out["AS"] = D_as
    if "q" in need:
        out["q"] = ((dlogpi(zq, wq, eps) + zq / NU1) * Wq).sum((1, 2))
    if "star" in need:
        zp, Wp = posterior(Q, x, kappa, v, W, eps)
        out["star"] = ((dlogpi(zp, W, eps) + zp / NU1) * Wp).sum((1, 2))
    if "CV" in need:
        xx = x[:, None, None]
        lt = logpi(zq, W, eps) - (xx - kappa * zq) ** 2 / (2 * v)
        sgn = np.where(zq >= 0, 1.0, -1.0)
        for eta in (ETAS if etas is None else etas):
            y = -sgn + (1 + eta) * (zq - sgn)                                            # y = m_j + (1+eta)(x1 - m_k)
            valid = np.abs(zq) < 1 + 1 / (1 + eta)                                       # y falls in cell j
            ly = logpi(y, W, eps) - (xx - kappa * y) ** 2 / (2 * v) + math.log1p(eta)
            om = valid * 0.5 * (1 + np.tanh(0.5 * (ly - lt)))                            # Barker
            out[f"CV{eta}"] = D_as + ((om * (y - zq) / wc) * Wq).sum((1, 2))
            om_out[eta] = (om * Wq).sum((1, 2))
    return out, om_out


def fp_step(p, Df, h, dx, theta):
    """ one conservative finite-volume step of d_nu p = -d_x(D p) + 1/2 d_xx p (zero flux at both ends); theta = 1: implicit Euler, 0.5: Crank-Nicolson. Df: drift at the nx+1 faces. """
    n = len(p)
    aL = Df / 2 + 1 / (2 * dx)                      # F_j = aL_j p_{j-1} + aR_j p_j,  F = D p_face - (1/2) d_x p
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


def make_grid(eps, par, nu0_scale=1.0):
    dx = eps / par["dxdiv"]
    L = max(1 + 8 * eps, 6 * math.sqrt(NU1))
    half = int(math.ceil(L / dx))
    nx = 2 * half
    xf = (np.arange(nx + 1) - half) * dx                                                 # faces, 0 is a face
    xc = 0.5 * (xf[1:] + xf[:-1])
    nu0 = (5 * dx) ** 2 * nu0_scale
    wend = (dx / 10) ** 2
    n1 = int(round(0.3 * par["nt"])); n2 = par["nt"] - n1
    g1 = nu0 * ((NU1 / 2) / nu0) ** (np.arange(n1 + 1) / n1)
    g2 = NU1 - (NU1 / 2) * (wend / (NU1 / 2)) ** (np.arange(n2 + 1) / n2)
    nus = np.concatenate([g1, g2[1:]])
    return dict(dx=dx, nx=nx, xf=xf, xc=xc, nus=nus, nu0=nu0, wend=wend)


def gauss_cells(g, var):
    s = math.sqrt(var)
    cdf = 0.5 * (1 + erf(g["xf"] / (s * SQ2)))
    return np.diff(cdf)                                                                  # exact cell masses


def run(eps, par, Q, wq, methods, nu0_scale=1.0, record_omega=False, steps=None):
    """ integrate the requested processes together. methods: subset of ['zero', 'q', 'star', 'AS', 'CV0.0', 'CV0.1', 'CV0.3']. Returns what_hat per method (and E_q[omega] time averages). """
    g = make_grid(eps, par, nu0_scale)
    dx, nx, xf, nus = g["dx"], g["nx"], g["xf"], g["nus"]
    pm = {m: gauss_cells(g, g["nu0"]) / dx for m in methods}                             # densities per cell
    need = set()
    for m in methods:
        if m == "q": need.add("q")
        elif m == "star": need.add("star")
        elif m == "AS": need.add("AS")
        elif m.startswith("CV"): need.add("CV")
    ts = t_of_nu(nus)
    etas = sorted({float(m[2:]) for m in methods if m.startswith("CV")})
    om_int = {e: 0.0 for e in etas}
    nstep = len(nus) - 1 if steps is None else steps
    for n in range(nstep):
        h = nus[n + 1] - nus[n]
        nu_m = 0.5 * (nus[n] + nus[n + 1])
        kappa = nu_m / NU1
        v = nu_m * (1 - kappa)
        X = kappa + 9 * math.sqrt(v + kappa ** 2 * eps ** 2)
        j0 = max(1, int(math.floor((-X - xf[0]) / dx)))
        j1 = min(nx - 1, int(math.ceil((X - xf[0]) / dx)))
        xs = xf[j0:j1 + 1]
        D = {}
        if need:
            D, om = drifts(Q, xs, nu_m, eps, wq, need, etas)
            if record_omega:
                vv = v + kappa ** 2 * eps ** 2
                pq = (wq * np.exp(-0.5 * (xs - kappa) ** 2 / vv) + (1 - wq) * np.exp(-0.5 * (xs + kappa) ** 2 / vv)) / math.sqrt(2 * math.pi * vv)
                dt = ts[n + 1] - ts[n]
                for e in etas:
                    om_int[e] += float((pq * om[e]).sum() * dx) * dt
        theta = 1.0 if n < 4 else 0.5
        for m in methods:
            if m == "zero":
                Df = np.zeros(nx + 1)
            else:
                key = m if m in ("q", "star", "AS") else "CV" + m[2:]
                d = D[key]
                Df = np.empty(nx + 1)
                Df[j0:j1 + 1] = d
                Df[:j0] = d[0]
                Df[j1 + 1:] = d[-1]
                Df[0] = Df[-1] = 0.0
            pm[m] = fp_step(pm[m], Df, h, dx, theta)
    xc = g["xc"]
    out = {m: float(pm[m][xc > 0].sum() * dx) for m in methods}
    res = dict(what=out, mass={m: float(pm[m].sum() * dx) for m in methods}, grid=g)
    if "zero" in methods:
        res["tv_zero"] = float(0.5 * np.abs(pm["zero"] * dx - gauss_cells(g, NU1 - g["wend"])).sum())
    if record_omega:
        res["omega"] = {e: om_int[e] / (ts[nstep] - ts[0]) for e in etas}
    return res


def main():
    par_name = "A"
    Q = Quad(LADDER["A"]["ng"])
    # pilot timing (rules_quadrature_1d.md): representative steps at eps = 0.25 with all drifts, projected over the whole job
    def pilot(par):
        Qp = Quad(par["ng"]); g = make_grid(0.25, par); nus = g["nus"]
        t0 = time.perf_counter(); cnt = 0
        for n in np.linspace(40, len(nus) - 40, 8).astype(int):
            xs = g["xf"][(np.abs(g["xf"]) < 3)]
            drifts(Qp, xs if len(xs) else g["xf"], 0.5 * (nus[n] + nus[n + 1]), 0.25, WP, {"q", "star", "AS", "CV"})
            for _ in range(7):
                fp_step(np.ones(g["nx"]) / g["nx"], np.zeros(g["nx"] + 1), 1e-3, g["dx"], 0.5)
            cnt += 1
        return (time.perf_counter() - t0) / cnt
    for name in ("A", "B", "C"):
        par = LADDER[name]
        proj = pilot(par) * par["nt"] * sum(0.25 / e for e in EPS_LIST) * 1.4
        print(f"pilot rung {name}: projected {proj:.1f} s")
        if proj <= 40 or name == "C":
            par_name = name
            break
    par = dict(LADDER[par_name])
    if QUICK:
        par["nt"] = 100
    Q = Quad(par["ng"])
    print(f"grid rung {par_name}: dx = eps/{par['dxdiv']}, {par['nt']} steps, {par['ng']} Gauss-Hermite nodes per component")
    methods = ["zero", "q", "star", "AS", "CV0.0", "CV0.1", "CV0.3"]
    eps_list = [0.6] if QUICK else EPS_LIST
    R = {}
    for eps in eps_list:
        r = run(eps, par, Q, WP, methods, record_omega=True)
        R[eps] = r
        floor = max(abs(r["what"]["q"] - WP), abs(r["what"]["star"] - W), 1e-6)
        r["floor"] = floor
        print(f"eps {eps}: TV(u=0) {r['tv_zero']:.2e}; w_hat q {r['what']['q']:.5f} (target {WP}), star {r['what']['star']:.5f} (target {W}); floor {floor:.2e}; masses {min(r['mass'].values()):.6f}..{max(r['mass'].values()):.6f}; elapsed {time.perf_counter() - T_START:.1f} s")
    # t0 check at eps = 0.3
    chk = None
    if not QUICK:
        r_half = run(0.3, par, Q, WP, ["q", "AS"], nu0_scale=0.5)
        chk = {m: abs(r_half["what"][m] - R[0.3]["what"][m]) for m in ("q", "AS")}
        print(f"t0/2 check at eps 0.3: |dw_hat| q {chk['q']:.2e}, AS {chk['AS']:.2e} (floor {R[0.3]['floor']:.2e}); elapsed {time.perf_counter() - T_START:.1f} s")
    # iteration: CV eta = 0 at eps = 0.3
    its = []
    if not QUICK:
        wq = WP
        for n in range(1, 6):
            r = run(0.3, par, Q, wq, ["CV0.0"])
            wq = r["what"]["CV0.0"]
            its.append(wq)
            print(f"iteration {n}: w_hat = {wq:.5f}  |w_hat - w| = {abs(wq - W):.5f}")
    total = time.perf_counter() - T_START
    # tables
    rows = []
    print(f"\n{'eps':>5} {'floor':>9} | {'w_hat AS':>9} {'rho AS':>12} | " + " | ".join(f"{'w CV' + e[2:]:>9} {'rho CV' + e[2:]:>10}" for e in ("CV0.0", "CV0.1", "CV0.3")) + " | E_q[omega] eta=0, 0.1, 0.3")
    for eps, r in R.items():
        wq_hat = r["what"]["q"]; fl = r["floor"]
        def rho(m):
            d = r["what"][m] - wq_hat
            return d / (W - WP), abs(d) < 10 * fl
        cells = []
        for m in ("AS", "CV0.0", "CV0.1", "CV0.3"):
            rv, small = rho(m)
            cells.append(f"{r['what'][m]:9.5f} {('<plancher' if small else format(rv, '10.4g')):>10}")
        om = r["omega"]
        print(f"{eps:5.2f} {fl:9.2e} | " + " | ".join(cells) + f" | {om[0.0]:.4f}, {om[0.1]:.4f}, {om[0.3]:.4f}")
        rows.append(dict(eps=eps, floor=fl, what=r["what"], rho={m: rho(m)[0] for m in ("AS", "CV0.0", "CV0.1", "CV0.3")}, below_floor={m: bool(rho(m)[1]) for m in ("AS", "CV0.0", "CV0.1", "CV0.3")}, omega=om, tv_zero=r["tv_zero"]))
    # verdicts
    verdict = {}
    if not QUICK:
        Qs = [x for x in rows if not x["below_floor"]["AS"] and x["rho"]["AS"] > 0]
        slope = r2 = None
        if len(Qs) >= 3:
            X_ = np.array([1 / (2 * x["eps"] ** 2) for x in Qs]); Y_ = np.log([x["rho"]["AS"] for x in Qs])
            A_ = np.vstack([X_, np.ones_like(X_)]).T
            (slope, icpt), res, _, _ = np.linalg.lstsq(A_, Y_, rcond=None)
            r2 = 1 - ((Y_ - (slope * X_ + icpt)) ** 2).sum() / ((Y_ - Y_.mean()) ** 2).sum()
            part1 = bool(r2 >= 0.95 and -1.5 <= slope <= -0.5)
        else:
            part1 = None
        part2 = []
        for x in rows:
            if x["eps"] <= 0.35:
                part2.append((x["rho"]["AS"] < 0.05) if not x["below_floor"]["AS"] else (10 * x["floor"] / (W - WP) < 0.05))
        part2 = bool(all(part2))
        h_as = "non concluante" if part1 is None and part2 else ("soutenue" if part1 and part2 else "non soutenue")
        c1 = bool(all(x["rho"]["CV0.0"] >= 0.5 for x in rows))
        c2 = any(abs(w_ - W) < 0.01 for w_ in its[:3])
        h_cv = "soutenue" if c1 and c2 else "non soutenue"
        verdict = dict(H_AS=h_as, H_AS_part1=part1, H_AS_part2=part2, slope=None if slope is None else float(slope), R2=None if r2 is None else float(r2), n_Q=len(Qs), H_CV=h_cv, H_CV_rho=c1, H_CV_iter=bool(c2))
        print(f"\nH_AS: {h_as} (part 1 affine, slope in [-1.5,-0.5], R2>=0.95: {part1}, slope {slope}, R2 {r2}, |Q| = {len(Qs)}; part 2 rho_AS < 0.05 for eps <= 0.35: {part2})")
        print(f"H_CV: {h_cv} (rho_CV(eta=0) >= 0.5 for all eps: {c1}; |w_hat - w| < 0.01 within 3 iterations: {c2}; iterations {np.round(its, 5).tolist()})")
        # figure
        fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor="#fcfcfb")
        COL = {"AS": "#2a78d6", "CV0.0": "#eb6834", "CV0.1": "#1baf7a", "CV0.3": "#6250d6"}
        MK = {"AS": "o", "CV0.0": "s", "CV0.1": "^", "CV0.3": "D"}
        for m in COL:
            pts = [(1 / (2 * x["eps"] ** 2), x["rho"][m]) for x in rows if not x["below_floor"][m] and x["rho"][m] > 0]
            if pts:
                ax.plot([p[0] for p in pts], np.log([p[1] for p in pts]), color=COL[m], marker=MK[m], ms=7, lw=1.8, label="AS alone" if m == "AS" else f"AS + darting, η = {m[2:]}")
        if slope is not None:
            xs_ = np.linspace(min(1 / (2 * x["eps"] ** 2) for x in rows), max(1 / (2 * x["eps"] ** 2) for x in rows), 20)
            ax.plot(xs_, slope * xs_ + icpt, color="#2a78d6", ls=":", lw=1.2)
            ax.text(0.03, 0.05, f"AS fit: slope {slope:.2f}, R² {r2:.3f} (ε with ρ above 10 × floor)", transform=ax.transAxes, fontsize=8, color="#52514e")
        ax.set(xlabel="1/(2ε²)", ylabel="ln ρ  (fraction of the weight correction in one outer iteration)", title="1D quadrature: ρ versus 1/(2ε²)")
        ax.grid(color="#e4e3de", lw=0.8); ax.spines[["top", "right"]].set_visible(False); ax.set_facecolor("#fcfcfb"); ax.legend(frameon=False, fontsize=8, loc="center right")
        fig.tight_layout(); fig.savefig("figures/quadrature_1d_logrho.png", dpi=150, facecolor=fig.get_facecolor())
    json.dump(dict(rung=par_name, par=par, rows=rows, t0_check=chk, iterations=its, verdict=verdict, total_time_s=total), open("figures/quadrature_1d_results.json" if not QUICK else "/tmp/quad_quick.json", "w"), indent=1)
    print(f"\nTOTAL TIME {total:.1f} s (budget 60 s){'  *** OVER BUDGET ***' if total > 60 else ''}")


if __name__ == "__main__":
    main()
