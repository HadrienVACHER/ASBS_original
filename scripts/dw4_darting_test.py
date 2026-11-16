"""
DW-4: offline test of a minimum-to-minimum DARTING kernel as a Barker control variate.

Question: can a Barker kernel whose jumps are guided by the local quadratic model of each basin
(minimum + Hessian, reliable only near the bottom of the wells) move mass between basins much more
than the 24 reflections (~1 % out-flux from the dominant basins)?

Kernel (generalized darting, Sminchisescu & Welling 2007; smart darting, Andricioaei et al. 2001).
For x in basin type a, target type b != a:
  1. m = quench(x) (precise minimum, same labelling and orientation as x).
  2. For each (perm, flip) such that m lies on the rotation orbit of (perm, flip)(src_ab) (n_a choices,
     = stabiliser size), take the rotation angle from the PROCRUSTES alignment of x itself onto
     (perm, flip)(src_ab) -- not from the quench, which lands at an arbitrary angle on the ring of minima.
     With g = (perm, flip, theta_x):
       z  = g^{-1} x,  d = coords(z - src_ab)   (Procrustes optimality => d is orthogonal to the rotation tangent)
       d' = M_ab d,    y = g(tgt_ab + d')        (d' orthogonal to the tangent of tgt => theta_y = theta_x)
     where, in a fixed frame per unordered pair {a,b}, src_ab/tgt_ab are copies of the two minima
     (tgt = best permutation/reflection/rotation alignment of mu_b onto mu_a) and
       M_ab = E_tgt diag(sqrt(lam_src/lam_tgt)) E_src^T
     maps the 5 vibrational Hessian eigen-directions (ascending) of one basin onto those of the other
     (on the slices orthogonal to the rotation tangents); M_ba = M_ab^{-1} on the slices.
  3. The move is valid only if quench(y) lies on the rotation orbit of (perm, flip)(tgt_ab) AND the
     Procrustes angle of y onto (perm, flip)(tgt_ab) equals theta_x; else weight 0.
     Jacobian of x -> y (polar decomposition of R^6 = rotation angle x 5-dim slice):
       |det M_ab| * f_tgt(d') / f_src(d),   f_mu(d) = |mu| + <d, mu>/|mu|.
  4. Proposal probabilities: b uniform among the K-1 other types, g uniform among the n_a alignments.
     Reverse move from y uses exactly one of its n_b alignments. Barker weight with Hastings ratio:
       log R = -E(y)/tau + E(x)/tau + log Jacobian + log(n_a/n_b) + gaussian(x_t) term,
       omega = 1 / (1 + exp(-log R)),
     gaussian term (kappa > 0): (|x_t - kappa x|^2 - |x_t - kappa y|^2) / (2 v).
  CV term (Rao-Blackwellised over all choices): h(x) = sum_{b,g} omega/((K-1) n_a) * (y - x) / nu_{1|t}.

What the script measures (report only, except the GO rule below):
  T0  exactness: M_ba M_ab = identity on the slice; round trip x -> y -> x error after precise quench.
  T1  detailed balance on the reference: lumped flux balance w_a K_ab = w_b K_ba (bootstrap z).
  T2  out-flux per basin at kappa = 0, 0.1, 0.3, 0.5, 0.7 (x_t drawn from the bridge given x).
  T3  signal on AS samples: Rao-Blackwellised one-step change Delta_b of basin proportions at kappa = 0.

PRE-REGISTERED GO RULE (fixed before running on the real data; do not change after seeing results):
  GO iff all of
   (i)   T0: M_ba M_ab = identity on the slice to 1e-9 and round-trip max error < 1e-6 on valid moves;
   (ii)  T1: for every pair with both lumped fluxes > 1e-3, |log(w_a K_ab / (w_b K_ba))| < 3 bootstrap SE;
   (iii) T2: out-flux at kappa = 0 >= 0.10 for each of the two dominant basins (triangle+pendant, 2-edge path);
   (iv)  T3: for every basin that enters (|log(AS_a/ref_a)| >= 0.2 AND mismatch z >= 2), sign(Delta_b) =
         sign(ref_b - AS_b) and |Delta_b|/SE >= 2.
  Report-only: the kappa > 0 out-fluxes, valid-landing fraction, mean Barker weight.

ADAPT BEFORE RUNNING ON THE REAL DATA (Claude Code):
  * energy()/grad(): replace by the repo's DW-4 energy (same tau), and check on 100 reference samples that
    both agree to 1e-8 (my standalone formula may differ from the repo's by a convention).
  * quench_precise(): may be replaced by the repo's Newton quench, but it MUST converge to
    ||grad|| < 1e-8: the exactness of the kernel (hence the exact zero mean of the CV) rests on the
    alignment g being a deterministic function of x, which requires a precise quench.
  * Basin labels: this script labels basins by the short-pair graph of the quenched minimum
    (edges, sorted degrees). Check the mapping to the repo's names on the reference.

Usage:
  python dw4_darting_test.py --ref data/test_split_DW4.npy --as results/dw4_e1500_ed8bb67_samples.pt \
      --n-ref 3000 --n-as 5000 --nu1 4.0 --seed 0
"""
import argparse, itertools, time
import numpy as np

# ----------------------------------------------------------------------------- energy (REPLACE BY REPO'S)
B_, C_, D0, TAU = -4.0, 0.9, 4.0, 1.0
PAIRS = list(itertools.combinations(range(4), 2))


def pair_e(d):
    return B_ * (d - D0) ** 2 + C_ * (d - D0) ** 4


def energy(X):                                   # X (n, 4, 2) -> (n,)
    return sum(pair_e(np.linalg.norm(X[:, i] - X[:, j], axis=-1)) for i, j in PAIRS) / TAU


def grad(X):                                     # (n, 4, 2)
    g = np.zeros_like(X)
    for i, j in PAIRS:
        r = X[:, i] - X[:, j]; d = np.linalg.norm(r, axis=-1, keepdims=True)
        de = 2 * B_ * (d - D0) + 4 * C_ * (d - D0) ** 3
        g[:, i] += de * r / d; g[:, j] -= de * r / d
    return g / TAU


# ----------------------------------------------------------------------------- geometry
def rc(X):
    return X - X.mean(-2, keepdims=True)


BV = np.linalg.svd(np.eye(8) - np.kron(np.ones((4, 4)) / 4, np.eye(2)))[0][:, :6]   # (8, 6) zero-CoM basis
PERMS = [np.array(p) for p in itertools.permutations(range(4))]
FLIP = np.diag([1.0, -1.0])


def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def g_apply(g, X):                               # g = (perm, flip, theta), X (4, 2)
    p, f, t = g
    Y = X[p] @ FLIP.T if f else X[p]
    return Y @ rot(t).T


def g_inv_apply(g, X):
    p, f, t = g
    Y = X @ rot(-t).T
    if f:
        Y = Y @ FLIP.T
    Z = np.empty_like(Y); Z[p] = Y
    return Z


def _fit_angle(Y, m):
    return np.arctan2(np.sum(Y[:, 0] * m[:, 1] - Y[:, 1] * m[:, 0]), np.sum(Y * m))


def f_vol(mu, d):                                # rotation volume factor |mu| + <d, mu>/|mu| (d, mu as (4, 2))
    nm = np.linalg.norm(mu)
    return nm + np.sum(d * mu) / nm


def procrustes(p, f, mu, x):                     # angle aligning (p, f)(mu) onto x
    Y = mu[p] @ FLIP.T if f else mu[p]
    return _fit_angle(Y, x)


def alignments(m, mu, tol=1e-4):                # all g with g(mu) = m (precise minima => tight tol)
    out = []
    for p in PERMS:
        for f in (0, 1):
            Y = mu[p] @ FLIP.T if f else mu[p]
            t = _fit_angle(Y, m)
            if np.abs(Y @ rot(t).T - m).max() < tol:
                out.append((p, f, t))
    return out


def best_alignment(m, mu):
    best = None
    for p in PERMS:
        for f in (0, 1):
            Y = mu[p] @ FLIP.T if f else mu[p]
            t = _fit_angle(Y, m); r = np.abs(Y @ rot(t).T - m).sum()
            if best is None or r < best[0]:
                best = (r, (p, f, t))
    return best[1]


def num_hess(x, h=1e-5):                         # x (8,) -> (8, 8)
    H = np.zeros((8, 8))
    for i in range(8):
        e = np.zeros(8); e[i] = h
        H[:, i] = (grad((x + e).reshape(1, 4, 2)) - grad((x - e).reshape(1, 4, 2))).ravel() / (2 * h)
    return 0.5 * (H + H.T)


def quench_precise(X, gd_iters=3000, lr=0.015, newton=8, gtol=1e-10):
    X = rc(X.copy())
    for _ in range(gd_iters):
        X -= lr * np.clip(grad(X), -20, 20)
    for n in range(len(X)):
        for _ in range(newton):
            g = grad(X[n:n + 1]).ravel()
            if np.linalg.norm(g) < gtol:
                break
            Hv = BV.T @ num_hess(X[n].ravel()) @ BV
            step = np.linalg.lstsq(Hv, BV.T @ g, rcond=1e-8)[0]
            X[n] = rc((X[n].ravel() - BV @ step).reshape(1, 4, 2))[0]
    return X


def label(Q):                                    # short-pair graph signature of quenched minima
    out = []
    for x in Q:
        deg = np.zeros(4, int); e = 0
        for i, j in PAIRS:
            if np.linalg.norm(x[i] - x[j]) < D0:
                deg[i] += 1; deg[j] += 1; e += 1
        out.append(f"{e}e" + "".join(map(str, sorted(deg, reverse=True))))
    return np.array(out)


def frame(mu):                                   # tangent u, vib eigvecs (6, 5) ascending, eigenvalues
    J = np.stack([-mu[:, 1], mu[:, 0]], 1).ravel()
    u = BV.T @ J; u /= np.linalg.norm(u)
    H6 = BV.T @ num_hess(mu.ravel()) @ BV
    P = np.eye(6) - np.outer(u, u)
    w, V = np.linalg.eigh(P @ H6 @ P)
    idx = np.argsort(-np.abs(w))[:5]; idx = idx[np.argsort(w[idx])]
    V = V[:, idx]; V = V * np.sign(V[np.abs(V).argmax(0), range(5)])
    return u, V, w[idx]


# ----------------------------------------------------------------------------- kernel
ORDER = ["4e3221", "2e2110", "2e1111", "5e3322", "3e3111"]   # triangle+pendant, 2-edge path, two pairs, K4-, star


def build_model(minima):
    mus = [rc(minima[s][None])[0] for s in ORDER]
    K = len(mus); pairs = {}; maxerr = 0.0
    stab = lambda mu: len(alignments(mu, mu))
    for a in range(K):
        for b in range(a + 1, K):
            mua = mus[a]; mub = rc(g_apply(best_alignment(mua, mus[b]), mus[b])[None])[0]
            ua, Ea, la = frame(mua); ub, Eb, lb = frame(mub)
            assert la.min() > 0 and lb.min() > 0, "non-positive vibrational curvature"
            Mab = Eb @ np.diag(np.sqrt(la / lb)) @ Ea.T
            Mba = Ea @ np.diag(np.sqrt(lb / la)) @ Eb.T
            maxerr = max(maxerr, np.abs(Mba @ Mab - Ea @ Ea.T).max(), np.abs(Mab @ Mba - Eb @ Eb.T).max())
            ld = 0.5 * np.sum(np.log(la / lb))
            pairs[(a, b)] = dict(src=mua, tgt=mub, M=Mab, n_tgt=stab(mub), logdet=ld)
            pairs[(b, a)] = dict(src=mub, tgt=mua, M=Mba, n_tgt=stab(mua), logdet=-ld)
    return pairs, maxerr


def proposals(x, m, a, pairs, K):
    out = []
    for b in range(K):
        if b == a:
            continue
        P = pairs[(a, b)]; al = alignments(m, P["src"])
        for (pp, ff, _) in al:
            g = (pp, ff, procrustes(pp, ff, P["src"], x))
            z = rc(g_inv_apply(g, x)[None])[0]
            d8 = z - P["src"]; d = BV.T @ d8.ravel()
            dp8 = (BV @ (P["M"] @ d)).reshape(4, 2)
            y = rc(g_apply(g, P["tgt"] + dp8)[None])[0]
            logjac = P["logdet"] + np.log(f_vol(P["tgt"], dp8)) - np.log(f_vol(P["src"], d8))
            out.append(dict(b=b, g=g, y=y, n_a=len(al), n_b=P["n_tgt"], logdet=logjac))
    return out


def evaluate(X, pairs, K, kappas, nu1, rng, roundtrip=0):
    """Per sample: probability of moving to each type (K x len(kappas)), plus diagnostics."""
    Q = quench_precise(X); lab = label(Q)
    a_of = np.array([ORDER.index(s) if s in ORDER else -1 for s in lab])
    ok = a_of >= 0
    X, Q, a_of = X[ok], Q[ok], a_of[ok]
    props = [proposals(X[n], Q[n], a_of[n], pairs, K) for n in range(len(X))]
    flat = [p for pr in props for p in pr]
    QY = quench_precise(np.stack([p["y"] for p in flat])) if flat else np.zeros((0, 4, 2))
    probs = np.zeros((len(X), K, len(kappas)))
    k = 0; valid_all = []; rt = []; omegas = []
    E_x = energy(X)
    for n, pr in enumerate(props):
        a = a_of[n]
        xt = []
        for kap in kappas:
            v = nu1 * kap * (1 - kap)
            xt.append(kap * X[n] + np.sqrt(v) * rc(rng.normal(size=(1, 4, 2)))[0] if kap > 0 else None)
        for p in pr:
            qy = QY[k]; k += 1; P = pairs[(a, p["b"])]
            pp, ff, th = p["g"]
            on_orbit = np.abs(rc(g_apply((pp, ff, procrustes(pp, ff, P["tgt"], qy)), P["tgt"])[None])[0] - qy).max() < 1e-4
            same_angle = abs(np.angle(np.exp(1j * (procrustes(pp, ff, P["tgt"], p["y"]) - th)))) < 1e-8
            valid = on_orbit and same_angle
            valid_all.append(valid)
            if not valid:
                continue
            base = -energy(p["y"][None])[0] + E_x[n] + p["logdet"] + np.log(p["n_a"] / p["n_b"])
            for ki, kap in enumerate(kappas):
                lr = base
                if kap > 0:
                    v = nu1 * kap * (1 - kap)
                    lr += (np.sum((xt[ki] - kap * X[n]) ** 2) - np.sum((xt[ki] - kap * p["y"]) ** 2)) / (2 * v)
                om = 1 / (1 + np.exp(-np.clip(lr, -50, 50)))
                if ki == 0:
                    omegas.append(om)
                probs[n, p["b"], ki] += om / ((K - 1) * p["n_a"])
            if len(rt) < roundtrip:
                Pr = pairs[(p["b"], a)]; best = np.inf
                for (p2, f2, _) in alignments(qy, Pr["src"]):
                    g2 = (p2, f2, procrustes(p2, f2, Pr["src"], p["y"]))
                    z = rc(g_inv_apply(g2, p["y"])[None])[0]
                    d = BV.T @ (z - Pr["src"]).ravel()
                    xr = rc(g_apply(g2, Pr["tgt"] + (BV @ (Pr["M"] @ d)).reshape(4, 2))[None])[0]
                    best = min(best, np.abs(xr - X[n]).max())
                rt.append(best)
    return dict(a=a_of, probs=probs, valid=np.mean(valid_all) if valid_all else np.nan,
                rt=np.array(rt), omega=np.mean(omegas) if omegas else np.nan)


def load(path, n, rng):
    if path.endswith(".pt"):
        import torch
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            obj = obj.get("samples", next(iter(obj.values())))
        arr = obj.detach().cpu().numpy() if hasattr(obj, "detach") else np.asarray(obj)
    else:
        arr = np.load(path)
    arr = arr.reshape(len(arr), 4, 2).astype(np.float64)
    idx = rng.choice(len(arr), min(n, len(arr)), replace=False)
    return rc(arr[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--as", dest="as_path", default=None)
    ap.add_argument("--n-ref", type=int, default=3000); ap.add_argument("--n-as", type=int, default=5000)
    ap.add_argument("--nu1", type=float, default=4.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nboot", type=int, default=300)
    args = ap.parse_args(); rng = np.random.default_rng(args.seed); t0 = time.time()
    kappas = [0.0, 0.1, 0.3, 0.5, 0.7]

    # minima: lowest-energy quenched representative of each type, from the reference
    R = load(args.ref, args.n_ref, rng)
    QR = quench_precise(R); LR = label(QR); ER = energy(QR)
    minima = {s: QR[LR == s][np.argmin(ER[LR == s])] for s in ORDER}
    pairs, maxerr = build_model(minima); K = len(ORDER)
    print(f"[T0] max |M_ba M_ab - Id_slice| = {maxerr:.2e}   ({time.time()-t0:.0f}s)", flush=True)

    res = evaluate(R, pairs, K, kappas, args.nu1, rng, roundtrip=500)
    a = res["a"]; P0 = res["probs"]
    w = np.array([np.mean(a == i) for i in range(K)])
    print(f"[T0] round trip max error {res['rt'].max():.2e} (n={len(res['rt'])}); valid landing {res['valid']:.3f}; "
          f"mean Barker weight (valid, kappa=0) {res['omega']:.3f}   ({time.time()-t0:.0f}s)", flush=True)
    print("reference weights:", dict(zip(ORDER, np.round(w, 4))))

    # T2: out-flux per basin and kappa
    print("[T2] out-flux (prob. of changing basin per kernel step), kappa =", kappas)
    for i in range(K):
        sel = a == i
        print(f"   {ORDER[i]}: " + "  ".join(f"{P0[sel, :, ki].sum(1).mean():.3f}" for ki in range(len(kappas))))

    # T1: lumped flux balance with bootstrap
    print("[T1] flux balance w_a K_ab vs w_b K_ba (kappa = 0), log-ratio / SE")
    N = len(a)
    def lump(idx):
        aa = a[idx]; pp = P0[idx, :, 0]
        F = np.zeros((K, K))
        for i in range(K):
            F[i] = pp[aa == i].sum(0) / len(idx)          # = w_i K_ij
        return F
    F = lump(np.arange(N)); boots = np.stack([lump(rng.integers(0, N, N)) for _ in range(args.nboot)])
    t1_ok = True
    for i in range(K):
        for j in range(i + 1, K):
            if F[i, j] > 1e-3 and F[j, i] > 1e-3:
                lr = np.log(F[i, j] / F[j, i])
                se = np.std(np.log(boots[:, i, j] / np.maximum(boots[:, j, i], 1e-12)))
                t1_ok &= abs(lr) < 3 * se
                print(f"   {ORDER[i]}-{ORDER[j]}: {F[i, j]:.5f} vs {F[j, i]:.5f}  log-ratio {lr:+.3f}  SE {se:.3f}")

    t0_ok = maxerr < 1e-9 and res["rt"].max() < 1e-6
    t2_ok = all(P0[a == i, :, 0].sum(1).mean() >= 0.10 for i in (0, 1))

    # T3: signal on AS samples
    t3_ok = None
    if args.as_path:
        A = load(args.as_path, args.n_as, rng)
        ra = evaluate(A, pairs, K, [0.0], args.nu1, rng)
        aa, pa = ra["a"], ra["probs"][:, :, 0]
        q = np.array([np.mean(aa == i) for i in range(K)])
        out = pa.sum(1)
        per = np.stack([pa[:, b] - (aa == b) * out for b in range(K)], 1)     # per-sample contribution to Delta_b
        D = per.mean(0); SE = per.std(0) / np.sqrt(len(aa))
        print("[T3] basin   ref     AS      Delta_b    z     enters")
        t3_ok = True
        for b in range(K):
            mis_z = abs(q[b] - w[b]) / np.sqrt(q[b] * (1 - q[b]) / len(aa) + w[b] * (1 - w[b]) / N)
            enters = q[b] > 0 and w[b] > 0 and abs(np.log(q[b] / w[b])) >= 0.2 and mis_z >= 2
            good = np.sign(D[b]) == np.sign(w[b] - q[b]) and abs(D[b]) / SE[b] >= 2
            if enters:
                t3_ok &= bool(good)
            print(f"   {ORDER[b]}  {w[b]:.4f}  {q[b]:.4f}  {D[b]:+.5f}  {D[b]/SE[b]:+6.1f}  {enters}{' OK' if enters and good else ''}")

    print(f"\nGO RULE: T0 {t0_ok}  T1 {t1_ok}  T2 {t2_ok}  T3 {t3_ok}  =>  "
          f"{'GO' if (t0_ok and t1_ok and t2_ok and t3_ok) else 'NO-GO'}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
