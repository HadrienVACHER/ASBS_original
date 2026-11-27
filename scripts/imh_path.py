""" rules/rules_imh_path.md: independent Metropolis-Hastings in path space, replayed on saved pools of path log weights. CPU, every call < 1 min.
    python scripts/imh_path.py check        numerical equality of the path log weight with girsanov.log_path_weights (500 paths, float64) for cells C and D
    python scripts/imh_path.py A|B|C|D      one cell: 3 pools, IMH replay, IS on the same pool, decomposition of E_q[w^2]
    python scripts/imh_path.py verdict """
import json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import dart_lib as L, basin_lib as B, gmm_dk
T0 = time.time(); OUT = ROOT / "figures" / "imh_path"; OUT.mkdir(parents=True, exist_ok=True)
FLOOR_A = 0.0100


def gmm_target(D, K, DL):
    mu = gmm_dk.means(D, K, DL).double().numpy(); wt = gmm_dk.weights(K).double().numpy(); wt /= wt.sum(); pi = L.Mix("G", mu, wt, 1.0)
    return pi, mu, K, gmm_dk.true_proportions(D, K, DL).numpy(), B.polish_minima(pi)


def split3(n):
    e = np.cumsum([0, (n + 2) // 3, (n + 1) // 3, n // 3]); return [slice(int(e[i]), int(e[i + 1])) for i in range(3)]


def pools(c):
    """ returns (K, pi_true, [(name, lw, labels)]) : labels in 0..K-1 or -1 """
    import torch
    if c == "A":
        pt = gmm_dk.true_proportions(128, 8, 6.0).numpy(); out = []
        for e in (1000, 1250, 1500):
            z = np.load(ROOT / f"figures/gmmdk_step4/logw/is_logw_gmmdk4_d128_sx0.5_ep{e}.npz"); out.append((f"ep{e}", z["lw"], z["cells"]))
        return 8, pt, out
    if c == "B":
        pt = gmm_dk.true_proportions(128, 8, 6.0).numpy(); out = []; R = ROOT / "results/darting_large_sigma"
        for e, tag in ((1000, "d128e1000"), (1250, "d128e1250"), (1500, "d128")):
            lw = np.concatenate([np.load(R / f"sim_{tag}_s{s}_p0of1.npz")["lw"] for s in (1, 2, 3)]); lab = np.concatenate([np.load(R / f"raw_{tag}_s{s}.npz")["basins"] for s in (1, 2, 3)])
            out.append((f"ep{e}", lw, lab))
        return 8, pt, out
    if c == "C":
        z = np.load(ROOT / "figures/mosw_posthoc/target_d10.npz"); pi = L.Mix("S", z["mu"], z["wt"], 1.0); M = z["M"]
        pt = np.array(json.load(open(ROOT / "figures/mog_mos_truth_mosw_d10.json"))["p"])
        R = ROOT / "results/mosw_step3/step3_results/d10"; lw = np.load(R / "is_logw_d10_ep250.npz")["lw"]; x = np.load(R / "endpoints_d10_ep250.npy").astype(np.float64)
        lab, fc, _ = B.basins(pi, x, M); print(f"C: quench converged {fc:.4f}, assigned {(lab >= 0).mean():.4f}", flush=True)
        return 10, pt, [(f"bloc{i + 1}", lw[s], lab[s]) for i, s in enumerate(split3(len(lw)))]
    if c == "D":
        pi, mu, K, pt, M = gmm_target(32, 8, 8.0)
        z = np.load(ROOT / "results/darting_large_sigma/sim_k8x0.5_s1_p0of1.npz")
        st = torch.load(ROOT / "results/wf_start/k8d32/start_samples.pt", map_location="cpu", weights_only=False)
        x = np.concatenate([z["x"].astype(np.float64), st["buffer_x1"].double().numpy()]); lw = np.concatenate([z["lw"], st["logw"].double().numpy()])
        lab, fc, _ = B.basins(pi, x, M); print(f"D: quench converged {fc:.4f}, assigned {(lab >= 0).mean():.4f}", flush=True)
        return K, pt, [(f"bloc{i + 1}", lw[s], lab[s]) for i, s in enumerate(split3(len(lw)))]
    raise ValueError(c)


def chain(lw, seed):
    """ independent MH replayed on the pool in order; returns (indices of the states, accepted flags) """
    rng = np.random.default_rng(seed); lu = np.log(rng.random(len(lw))); n = len(lw); idx = np.empty(n, int); acc = np.zeros(n, bool); cur = 0; idx[0] = 0
    for i in range(1, n):
        if lu[i] < lw[i] - lw[cur]: cur = i; acc[i] = True
        idx[i] = cur
    return idx, acc


def iat_geyer(y):
    """ integrated autocorrelation time of a series: Geyer initial monotone positive sequence """
    n = len(y); z = y - y.mean(); v = float(z @ z) / n
    if v <= 0: return None
    f = np.fft.rfft(z, 2 * n); g = np.fft.irfft(f * np.conj(f))[:n] / n
    m2 = n // 2; G = g[:2 * m2:2] + g[1:2 * m2:2]
    pos = np.nonzero(G <= 0)[0]; m = int(pos[0]) if len(pos) else len(G)
    G = np.minimum.accumulate(G[:m])
    return float(max((-g[0] + 2 * G.sum()) / g[0], 1e-12))


def iat_batch(y, nb=30):
    n = (len(y) // nb) * nb; z = y[:n].reshape(nb, -1).mean(1); v = y[:n].var()
    return float((n // nb) * z.var(ddof=1) / v) if v > 0 else None


def one_pool(K, pt, name, lw, lab, seed):
    n = len(lw); idx, acc = chain(lw, seed); burn = n // 10; s = idx[burn:]; a = acc[burn:]
    longest = cur = 0
    for v in a:
        cur = 0 if v else cur + 1; longest = max(longest, cur)
    states = lab[s]; m = states >= 0
    p_imh = np.bincount(states[m], minlength=K) / m.sum()
    w = np.exp(lw - logsumexp(lw)); ml = lab >= 0
    p_is = np.bincount(lab[ml], weights=w[ml], minlength=K) / w[ml].sum()
    ess_is = float(1.0 / (n * (w ** 2).sum()))
    ess_imh_k, tau_b = {}, {}
    for k in range(K):
        y = (states == k).astype(float); t = iat_geyer(y)
        if t is not None: ess_imh_k[k] = 1.0 / t; tau_b[k] = iat_batch(y)
    # per-indicator ESS of the self-normalised IS (report only)
    ess_is_k = {}
    for k in range(K):
        f = (lab == k).astype(float); ph = float((w * f).sum()); var = float((w ** 2 * (f - ph) ** 2).sum()); v0 = ph * (1 - ph)
        if var > 0 and 0 < ph < 1: ess_is_k[k] = v0 / (n * var)
    # decomposition of E_q[w^2] with normalised weights (mean 1 on the pool)
    wn = w * n; q = np.array([(lab == k).mean() for k in range(K)]); mean_k = np.array([wn[lab == k].mean() if (lab == k).any() else 0.0 for k in range(K)])
    m2_k = np.array([(wn[lab == k] ** 2).mean() if (lab == k).any() else 0.0 for k in range(K)]); pih = q * mean_k
    unas = float((wn[~ml] ** 2).sum() / n)
    chi2 = [float(m2_k[k] / mean_k[k] ** 2 - 1) if q[k] > 0 and mean_k[k] > 0 else None for k in range(K)]
    T = float(sum(pih[k] ** 2 / q[k] for k in range(K) if q[k] > 0)); Ew2 = float((q * m2_k).sum() + unas); Ew2a = float((q * m2_k).sum())
    T_exact = float(sum(pt[k] ** 2 / q[k] for k in range(K))) if np.all(q > 0) else float("inf")
    imh_vals = list(ess_imh_k.values()); is_vals = list(ess_is_k.values())
    return dict(pool=name, n=n, accept=float(a.mean()), longest_reject=int(longest), ess_imh=float(np.median(imh_vals)), ess_imh_min=float(min(imh_vals)), ess_imh_by_k={int(k): v for k, v in ess_imh_k.items()},
                tau_batch_med=float(np.median([t for t in tau_b.values() if t])), ess_is=ess_is, ess_is_indic_med=(float(np.median(is_vals)) if is_vals else None), ratio=float(np.median(imh_vals) / ess_is),
                tv_imh=L.tv(p_imh, pt), tv_is=L.tv(p_is, pt), max_w=float(w.max()), p_imh=p_imh.tolist(), p_is=p_is.tolist(), q=q.tolist(), absent=[int(k) for k in range(K) if q[k] == 0],
                absent_mass_imh=float(sum(p_imh[k] for k in range(K) if q[k] == 0)), few_proposals={int(k): int((lab == k).sum()) for k in range(K) if 0 < (lab == k).sum() <= 10},
                frac_unassigned=float((~ml).mean()), decomposition=dict(Ew2=Ew2, Ew2_assigned=Ew2a, T_prop=T, share_mass=(math.log(max(T, 1e-12)) / math.log(Ew2a) if Ew2a > 1 + 1e-9 else None), T_prop_exact=T_exact, chi2=chi2))


c = sys.argv[1]
if c == "check":
    import torch, train, hydra
    from omegaconf import OmegaConf
    from adjoint_samplers.components.sde import ControlledSDE
    from adjoint_samplers.utils import girsanov, train_utils
    torch.set_default_dtype(torch.float64); res = {}
    for name, cfgp, ckpt in (("C", "results/mosw_step3/asbs/results/mosw3_d10/config.yaml", "results/mosw_step3/asbs/results/mosw3_d10/controller_ep250.pt"),
                             ("D", "results/gmm_dk_window/w1/gmmdk_d32_K8_D8_s0_sx0.5/config.yaml", "results/gmm_dk_window/w1/gmmdk_d32_K8_D8_s0_sx0.5/controller_final.pt")):
        cfg = OmegaConf.load(ROOT / cfgp); OmegaConf.resolve(cfg); ts = train_utils.get_timesteps(**cfg.timesteps).double(); energy = hydra.utils.instantiate(cfg.energy, device="cpu")
        for mod in (energy.dist,) if hasattr(energy, "dist") else ():
            for a_, v_ in list(vars(mod).items()):
                if torch.is_tensor(v_) and v_.is_floating_point(): setattr(mod, a_, v_.double())
        ref = hydra.utils.instantiate(cfg.ref_sde); ctrl = hydra.utils.instantiate(cfg.controller); ctrl.load_state_dict(torch.load(ROOT / ckpt, map_location="cpu", weights_only=False)["controller"]); ctrl.double(); sde = ControlledSDE(ref, ctrl); N, D = 500, int(cfg.dim)
        torch.manual_seed(123)
        with torch.no_grad():
            lw1, x1 = girsanov.log_path_weights(sde, energy, torch.zeros(N, D), ts)
        torch.manual_seed(123); sde.train(False); x = torch.zeros(N, D); logp0 = torch.zeros(N); logq = torch.zeros(N); V = 0.0
        lg = lambda y, m, v: -0.5 * ((y - m) ** 2).sum(-1) / v - 0.5 * D * torch.log(2 * math.pi * v)
        with torch.no_grad():
            for i in range(len(ts) - 1):
                t, dt = ts[i], ts[i + 1] - ts[i]; v = sde.diff(t) ** 2 * dt; mdt = sde.drift(t, x) * dt; eps = sde.randn_like(x); xn = sde.propagate(x, mdt + v.sqrt() * eps)
                logp0 += lg(xn, x, v); logq += lg(xn, x + mdt, v); x = xn; V += float(v)
            lw2 = -energy.eval(x) - lg(x, torch.zeros_like(x), torch.tensor(V)) + logp0 - logq
        d = float((lw1 - lw2).abs().max()); res[name] = dict(max_abs_diff=d, max_abs_diff_endpoints=float((x - x1).abs().max()), lw_sd=float(lw1.std()), V=V, ok=bool(d <= 1e-6))
        print(name, res[name], flush=True)
    json.dump(res, open(OUT / "check.json", "w"), indent=1)
    if not all(v["ok"] for v in res.values()): sys.exit("check failed: stop")
elif c in "ABCD":
    assert json.load(open(OUT / "check.json")) and all(v["ok"] for v in json.load(open(OUT / "check.json")).values()), "run check first"
    K, pt, P = pools(c); rows = [one_pool(K, pt, name, np.asarray(lw, float), np.asarray(lab, int), 1000 + i) for i, (name, lw, lab) in enumerate(P)]
    res = dict(cell=c, K=K, pools=rows)
    for key in ("accept", "longest_reject", "ess_imh", "ess_imh_min", "ess_is", "ess_is_indic_med", "ratio", "tv_imh", "tv_is", "max_w"):
        v = [r[key] for r in rows if r[key] is not None]; res[key] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else None)
    json.dump(res, open(OUT / f"cell_{c}.json", "w"), indent=1, default=float)
    for r in rows:
        dc = r["decomposition"]; print(f"{c} {r['pool']}: acc {r['accept']:.4f} longest {r['longest_reject']} | ESS imh {r['ess_imh']:.2e} (min {r['ess_imh_min']:.2e}) is {r['ess_is']:.2e} (indic {r['ess_is_indic_med']}) ratio {r['ratio']:.3f} | TV imh {r['tv_imh']:.4f} is {r['tv_is']:.4f} | maxw {r['max_w']:.3f} | E[w2] {dc['Ew2']:.3g} T {dc['T_prop']:.3g} share {dc['share_mass']} T_exact {dc['T_prop_exact']:.3g} | absent {r['absent']} few {r['few_proposals']}", flush=True)
    print(f"{c}: {time.time() - T0:.1f} s")
elif c == "verdict":
    R = {k: json.load(open(OUT / f"cell_{k}.json")) for k in "ABCD"}
    def marg(m, s, th): return bool(s is not None and (m - s - th) * (m + s - th) < 0)
    I1 = {k: dict(ratio=R[k]["ratio"], ok=bool(R[k]["ratio"]["mean"] <= 2.0), marginal=marg(R[k]["ratio"]["mean"], R[k]["ratio"]["sd"], 2.0)) for k in "ABCD"}
    I1v = all(v["ok"] for v in I1.values())
    I2 = {k: dict(tv_imh=R[k]["tv_imh"], tv_is=R[k]["tv_is"], ok=bool(R[k]["tv_imh"]["mean"] >= 0.5 * R[k]["tv_is"]["mean"])) for k in "BC"}
    I3 = dict(absent=[p["absent"] for p in R["D"]["pools"]], mass=[p["absent_mass_imh"] for p in R["D"]["pools"]], ok=bool(all(p["absent_mass_imh"] == 0 for p in R["D"]["pools"])))
    rng = np.random.default_rng(7); pt = gmm_dk.true_proportions(128, 8, 6.0).numpy(); nA = R["A"]["pools"][0]["n"] - R["A"]["pools"][0]["n"] // 10
    floor_n = float(np.median([L.tv(rng.multinomial(nA, pt / pt.sum()) / nA, pt) for _ in range(200)]))
    m = R["A"]["tv_imh"]["mean"]; I4 = dict(tv_imh=R["A"]["tv_imh"], threshold=2 * FLOOR_A, floor_at_chain_length=floor_n, ok=bool(m <= 2 * FLOOR_A), ok_at_chain_floor=bool(m <= 2 * floor_n), marginal=marg(m, R["A"]["tv_imh"]["sd"], 2 * FLOOR_A))
    V = dict(I1=dict(per_cell=I1, value=I1v), I2=dict(per_cell=I2, value=all(v["ok"] for v in I2.values())), I3=I3, I4=I4)
    V["all_true"] = bool(V["I1"]["value"] and V["I2"]["value"] and I3["ok"] and I4["ok"]); json.dump(V, open(OUT / "verdict.json", "w"), indent=1, default=float); print(json.dumps(V, indent=1, default=float))
