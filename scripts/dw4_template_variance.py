""" rules_dw4_template_variance.md: do richer G-averaged per-basin templates (mixtures of J Gaussians) reduce the spread of log u_a = log N_bar_a + E/tau on DW-4? CPU, each call < 1 min.
    python scripts/dw4_template_variance.py prep
    python scripts/dw4_template_variance.py fit J          (resumable: stops after ~45 s and continues from the checkpoint at the next call; prints DONE when converged)
    python scripts/dw4_template_variance.py eval J
    python scripts/dw4_template_variance.py verdict
    Everything is fitted and evaluated locally in the deterministic Helmert basis of dw4_division_step0 (no cluster parameter). """
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import dw4_division_step0 as s0  # noqa: E402  (sets float64 default)
import dw4_division_b as b  # noqa: E402
from adjoint_samplers.utils.dw4_quench import quench  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, state_ids  # noqa: E402

T0 = time.time()
N0, TFLOOR, MAX_IT, TOL, CHUNK, BUDGET = 50.0, 5e-3, 40, 1e-4, 200, 45.0
ANG = torch.arange(s0.N_ANG) * (2 * math.pi / s0.N_ANG)
TH8 = torch.arange(8) * (2 * math.pi / 8)
TYPES = [7, 2, 3, 9, 5]                                       # triangle+pendant, path of 2, two disjoint pairs, K4 minus an edge, star
SMALL = [3, 9, 5]
SD_REF = {7: 2.66, 2: 1.68, 3: 6.69, 9: 6.46, 5: 3.30}         # step 1 (EM), reference half B
OUT = ROOT / "figures" / "dw4_template_variance"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "results" / "dw4_template_variance_data.pt"
nm = lambda k: NAMES[KEYS[k]]


def mahal(x, mu_c, P):
    """ -0.5 Mahalanobis on the 48 x 256 grid for x (c, 8): (c, 48, 256), and the images (c, 48, 4, 2) """
    imgs = s0.group_act(s0.centre(x))
    w8 = s0.to_v(s0.rot(imgs[:, :, None], TH8[None, None, :])) - mu_c
    m8 = torch.einsum("nhaj,jk,nhak->nha", w8, P, w8)
    cf = torch.nn.functional.pad(torch.fft.rfft(m8, dim=-1), (0, s0.N_ANG // 2 + 1 - 5))
    return -0.5 * torch.fft.irfft(cf, n=s0.N_ANG, dim=-1) * (s0.N_ANG / 8), imgs


def log_mix(x, comps, scale=1.0):
    """ log N_bar(x) for the mixture comps = [(c, mu, Sigma)] with covariances scale * Sigma, x (n, 8) """
    out = []
    for lo in range(0, len(x), 1000):
        xs = x[lo:lo + 1000]
        terms = [math.log(c) + s0.log_qbar(xs, mu, scale * S) for c, mu, S in comps]
        out.append(torch.logsumexp(torch.stack(terms), 0))
    return torch.cat(out)


def tangent(mu_c):
    mu8 = (s0.B @ mu_c).reshape(4, 2)
    t = s0.to_v(torch.stack([-mu8[:, 1], mu8[:, 0]], -1))
    return t / t.norm()


def floor_t(mu, S):
    S = 0.5 * (S + S.T); t = tangent(mu); vt = float(t @ S @ t)
    return S + (TFLOOR - vt) * torch.outer(t, t) if vt < TFLOOR else S


def estep(x, comps):
    """ mixture E-step: per component S0, S1, S2 and the mean log-likelihood """
    J = len(comps)
    S0 = [0.0] * J; S1 = [torch.zeros(6) for _ in range(J)]; S2 = [torch.zeros(6, 6) for _ in range(J)]; ll = 0.0
    pre = [(torch.linalg.inv(S), torch.logdet(2 * math.pi * S)) for _, _, S in comps]
    for lo in range(0, len(x), CHUNK):
        xs = x[lo:lo + CHUNK]
        lrs, lse = [], []
        for (c, mu, S), (P, ld) in zip(comps, pre):
            lr, imgs = mahal(xs, mu, P)
            lrs.append(lr); lse.append(torch.logsumexp(lr, dim=(1, 2)) - 0.5 * ld - math.log(48 * s0.N_ANG) + math.log(c))
        L = torch.stack(lse, 1)                                                   # (c, J)
        tot = torch.logsumexp(L, 1); ll += float(tot.sum())
        R = torch.exp(L - tot[:, None])                                           # component responsibilities
        for j in range(J):
            r = torch.exp(lrs[j] - torch.logsumexp(lrs[j], dim=(1, 2))[:, None, None]) * R[:, j, None, None]
            ci, hi = (r.sum(-1) > 1e-7).nonzero(as_tuple=True)
            if len(ci) == 0:
                continue
            y = s0.to_v(s0.rot(imgs[ci, hi][:, None], ANG[None, :]))
            rr = r[ci, hi]
            S0[j] += float(rr.sum()); S1[j] += torch.einsum("pt,ptj->j", rr, y); S2[j] += torch.einsum("pt,ptj,ptk->jk", rr, y, y)
    return S0, S1, S2, ll / len(x)


def kmeans_init(y, J, gen):
    n = len(y); C = y[torch.randint(n, (1,), generator=gen)]
    for _ in range(1, J):                                                         # k-means++
        d = torch.cdist(y, C).min(1).values ** 2
        C = torch.cat([C, y[torch.multinomial(d / d.sum(), 1, generator=gen)]])
    for _ in range(25):
        lab = torch.cdist(y, C).argmin(1)
        C = torch.stack([y[lab == j].mean(0) if (lab == j).sum() > 0 else C[j] for j in range(J)])
    return torch.cdist(y, C).argmin(1)


def init_comps(x, mu8, J, gen):
    mu, S = s0.fit_basin(x, mu8)[:2]
    if J == 1:
        return [(1.0, mu, S)]
    y = s0.to_v(s0.align(x, mu8)); lab = kmeans_init(y, J, gen); comps = []
    for j in range(J):
        yj = y[lab == j]; nj = len(yj)
        if nj < 2:
            comps.append((max(nj, 1) / len(y), mu.clone(), S.clone())); continue
        d = yj - yj.mean(0); Sj = (d.T @ d / (nj - 1) * nj + N0 * S) / (nj + N0)
        comps.append((nj / len(y), yj.mean(0), floor_t(yj.mean(0), Sj)))
    tot = sum(c for c, _, _ in comps)
    return [(c / tot, m, S) for c, m, S in comps]


def mix_draw(comps, scale, n, gen):
    counts = torch.multinomial(torch.tensor([c for c, _, _ in comps]), n, replacement=True, generator=gen).bincount(minlength=len(comps))
    return torch.cat([b.draw(mu, scale * S, int(k), gen) for (c, mu, S), k in zip(comps, counts) if k > 0])


cmd = sys.argv[1]
if cmd == "prep":
    m = torch.load(ROOT / "results" / "dw4_e1500_ed8bb67_samples.pt", map_location="cpu", weights_only=False)["model_samples"].double()
    qm, gm, _ = quench(m); lab = state_ids(qm.double()); E = s0.energy(m)
    pt = torch.load(ROOT / "results" / "pt_samples_T1.pt", map_location="cpu", weights_only=False)
    X, Lb = pt["states"].double(), pt["basin_state_id"]
    XA, LA = X[:32].reshape(-1, 8), Lb[:32].reshape(-1); XB, LB = X[32:].reshape(-1, 8), Lb[32:].reshape(-1)
    g = torch.Generator().manual_seed(0); mu8, fit, ev, ptb = {}, {}, {}, {}
    for k in TYPES:
        ia = (LA == k).nonzero().flatten(); ia = ia[torch.randperm(len(ia), generator=g)[:200]]
        q, _, _ = quench(XA[ia]); q = q.double(); qk = q[state_ids(q) == k]; Eq = s0.energy(qk)
        mu8[k] = s0.centre(qk[Eq.argmin()][None])[0]
        idx = (lab == k).nonzero().flatten(); fit[k], ev[k] = idx[0::2], idx[1::2]
        ib = (LB == k).nonzero().flatten(); ptb[k] = ib[torch.randperm(len(ib), generator=g)[:2000]]
    torch.save(dict(m=m, lab=lab, E=E, mu8=mu8, fit=fit, ev=ev, ptB=XB, ptB_idx=ptb, quench_conv=float((gm < 1e-4).double().mean())), DATA)
    info = {nm(k): dict(n_as=int((lab == k).sum()), n_fit=len(fit[k]), n_eval=len(ev[k]), n_ptB=len(ptb[k]), n_ptB_available=int((LB == k).sum())) for k in TYPES}
    json.dump(dict(info=info, quench_conv=float((gm < 1e-4).double().mean()), time_s=time.time() - T0), open(OUT / "prep.json", "w"), indent=1)
    print(json.dumps(info, indent=1), f"\nprep {time.time() - T0:.1f} s")

elif cmd == "prepref":                                                            # report-only set: half B (odd indices) of the DEM reference
    D = torch.load(DATA, weights_only=False)
    refd = torch.tensor(np.load(ROOT / "data" / "test_split_DW4.npy", allow_pickle=True)).double()
    Bi = torch.arange(1, len(refd), 2); qr, _, _ = quench(refd[Bi]); lr = state_ids(qr.double())
    D["ref"] = refd[Bi]; D["refB"] = {k: (lr == k).nonzero().flatten() for k in TYPES}; torch.save(D, DATA)
    print({nm(k): len(D["refB"][k]) for k in TYPES}, f"prepref {time.time() - T0:.1f} s")

elif cmd == "fit":
    J = int(sys.argv[2]); D = torch.load(DATA, weights_only=False); ck = ROOT / "results" / f"dw4_template_variance_fit_J{J}.pt"
    xs = {k: D["m"][D["fit"][k]] for k in TYPES}
    if ck.exists():
        st = torch.load(ck, weights_only=False)
    else:
        g = torch.Generator().manual_seed(0)
        st = dict(comps={k: init_comps(xs[k], D["mu8"][k], J, g) for k in TYPES}, hist={k: [] for k in TYPES}, it=0, done=False, time=[])
    while not st["done"] and time.time() - T0 < BUDGET:
        ml = {k: estep(xs[k], st["comps"][k]) for k in TYPES}
        tot = sum(sum(ml[k][0]) for k in TYPES)
        pool = sum(ml[k][0][j] * (ml[k][2][j] / ml[k][0][j] - torch.outer(ml[k][1][j] / ml[k][0][j], ml[k][1][j] / ml[k][0][j]))
                   for k in TYPES for j in range(J) if ml[k][0][j] > 0) / tot
        new = {}
        for k in TYPES:
            S0, S1, S2, ll = ml[k]; n = len(xs[k]); comps = []
            for j in range(J):
                if S0[j] <= 1e-8:
                    c_, mu, S = st["comps"][k][j]; comps.append((1e-6, mu, S)); continue
                mu = S1[j] / S0[j]; Sml = S2[j] / S0[j] - torch.outer(mu, mu)
                comps.append((S0[j] / n, mu, floor_t(mu, (S0[j] * Sml + N0 * pool) / (S0[j] + N0))))
            ctot = sum(c for c, _, _ in comps); new[k] = [(c / ctot, mu, S) for c, mu, S in comps]
            st["hist"][k].append(ll)
        st["comps"] = new; st["it"] += 1
        delta = max(abs(st["hist"][k][-1] - st["hist"][k][-2]) for k in TYPES) if st["it"] > 1 else float("inf")
        st["done"] = bool(delta < TOL or st["it"] >= MAX_IT)
        torch.save(st, ck)
        print(f"J={J} it {st['it']}: mean log-lik " + ", ".join(f"{st['hist'][k][-1]:.3f}" for k in TYPES) + f"  max|change| {delta:.2e}  ({time.time() - T0:.0f} s)", flush=True)
    st["time"].append(time.time() - T0); torch.save(st, ck)
    print(("DONE" if st["done"] else "CONTINUE") + f" J={J} after {st['it']} iterations; this call {time.time() - T0:.1f} s")

elif cmd == "eval":
    J = int(sys.argv[2]); D = torch.load(DATA, weights_only=False); st = torch.load(ROOT / "results" / f"dw4_template_variance_fit_J{J}.pt", weights_only=False)
    assert st["done"], "fit not converged / finished"
    g = torch.Generator().manual_seed(0); res = dict(J=J, iterations=st["it"], loglik={nm(k): st["hist"][k][-1] for k in TYPES}, basins={})
    EB = s0.energy(D["ptB"]); Eref = s0.energy(D["ref"])
    for k in TYPES:
        comps = st["comps"][k]
        ps = [float(b.pair_in_window(mix_draw(comps, 1 - e, b.N_DRAW, g)).double().mean()) for e in b.ETAS]
        ok = [e for e, p in zip(b.ETAS, ps) if p < 0.01]; eta = ok[0] if ok else b.ETAS[-1]
        sets = dict(as_eval=(D["m"][D["ev"][k]], D["E"][D["ev"][k]]), ptB=(D["ptB"][D["ptB_idx"][k]], EB[D["ptB_idx"][k]]), refB=(D["ref"][D["refB"][k]], Eref[D["refB"][k]]))
        r = dict(eta=eta, eta_flagged=not ok, leak=ps)
        for sn, (x, E) in sets.items():
            lu = log_mix(x, comps, 1 - eta) + E / b.TAU
            r[sn] = dict(n=len(x), sd=float(lu.std()), khat=b.khat(torch.exp(lu - lu.max()).numpy()), sd_eta0=float((log_mix(x, comps, 1.0) + E / b.TAU).std()))
            if J == 5 and sn in ("ptB", "as_eval"):
                r[sn]["sd_cov_x2"] = float((log_mix(x, comps, 2 * (1 - eta)) + E / b.TAU).std())
        res["basins"][nm(k)] = r
        print(f"J={J} {nm(k):26} eta {eta}{'*' if not ok else ''}: SD AS-eval {r['as_eval']['sd']:.3f} (khat {r['as_eval']['khat']:.2f}) | PT B {r['ptB']['sd']:.3f} (khat {r['ptB']['khat']:.2f}) | ref B {r['refB']['sd']:.3f}"
              + (f" | cov x2: PT B {r['ptB']['sd_cov_x2']:.3f}" if J == 5 else "") + f"  ({time.time() - T0:.0f} s)", flush=True)
    res["time_s"] = time.time() - T0
    json.dump(res, open(OUT / f"eval_J{J}.json", "w"), indent=1); print(f"eval J={J}: {res['time_s']:.1f} s")

elif cmd == "verdict":
    R = {J: json.load(open(OUT / f"eval_J{J}.json")) for J in (1, 3, 5, 10) if (OUT / f"eval_J{J}.json").exists()}
    c1 = {nm(k): R[1]["basins"][nm(k)]["ptB"]["sd"] / SD_REF[k] - 1 for k in TYPES}
    c1r = {nm(k): R[1]["basins"][nm(k)]["refB"]["sd"] / SD_REF[k] - 1 for k in TYPES}
    coh = all(abs(v) <= 0.15 for v in c1.values())
    V = dict(coherence=dict(rel_dev_ptB=c1, rel_dev_refB_reported=c1r, ok=coh))
    if coh and 5 in R:
        b5 = R[5]["basins"]
        V["V1"] = dict(sd_ptB={nm(k): b5[nm(k)]["ptB"]["sd"] for k in SMALL}, ok=all(b5[nm(k)]["ptB"]["sd"] <= 2.0 for k in SMALL))
        V["V2"] = dict(khat_as_eval={nm(k): b5[nm(k)]["as_eval"]["khat"] for k in SMALL}, khat_ptB_reported={nm(k): b5[nm(k)]["ptB"]["khat"] for k in SMALL},
                       ok=all(b5[nm(k)]["as_eval"]["khat"] < 0.5 for k in SMALL))
        V["negative_control"] = dict(sd_ptB_x2={nm(k): b5[nm(k)]["ptB"]["sd_cov_x2"] for k in SMALL}, ok=all(b5[nm(k)]["ptB"]["sd_cov_x2"] > b5[nm(k)]["ptB"]["sd"] for k in SMALL))
        V["reading"] = ("gabarits trop pauvres : la piste se rouvre sur d'autres cibles" if (V["V1"]["ok"] and V["V2"]["ok"]) else "division fermée (R) quel que soit J")
    else:
        V["reading"] = "arrêt : cohérence non tenue (J = 1 ne reproduit pas l'étape 1 à 15 % près sur PT B)" if not coh else "J = 5 non évalué"
    json.dump(V, open(OUT / "verdict.json", "w"), indent=1); print(json.dumps(V, indent=1, ensure_ascii=False))
