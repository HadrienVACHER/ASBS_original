""" rules_ns_damping.md: verdicts R1, R2 (pre-registered) and R3 (descriptive) from the evaluation JSONs (scripts/ns_damping_eval.py). Reads only; prints and writes OUT/verdicts.json.
    python scripts/ns_damping_rules.py OUT [--final_epoch 1500]
    Run names: nsd_d32_D{4,8}_{as,damp,ns}_sx{1,2}  (as = (a) plain AS, reused runs; damp = (b) damping eta = 0.5; ns = (c) Newton-Stein h = 1). """
import argparse, json, math
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--final_epoch", type=int, default=1500)
ap.add_argument("--r3_from", type=int, default=1000); ap.add_argument("--r3_to", type=int, default=1450)
args = ap.parse_args()
OUT, EF = Path(args.out), str(args.final_epoch)
ARMS, MULTS = ("as", "damp", "ns"), ("1", "2")


def q(delta, arm, m):
    p = OUT / f"q_nsd_d32_D{delta}_{arm}_sx{m}.json"
    if not p.exists():
        return None
    r = json.load(open(p))["epochs"].get(EF)
    return r if r and r.get("status") == "evaluated" else None


def osc(series, D=32, nmin=50, zmin=4.0):
    """ series: {epoch: {n: [K], err: [K x D or None]}} in increasing epoch order. Returns (O, n qualifying pairs, median z). z = n ||e||^2 / D (about 1 if the error is pure Monte Carlo noise). """
    eps = sorted(series); zs, neg, tot = [], 0, 0
    for E in eps:
        s = series[E]
        zs += [s["n"][k] * float(np.sum(np.square(s["err"][k]))) / D for k in range(len(s["n"])) if s["n"][k] >= nmin]
    for E1, E2 in zip(eps[:-1], eps[1:]):
        a, b = series[E1], series[E2]
        for k in range(len(a["n"])):
            if a["n"][k] < nmin or b["n"][k] < nmin:
                continue
            ea, eb = np.array(a["err"][k]), np.array(b["err"][k])
            if a["n"][k] * ea @ ea / D < zmin or b["n"][k] * eb @ eb / D < zmin:
                continue
            tot += 1; neg += int(ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb)) < 0)
    O = neg / tot if tot else float("nan")
    return O, tot, float(np.median(zs)) if zs else float("nan")


def label(O, tot):
    if tot < 4:
        return "signal trop faible (moins de 4 paires au-dessus du bruit)"
    return "oscille" if O >= 0.5 else ("n'oscille pas" if O < 0.25 else "intermediaire")


V = dict(final_epoch=args.final_epoch, table={})
for delta in ("4", "8"):
    for m in MULTS:
        for arm in ARMS:
            r = q(delta, arm, m)
            V["table"][f"D{delta}_{arm}_sx{m}"] = None if r is None else dict(Q=r["Q"]["Q"], Q1=r["Q"]["Q1"], Q2=r["Q"]["Q2"], ratio=r["Q"]["ratio"], coverage=r["Q"]["basin_coverage"],
                                                                         frac_converged=r["Q"]["frac_converged"], ess=r["IS_summary"]["ess"], khat=r["IS_summary"]["khat"], tv_raw=r["raw"]["tv_voronoi"])
# ---- R1: (32, 8, 4); at x1 AND at x2: (c) passes Q, and for each of (a), (b): if it passes Q, ESS(c) >= factor x ESS(arm) (2 for (a), 1.5 for (b)); if it fails Q, the condition holds
r1, r1_detail, r1_eval = True, {}, True
for m in MULTS:
    rc, d = q("4", "ns", m), {}
    if rc is None:
        r1_eval = False; r1 = False; d["ns"] = "non evalue"; r1_detail[m] = d; continue
    d["ns_passes_Q"] = rc["Q"]["Q"]
    ok = rc["Q"]["Q"]
    for arm, fac in (("as", 2.0), ("damp", 1.5)):
        ra = q("4", arm, m)
        if ra is None:
            r1_eval = False; ok = False; d[arm] = "non evalue"; continue
        if not ra["Q"]["Q"]:
            d[arm] = dict(passes_Q=False, condition="vraie ((c) passe Q, ce bras non)" if rc["Q"]["Q"] else "fausse ((c) ne passe pas Q)"); continue
        c_ok = bool(rc["IS_summary"]["ess"] >= fac * ra["IS_summary"]["ess"])
        d[arm] = dict(passes_Q=True, ess_arm=ra["IS_summary"]["ess"], ess_ns=rc["IS_summary"]["ess"], ratio=rc["IS_summary"]["ess"] / max(ra["IS_summary"]["ess"], 1e-300), factor=fac, condition=c_ok)
        ok = ok and c_ok
    d["ok"] = bool(ok); r1 = r1 and ok; r1_detail[m] = d
V["R1"] = dict(value=bool(r1), evaluable=r1_eval, detail=r1_detail)
# ---- R2: (32, 8, 8): (c) passes Q at >= 1 multiplier, (a) and (b) at none
st = {arm: [q("8", arm, m) for m in MULTS] for arm in ARMS}
r2_eval = all(x is not None for v in st.values() for x in v)
pas = {arm: [bool(x is not None and x["Q"]["Q"]) for x in v] for arm, v in st.items()}
V["R2"] = dict(value=bool(r2_eval and any(pas["ns"]) and not any(pas["as"]) and not any(pas["damp"])), evaluable=r2_eval, passes_Q=pas)
# ---- R3 (descriptive): eval dumps r3_from..r3_to (every 50 epochs, 5000 samples); also snapshots (20 000 paths, every 250 epochs) where available
R3 = {}
for delta in ("4", "8"):
    for m in MULTS:
        for arm in ARMS:
            p = OUT / f"q_nsd_d32_D{delta}_{arm}_sx{m}.json"
            if not p.exists():
                continue
            J = json.load(open(p))
            dumps = {int(E): s for E, s in J.get("eval_dumps", {}).items() if args.r3_from <= int(E) <= args.r3_to}
            snaps = {int(E): r["modes"] for E, r in J["epochs"].items() if r.get("status") == "evaluated"}
            e = {}
            if len(dumps) >= 2:
                O, tot, mz = osc(dumps); e["dumps"] = dict(epochs=sorted(dumps), O=O, pairs=tot, median_z=mz, label=label(O, tot))
            else:
                e["dumps"] = "non disponible"
            if len(snaps) >= 2:
                O, tot, mz = osc(snaps); e["snapshots"] = dict(epochs=sorted(snaps), O=O, pairs=tot, median_z=mz, label=label(O, tot))
            else:
                e["snapshots"] = "non disponible" if not snaps else dict(epochs=sorted(snaps), median_z=osc(snaps)[2])
            R3[f"D{delta}_{arm}_sx{m}"] = e
V["R3"] = R3
V["decision"] = ("Newton-Stein vaut la peine d'etre poursuivi (R1 ou R2 vrai)" if (V["R1"]["value"] or V["R2"]["value"])
                 else "on arrete : ni R1 ni R2 (Newton-Stein n'apporte rien de plus que l'amortissement sur ces tests)")
json.dump(V, open(OUT / "verdicts.json", "w"), indent=1, default=float)
for k, v in V["table"].items():
    print(k, "absent" if v is None else f"Q {v['Q']} (Q1 {v['Q1']} {v['ratio']:.2f}x, Q2 {v['Q2']} cov {v['coverage']}/8 conv {v['frac_converged']:.3f}) ESS {v['ess']:.4f} khat {v['khat']:.2f} TVraw {v['tv_raw']:.3f}")
print("R1", V["R1"]["value"], "(evaluable)" if V["R1"]["evaluable"] else "(NON evaluable)", json.dumps(V["R1"]["detail"], default=float))
print("R2", V["R2"]["value"], "(evaluable)" if V["R2"]["evaluable"] else "(NON evaluable)", V["R2"]["passes_Q"])
for k, v in R3.items():
    print("R3", k, {s: (x if isinstance(x, str) else {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in x.items() if kk != "epochs"}) for s, x in v.items()})
print("DECISION:", V["decision"])
