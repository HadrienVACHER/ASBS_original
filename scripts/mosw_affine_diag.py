""" rules_mosw_affine_freeze.md. CPU, every call < 1 min.
    python scripts/mosw_affine_diag.py M ARM        ARM in {trans, affine}: darting of the raw MoS-w d=10 epoch-250 points (first 5000) with the existing bank, S in {10, 50, 200} (one call, snapshots)
    python scripts/mosw_affine_diag.py F CELL       CELL in {k8d32, mosw10, d128}: MALA step tuned on the first batch of the start file vs on 3 later batches of the (s) run
    python scripts/mosw_affine_diag.py verdict """
import json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import dart_lib as L, basin_lib as B, selftrain_analyse_lib as A
T0 = time.perf_counter(); OUT = ROOT / "figures" / "mosw_affine"; OUT.mkdir(parents=True, exist_ok=True)
SW = (10, 50, 200)


def jsave(p, r): r["time_s"] = time.perf_counter() - T0; json.dump(r, open(p, "w"), indent=1, default=float)


cmd = sys.argv[1]
if cmd == "M":
    arm = sys.argv[2]; pi, mu, K, pt, M, med = A.target("mosw10"); bank = np.load(ROOT / "figures" / "mosw_posthoc" / "bank_d10.npy")
    x = np.load(ROOT / "results" / "mosw_step3" / "step3_results" / "d10" / "endpoints_d10_ep250.npy").astype(np.float64)[:5000]
    raw = A.stats(x, pi, mu, K, pt, M, med); raw.pop("p_vor"); raw.pop("p_basin")
    dr = L.darting(pi, x, mu, pt, SW=SW, seed=99, keep_final=True, minima=bank, force_translation=(arm == "trans"))
    res = dict(arm=arm, raw=raw, map=dr.get("map"), hessian_spread=dr.get("hessian_spread"), mala_step=dr["mala_step"], by_S={})
    for S in SW:
        v = dr["by_S"][S]; Xf = v.pop("X_final"); st = A.stats(Xf, pi, mu, K, pt, M, med); st.pop("p_vor"); st.pop("p_basin")
        res["by_S"][str(S)] = dict(**st, jump_acc=v["jump_acc"], mala_acc=v["mala_acc"], cost=int(v["cost"]))
    jsave(OUT / f"M_{arm}.json", res)
    print(f"M {arm} (map {res['map']}, Hessian spread {res['hessian_spread']:.2e}): raw Q1 {raw['q1_ratio']:.2f} TV basin {raw['tv_basin']:.3f} | " + " | ".join(f"S{S}: Q1 {v['q1_ratio']:.3f} TV basin {v['tv_basin']:.4f} cov {v['cov_basin']} vor {v['tv_vor']:.4f} acc {v['jump_acc']:.2f}" for S, v in res["by_S"].items()) + f" ({res['time_s']:.1f} s)")
elif cmd == "F":
    c = sys.argv[2]; pi, mu, K, pt, M, med = A.target(c); bank = np.load(ROOT / "figures" / "mosw_posthoc" / "bank_d10.npy") if c == "mosw10" else M
    def step(X):
        return L.darting(pi, X[:500].astype(np.float64), mu, pt, SW=(1,), seed=99, minima=bank)["mala_step"]
    st0 = torch.load(ROOT / "results" / "wf_start" / {"k8d32": "k8d32", "mosw10": "mosw10", "d128": "d128"}[c] / "start_samples.pt", map_location="cpu", weights_only=False)["model_samples"].double().numpy()
    hs0 = step(st0); res = dict(cell=c, frozen_step=hs0, later={})
    for e in (50, 250, 450):
        xe = torch.load(ROOT / "results" / "selftrain_runs" / f"sd_s_{c}" / "eval_samples" / f"ep{e:05d}.pt", map_location="cpu", weights_only=False).double().numpy()
        h = step(xe); res["later"][str(e)] = dict(tuned=h, ratio_frozen_over_tuned=hs0 / h)
    res["ok"] = all(0.9 <= v["ratio_frozen_over_tuned"] <= 1.1 for v in res["later"].values())
    jsave(OUT / f"F_{c}.json", res)
    print(f"F {c}: frozen {hs0:.4f}; " + " ".join(f"ep{e} tuned {v['tuned']:.4f} ratio {v['ratio_frozen_over_tuned']:.3f}" for e, v in res["later"].items()) + f" -> in [0.9, 1.1]: {res['ok']} ({res['time_s']:.1f} s)")
elif cmd == "verdict":
    R = {a: json.load(open(OUT / f"M_{a}.json")) for a in ("trans", "affine")}; V = dict(M1=dict(cells={}))
    for a, r in R.items():
        for S, v in r["by_S"].items():
            V["M1"]["cells"][f"{a}_S{S}"] = dict(q1=v["q1_ratio"], tv_basin=v["tv_basin"], ok=bool(v["q1_ratio"] <= 1.5 and v["tv_basin"] <= 0.02))
    V["M1"]["value"] = any(c["ok"] for c in V["M1"]["cells"].values()); okS = sorted(int(k.split("_S")[1]) for k, c in V["M1"]["cells"].items() if c["ok"])
    V["M1"]["smallest_S"] = okS[0] if okS else None; V["M1"]["map"] = {a: R[a]["map"] for a in R}; V["M1"]["hessian_spread"] = R["affine"]["hessian_spread"]
    F = {c: json.load(open(OUT / f"F_{c}.json")) for c in ("k8d32", "mosw10", "d128")}; V["F1"] = dict(per_cell={c: f["ok"] for c, f in F.items()}, ratios={c: [v["ratio_frozen_over_tuned"] for v in f["later"].values()] for c, f in F.items()}, value=all(f["ok"] for f in F.values()))
    jsave(OUT / "verdict.json", V); print(json.dumps(V, indent=1))
