""" rules_dw4_guard_power.md: power-calibrated re-analysis of guard (iii) of the DW-4 darting confirmation. CPU, each call < 1 min.
    python scripts/dw4_guard_power.py floor --reps 0:10
    python scripts/dw4_guard_power.py arm NAME --reps 0:10          NAME in D30_s2 D30_s3 D30_s4 C2 C3 C4
    python scripts/dw4_guard_power.py calib SEED                     SEED in 2 3 4 (needs floor and arm D30_s<SEED> complete)
    python scripts/dw4_guard_power.py ctrl SEED --reps 0:10          degraded D30_s<SEED> with the calibrated sigma
    python scripts/dw4_guard_power.py verdict
    Results accumulate in figures/dw4_guard_power/<key>.json (one file per floor / arm / calib / ctrl, replicate index -> values, plus the time of each call). """
import argparse, contextlib, io, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import ot as pot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adjoint_samplers.energies import DoubleWellEnergy  # noqa: E402
from adjoint_samplers.utils.eval_utils import dist_point_clouds_fast  # noqa: E402
from adjoint_samplers.utils.graph_utils import remove_mean  # noqa: E402

N = 2000
OUT = ROOT / "figures" / "dw4_guard_power"; OUT.mkdir(parents=True, exist_ok=True)
ARMS = {"D30_s2": "dw4_cont_D30_s2", "D30_s3": "dw4_cont_D30_s3", "D30_s4": "dw4_cont_D30_s4", "C2": "dw4_cont_C2_s2", "C3": "dw4_cont_C3_s3", "C4": "dw4_cont_C4_s4"}
SIGMAS = (0.05, 0.1, 0.2, 0.3, 0.5)
energy = DoubleWellEnergy(dim=8, n_particles=4)
rm = lambda x: remove_mean(x, 4, 2)
PT = torch.load(ROOT / "results" / "pt_samples_T1.pt", map_location="cpu", weights_only=False)["states"].float()   # (64, 1600, 8)
NC, NT, _ = PT.shape


def w2(x, ref):
    """ energy_w2 and eq_w2 of the official DW-4 evaluation (evaluator.SyntheticEenergyEvaluator with fast_eq) """
    x, ref = rm(x), rm(ref)
    e = pot.emd2_1d(energy.eval(ref).numpy(), energy.eval(x).numpy()) ** 0.5
    M = dist_point_clouds_fast(x.reshape(-1, 4, 2), ref.reshape(-1, 4, 2))
    a = torch.ones(len(x)) / len(x)
    q = pot.emd2(M=M ** 2, a=a, b=a) ** 0.5
    return float(e), float(q)


def pt_draw(chains, g):
    c = chains[torch.randint(len(chains), (N,), generator=g)]
    t = torch.randint(NT, (N,), generator=g)
    return PT[c, t]


def ref_draw(r):                     # common reference of bootstrap replicate r (same for every arm)
    return pt_draw(torch.arange(NC), torch.Generator().manual_seed(50_000 + r))


def arm_samples(name):
    return torch.load(ROOT / "results" / "confirm" / ARMS[name] / "samples_final.pt", map_location="cpu", weights_only=False)["model_samples"].float()


def load(key):
    p = OUT / f"{key}.json"
    return json.load(open(p)) if p.exists() else {"reps": {}, "calls": []}


def save(key, d):
    json.dump(d, open(OUT / f"{key}.json", "w"), indent=1)


def summary(key):
    d = load(key); v = np.array([d["reps"][k] for k in sorted(d["reps"], key=int)])
    return v  # (n_reps, 2): energy_w2, eq_w2


ap = argparse.ArgumentParser()
ap.add_argument("cmd"); ap.add_argument("arg", nargs="?"); ap.add_argument("--reps", default="0:20")
a = ap.parse_args()
t0 = time.time()
r0, r1 = (int(x) for x in a.reps.split(":"))

if a.cmd == "floor":
    d = load("floor")
    for r in range(r0, r1):
        g = torch.Generator().manual_seed(2000 + r)
        perm = torch.randperm(NC, generator=g)
        d["reps"][str(r)] = w2(pt_draw(perm[: NC // 2], g), pt_draw(perm[NC // 2:], g))
    d["calls"].append(dict(reps=a.reps, time_s=time.time() - t0)); save("floor", d)
    print(f"floor reps {a.reps}: {time.time() - t0:.1f} s")

elif a.cmd == "arm":
    X = arm_samples(a.arg); d = load(f"arm_{a.arg}")
    for r in range(r0, r1):
        idx = torch.randint(len(X), (N,), generator=torch.Generator().manual_seed(10_000 + r))
        d["reps"][str(r)] = w2(X[idx], ref_draw(r))
    d["calls"].append(dict(reps=a.reps, time_s=time.time() - t0)); save(f"arm_{a.arg}", d)
    print(f"arm {a.arg} reps {a.reps}: {time.time() - t0:.1f} s")

elif a.cmd == "calib":
    s = int(a.arg); fl = summary("floor"); dv = summary(f"arm_D30_s{s}")
    assert len(fl) == 20 and len(dv) == 20, "floor and arm must be complete (20 replicates)"
    med, sdf = float(np.median(fl[:, 1])), float(np.std(fl[:, 1], ddof=1))
    g_s = max(float(dv[:, 1].mean()) - med, sdf)
    X = arm_samples(f"D30_s{s}"); tried, chosen = {}, None
    for sig in SIGMAS:
        vals = []
        for r in range(2):
            gen = torch.Generator().manual_seed(70_000 + 100 * s + r)
            idx = torch.randint(len(X), (N,), generator=gen)
            vals.append(w2(X[idx] + sig * torch.randn(N, 8, generator=gen), ref_draw(900 + r))[1])
        tried[str(sig)] = float(np.mean(vals))
        if np.mean(vals) - med >= 2 * g_s:
            chosen = sig; break
    flagged = chosen is None
    chosen = SIGMAS[-1] if chosen is None else chosen
    save(f"calib_s{s}", dict(seed=s, floor_median=med, floor_sd=sdf, gap=g_s, target=2 * g_s, tried=tried, sigma=chosen, flagged_no_sigma_reaches=flagged, time_s=time.time() - t0))
    print(f"calib s{s}: gap {g_s:.4f} target {2 * g_s:.4f} tried {tried} -> sigma {chosen}{' (FLAGGED)' if flagged else ''}; {time.time() - t0:.1f} s")

elif a.cmd == "ctrl":
    s = int(a.arg); sig = json.load(open(OUT / f"calib_s{s}.json"))["sigma"]
    X = arm_samples(f"D30_s{s}"); d = load(f"ctrl_s{s}"); d["sigma"] = sig
    for r in range(r0, r1):
        gen = torch.Generator().manual_seed(80_000 + 100 * s + r)
        idx = torch.randint(len(X), (N,), generator=torch.Generator().manual_seed(10_000 + r))
        d["reps"][str(r)] = w2(X[idx] + sig * torch.randn(N, 8, generator=gen), ref_draw(r))
    d["calls"].append(dict(reps=a.reps, time_s=time.time() - t0)); save(f"ctrl_s{s}", d)
    print(f"ctrl s{s} (sigma {sig}) reps {a.reps}: {time.time() - t0:.1f} s")

elif a.cmd == "verdict":
    fl = summary("floor"); V = dict(n=N, floor={})
    for i, m in enumerate(("energy_w2", "eq_w2")):
        V["floor"][m] = dict(median=float(np.median(fl[:, i])), p95=float(np.percentile(fl[:, i], 95)), sd=float(np.std(fl[:, i], ddof=1)), n_reps=len(fl))
    arms = {k: summary(f"arm_{k}") for k in ARMS}
    V["arms"] = {k: {m: dict(mean=float(v[:, i].mean()), sd=float(np.std(v[:, i], ddof=1)), n_reps=len(v)) for i, m in enumerate(("energy_w2", "eq_w2"))} for k, v in arms.items()}

    def g3(D, Cs, i):
        deltas, sds = [], []
        for s in (2, 3, 4):
            dv, cv = D[s][:, i], Cs[s][:, i]
            deltas.append(float(dv.mean() - cv.mean())); sds.append(float(math.sqrt(np.var(dv, ddof=1) + np.var(cv, ddof=1))))
        mean = float(np.mean(deltas)); se = float(math.sqrt(sum(x * x for x in sds)) / 3)
        per_seed_ok = [bool(dl <= 3 * sd) for dl, sd in zip(deltas, sds)]
        return dict(delta=dict(zip(("s2", "s3", "s4"), deltas)), sd_delta=dict(zip(("s2", "s3", "s4"), sds)), mean_delta=mean, se_mean=se,
                    mean_ok=bool(mean <= 2 * se), per_seed_ok=per_seed_ok, G3=bool(mean <= 2 * se and all(per_seed_ok)))
    D = {s: arms[f"D30_s{s}"] for s in (2, 3, 4)}; Cs = {s: arms[f"C{s}"] for s in (2, 3, 4)}
    V["G3_eq_w2"] = g3(D, Cs, 1); V["G3_energy_w2_descriptive"] = g3(D, Cs, 0)
    Dd = {s: summary(f"ctrl_s{s}") for s in (2, 3, 4)}
    V["ctrl"] = {f"s{s}": dict(sigma=json.load(open(OUT / f"ctrl_s{s}.json"))["sigma"], eq_w2_mean=float(Dd[s][:, 1].mean()), eq_w2_sd=float(np.std(Dd[s][:, 1], ddof=1)),
                               calib=json.load(open(OUT / f"calib_s{s}.json"))) for s in (2, 3, 4)}
    V["G3_on_degraded"] = g3(Dd, Cs, 1)
    V["positive_control_detected"] = bool(not V["G3_on_degraded"]["G3"])
    G = V["G3_eq_w2"]["G3"]
    V["decision"] = ("non confirmé" if not G else ("confirmé" if V["positive_control_detected"] else "inconclusif"))
    V["decision_note"] = "(i) et (ii) : verdicts d'origine (vrais sur les graines 2, 3, 4), non recalculés"
    V["times"] = {p.stem: ([c["time_s"] for c in json.load(open(p)).get("calls", [])] or [json.load(open(p)).get("time_s")]) for p in sorted(OUT.glob("*.json")) if p.stem != "verdict"}
    json.dump(V, open(OUT / "verdict.json", "w"), indent=1)
    print(json.dumps({k: V[k] for k in ("floor", "arms", "G3_eq_w2", "G3_energy_w2_descriptive", "G3_on_degraded", "positive_control_detected", "decision")}, indent=1))
