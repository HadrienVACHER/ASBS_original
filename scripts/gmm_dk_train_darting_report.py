""" Report of rules_train_darting.md. python scripts/gmm_dk_train_darting_report.py OUT_PREFIX RESULTS_DIR [IS_JSON...]
    RESULTS_DIR holds gmmdk_cont_<a|b>_d<d>_K8_D<Delta>_s0/{eval_samples/ep*.pt, samples_final.pt}; the start samples are results/gmm_dk_cont/<start run>/start_samples.pt; IS_JSON: outputs of gmm_dk_analysis.py --is_only on the final controllers. Report only. """
import glob, json, math, re, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import gmm_dk as G
import darting_vs_is_stepA as A

prefix, rdir = sys.argv[1], Path(sys.argv[2])
isj = {r["run"]: r for f in sys.argv[3:] for r in json.load(open(f))}
CELLS = {"(8,8,8)": (8, 8, 8.0, "gmmdk_d8_K8_D8_s0_sx0.5"), "(32,8,6)": (32, 8, 6.0, "gmmdk_d32_K8_D6_s0_sx0.5")}
K = 8


def metrics(x, d, delta, ptrue):
    A.MM = delta / math.sqrt(2)
    ph = A.props(A.cells(np.asarray(x, dtype=np.float64)[:, :K]))
    cov = ph >= 0.25 * ptrue
    return dict(tv=A.tv(ph, ptrue), cov=int(cov.sum()), maxlog=float(np.abs(np.log(ph[cov] / ptrue[cov])).max()) if cov.any() else float("nan"), p_hat=ph.tolist())


out, verdict_rows = {}, {}
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), facecolor="#fcfcfb", sharey=False)
for ax, (cell, (d, kk, delta, start)) in zip(axs, CELLS.items()):
    ptrue = G.true_proportions(d, K, delta).numpy()
    rng = np.random.default_rng(3); A.MM = delta / math.sqrt(2)
    floor = float(np.median([A.tv(A.props(A.cells(A.draw_pi_K(rng, 10000, "A1"))), ptrue) for _ in range(100)]))
    st = torch.load(ROOT / "results/gmm_dk_cont" / start / "start_samples.pt", weights_only=False)["model_samples"].numpy()
    res = {"floor": floor, "start": metrics(st, d, delta, ptrue), "arms": {}}
    for arm in ("a", "b"):
        name = f"gmmdk_cont_{arm}_d{d}_K8_D{int(delta)}_s0"
        files = sorted(glob.glob(str(rdir / name / "eval_samples" / "ep*.pt")))
        curve = [(int(re.search(r"ep(\d+)", f).group(1)), metrics(torch.load(f, weights_only=False).numpy(), d, delta, ptrue)["tv"]) for f in files]
        fin = metrics(torch.load(rdir / name / "samples_final.pt", weights_only=False)["model_samples"].numpy(), d, delta, ptrue)
        res["arms"][arm] = dict(curve=curve, final=fin, IS=isj.get(name))
        ax.plot([e for e, _ in curve], [v for _, v in curve], marker="o" if arm == "a" else "s", ms=5, lw=1.8, color="#2a78d6" if arm == "a" else "#eb6834", label="(a) AS alone" if arm == "a" else "(b) AS + darting, γ = 1")
    ax.axhline(floor, color="#52514e", ls=":", lw=1); ax.axhline(res["start"]["tv"], color="#52514e", ls="--", lw=1)
    ax.text(0.02, 0.92, "dashed: start; dotted: exact-sampler floor", transform=ax.transAxes, fontsize=7.5, color="#52514e")
    ax.set(title=f"cell {cell}: raw sampler TV versus epoch", xlabel="continuation epoch", yscale="log"); ax.grid(color="#e4e3de", lw=0.8); ax.spines[["top", "right"]].set_visible(False); ax.set_facecolor("#fcfcfb")
    out[cell] = res
axs[0].set_ylabel("TV of the mode proportions (n = 10⁴)"); axs[0].legend(frameon=False, fontsize=8, loc="center right")
fig.tight_layout(); fig.savefig(prefix + "_curves.png", dpi=150, facecolor=fig.get_facecolor())
print(f"{'cell':>9} {'arm':>6} | {'TV raw':>7} {'cov':>4} {'max|log r|':>10} | {'TV IS':>7} {'ESS':>6} {'khat':>5}   (floor n=1e4)")
rt, rf, rs = [], [], []
for cell, res in out.items():
    s = res["start"]
    print(f"{cell:>9} {'start':>6} | {s['tv']:7.4f} {s['cov']:2d}/8 {s['maxlog']:10.3f} |   (start IS values: see RESULTS.md)  floor {res['floor']:.4f}")
    for arm in ("a", "b"):
        f, i = res["arms"][arm]["final"], res["arms"][arm]["IS"]
        print(f"{cell:>9} {'('+arm+')':>6} | {f['tv']:7.4f} {f['cov']:2d}/8 {f['maxlog']:10.3f} | " + (f"{i['is_tv_weighted']:7.4f} {i['is_ess_frac']:6.3f} {i['is_khat']:5.2f}" if i else "   IS not available"))
    a, b = res["arms"]["a"], res["arms"]["b"]
    rt.append(b["final"]["tv"] <= 0.5 * a["final"]["tv"] and b["final"]["cov"] == 8)
    rf.append(b["final"]["tv"] <= 3 * res["floor"])
    rs.append(bool(b["IS"] and a["IS"] and b["IS"]["is_tv_weighted"] <= 2 * res["floor"] and b["IS"]["is_ess_frac"] >= 0.5 * a["IS"]["is_ess_frac"]))
    print(f"{cell:>9} curves (epoch: TV a / b): " + "; ".join(f"{e}: {va:.3f}/{vb:.3f}" for (e, va), (_, vb) in zip(a["curve"], b["curve"])))
print(f"\nR_train (both cells: TV_raw(b) <= 0.5 TV_raw(a) and coverage 8/8): {'TRUE' if all(rt) else 'FALSE'}  per cell {rt}")
print(f"R_fort (descriptive, TV_raw(b) <= 3 floor): per cell {rf}")
print(f"R_surete (TV after IS(b) <= 2 floor and ESS(b) >= 0.5 ESS(a)): {'TRUE' if all(rs) else 'FALSE'}  per cell {rs}")
json.dump(dict(out=out, R_train=bool(all(rt)), R_train_cells=rt, R_fort=rf, R_surete=bool(all(rs)), R_surete_cells=rs), open(prefix + ".json", "w"), indent=1, default=float)
