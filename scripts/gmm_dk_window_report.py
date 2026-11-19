""" Report of the sigma_max window study (predictions_dk.md, addendum 3). Report only.
    python scripts/gmm_dk_window_report.py OUT_PREFIX METRICS_JSON... --is IS_JSON...
      METRICS_JSON: outputs of gmm_dk_analysis.py (with or without IS), --is: outputs of --is_only runs (merged by run name).
    Writes OUT_PREFIX_table.json, OUT_PREFIX_d32_K8.png and OUT_PREFIX_d8_K8.png and prints the full table and the best sigma_max per (d, K, Delta). """
import json, math, re, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

prefix = sys.argv[1]
args = sys.argv[2:]
i = args.index("--is") if "--is" in args else len(args)
rows = {}
for f in args[:i]:
    for r in json.load(open(f)):
        rows[r["run"]] = dict(r)
for f in args[i + 1:]:
    for r in json.load(open(f)):
        rows.setdefault(r["run"], {}).update({k: v for k, v in r.items() if k.startswith("is_") or k in ("run", "d", "K", "delta", "p")})

def nu1(smax, smin=0.001, steps=200):
    t = torch.linspace(0, 1, steps).double()
    g2 = smin ** 2 * (smax / smin) ** (2 * (1 - t[:-1])) * 2 * math.log(smax / smin)
    return float((g2 * (t[1:] - t[:-1])).sum())

out = []
for name, r in rows.items():
    m = re.match(r"gmmdk_d(\d+)_K(\d+)_D([\d.]+)_s(\d+)(?:_sx([\d.]+))?$", name)
    d, K, D, seed, mult = int(m[1]), int(m[2]), float(m[3]), int(m[4]), float(m[5]) if m[5] else 1.0
    sig = mult * 2 * math.sqrt(1 + D ** 2 / (2 * d))
    row = dict(run=name, d=d, K=K, delta=D, seed=seed, mult=mult, sigma_max=sig, rho=math.sqrt(d * nu1(sig)) / (D / math.sqrt(2)))
    for k in ("coverage", "tv", "floor_median", "beta", "beta_se", "ratios", "p", "p_hat", "lost", "is_ess_frac", "is_khat", "is_tv_weighted"):
        row[k] = r.get(k)
    out.append(row)
out.sort(key=lambda r: (r["d"], r["K"], r["delta"], r["seed"], r["mult"]))
json.dump(out, open(prefix + "_table.json", "w"), indent=1)
f = lambda v, fmt: "   n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)
print(f"{'d':>3} {'Dl':>3} {'s':>1} {'mult':>5} {'sigma':>6} {'rho':>5} {'cov':>4} {'TV/fl':>6} {'beta':>6} {'ESS':>6} {'khat':>5} {'TV_IS':>6}")
for r in out:
    print(f"{r['d']:3d} {r['delta']:3.0f} {r['seed']:1d} {r['mult']:5.2f} {r['sigma_max']:6.3f} {r['rho']:5.2f} {f(r['coverage'],'d'):>2}/{r['K']:<1d} {f(None if r['tv'] is None else r['tv']/r['floor_median'],'6.1f')} {f(r['beta'],'+6.2f')} {f(r['is_ess_frac'],'6.3f')} {f(r['is_khat'],'5.2f')} {f(r['is_tv_weighted'],'6.3f')}")
print("\nBest sigma_max per (d, K, Delta), seed 0 (max coverage, then min TV):")
for key in sorted({(r["d"], r["K"], r["delta"]) for r in out}):
    c = [r for r in out if (r["d"], r["K"], r["delta"]) == key and r["seed"] == 0 and r["tv"] is not None]
    b = sorted(c, key=lambda r: (-r["coverage"], r["tv"]))[0]
    print(f"  d={key[0]} K={key[1]} Delta={key[2]:.0f}: mult x{b['mult']} (sigma_max {b['sigma_max']:.3f}), coverage {b['coverage']}/{key[1]}, TV {b['tv']:.3f}, beta {f(b['beta'],'+.2f')} ± {f(b['beta_se'],'.2f')}, ratios {b['ratios']}")

COL = {4.0: "#2a78d6", 6.0: "#eb6834", 8.0: "#1baf7a"}; MK = {4.0: "o", 6.0: "s", 8.0: "^"}
INK, MUTED, BG = "#0b0b0b", "#52514e", "#fcfcfb"
for (d, K), tag in (((32, 8), "d32_K8"), ((8, 8), "d8_K8")):
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor=BG)
    for D in [x for x in sorted({r["delta"] for r in out if (r["d"], r["K"]) == (d, K)}) if x in ((4.0, 6.0, 8.0) if d == 32 else (8.0,))]:   # the cells of the sigma_max window study
        for seed, filled in ((0, True), (1, False)):
            c = sorted([r for r in out if (r["d"], r["K"], r["delta"], r["seed"]) == (d, K, D, seed) and r["coverage"] is not None], key=lambda r: r["sigma_max"])
            if not c: continue
            x = [r["sigma_max"] for r in c]
            kw = dict(color=COL[D], marker=MK[D], ms=8, mew=1.8, mfc=COL[D] if filled else BG, ls="-" if filled else ":", lw=2 if filled else 1.2)
            lab = f"Δ = {D:.0f}, seed {seed}"
            ax[0].plot(x, [r["coverage"] / K for r in c], label=lab, **kw)
            b = [(r["sigma_max"], r["beta"], r["beta_se"]) for r in c if r["beta"] is not None and np.isfinite(r["beta"])]
            if b: ax[1].errorbar([t[0] for t in b], [t[1] for t in b], yerr=[t[2] if np.isfinite(t[2]) else 0 for t in b], capsize=3, **kw)
            e = [(r["sigma_max"], r["is_ess_frac"]) for r in c if r["is_ess_frac"] is not None]
            if e: ax[2].plot([t[0] for t in e], [t[1] for t in e], **kw)
        rule = 2 * math.sqrt(1 + D ** 2 / (2 * d))
        for a in ax: a.axvline(rule, color=COL[D], lw=0.8, alpha=0.5, ls="--")
    ax[0].set(title="Coverage (fraction of modes with p̂ ≥ 0.25 p)", ylim=(-0.03, 1.08)); ax[1].set(title="Shrinkage β (0: right weights)", ylim=(-0.8, 1.2)); ax[2].set(title="Path-IS ESS fraction", ylim=(0, 1.0))
    ax[1].axhline(0, color=MUTED, lw=1, ls=":")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("sigma_max  (dashed: the rule, one per Δ)", color=MUTED); a.set_facecolor(BG); a.grid(color="#e4e3de", lw=0.8)
        a.spines[["top", "right"]].set_visible(False); a.title.set_fontsize(10.5); a.title.set_color(INK); a.tick_params(colors=MUTED)
    ax[0].legend(frameon=False, fontsize=8, loc="center right")
    ax[1].text(0.02, 0.04, "β off-scale where only 2-3 modes are covered (e.g. Δ = 8: −5.5 to +3)", transform=ax[1].transAxes, fontsize=7.5, color=MUTED)
    fig.suptitle(f"AS, d = {d}, K = {K}: coverage, β and path-IS ESS versus sigma_max, one curve per Δ (open markers: seed 1)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{prefix}_{tag}.png", dpi=150, facecolor=BG)
print("saved figures")
