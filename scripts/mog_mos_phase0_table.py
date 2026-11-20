""" Summary table of rules_mog_mos_phase0.md from figures/mog_mos_phase0_*.json (no recomputation). """
import glob, json
rows = [json.load(open(f)) | {"file": f} for f in sorted(glob.glob("figures/mog_mos_phase0_*_disc*.json"))]
order = {"mog": 0, "mos": 1, "mogextra": 2}
rows.sort(key=lambda r: (order[r["target"]], r["d"], r["discover"]))
print(f"{'target':>9} {'d':>3} {'disc':>5} {'minima':>7} {'floor':>6} | {'IS(n) TV/ESS/khat':>22} | " + " | ".join(f"S={S}: TV_dart / TV_IS(eq cost) [ESS_IS] jump-acc  G" for S in (1, 5, 10)) + " | time")
for r in rows:
    d = r.get("darting")
    s = f"{r['target']:>9} {r['d']:3d} {r['discover']:5d} {r['minima_found']:>3d}/{r['K']:<3d} {r['floor']:6.4f} | {r['IS_n']['tv']:7.4f} {r['IS_n']['ess']:6.4f} {r['IS_n']['khat']:5.2f} | "
    if d:
        parts = []
        for S in ("1", "5", "10"):
            x, i = d[S], r["IS_equal_cost"][S]
            parts.append(f"{x['tv']:.4f} / {i['tv']:.4f} [{i['ess']:.4f}] {x['jump_acc']:.2f} {'G' if r['G']['by_S'][S] else '-'} (cost {x['cost']:.1e})")
        s += " | ".join(parts)
    else:
        s += "darting n/a: " + str(r.get("error"))
    print(s + f" | {r['time_s']:.1f} s")
print("\nG true (standard protocol, 500 discovery samples):", [(r["target"], r["d"], [S for S, v in r["G"]["by_S"].items() if v]) for r in rows if r.get("G") and r["G"]["any"] and r["discover"] == 500])
print("G true (exploratory, 5000 discovery samples):", [(r["target"], r["d"], [S for S, v in r["G"]["by_S"].items() if v]) for r in rows if r.get("G") and r["G"]["any"] and r["discover"] == 5000] or "none")
