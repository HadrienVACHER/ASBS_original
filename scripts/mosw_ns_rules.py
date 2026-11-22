""" rules_mosw_ns.md: verdicts S1, S2 and the written decision from OUT/mns_d{50,10}_{as,damp,ns}/out.json (scripts/mosw_step3_eval.py --q_only). Reads only; prints and writes OUT/verdicts.json.
    python scripts/mosw_ns_rules.py OUT """
import json, sys
from pathlib import Path
OUT = Path(sys.argv[1])
def ep(d, arm):
    p = OUT / f"mns_d{d}_{arm}" / "out.json"
    return {} if not p.exists() else {int(E): r for E, r in json.load(open(p))["epochs"].items() if r.get("status") == "evaluated"}
V = dict(table={})
for d in (50, 10):
    for arm in ("as", "damp", "ns"):
        V["table"][f"d{d}_{arm}"] = {E: dict(Q1=r["Q"]["Q1"], ratio=r["Q"]["ratio"], coverage=r["Q"]["basin_coverage"], Q=r["Q"]["Q"], basin_tv=r["raw"]["basin_tv"]) for E, r in sorted(ep(d, arm).items())}
late = (750, 1000, 1250, 1500)
s1_eval = all(E in ep(50, arm) for arm in ("as", "damp", "ns") for E in late)
q1 = {arm: [bool(ep(50, arm).get(E, {}).get("Q", {}).get("Q1", False)) for E in late] for arm in ("as", "damp", "ns")}
V["S1"] = dict(value=bool(s1_eval and any(q1["ns"]) and not any(q1["as"]) and not any(q1["damp"])), evaluable=s1_eval, Q1_at_750_1500=q1)
c = {arm: ep(10, arm).get(1500, {}).get("Q", {}).get("basin_coverage") for arm in ("as", "damp", "ns")}
s2_eval = all(v is not None for v in c.values())
V["S2"] = dict(value=bool(s2_eval and c["ns"] >= 8 and c["as"] <= 5 and c["damp"] <= 5), evaluable=s2_eval, coverage_ep1500=c)
V["decision"] = "Newton-Stein rouvert (S1 ou S2 vrai)" if (V["S1"]["value"] or V["S2"]["value"]) else "fermeture definitive : l'echec d'AS sur le MoS ne vient pas de l'instabilite (regle ecrite)"
json.dump(V, open(OUT / "verdicts.json", "w"), indent=1, default=float)
for k, v in V["table"].items():
    print(k, " ".join(f"{E}:{x['ratio']:.2f}{'*' if x['Q1'] else ''}/{x['coverage']}/{x['basin_tv']:.3f}" for E, x in v.items()))
print("S1", V["S1"]); print("S2", V["S2"]); print("DECISION:", V["decision"])
