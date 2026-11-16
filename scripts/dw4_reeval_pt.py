""" Re-evaluation of the DW-4 continuation and Barker rules against p_PT (rules_reeval_pt.md, committed in 4a1342f before any history value was read).
    Two truth spaces: R = raw states (state/<k>, state_ids on non-quenched coordinates; the requested adjustment), Q = quenched basins (basin/<k>, the letter of the written rules).     python scripts/dw4_reeval_pt.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dw4_barker_rule as br  # noqa: E402
from adjoint_samplers.utils.dw4_quench import quench  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, state_freq, state_ids, state_tv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"C1": "dw4_cont_C1", "C2": "dw4_cont_C2", "G3": "dw4_cont_G3", "G10": "dw4_cont_G10", "G30": "dw4_cont_G30"}
LAST, COMMIT, K = 20, "aadf33b", len(KEYS)
NAME = [NAMES[k] for k in KEYS]


def truth():
    d = torch.load(ROOT / "results" / "pt_samples_T1.pt", map_location="cpu", weights_only=False)
    st, ids = d["states"], d["basin_state_id"]                                                   # (64, n, 8), (64, n)
    raw = torch.stack([state_ids(st[c].double()) for c in range(st.shape[0])])
    out = {}
    for tag, lab in (("R", raw), ("Q", ids)):
        cm = torch.stack([torch.bincount(lab[c], minlength=K).double() / lab.shape[1] for c in range(lab.shape[0])])      # (64, 11)
        out[tag] = (cm.mean(0).numpy(), (cm.std(0) / math.sqrt(len(cm))).numpy())
    return out


def continuation(tag, cols, tru, rows):
    """ the written rule with the truth replaced; rows[arm] = (n_eval, 11) proportions of the last 20 evaluations """
    rng = np.random.default_rng(0)
    S, SE, P = {}, {}, {}
    for arm in ARMS:
        X = rows[arm]
        P[arm] = X.mean(0)
        S[arm] = float(np.abs(P[arm] - tru).sum())
        SE[arm] = float(np.std([float(np.abs(X[rng.integers(len(X), size=len(X))].mean(0) - tru).sum()) for _ in range(1000)]))
    noise = abs(S["C1"] - S["C2"]) + 2 * max(SE.values())
    base = min(S["C1"], S["C2"])
    print(f"  [{tag}] S = " + ", ".join(f"{a} {S[a]:.4f} +- {SE[a]:.4f}" for a in ARMS))
    print(f"  [{tag}] noise = |S_C1 - S_C2| + 2 SE_max = {abs(S['C1'] - S['C2']):.4f} + 2 x {max(SE.values()):.4f} = {noise:.4f}; min(S_C1, S_C2) = {base:.4f}; effect threshold = {base - noise:.4f}")
    effect = {g: S[g] < base - noise for g in ("G3", "G10", "G30")}
    mono = (S["G3"] + noise >= S["G10"]) and (S["G10"] + noise >= S["G30"])
    alive = (effect["G10"] or effect["G30"]) and mono
    print(f"  [{tag}] effect: " + ", ".join(f"{g} {effect[g]}" for g in effect) + f"; S decreasing in gamma within noise: {mono}  =>  " + ("METHODE VIVANTE" if alive else "METHODE NON demontree vivante"))
    return S, SE, P, alive


def main():
    import wandb
    tr = truth()
    for tag in ("R", "Q"):
        p, se = tr[tag]
        print(f"truth p_PT^{tag} (11 classes, SE over chains): " + ", ".join(f"{NAME[k]} {p[k]:.4f} ({se[k]:.4f})" for k in range(K) if p[k] >= 0.001))
    runs = list(wandb.Api(timeout=120).runs("CCDS_NTU/as", per_page=200, order="-created_at"))
    data = {"R": {}, "Q": {}}
    eq, ok = {}, {}
    for arm, nm in ARMS.items():
        sel = [r for r in runs if r.name == nm and r.config.get("git_commit", "")[:7] == COMMIT]
        assert sel and sel[0].state == "finished", (arm, nm)
        r = sel[0]
        keys = ["basin_tv"] + [f"basin/{k}" for k in range(K)] + [f"state/{k}" for k in range(K)]
        h = r.history(samples=6000, x_axis="_step", keys=keys).dropna(subset=["basin_tv"]).set_index("_step").iloc[-LAST:]
        data["Q"][arm] = h[[f"basin/{k}" for k in range(K)]].values
        data["R"][arm] = h[[f"state/{k}" for k in range(K)]].values
        nan_state = int(np.isnan(data["R"][arm]).sum())
        loss = r.history(samples=6000, x_axis="_step", keys=["adjoint_loss"])["adjoint_loss"].dropna().values
        e = r.history(samples=6000, x_axis="_step", keys=["eq_w2"])
        eq[arm] = float(e["eq_w2"].dropna().values[-2:].mean()) if "eq_w2" in e and e["eq_w2"].notna().any() else float("nan")
        ok[arm] = bool(np.isfinite(loss).all())
        print(f"  {arm} {nm}: evaluations epochs {int(h.index[0])}..{int(h.index[-1])} ({len(h)} rows); NaN in the raw state columns: {nan_state}; eq_w2 {eq[arm]:.3f}; finite loss {ok[arm]}", flush=True)
    eq_ctrl = np.nanmean([eq["C1"], eq["C2"]])
    print(f"  instability check: G30 finite loss {ok['G30']}, eq_w2(G30) {eq['G30']:.3f} vs 1.5 x controls {1.5 * eq_ctrl:.3f}")
    print("\n=== CONTINUATION (fait foi) ===")
    res = {}
    for tag in ("R", "Q"):
        res[tag] = continuation(tag, None, tr[tag][0], data[tag])
    print("\n=== Paired differences Barker - AS (G - C1, same seed), mean over the last 20 evaluations, bootstrap SE (1000); sign = sign(p_PT - p_C1)? (report only) ===")
    rng = np.random.default_rng(1)
    for tag in ("R", "Q"):
        p = tr[tag][0]
        print(f"  [{tag}]   state (p_PT; p_C1)            " + "".join(f"{g:>24}" for g in ("G3", "G10", "G30")))
        for k in range(K):
            if p[k] < 0.005:
                continue
            c1 = data[tag]["C1"][:, k]
            row = f"  {NAME[k]:26} ({p[k]:.4f}; {c1.mean():.4f})"
            for g in ("G3", "G10", "G30"):
                dd = data[tag][g][:, k] - c1
                bs = [dd[i].mean() for i in (rng.integers(len(dd), size=len(dd)) for _ in range(1000))]
                good = np.sign(dd.mean()) == np.sign(p[k] - c1.mean())
                row += f"  {dd.mean():+.4f} +- {np.std(bs):.4f} {'ok ' if good else 'NON'}"
            print(row)

    print("\n=== BARKER (primary TV on the 5000 final samples; bootstrap 200, seed 0) ===")
    load = lambda f: torch.load(f, map_location="cpu", weights_only=False)["model_samples"].double()[:5000]
    cv = load(ROOT / "results" / "dw4_as_barker_e1500_seed0_samples_final.pt")
    ass = [load(ROOT / "results" / "dw4_e1500_ed8bb67_samples.pt"), load(ROOT / "results" / "dw4_e1500_94252_samples.pt")]
    for tag in ("R", "Q"):
        tru = torch.tensor(tr[tag][0])
        if tag == "Q":
            tvs = [br.tv_with_se(x, tru) for x in [cv] + ass]
        else:
            tvs = []
            for x in [cv] + ass:
                g = torch.Generator().manual_seed(0)
                ids = state_ids(x)
                base = state_tv(state_freq(ids), tru)
                tvs.append((base, float(np.std([state_tv(state_freq(ids[torch.randint(len(ids), (len(ids),), generator=g)]), tru) for _ in range(200)]))))
        (tv_cv, se_cv), tv_as = tvs[0], tvs[1:]
        j = int(np.argmin([t for t, _ in tv_as]))
        thr = tv_as[j][0] - 2 * math.sqrt(se_cv ** 2 + tv_as[j][1] ** 2)
        print(f"  [{tag}] TV: AS ed8bb67 {tv_as[0][0]:.4f} +- {tv_as[0][1]:.4f}, AS 94252 {tv_as[1][0]:.4f} +- {tv_as[1][1]:.4f}, CV {tv_cv:.4f} +- {se_cv:.4f}; threshold {thr:.4f}  =>  primary " + ("PASSES" if tv_cv < thr else "FAILS"))
        # direction (b): lists derived from the truth
        if tag == "Q":
            props = [torch.tensor(state_freq(state_ids(quench(x.float())[0])).numpy()) for x in [cv] + ass]
        else:
            props = [state_freq(state_ids(x)) for x in [cv] + ass]
        p_cv, p_as = props[0].numpy(), (props[1] + props[2]).numpy() / 2
        direc, lines = True, []
        for k in range(K):
            if tr[tag][0][k] < 0.005:
                continue
            over = p_as[k] > tr[tag][0][k]
            ok_k = (p_cv[k] < p_as[k]) if over else (p_cv[k] > p_as[k])
            direc &= bool(ok_k)
            lines.append(f"{NAME[k]} {'over' if over else 'under'} (truth {tr[tag][0][k]:.4f}, AS {p_as[k]:.4f}, CV {p_cv[k]:.4f}) {'ok' if ok_k else 'NO'}")
        print(f"  [{tag}] direction (b, lists derived from the truth): {'PASSES' if direc else 'FAILS'}: " + "; ".join(lines))
    print("\n  (mechanistic and sanity: no truth involved, unchanged; direction (a) with the hard-coded lists: unchanged)")


if __name__ == "__main__":
    main()
