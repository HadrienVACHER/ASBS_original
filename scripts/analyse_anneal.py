""" γ annealing at d = 36 (big controller, 3 seeds): AS-big, frozen Barker (γ = 3 throughout) and Barker with γ decaying linearly to 0
    over the last 20% of the epochs. Per cell and per seed: |log R̄|, light quarter, min R̄, energy_w2, mode_spread and the fraction of
    samples OFF every mode (1 - frac_in_modes), all averaged over the last 30 evaluations. W&B reads only (no heavy computation).

      python scripts/analyse_anneal.py
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils.dist_utils import GMM  # noqa: E402

D = 36
CELLS = [("AS big", "gmm36d_as_big_cap"), ("Barker g=3 (frozen)", "gmm36d_as_barker_g3_thr_big"),
         ("Barker g=3 annealed", "gmm36d_as_barker_g3_thr_big_anneal")]


def main():
    import wandb
    runs = list(wandb.Api(timeout=120).runs("CCDS_NTU/adjoint_sampler_cv", per_page=300, order="-created_at"))
    g = GMM(dim=D, K=D, seed=0, layout="spread", weight_law="loguniform", weight_ratio=20, radius=4.25)
    w = g.weights.numpy().astype(float); w /= w.sum()
    light = np.argsort(w)[:D // 4]
    per = {}
    for label, base in CELLS:
        seen = {}
        for r in runs:
            m = re.match(rf"^{re.escape(base)}_K{D}_s(\d)_spread", r.name)
            if m and r.state == "finished" and int(m.group(1)) not in seen:
                seen[int(m.group(1))] = r
        for s, r in sorted(seen.items()):
            keys = ["energy_w2", "mode_spread", "frac_in_modes"] + [f"mode_ratio/{j}" for j in range(D)]
            h = r.history(keys=keys, samples=6000, pandas=True, x_axis="_step").dropna(subset=["energy_w2"])
            Rm = h[keys[3:]].values[-30:].mean(0)
            per[(label, s)] = dict(logR=np.mean(np.abs(np.log(np.maximum(Rm, 1e-3)))), light=Rm[light].mean(), minR=Rm.min(),
                                   ew2=h["energy_w2"].values[-30:].mean(), spread=h["mode_spread"].values[-30:].mean(),
                                   off=1 - h["frac_in_modes"].values[-30:].mean(), floor=1 - r.summary.get("frac_in_modes_floor", 1.0))
        if not seen:
            print(f"{label}: aucun run terminé")
    print(f"{'cellule':22s}{'n':>3}{'|log R|':>15}{'quart léger':>15}{'min R':>13}{'energy_w2':>13}{'spread':>9}{'hors mode (%)':>16}")
    for label, _ in CELLS:
        v = [per[k] for k in per if k[0] == label]
        if not v:
            continue
        f = lambda key: (np.mean([x[key] for x in v]), np.std([x[key] for x in v]))
        a, b, c, e, sp, off = f("logR"), f("light"), f("minR"), f("ew2"), f("spread"), f("off")
        print(f"{label:22s}{len(v):3d}{a[0]:8.3f}±{a[1]:.3f}{b[0]:9.3f}±{b[1]:.3f}{c[0]:7.3f}±{c[1]:.3f}{e[0]:7.2f}±{e[1]:.2f}{sp[0]:9.3f}{100 * off[0]:9.2f}±{100 * off[1]:.2f}")
    fl = [x["floor"] for x in per.values()]
    if fl:
        print(f"(hors mode des échantillons exacts: {100 * np.mean(fl):.2f} %)")
    print("\npar graine, annealed - frozen / annealed - AS big:")
    for s in (0, 1, 2):
        try:
            an, fr, a0 = per[("Barker g=3 annealed", s)], per[("Barker g=3 (frozen)", s)], per[("AS big", s)]
        except KeyError:
            continue
        print(f"  graine {s}: |log R| {an['logR'] - fr['logR']:+.3f} / {an['logR'] - a0['logR']:+.3f} ; energy_w2 {an['ew2'] - fr['ew2']:+.2f} / {an['ew2'] - a0['ew2']:+.2f} ; "
              f"hors mode {100 * (an['off'] - fr['off']):+.2f} / {100 * (an['off'] - a0['off']):+.2f} pts")


if __name__ == "__main__":
    main()
