""" Analyse Rao-Blackwellisée du noyau de réflexions DW-4 (24 applications : 12 réflexions de droite + 12 réflexions ponctuelles), CPU seulement, sur le samples_final.pt d'un run terminé.

      python scripts/dw4_reflection_rb.py --samples results/dw4_as_states_s2/seed0/samples_final.pt

    STATISTIQUE PRIMAIRE (appariée, sans tirage de r ni d'acceptation) : pour chaque classe b,
        Delta_b = moyenne_n (1/R) sum_r w_r(x_n) [ 1{s(T_r x_n) = b} - 1{s(x_n) = b} ],  R = 24,  w_r = pi(T_r x) / (pi(x) + pi(T_r x))  (énergie seule, kappa = 0),
    erreur standard à partir des valeurs par échantillon. s = classe du graphe des paires courtes (d < 4), lue soit sur la configuration BRUTE soit sur son BASSIN (trempe de x et de
    chaque T_r x). Rapporte par classe : proportion AS, proportion référence, Delta_b sur AS (± SE), Delta_b sur la référence (± SE).
    SANITÉ : sur la référence, |Delta_b| / SE < 3 pour chaque classe.
    PASSAGE PRÉ-ENREGISTRÉ : pour toute classe avec |log(AS/réf)| >= 0.2, signe(Delta_b sur AS) = signe(réf_b - AS_b) avec |Delta_b| / SE >= 2.
    SECONDAIRE (rapporté seulement) : matrice de flux RB K_ab sur AS, rapports BAR implicites sur les bassins contre les rapports de référence, tableau TV à 1 / 5 / 20 pas.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils import dw4_reflections as R  # noqa: E402
from adjoint_samplers.utils.dw4_quench import quench  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, state_freq, state_ids, state_tv  # noqa: E402

N_AS = 5000
RARE = 0.005                                   # reference frequency below which a class is flagged as SE-limited (reported, rule still applied as written)


def label_raw(z):
    return state_ids(z.float())


def label_basin(z):
    return state_ids(quench(z.float())[0])


def stats(x, label, variant=True):
    D, Fl, lab_x, K = R.rb_terms(x, label, variant, n_classes=len(KEYS))
    mean, se = D.mean(0), D.std(0) / np.sqrt(len(D))
    return dict(D=mean, se=se, K=Fl.mean(0), prop=torch.bincount(lab_x, minlength=len(KEYS)).double() / len(lab_x), n=len(D))


def table(name, a, r):
    """ prints the per-class table and returns (sanity_ok, pass_ok, details) """
    print(f"\n=== {name} ===")
    print(f"{'classe':28}{'AS':>8}{'réf':>8}{'|log AS/réf|':>14}{'Delta AS':>12}{'SE':>9}{'|D|/SE':>8}{'Delta réf':>12}{'SE':>9}{'|D|/SE':>8}")
    sanity, ok, rows = True, True, []
    for k, key in enumerate(KEYS):
        pa, pr = float(a["prop"][k]), float(r["prop"][k])
        if max(pa, pr, float(a["se"][k]), float(r["se"][k])) == 0:
            continue
        lg = abs(np.log(pa / pr)) if pa > 0 and pr > 0 else (float("inf") if (pa > 0) != (pr > 0) else 0.0)
        da, sa, dr, sr = float(a["D"][k]), float(a["se"][k]), float(r["D"][k]), float(r["se"][k])
        za = abs(da) / sa if sa > 0 else float("nan")
        zr = abs(dr) / sr if sr > 0 else float("nan")
        if sr > 0 and zr >= 3:
            sanity = False
        need = lg >= 0.2
        passed = None
        if need:
            passed = bool(np.sign(da) == np.sign(pr - pa) and sa > 0 and za >= 2)
            ok &= passed
        rows.append((NAMES[key], need, passed, pr < RARE))
        flag = ("  PASS" if passed else "  FAIL") if need else ""
        flag += "  (rare: SE-limited)" if pr < RARE and need else ""
        print(f"{NAMES[key]:28}{pa:8.4f}{pr:8.4f}{lg:14.2f}{da:+12.5f}{sa:9.5f}{za:8.1f}{dr:+12.5f}{sr:9.5f}{zr:8.1f}{flag}")
    print(f"sanité (référence, |Delta|/SE < 3 pour toute classe) : {sanity}")
    ok_main = all(p for (_, need, p, rare) in rows if need and not rare)
    print(f"PASSAGE pré-enregistré (toutes les classes avec |log| >= 0.2) : {ok}   [classes de fréquence de référence >= {100 * RARE:.1f} % seulement : {ok_main}, rapporté]")
    return sanity, ok


def flux_report(a, r, label):
    print(f"\n--- flux RB K_ab (AS, {label}), états principaux ; rapport BAR implicite pi_b/pi_a ~ (K_ab / p_a) / (K_ba / p_b) contre le rapport de référence ---")
    main = [k for k in range(len(KEYS)) if float(r["prop"][k]) >= 0.01]
    print(f"{'a -> b':52}{'K_ab':>10}{'K_ba':>10}{'implicite':>11}{'réf b/a':>9}{'AS b/a':>9}")
    for i, ka in enumerate(main):
        for kb in main[i + 1:]:
            kab, kba = float(a["K"][ka, kb]), float(a["K"][kb, ka])
            if kab + kba < 1e-6:
                continue
            pa_, pb_ = float(a["prop"][ka]), float(a["prop"][kb])
            imp = (kab / pa_) / (kba / pb_) if kba > 0 and pa_ > 0 and pb_ > 0 else float("nan")
            print(f"{NAMES[KEYS[ka]][:24]:25} -> {NAMES[KEYS[kb]][:24]:24}{kab:10.5f}{kba:10.5f}{imp:11.3f}{float(r['prop'][kb] / r['prop'][ka]):9.3f}{pb_ / pa_:9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--line-only", action="store_true", help="12 line reflections only (not the default)")
    a = ap.parse_args()
    variant = not a.line_only
    root = Path(__file__).resolve().parents[1]
    ref = torch.tensor(np.load(root / "data" / "test_split_DW4.npy", allow_pickle=True)).double()
    model = torch.load(a.samples, map_location="cpu", weights_only=False)["model_samples"].double()[:N_AS]
    print(f"R = {24 if variant else 12} maps ; AS: {len(model)} samples, reference: {len(ref)} samples")
    results = {}
    for lab, fn in (("états BRUTS (d < 4 sur la configuration)", label_raw), ("BASSINS (d < 4 après trempe)", label_basin)):
        sa, sr = stats(model, fn, variant), stats(ref, fn, variant)
        results[lab] = (sa, sr, table(lab, sa, sr))
    for lab, (sa, sr, _) in results.items():
        if "BASSINS" in lab:
            flux_report(sa, sr, "bassins")
    g = torch.Generator().manual_seed(0)
    print(f"\n--- tableau TV 1 / 5 / 20 pas (noyau avec tirage), à titre indicatif ; plancher p95 ≈ 0.022 ---")
    freq_ref = state_freq(state_ids(ref.float()))
    sub = [torch.randperm(len(ref), generator=g) for _ in range(25)]
    x, done = model.clone(), 0
    for s in (0, 1, 5, 20):
        for _ in range(s - done):
            x, _r = R.kernel_step(x, variant, g)
        done = s
        tv = float(np.mean([state_tv(state_freq(state_ids(x.float())), state_freq(state_ids(ref[p[:N_AS]].float()))) for p in sub]))
        print(f"  {s:2d} pas : TV {tv:.4f}")


if __name__ == "__main__":
    main()
