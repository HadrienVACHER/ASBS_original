""" ABANDONNÉ (décision du relecteur : l'alignement de m avec c* n'est pas garanti même sans bug) ; conservé pour mémoire. ATTENTION : NON VALIDÉ. Sur un jeu synthétique mal pondéré (référence rééchantillonnée avec des rapports d'états connus, indépendant des pools de référence) ce diagnostic rend corr ~ -0.7
    et beta ~ -0.05 alors que le flux d'états à kappa > 0 (scripts/dw4_flux_kappa.py) a le bon signe pour toutes les classes. L'estimateur de E[h | x_t] par poids gaussiens sur des
    échantillons a un résidu non nul (rms ~ 0.04, indépendant de la taille de pool jusqu'à ESS ~ 17000) alors que E_pi[h | x_t] = 0 exactement. Ne PAS soumettre ni interpréter tant que
    cet écart n'est pas compris.

DIAGNOSTIC hors entraînement : le terme de Barker porte-t-il la correction EXACTE des poids ? (job court sur le cluster, CPU ou GPU)

      python scripts/dw4_barker_diagnostic.py --as-samples results/dw4_e1500_ed8bb67_samples.pt [--gamma 3] [--sanity]

    Entrées : échantillons de référence et échantillons d'AS (ed8bb67), chacun augmenté par des éléments aléatoires de O(2) x S4 (exact : pi est invariante). Pour kappa dans
    {0.05, 0.1, 0.2, 0.3, 0.5, 0.7}, nu1 = sigma_max^2, v = nu1 kappa (1 - kappa), nu_{1|t} = nu1 (1 - kappa) :
      - x_t = kappa x1 + sqrt(v) eps (centre de masse nul), x1 ~ AS (sous-ensemble C, indépendant des moitiés A et B) ;
      - correction exacte  c*(x_t) = (E_ref[X1 | x_t] - E_AS[X1 | x_t]) / nu_{1|t}, espérances par poids gaussiens auto-normalisés N(x_t; kappa x, v) sur les échantillons ;
      - terme de Barker    m(x_t) = gamma (E_AS[h | x_t] - E_ref[h | x_t]) / nu_{1|t}  (E_ref[h | x_t] = 0 exactement : le soustraire est un variate de contrôle apparié),
        h avec les poids dépendant de x_t (24 réflexions) ;
      - validation croisée : chaque ensemble est coupé en deux moitiés indépendantes A / B (par échantillon d'origine, AVANT l'augmentation) ;
        beta = <m_A, c*_B> / <c*_A, c*_B>,  corr = <m_A, c*_B> / sqrt(<m_A, m_B> <c*_A, c*_B>), moyennés sur les échanges A <-> B.
    Par kappa : rms|c*| et rms|m| (signal = racine du produit croisé, et valeur brute), corr, beta, gamma nécessaire pour beta = 0.5.
    --sanity : AS remplacé par une moitié indépendante de la référence : c* et m doivent être ~ 0.
    LECTURE PRÉ-ENREGISTRÉE (fenêtre kappa dans [0.1, 0.7]) :
      corr <= 0 : problème de signe / de câblage (comparer la construction de la cible au matcher Barker du GMM) ;
      corr > 0 et gamma nécessaire <= 10 : le terme est adéquat en principe, l'échec vient de la dynamique d'entraînement ou du bruit ;
      corr > 0 et gamma nécessaire > 10 : les réflexions sont trop faibles ; DW-4 avec ce noyau est fermé (prochaine étape : applications alignées sur les modes).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils import dw4_reflections as dwr  # noqa: E402

KAPPAS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
WINDOW = (0.1, 0.2, 0.3, 0.5, 0.7)


def augment(x, K, gen):
    """ K random elements of O(2) x S4 per sample (exact for pi): (n, 8) -> (n K, 8) """
    n = len(x)
    perm = torch.rand(n, K, 4, generator=gen).argsort(-1)
    th = torch.rand(n, K, generator=gen) * 2 * np.pi
    flip = (torch.rand(n, K, generator=gen) < 0.5).to(x.dtype) * 2 - 1
    c, s = torch.cos(th).to(x.dtype), torch.sin(th).to(x.dtype)
    zz = x.reshape(n, 1, 4, 2).expand(n, K, 4, 2)
    zz = torch.gather(zz, 2, perm[..., None].expand(n, K, 4, 2))
    y = zz[..., 1] * flip[..., None]                                    # reflection y -> -y, then rotation
    xr = c[..., None] * zz[..., 0] - s[..., None] * y
    yr = s[..., None] * zz[..., 0] + c[..., None] * y
    return torch.stack([xr, yr], -1).reshape(n * K, 8)


def posterior_stats(pool, xt, kappa, v, topk, variant):
    """ E[X1 | x_t] and E[h | x_t] (T, 8) each, and the mean effective sample size, from the weighted samples `pool` """
    T = len(xt)
    Ex, Eh, ess = torch.zeros(T, 8, dtype=pool.dtype), torch.zeros(T, 8, dtype=pool.dtype), 0.0
    for t in range(T):
        logw = -((xt[t] - kappa * pool) ** 2).sum(-1) / (2 * v)
        top = logw.topk(min(topk, len(pool))).indices
        w = torch.softmax(logw[top], 0)
        P = pool[top]
        h, _ = dwr.barker_h(P, xt[t].expand(len(P), 8), kappa, v, variant)
        Ex[t], Eh[t] = (w[:, None] * P).sum(0), (w[:, None] * h).sum(0)
        ess += float(1.0 / (w ** 2).sum())
    return Ex, Eh, ess / T


def inner(a, b):
    return float((a * b).sum(-1).mean())


def run(ref_pools, as_pools, as_c, kappas, gamma, nu1, T, topk, variant, gen, label):
    (refA, refB), (asA, asB) = ref_pools, as_pools
    print(f"\n=== {label} ===")
    print(f"{'kappa':>6}{'ESS':>7}{'rms|c*| brut':>14}{'signal':>8}{'rms|m| brut':>13}{'signal':>8}{'corr':>8}{'beta':>9}{'gamma pour beta=0.5':>21}")
    rows = {}
    for kappa in kappas:
        v, nu1t = nu1 * kappa * (1 - kappa), nu1 * (1 - kappa)
        x1 = as_c[torch.randperm(len(as_c), generator=gen)[:T]]
        xt = kappa * x1 + v ** 0.5 * dwr.recentre(torch.randn(T, 4, 2, generator=gen, dtype=x1.dtype)).reshape(T, 8)
        out = {}
        for nm, pool in (("refA", refA), ("refB", refB), ("asA", asA), ("asB", asB)):
            out[nm] = posterior_stats(pool, xt, kappa, v, topk, variant)
        ess = np.mean([o[2] for o in out.values()])
        cA = (out["refA"][0] - out["asA"][0]) / nu1t
        cB = (out["refB"][0] - out["asB"][0]) / nu1t
        mA = gamma * (out["asA"][1] - out["refA"][1]) / nu1t
        mB = gamma * (out["asB"][1] - out["refB"][1]) / nu1t
        cc, mm = inner(cA, cB), inner(mA, mB)
        betas, corrs = [], []
        for (m1, c1, c2, m2) in ((mA, cA, cB, mB), (mB, cB, cA, mA)):
            cross, cc_ = inner(m1, c2), inner(c1, c2)
            betas.append(cross / cc_ if cc_ != 0 else float("nan"))
            den = mm * cc_
            corrs.append(cross / np.sqrt(den) if den > 0 else float("nan"))
        beta, corr = float(np.nanmean(betas)), float(np.nanmean(corrs))
        need = gamma * 0.5 / beta if beta > 0 else float("inf")
        rows[kappa] = (corr, beta, need)
        print(f"{kappa:6.2f}{ess:7.0f}{np.sqrt(0.5 * (inner(cA, cA) + inner(cB, cB))):14.4f}{np.sqrt(max(cc, 0)):8.4f}{np.sqrt(0.5 * (inner(mA, mA) + inner(mB, mB))):13.4f}{np.sqrt(max(mm, 0)):8.4f}"
              f"{corr:8.3f}{beta:9.4f}{need:21.1f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-samples", required=True)
    ap.add_argument("--gamma", type=float, default=3.0)
    ap.add_argument("--sigma-max", type=float, default=2.0)
    ap.add_argument("--n-xt", type=int, default=300)
    ap.add_argument("--aug", type=int, default=10)
    ap.add_argument("--topk", type=int, default=1500)
    ap.add_argument("--kappas", type=float, nargs="*", default=list(KAPPAS))
    ap.add_argument("--ref-n", type=int, default=0, help="use only the first n reference samples (tests: the synthetic AS is built from the others)")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--line-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    gen = torch.Generator().manual_seed(a.seed)
    variant = not a.line_only
    ref = dwr.recentre(torch.tensor(np.load(root / "data" / "test_split_DW4.npy", allow_pickle=True)).double().reshape(-1, 4, 2)).reshape(-1, 8)
    ass = dwr.recentre(torch.load(a.as_samples, map_location="cpu", weights_only=False)["model_samples"].double().reshape(-1, 4, 2)).reshape(-1, 8)
    if a.ref_n:
        ref = ref[: a.ref_n]
    ref, ass = ref[torch.randperm(len(ref), generator=gen)], ass[torch.randperm(len(ass), generator=gen)]
    nC = max(a.n_xt, len(ass) // 5)
    asC, asA, asB = ass[:nC], ass[nC:nC + (len(ass) - nC) // 2], ass[nC + (len(ass) - nC) // 2:]
    refA, refB = ref[: len(ref) // 2], ref[len(ref) // 2:]
    nu1 = a.sigma_max ** 2
    print(f"AS: C {len(asC)} / A {len(asA)} / B {len(asB)} ; reference: A {len(refA)} / B {len(refB)} ; augmentation x{a.aug} ; gamma {a.gamma} ; nu1 = {nu1} ; R = {24 if variant else 12}")
    aug = lambda x: dwr.recentre(augment(x, a.aug, gen).reshape(-1, 4, 2)).reshape(-1, 8)
    rows = run((aug(refA), aug(refB)), (aug(asA), aug(asB)), asC, a.kappas, a.gamma, nu1, a.n_xt, a.topk, variant, gen, "AS contre référence")
    if a.sanity:
        q = len(ref) // 4
        run((aug(ref[:q]), aug(ref[q:2 * q])), (aug(ref[2 * q:3 * q]), aug(ref[3 * q:])), ref[len(ref) - a.n_xt:], a.kappas, a.gamma, nu1, a.n_xt, a.topk, variant, gen,
            "SANITÉ : AS remplacé par une autre moitié de la référence (c* et m doivent être ~ 0, brut ~ bruit)")
    win = [k for k in a.kappas if k in WINDOW and k in rows]
    if win:
        corr = np.nanmean([rows[k][0] for k in win])
        beta = np.nanmean([rows[k][1] for k in win])
        need = a.gamma * 0.5 / beta if beta > 0 else float("inf")
        print(f"\nFENÊTRE kappa dans {win} : corr moyenne {corr:.3f}, beta moyen {beta:.4f}, gamma nécessaire pour beta = 0.5 : {need:.1f}")
        if not corr > 0:
            print("LECTURE : corr <= 0 -> problème de signe / de câblage : comparer la construction de la cible au matcher Barker du GMM.")
        elif need <= 10:
            print("LECTURE : corr > 0 et gamma nécessaire <= 10 -> le terme est adéquat en principe ; l'échec vient de la dynamique d'entraînement ou du bruit.")
        else:
            print("LECTURE : corr > 0 et gamma nécessaire > 10 -> les réflexions sont trop faibles ; DW-4 avec ce noyau est fermé (prochaine étape : applications alignées sur les modes, question de conception).")


if __name__ == "__main__":
    main()
