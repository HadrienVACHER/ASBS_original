""" Baselines POST-HOC appliquées aux échantillons finaux de AS, à budget d'énergie apparié, et comparaison des PROPORTIONS pour toutes les méthodes (ManyWell-32).

      python scripts/manywell_posthoc.py --init results/manywell_pretrain/seed0/samples_final.pt \
          --as-samples results/manywell_corr_as/seed0/samples_final.pt --barker-samples results/manywell_corr_barker/seed0/samples_final.pt \
          [--barker-commit <hash>]   # budget lu sur W&B ; sinon --budget <évaluations par échantillon>

    Baselines (le modèle AS n'est PAS modifié) :
      reweight : poids de puits léger estimés par coordonnée à partir des énergies (BAR avec l'application affine entre les deux puits et son jacobien ; estimateur
                 exponentiel d'un seul côté si un puits a < 20 échantillons), ré-échantillonnage multinomial du jeu entier avec le poids produit sur les 16 coordonnées
                 prod_i b_k(i) / a_k(i) (b : poids cible estimé, a : proportion empirique) ; un puits sans échantillon ne peut pas être récupéré.
      mcmc     : noyau de l'orbite de Barker (sous-groupe s = 3, banc de puits du buffer initial, comme pour Barker) appliqué directement aux échantillons, nombre de pas
                 = budget / 7 évaluations d'énergie par pas.
    Métriques pré-enregistrées pour toutes les méthodes : |f_i - p| moyen (primaire), puits conservés (f_i > 0.01), TV du nombre de puits légers, energy_w2.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.components.evaluator import ManyWellEvaluator  # noqa: E402
from adjoint_samplers.components.matcher import AdjointVEBarkerMatcher, _denoise_separable, _product_wells  # noqa: E402
from adjoint_samplers.energies.dist_energy import DistEnergy  # noqa: E402
from adjoint_samplers.utils.dist_utils import ManyWell  # noqa: E402

P_LEFT = 0.15569


def fit_wells(energy, x_init):
    xd, conv, lam, *_ = _denoise_separable(energy, x_init, 12)
    wells, n = _product_wells(energy, xd, conv, lam)
    assert n == energy.dim // 2, f"only {n} double wells found in the initial buffer"
    return wells


def metrics(evaluator, x):
    r = evaluator(x)
    f = np.array([r[f"well_ratio/{i}"] for i in range(evaluator.dist.n_wells)]) * evaluator.p_left
    return dict(dev=float(np.abs(f - P_LEFT).mean()), kept=int((f > 0.01).sum()), tv=r["light_count_tv"], w2=r["energy_w2"], f=f)


def bar_log_ratio(wl, wr, nf_min=20):
    """ log(Z_R / Z_L) from the works of the maps L->R (wl) and R->L (wr) (inf = the image falls outside the target well); BAR, one-sided if a side is small """
    ok_l, ok_r = len(wl) >= nf_min, len(wr) >= nf_min
    if not ok_l and not ok_r:
        return None
    lme = lambda w: float(torch.logsumexp(-w, 0) - np.log(len(w)))                       # log mean exp(-W)
    if ok_l and not ok_r:
        return lme(wl)                                                                   # Z_R / Z_L = E_L exp(-W_F)
    if ok_r and not ok_l:
        return -lme(wr)                                                                  # Z_L / Z_R = E_R exp(-W_R)
    M = np.log(len(wl) / len(wr))
    fer = lambda z: torch.sigmoid(-z)
    lo, hi = -40.0, 40.0
    for _ in range(100):                                                                 # ΔF = F_R - F_L = -log(Z_R / Z_L)
        dF = 0.5 * (lo + hi)
        g = float(fer(wl - dF + M).sum() - fer(wr + dF - M).sum())
        lo, hi = (lo, dF) if g > 0 else (dF, hi)                                       # g increases with dF
    return -dF


def reweight(x, wells, g, gen):
    """ per-coordinate target light-well weights b from the energies (BAR over the orbit map), joint multinomial resampling with prod_i b/a """
    n, D = x.shape
    logw = torch.zeros(n, dtype=torch.float64)
    info = []
    for j, i in enumerate(wells["idx"].tolist()):
        xi = x[:, i].double()
        mu, sg, sp = wells["mu"][j].double(), wells["sigma"][j].double(), float(wells["split"][j])
        e = lambda v: g.a * v + g.b * v ** 2 + g.c * v ** 4
        left = xi < sp
        yl = mu[1] + sg[1] * (xi[left] - mu[0]) / sg[0]                                  # L -> R
        yr = mu[0] + sg[0] * (xi[~left] - mu[1]) / sg[1]                                 # R -> L
        wf = e(yl) - e(xi[left]) - torch.log(sg[1] / sg[0])
        wr = e(yr) - e(xi[~left]) - torch.log(sg[0] / sg[1])
        wf = torch.where(yl > sp, wf, torch.full_like(wf, float("inf")))
        wr = torch.where(yr < sp, wr, torch.full_like(wr, float("inf")))
        lr = bar_log_ratio(wf, wr)
        a_l = float(left.double().mean())
        if lr is None or a_l in (0.0, 1.0):
            info.append((i, a_l, None))
            continue
        b_l = 1.0 / (1.0 + np.exp(lr))
        info.append((i, a_l, b_l))
        logw += torch.where(left, torch.tensor(np.log(b_l / a_l)), torch.tensor(np.log((1 - b_l) / (1 - a_l))))
    w = torch.softmax(logw, 0)
    ess = float(1.0 / (w ** 2).sum())
    idx = torch.multinomial(w, n, replacement=True, generator=gen)
    return x[idx], ess, info


@torch.no_grad()
def orbit_mcmc(x, wells, energy, steps, s=3, seed=0):
    """ `steps` Gibbs steps on the product-subgroup orbit (same orbit code as the Barker matcher): z ~ softmax(-E(z) + logdet) over the 2^s orbit, subset S uniform """
    torch.manual_seed(seed)
    m = SimpleNamespace(wells=wells, product_subset=s)
    E = energy.eval(x)
    for _ in range(steps):
        jumps, valid, logdet = AdjointVEBarkerMatcher._orbit_product(m, x)
        cand = x.unsqueeze(1) + jumps                                                    # (N, 2^s - 1, D)
        Ec = torch.full_like(valid, float("inf"))
        mask = valid.bool()
        Ec[mask] = energy.eval(cand[mask])
        logits = torch.cat([-E.unsqueeze(1), -Ec + logdet], 1)
        logits[:, 1:][~mask] = -float("inf")
        k = torch.distributions.Categorical(logits=logits).sample()
        sel = k > 0
        row = torch.arange(len(x))[sel]
        x = x.clone()
        x[row] = cand[row, k[sel] - 1]
        E[row] = Ec[row, k[sel] - 1]
    return x


def budget_from_wandb(commit, n_final, epochs=2000, per_epoch=512):
    import wandb
    for r in wandb.Api(timeout=120).runs("CCDS_NTU/adjoint_sampler_cv", per_page=200, order="-created_at"):
        if r.name == "manywell_corr_barker" and r.config.get("git_commit", "")[:7] == commit[:7] and r.state == "finished":
            keys = ["mh/energy_evals_per_sample", "denoise/E_evals_per_sample", "denoise/grad_evals_per_sample"]
            h = r.history(keys=keys, samples=6000, pandas=True, x_axis="_step")
            per = sum(float(h[k].dropna().mean()) for k in keys)
            print(f"Barker extra evaluations per new sample: {per:.1f} (orbit + denoising) x {epochs} epochs x {per_epoch} samples / {n_final} final samples")
            return per * epochs * per_epoch / n_final
    raise SystemExit("Barker run not found on W&B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--as-samples", required=True)
    ap.add_argument("--barker-samples", default=None)
    ap.add_argument("--barker-commit", default=None)
    ap.add_argument("--budget", type=float, default=None)
    a = ap.parse_args()
    g = ManyWell(dim=32)
    energy = DistEnergy(g)
    evaluator = ManyWellEvaluator(energy)
    load = lambda p: torch.load(p, map_location="cpu", weights_only=False)["model_samples"].float()
    x_init, x_as = load(a.init), load(a.as_samples)
    budget = a.budget if a.budget is not None else budget_from_wandb(a.barker_commit, len(x_as))
    steps = int(budget // 7)
    wells = fit_wells(energy, x_init)
    gen = torch.Generator().manual_seed(0)
    rows = {"départ (pré-entraîné)": metrics(evaluator, x_init), "AS": metrics(evaluator, x_as)}
    x_rw, ess, info = reweight(x_as, wells, g, gen)
    rows["AS + reweight"] = metrics(evaluator, x_rw)
    rows[f"AS + MCMC orbite ({steps} pas)"] = metrics(evaluator, orbit_mcmc(x_as.clone(), wells, energy, steps))
    if a.barker_samples:
        rows["Barker"] = metrics(evaluator, load(a.barker_samples))
    print(f"budget d'énergie apparié : {budget:.0f} évaluations par échantillon ; reweight: ESS des poids produit {ess:.1f}/{len(x_as)} "
          f"(échantillons distincts après ré-échantillonnage: {len(torch.unique(x_rw[:, 0]))}) ; MCMC: {steps} pas x 7 évaluations")
    print(f"{'':32}{'|f-0.156| (primaire)':>22}{'puits conservés':>17}{'TV joint':>10}{'energy_w2':>11}")
    for k, v in rows.items():
        print(f"{k:32}{v['dev']:22.4f}{v['kept']:17d}{v['tv']:10.3f}{v['w2']:11.2f}")
    print("\nf_i par coordonnée :")
    for k, v in rows.items():
        print(f"  {k:30}", np.round(v["f"], 3))
    print("\nreweight : (coordonnée, a = proportion empirique, b = poids estimé) :", [(i, round(al, 3), None if b is None else round(b, 3)) for i, al, b in info])


if __name__ == "__main__":
    main()
