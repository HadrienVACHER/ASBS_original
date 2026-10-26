""" Post-hoc baselines of scripts/manywell_posthoc.py on a small ManyWell (dim 8): samples of the symmetric q (light fraction 0.5), target p (0.156).
    BAR estimates the light-well weight from the energies, the reweighting and the orbit MCMC move the proportions to p.

    python tests/test_posthoc.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manywell_posthoc import fit_wells, orbit_mcmc, reweight  # noqa: E402
from adjoint_samplers.energies.dist_energy import DistEnergy  # noqa: E402
from adjoint_samplers.utils.dist_utils import ManyWell  # noqa: E402


def light(x, split=-0.0417):
    return (x[:, 0::2] < split).float().mean(0)


def test_posthoc():
    torch.manual_seed(0)
    p, q = ManyWell(dim=8), ManyWell(dim=8, a=0.0, b=-6.0)
    energy = DistEnergy(p)
    wells = fit_wells(energy, q.sample([4000]))
    x = q.sample([6000])                                                  # a sampler that covers both wells with the wrong weights
    assert abs(float(light(x).mean()) - 0.5) < 0.03
    gen = torch.Generator().manual_seed(0)
    xr, ess, info = reweight(x, wells, p, gen)
    b = [bb for _, _, bb in info]
    assert all(bb is not None and abs(bb - p.light_mass) < 0.02 for bb in b), b
    fr = light(xr)
    assert (fr - p.light_mass).abs().max() < 0.06, fr
    xm = orbit_mcmc(x.clone(), wells, energy, steps=40)
    fm = light(xm)
    assert (fm - p.light_mass).abs().max() < 0.04, fm
    # an undiscovered well cannot be recovered by reweighting: samples with NO light well on coordinate 0
    x2 = x.clone()
    x2[:, 0] = x2[:, 0].abs()
    xr2, _, info2 = reweight(x2, wells, p, gen)
    assert float(light(xr2)[0]) == 0.0
    fm2 = light(orbit_mcmc(x2.clone(), wells, energy, steps=40))
    print(f"ok  BAR light-well weights {[round(v, 3) for v in b]} (exact {p.light_mass:.3f}); light fractions 0.5 -> reweight {[round(v, 2) for v in fr.tolist()]}, "
          f"MCMC {[round(v, 2) for v in fm.tolist()]}; reweighting cannot create a missing well (0 stays 0), the orbit MCMC does ({float(fm2[0]):.2f}); weights ESS {ess:.0f}/6000")


if __name__ == "__main__":
    test_posthoc()
    print("all passed")
