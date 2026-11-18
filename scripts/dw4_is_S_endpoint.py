""" Endpoint IS with the encompassing proposals of scripts/dw4_is_inflate.py, saved with R / Q labels for scripts/dw4_is_S_report.py (report only; local).
    Same construction (components refitted by EM on the AS samples ed8bb67, deterministic basis; N draws, seed 7); N can be reduced so that one call stays short.
      python scripts/dw4_is_S_endpoint.py --fam gauss --s 1 --n 20000 --out PATH.pt """
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dw4_division_b as b  # noqa: E402
import dw4_division_step0 as s0  # noqa: E402
import dw4_division_trunc as tr  # noqa: E402
import dw4_is_inflate as inf  # noqa: E402
from adjoint_samplers.utils.dw4_quench import quench  # noqa: E402
from adjoint_samplers.utils.dw4_states import state_ids  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fam", required=True)
ap.add_argument("--s", type=float, required=True)
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--seed", type=int, default=inf.SEED)
ap.add_argument("--out", required=True)
a = ap.parse_args()
t0 = time.time()
ref, q, lab_r, types, mu8 = b.load_reference()
cache = Path(a.out).parent / "em_comps.pt"                                                  # the EM fit takes ~40 s: cached next to the output
if cache.exists():
    comps = torch.load(cache, weights_only=False)
else:
    fit = tr.load_as_fit(str(inf.ROOT / "results" / "dw4_e1500_ed8bb67_samples.pt"))["fit"]
    comps = [fit[k] for k in types]
    torch.save(comps, cache)
print(f"EM fit {time.time() - t0:.0f} s", flush=True)
inf.N = a.n
gen = torch.Generator().manual_seed(a.seed)
x, lr = inf.build(a.fam, a.s, comps, gen)
lw = -s0.energy(x) - lr
print(f"draws + density {time.time() - t0:.0f} s", flush=True)
qx, _, _ = quench(x)
torch.save({"x1": x, "lw": lw, "lab_R": state_ids(x.double()), "lab_Q": state_ids(qx.double())}, a.out)
print(f"{a.fam} s={a.s}: {a.n} draws in {time.time() - t0:.0f} s -> {a.out}")
