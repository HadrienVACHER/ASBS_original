""" Labels (R = raw states, Q = quenched basins) of saved path-IS chunks, appended in the same file (report only; one chunk per call so that a call stays short).
      python scripts/dw4_is_S_label.py PATH.pt """
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils.dw4_quench import quench  # noqa: E402
from adjoint_samplers.utils.dw4_states import state_ids  # noqa: E402

p = Path(sys.argv[1])
d = torch.load(p, weights_only=False)
t0 = time.time()
x1 = d["x1"].double()
d["lab_R"] = state_ids(x1)
qx, _, _ = quench(x1)
d["lab_Q"] = state_ids(qx.double())
torch.save(d, p)
print(f"{p.name}: labelled {len(x1)} samples in {time.time() - t0:.0f} s")
