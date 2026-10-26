""" DW-4 states: the 11 isomorphism classes, the reference distribution, the evaluator.   python tests/test_dw4_states.py """
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, PAIRS, state_freq, state_ids, state_tv  # noqa: E402


def test_classes():
    assert len(KEYS) == 11 and set(KEYS) == set(NAMES)
    # every labelled graph lands in the class of its relabelling by any permutation of the particles
    from adjoint_samplers.utils.dw4_states import _TABLE
    for code in range(64):
        adj = {p: (code >> b) & 1 for b, p in enumerate(PAIRS)}
        for perm in itertools.permutations(range(4)):
            code2 = sum(adj[PAIRS[b]] << PAIRS.index(tuple(sorted((perm[PAIRS[b][0]], perm[PAIRS[b][1]])))) for b in range(6))
            assert _TABLE[code] == _TABLE[code2]
    print("ok  11 graph classes on 4 particles, invariant under relabelling")


def test_reference():
    ref = torch.tensor(np.load(Path(__file__).resolve().parents[1] / "data" / "test_split_DW4.npy", allow_pickle=True)).float()
    f = state_freq(state_ids(ref))
    assert abs(f.sum() - 1) < 1e-9
    # rotation / translation invariance
    th = 0.7
    R = torch.tensor([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]).float()
    z = ref.reshape(-1, 4, 2) @ R.T + torch.tensor([1.0, -2.0])
    assert torch.equal(state_ids(z.reshape(-1, 8)), state_ids(ref))
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(ref), generator=g)
    tvs = [state_tv(state_freq(state_ids(ref[torch.randperm(len(ref), generator=g)[:5000]])), state_freq(state_ids(ref[torch.randperm(len(ref), generator=g)[:5000]]))) for _ in range(5)]
    print("ok  reference (10000 samples): state frequencies", {NAMES[k][:18]: round(float(v), 3) for k, v in zip(KEYS, f) if v > 0.005}, f"; TV between random 5000-subsamples ~ {np.mean(tvs):.3f}")


if __name__ == "__main__":
    test_classes()
    test_reference()
    print("all passed")
