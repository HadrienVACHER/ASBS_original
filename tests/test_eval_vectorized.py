""" dist_point_clouds_fast (vectorised) against dist_point_clouds (Python double loop) on DW-4 reference samples: same matrix of distances up to ties between equally good assignments,
    same equivariant W2, and the speed-up.     python tests/test_eval_vectorized.py """
import sys
import time
from pathlib import Path

import numpy as np
import ot as pot
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_samplers.utils.eval_utils import dist_point_clouds, dist_point_clouds_fast  # noqa: E402
from adjoint_samplers.utils.graph_utils import remove_mean  # noqa: E402


def load(n, off):
    ref = torch.tensor(np.load(Path(__file__).resolve().parents[1] / "data" / "test_split_DW4.npy", allow_pickle=True)).float()
    return remove_mean(ref[off:off + n], 4, 2).reshape(-1, 4, 2)


def test_equal():
    a, b = load(60, 0), load(70, 5000)
    t = time.time(); M0 = dist_point_clouds(a, b); t0 = time.time() - t
    t = time.time(); M1 = dist_point_clouds_fast(a, b); t1 = time.time() - t
    assert M0.shape == M1.shape == (60, 70)
    diff = (M0 - M1).abs()
    bad = float((diff > 1e-4).float().mean())
    assert bad < 2e-3 and float(diff.median()) < 1e-5, (bad, float(diff.max()))
    w0 = pot.emd2(M=M0.numpy() ** 2, a=np.ones(60) / 60, b=np.ones(70) / 70) ** 0.5
    w1 = pot.emd2(M=M1.numpy() ** 2, a=np.ones(60) / 60, b=np.ones(70) / 70) ** 0.5
    assert abs(w0 - w1) < 1e-3 * max(1.0, w0), (w0, w1)
    print(f"ok  60 x 70 pairs: median |ΔM| {float(diff.median()):.1e}, pairs differing by > 1e-4: {100 * bad:.2f} % (ties between assignments), max {float(diff.max()):.2e}; "
          f"equivariant W2 {w0:.5f} (loop) vs {w1:.5f} (vectorised); {t0:.2f} s vs {t1:.2f} s")


def test_speed():
    a, b = load(300, 0), load(300, 5000)
    t = time.time(); dist_point_clouds(a, b); t0 = time.time() - t
    t = time.time(); dist_point_clouds_fast(a, b); t1 = time.time() - t
    a, b = load(1000, 0), load(1000, 5000)
    t = time.time(); dist_point_clouds_fast(a, b); t2 = time.time() - t
    print(f"ok  300 x 300: loop {t0:.1f} s, vectorised {t1:.2f} s (x{t0 / t1:.0f}); vectorised 1000 x 1000: {t2:.1f} s")


if __name__ == "__main__":
    test_equal()
    test_speed()
    print("all passed")
