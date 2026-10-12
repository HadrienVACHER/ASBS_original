# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Benchmark and check the two speed options (`fast_sampling`, `buffer_on_device`).

    python scripts/bench_sampler.py --dim 165 --K 2

On a CUDA machine this
  1. checks that the CUDA-graph simulation gives the same x1 as the eager loop
     (noise switched off, so the comparison is deterministic),
  2. times eager vs CUDA-graph sampling,
and everywhere it
  3. checks DeviceBatchLoader against DataLoader and times both.
"""

import argparse
import time
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adjoint_samplers.components.buffer import BatchBuffer, DeviceBatchLoader
from adjoint_samplers.components.model import FourierMLP
from adjoint_samplers.components.sde import (
    ControlledSDE, VESDE, sdeint, enable_cuda_graph_sampling,
)
from adjoint_samplers.utils.train_utils import get_timesteps


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def timeit(fn, device, n):
    fn(); sync(device)  # warm up
    t = time.perf_counter()
    for _ in range(n):
        fn()
    sync(device)
    return (time.perf_counter() - t) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=165)
    ap.add_argument("--K", type=float, default=2)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--nfe", type=int, default=200)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    print(f"device={dev}  dim={a.dim}  batch={a.batch}  nfe={a.nfe}")

    # ---- model: random (non-zero) controller so the drift is non-trivial
    torch.manual_seed(0)
    controller = FourierMLP(dim=a.dim, activation=torch.nn.GELU(), channels=a.channels).to(dev)
    with torch.no_grad():
        for p in controller.out_layer.parameters():
            p.normal_(0, 0.1)
    ref = VESDE(sigma_min=0.001, sigma_max=a.K).to(dev)
    sde = ControlledSDE(ref, controller).to(dev)
    timesteps = get_timesteps(0.0, 1.0, steps=a.nfe).to(dev)
    x0 = torch.randn(a.batch, a.dim, device=dev) * a.K

    # ---- 1/2: sampling
    t_eager = timeit(lambda: sdeint(sde, x0, timesteps, only_boundary=True), dev, a.reps)
    print(f"sampling, eager      : {t_eager * 1e3:8.1f} ms / batch")

    if dev.startswith("cuda"):
        # deterministic check: no noise
        randn = sde.randn_like
        sde.randn_like = lambda x: torch.zeros_like(x)
        _, x1_eager = sdeint(sde, x0, timesteps, only_boundary=True)
        enable_cuda_graph_sampling(True)
        _, x1_graph = sdeint(sde, x0, timesteps, only_boundary=True)
        err = (x1_eager - x1_graph).abs().max().item()
        print(f"check (no noise)     : max |x1_eager - x1_graph| = {err:.2e}", "OK" if err < 1e-3 else "MISMATCH")
        enable_cuda_graph_sampling(False)
        sde.randn_like = randn

        # weights are read in place: capture, change the weights, replay
        sde.randn_like = lambda x: torch.zeros_like(x)
        enable_cuda_graph_sampling(True)
        sdeint(sde, x0, timesteps, only_boundary=True)  # capture
        with torch.no_grad():
            for p in controller.parameters():
                p.mul_(1.1)
        _, x1_graph = sdeint(sde, x0, timesteps, only_boundary=True)  # replay
        enable_cuda_graph_sampling(False)
        _, x1_eager = sdeint(sde, x0, timesteps, only_boundary=True)
        err = (x1_eager - x1_graph).abs().max().item()
        print(f"check (weights moved): max |x1_eager - x1_graph| = {err:.2e}", "OK" if err < 1e-3 else "MISMATCH")
        sde.randn_like = randn

        enable_cuda_graph_sampling(True)
        t_graph = timeit(lambda: sdeint(sde, x0, timesteps, only_boundary=True), dev, a.reps)
        enable_cuda_graph_sampling(False)
        print(f"sampling, CUDA graph : {t_graph * 1e3:8.1f} ms / batch   (x{t_eager / t_graph:.1f})")
    else:
        print("no CUDA: graph sampling skipped")

    # ---- 3: batch loading
    N = 10000
    buf = BatchBuffer(N)
    buf.add({"x0": torch.randn(N, a.dim), "x1": torch.randn(N, a.dim), "adjoint1": torch.randn(N, a.dim)})
    ds = buf.build_dataset(duplicates=10)

    ref_loader = DataLoader(ds, batch_size=a.batch, shuffle=True)
    dev_loader = DeviceBatchLoader(ds, a.batch, dev)
    assert len(ref_loader) == len(dev_loader)
    n_ref = sum(b["x0"].shape[0] for b in ref_loader)
    n_dev = sum(b["x0"].shape[0] for b in dev_loader)
    # every buffer row must be visited exactly `duplicates` times per pass
    counts = torch.zeros(N, dtype=torch.long)
    idx_of = {tuple(r.tolist()): i for i, r in enumerate(ds.total_data["x0"][:, :4])}
    for b in dev_loader:
        for r in b["x0"][:, :4].cpu():
            counts[idx_of[tuple(r.tolist())]] += 1
    ok = n_ref == n_dev == len(ds) and bool((counts == 10).all())
    print(f"check (loader)       : {n_dev} samples/pass, each row 10x -> {'OK' if ok else 'MISMATCH'}")

    def run(loader, n=100):
        it = iter(_cycle(loader))
        for _ in range(n):
            b = next(it)
            {k: v.to(dev) for k, v in b.items()}
        sync(dev)

    t0 = timeit(lambda: run(ref_loader), dev, 2)
    t1 = timeit(lambda: run(dev_loader), dev, 2)
    print(f"100 batches, DataLoader        : {t0 * 1e3:8.1f} ms")
    print(f"100 batches, DeviceBatchLoader : {t1 * 1e3:8.1f} ms   (x{t0 / t1:.1f})")


def _cycle(it):
    while True:
        for x in it:
            yield x


if __name__ == "__main__":
    main()
