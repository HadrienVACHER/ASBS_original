""" DW-4, path-space importance weights of the trained AS controller (offline, CPU, no training, report only): is a path-space independence MH / IS usable?

      python scripts/dw4_path_weights.py [--n 10000] [--controller results/dw4_e1500_94252_controller.pt]

    Target path measure P* = pi(x1) B_disc(path | x1) (x0 = 0, B_disc = the bridge of the DISCRETISED reference Euler scheme, so that the endpoint marginal of P* is exactly pi); P^u = the controlled Euler-Maruyama scheme that
    `sdeint` simulates (same timesteps: torch.linspace(0, 1, 100), i.e. 99 steps; sigma_max 2, sigma_min 0.001; noise on the zero-CoM subspace, x_{k+1} = x_k + g_k^2 u_k dt_k + g_k sqrt(dt_k) xi_k, xi ~ N(0, P)).
    The weight w = dP*/dP^u of a simulated path is exact in discrete time (ratio of Gaussian transition densities; xi is the noise that was drawn):
        log w = -E(x1) - log p_ref,disc(x1) - sum_k [ g_k sqrt(dt_k) <u_k, xi_k> + 1/2 g_k^2 dt_k |u_k|^2 ],   log p_ref,disc(x1) = -|x1|^2 / (2 V) - 3 log(2 pi V),  V = sum_k g_k^2 dt_k   (up to a constant).
    UNIT TEST first (toy, 2-D, no projection, constant control u, target N(m, s^2 I)): self-normalised IS with these weights reproduces the target mean and variance (|z| < 3.5, 4e5 paths), and the same estimator with the sign of the
    <u, xi> term flipped fails (|z| > 5 on the mean): the formula and its signs are validated before the real model is touched.
    REAL MODEL: the controller of `dw4_e1500_94252_controller.pt` = AS run 94252 (commit c166b5d), NOT ed8bb67 (the ed8bb67 controller was overwritten), config of results/dw4_as_states_e1500 (EGNN hidden 128, 5 layers, nfe 100).
    N paths (seed 0): ESS = (sum w)^2 / sum w^2, Pareto k_hat of w (top 20 %, scripts/dw4_division_b.khat), max weight share, SD of log w; independence MH on the paths (100 chains x 99 steps, accept min(1, w'/w), states of steps 50-99);
    basin proportions (quenched) of the raw samples, of the self-normalised IS (bootstrap SE) and of the MH states, vs ref_B (half B) with |log p / ref| and its sum (raw AS 94252: weights 0.434 / 0.375 / 0.113 / 0.062 / 0.016).
    PRE-REGISTERED (written before the first run, report only): path weights USABLE iff ESS > 50 % AND k_hat < 0.5 (the rule of scripts/dw4_is_inflate.py).
"""
import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dw4_division_b as b  # noqa: E402
import dw4_division_step0 as s0  # noqa: E402
from adjoint_samplers.components.model import EGNN_dynamics  # noqa: E402
from adjoint_samplers.components.sde import ControlledSDE, GraphVESDE, VESDE  # noqa: E402
from adjoint_samplers.utils.dw4_quench import quench  # noqa: E402
from adjoint_samplers.utils.dw4_states import KEYS, NAMES, state_ids  # noqa: E402
from adjoint_samplers.utils.train_utils import get_timesteps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def simulate(u_fn, g_fn, ts, x0, noise_fn, project, flip=False):
    """ controlled Euler-Maruyama; returns x1, S = -sum [g sqrt(dt) <u, xi> + 1/2 g^2 dt |u|^2] (float64) and V = sum g^2 dt """
    x = x0.clone()
    S = torch.zeros(len(x), dtype=torch.float64)
    V = 0.0
    for i in range(len(ts) - 1):
        t = ts[i]
        dt = ts[i + 1] - t
        g = g_fn(t)
        u = u_fn(t, x)
        xi = noise_fn(x)
        cross = g * dt.sqrt() * (u * xi).sum(-1)
        S += (-(-cross if flip else cross) - 0.5 * g ** 2 * dt * (u ** 2).sum(-1)).double()
        V += float(g ** 2 * dt)
        x = project(x + g ** 2 * u * dt + g * dt.sqrt() * xi)
    return x, S, V


def toy_test():
    torch.manual_seed(0)
    d, n = 2, 400_000
    sigma_min, sigma_max = 0.001, 2.0
    ref = VESDE(sigma_min, sigma_max)
    ts = get_timesteps(0.0, 1.0, steps=100)
    a = torch.tensor([0.2, -0.1])
    m, s = torch.tensor([1.0, -0.5]), 2.0
    out = {}
    for name, flip in (("exact formula", False), ("sign of <u, xi> flipped (control)", True)):
        x1, S, V = simulate(lambda t, x: a.expand_as(x), ref.diff, ts, torch.zeros(n, d), torch.randn_like, lambda x: x, flip=flip)
        logE = (x1 - m).pow(2).sum(-1) / (2 * s ** 2)
        lw = -logE.double() + 0.5 * (x1 ** 2).sum(-1).double() / V + S                                  # -E - log p_ref (constants cancel in the self-normalisation)
        w = torch.exp(lw - lw.max())
        mean = (w[:, None] * x1.double()).sum(0) / w.sum()
        var = (w[:, None] * (x1.double() - mean) ** 2).sum(0) / w.sum()
        ess = float(w.sum() ** 2 / (w ** 2).sum())
        se_m = s / math.sqrt(ess)
        zm = (mean - m.double()) / se_m
        out[name] = (float(zm.abs().max()), float(((var - s ** 2).abs() / (s ** 2 * math.sqrt(2 / ess))).max()), ess)
        print(f"  toy {name:36}: ESS {ess:9.0f}  z(mean) max {out[name][0]:6.2f}  z(var) max {out[name][1]:6.2f}")
    ok = out["exact formula"][0] < 3.5 and out["exact formula"][1] < 3.5 and out["sign of <u, xi> flipped (control)"][0] > 5
    print(f"unit test (|z| < 3.5 for the exact formula, > 5 for the flipped control): {'PASS' if ok else 'FAIL (stop)'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--controller", default=str(ROOT / "results" / "dw4_e1500_94252_controller.pt"))
    a = ap.parse_args()
    if not toy_test():
        sys.exit(1)
    torch.manual_seed(0)
    ref_sde = GraphVESDE(n_particles=4, spatial_dim=2, sigma_max=2, sigma_min=0.001)
    ctrl = EGNN_dynamics(n_particles=4, spatial_dim=2, hidden_nf=128, n_layers=5, act_fn=torch.nn.SiLU(), recurrent=True, tanh=True, attention=True, condition_time=True, agg="sum")
    ctrl.load_state_dict(torch.load(a.controller, map_location="cpu", weights_only=False)["controller"])
    ctrl.eval()
    sde = ControlledSDE(ref_sde, ctrl)
    ts = get_timesteps(0.0, 1.0, steps=100)
    import time
    t0 = time.time()
    xs, Ss = [], []
    V = None
    chunk = 2000
    for lo in range(0, a.n, chunk):
        x0 = torch.zeros(min(chunk, a.n - lo), 8)
        x1, S, V = simulate(lambda t, x: ctrl(t, x), ref_sde.diff, ts, x0, ref_sde.randn_like, lambda x: ref_sde.propagate(x, torch.zeros_like(x)))
        xs.append(x1)
        Ss.append(S)
        print(f"  simulated {lo + len(x0)} paths ({time.time() - t0:.0f} s)", flush=True)
    x1, S = torch.cat(xs).double(), torch.cat(Ss)
    E = s0.energy(x1)
    lw = -E + 0.5 * (x1 ** 2).sum(-1) / V + S                                                    # up to a constant
    w = torch.exp(lw - lw.max())
    N = len(w)
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    kh = b.khat(w.numpy())
    print(f"\npaths: {N}; V = {V:.4f}; log w: sd {float(lw.std()):.2f}, range {float(lw.max() - lw.min()):.1f}; ESS {ess:.0f} ({100 * ess / N:.2f} %); k_hat {kh:.3f}; max weight share {float(w.max() / w.sum()):.4f}")
    ref, q, lab_r, types, mu8 = b.load_reference()
    names = [NAMES[KEYS[k]] for k in types]
    Bi = torch.arange(1, len(ref), 2)
    refB = torch.stack([(lab_r[Bi] == k).double().mean() for k in types])
    qx, _, _ = quench(x1)
    lab = state_ids(qx.double())
    oh = torch.stack([(lab == k).double() for k in types], 1)
    raw = oh.mean(0)
    p_is = (w[:, None] * oh).sum(0) / w.sum()
    g2 = torch.Generator().manual_seed(1)
    boots = torch.stack([(lambda i: (w[i][:, None] * oh[i]).sum(0) / w[i].sum())(torch.randint(N, (N,), generator=g2)) for _ in range(500)])
    C, T, BURN = 100, 99, 50
    gen = torch.Generator().manual_seed(2)
    idx = torch.arange(C * (T + 1)).reshape(C, T + 1) if N >= C * (T + 1) else None
    p_mh, acc = None, float("nan")
    if idx is not None:
        state, accs, labs = idx[:, 0], [], []
        for t in range(1, T + 1):
            prop = idx[:, t]
            ok = torch.rand(C, generator=gen, dtype=torch.float64) < torch.exp((lw[prop] - lw[state]).clamp(max=0))
            state = torch.where(ok, prop, state)
            accs.append(ok.double().mean())
            if t >= BURN:
                labs.append(lab[state])
        acc = float(torch.stack(accs).mean())
        labs = torch.stack(labs)
        p_mh = torch.stack([(labs == k).double().mean() for k in types])
    e = lambda p: torch.log(p.clamp(min=1e-12) / refB).abs()
    print(f"IMH acceptance {acc:.3f}\n{'basin':26}{'ref_B':>8}{'raw':>8}{'p_IS':>8}{'SE':>8}{'p_MH':>8}{'|log raw/ref|':>15}{'|log IS/ref|':>14}{'|log MH/ref|':>14}")
    for i in range(len(types)):
        print(f"{names[i]:26}{float(refB[i]):8.4f}{float(raw[i]):8.4f}{float(p_is[i]):8.4f}{float(boots.std(0)[i]):8.4f}{float(p_mh[i]) if p_mh is not None else float('nan'):8.4f}{float(e(raw)[i]):15.3f}{float(e(p_is)[i]):14.3f}{float(e(p_mh)[i]) if p_mh is not None else float('nan'):14.3f}")
    print(f"  sum |log p/ref|: raw {float(e(raw).sum()):.3f}, IS {float(e(p_is).sum()):.3f}" + (f", MH {float(e(p_mh).sum()):.3f}" if p_mh is not None else ""))
    print(f"PRE-REGISTERED (usable iff ESS > 50 % and k_hat < 0.5): {ess / N > 0.5 and kh < 0.5}")


if __name__ == "__main__":
    main()
