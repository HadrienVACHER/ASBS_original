""" Mode-weight correction of an AS buffer on the GMM target: raw vs refined centres, IMH vs SNIS vs jump + MALA,
    with the energy cost counted. See adjoint_samplers/components/mode_correction.py.

    buffer     samples_final.pt of train.py (dump_final=true), or a stand-in with the symptoms of AS (weights flattened
               by --power, modes slightly wide); --buffer-select all | recent:N | oldest:N picks the samples used.
    fit        EM on the buffer (M = 2K components, tied covariance), equal weights.
    refine     descent on E from each fitted mean, minima merged below merge_tol * sqrt(tr(Sigma) / d).
    methods    AS buffer | IMH, SNIS with raw centres (inflate --inflate-raw) | IMH, SNIS with refined centres
               (inflate --inflate-refined) | jump + MALA with refined centres. IMH, SNIS and jump + MALA get the
               same budget of energy evaluations (IMH: chains x (1 + steps)); refinement gradients are counted apart.
    metrics    mode_tv on the labels of all post-burn-in states of all chains (nearest true mode), the one of the final
               states, the effective sample size, and the iid floor at that ESS and at 50 000 samples.

    python scripts/imh_refined_gmm.py --dim 36 --K 36 --buffer results/.../samples_final.pt --wandb
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adjoint_samplers.components.mode_correction import (
    EnergyCounter, TiedMixture, chain_ess, iid_mode_tv, mode_tv_from_counts, refine_centres, run_imh,
    run_jump_mala, snis_mode_weights,
)
from adjoint_samplers.energies.dist_energy import DistEnergy
from adjoint_samplers.utils.dist_utils import GMM
from imh_fitted_demo import fit_gmm, load_samples


def select_buffer(buf, how):
    """ all | recent:N | oldest:N (the buffer is stored oldest first) """
    if how == "all":
        return buf
    kind, n = how.split(":")
    n = int(n)
    return buf[-n:] if kind == "recent" else buf[:n]


def tv_of(share, pi):
    return 0.5 * (share - pi).abs().sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=36)
    ap.add_argument("--K", type=int, default=36)
    ap.add_argument("--radius", type=float, default=4.25)
    ap.add_argument("--weight-ratio", type=float, default=20.0)
    ap.add_argument("--target-seed", type=int, default=0)
    ap.add_argument("--buffer", default=None, help="samples_final.pt of train.py (None: stand-in)")
    ap.add_argument("--buffer-key", default="buffer_x1", choices=["buffer_x1", "model_samples"])
    ap.add_argument("--buffer-select", default="all", help="all | recent:N | oldest:N")
    ap.add_argument("--buffer-size", type=int, default=10000, help="stand-in buffer size")
    ap.add_argument("--power", type=float, default=0.3, help="stand-in buffer: weights pi^power")
    ap.add_argument("--model-std", type=float, default=1.025, help="stand-in buffer: std of the modes")
    ap.add_argument("--components", type=int, default=None, help="M (default 2K)")
    ap.add_argument("--inflate-raw", type=float, default=1.2)
    ap.add_argument("--inflate-refined", type=float, default=1.0)
    ap.add_argument("--merge-tol", type=float, default=1.0, help="x sqrt(tr(Sigma)/d)")
    ap.add_argument("--refine-steps", type=int, default=200)
    ap.add_argument("--refine-lr", type=float, default=0.1)
    ap.add_argument("--chains", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--burn", type=int, default=20)
    ap.add_argument("--n-mala", type=int, default=5)
    ap.add_argument("--mala-h0", type=float, default=0.5, help="MALA step h = h0 d^(-1/3)")
    ap.add_argument("--jump-cycles", type=int, default=None, help="default: the IMH budget of energy evaluations")
    ap.add_argument("--as-epochs", type=int, default=2000, help="AS training epochs, for the cost line only")
    ap.add_argument("--as-resample", type=int, default=1000, help="AS buffer refill per epoch, for the cost line only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--out", default="results/imh_refined_gmm")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev, M, C, d, K = args.device, None, args.chains, args.dim, args.K
    M = args.components or 2 * K

    torch.manual_seed(args.seed)
    p = GMM(dim=d, K=K, seed=args.target_seed, layout="spread", weight_law="loguniform",
            weight_ratio=args.weight_ratio, radius=args.radius)
    p.to(dev)
    energy = DistEnergy(p, device=dev)
    pi = p.weights.to(dev)
    assign = p.assign

    # ---- buffer (T1)
    if args.buffer:
        buf, _ = load_samples(args.buffer, args.buffer_key, dev)
        buf = buf.reshape(-1, d)
    else:
        w = pi ** args.power
        k = torch.multinomial(w / w.sum(), args.buffer_size, replacement=True)
        buf = p.means[k] + args.model_std * torch.randn(args.buffer_size, d, device=dev)
    n_total = len(buf)
    buf = select_buffer(buf, args.buffer_select)
    print(f"target d={d} K={K} weight_ratio={args.weight_ratio:g} radius={args.radius:g}: weights {pi.min():.4f}..{pi.max():.4f}")
    print(f"buffer: {len(buf)} of {n_total} samples used (select {args.buffer_select}), device {dev}")

    # ---- fit and refine (T2)
    _, mu, cov = fit_gmm(buf, M, cov_type="tied")
    cov_t = cov[0]
    ec_ref = EnergyCounter(energy)
    merge_tol = args.merge_tol * (cov_t.trace() / d).sqrt().item()
    centres, n_grad_refine = refine_centres(ec_ref, mu, n_steps=args.refine_steps, lr=args.refine_lr, merge_tol=merge_tol)
    covered = torch.unique(assign(centres)).numel()
    raw_covered = torch.unique(assign(mu)).numel()
    moved = torch.cdist(mu, centres).min(-1).values
    print(f"fit: M={M} tied; true modes with a raw centre {raw_covered}/{K}; refined M'={len(centres)} "
          f"(merge_tol {merge_tol:.2f}), true modes covered {covered}/{K}; "
          f"raw centre -> minimum distance mean {moved.mean():.2f} max {moved.max():.2f}; {n_grad_refine} gradient evaluations")

    r_raw = TiedMixture(mu, cov_t * args.inflate_raw)
    r_ref = TiedMixture(centres, cov_t * args.inflate_refined)

    x0 = buf[torch.randint(len(buf), (C,), device=dev)]
    floor_big = iid_mode_tv(pi, 50000)
    rows = {}

    def add_row(name, share, final_labels, counts, diag, ec, ess, setup_grad=0, n_centres=None):
        tv = tv_of(share, pi)
        rows[name] = dict(
            mode_tv=tv,
            mode_tv_final=tv_of(torch.bincount(final_labels, minlength=K).float() / len(final_labels), pi) if final_labels is not None else float("nan"),
            ess=ess, floor_at_ess=iid_mode_tv(pi, ess) if ess and ess == ess and ess >= 1 else float("nan"),
            n_E=ec.n_E if ec else 0, n_grad=(ec.n_grad if ec else 0) + setup_grad, M=n_centres or 0, **diag)

    # ---- AS buffer
    lab = assign(buf)
    add_row("AS buffer", torch.bincount(lab, minlength=K).float() / len(lab), lab, None, {}, None, float("nan"))

    # ---- IMH (raw / refined), T3
    T, burn = args.steps, args.burn
    imh = {}
    for name, r, setup, n_c in (("IMH raw", r_raw, 0, M), ("IMH refined", r_ref, n_grad_refine, len(centres))):
        ec = EnergyCounter(energy)
        x, counts, diag = run_imh(ec, r, x0, T, burn, assign, K)
        add_row(name, counts.sum(0) / counts.sum(), assign(x), counts, dict(acc=diag["acc"], max_log_w=diag["max_log_w"]),
                ec, chain_ess(counts, pi), setup, n_c)
        imh[name] = ec.n_E
    budget = imh["IMH refined"]

    # ---- SNIS at the same budget, T4
    for name, r, setup, n_c in (("SNIS raw", r_raw, 0, M), ("SNIS refined", r_ref, n_grad_refine, len(centres))):
        ec = EnergyCounter(energy)
        pk, ess_norm, _, _ = snis_mode_weights(ec, r, budget, assign, K)
        add_row(name, pk, None, None, dict(ess_norm=ess_norm), ec, ess_norm * budget, setup, n_c)

    # ---- jump + MALA (refined centres), T5
    h = args.mala_h0 * d ** (-1 / 3)
    cycles = args.jump_cycles or max(6, budget // (C * (args.n_mala + 2)))
    ec = EnergyCounter(energy)
    x, counts, diag = run_jump_mala(ec, r_ref, x0, cycles, cycles // 3, args.n_mala, h, assign, K)
    add_row("jump+MALA refined", counts.sum(0) / counts.sum(), assign(x), counts, diag, ec, chain_ess(counts, pi),
            n_grad_refine, len(centres))

    # ---- report
    print(f"\n=== mode_tv (pooled labels of the post-burn-in states), floor at 50000 iid samples {floor_big:.3f}; "
          f"IMH budget {budget} evaluations of E; chains {C}, IMH steps {T}, burn {burn}")
    cols = ("mode_tv", "mode_tv_final", "ess", "floor_at_ess", "acc", "acc_jump", "max_log_w", "n_E", "n_grad", "M")
    print(f"  {'':<20s}" + "".join(f"{c:>14s}" for c in cols))
    for name, row in rows.items():
        print(f"  {name:<20s}" + "".join(
            f"{row[c]:14.3f}" if isinstance(row.get(c), float) and abs(row[c]) < 1e5 else f"{row.get(c, '-')!s:>14s}" for c in cols))
    print(f"\ncost of the AS training itself (for scale): about {args.as_epochs * args.as_resample} gradient evaluations "
          f"({args.as_epochs} epochs x {args.as_resample} buffer samples); refinement above: {n_grad_refine}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(args=vars(args), M_refined=len(centres), covered=covered, raw_covered=raw_covered,
                   n_grad_refine=n_grad_refine, floor_50000=floor_big, rows=rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    if args.wandb:
        import wandb
        run = wandb.init(project="adjoint_sampler_cv",
                         name=f"refined_d{d}_K{K}_wr{args.weight_ratio:g}_rad{args.radius:g}", config=vars(args))
        table = wandb.Table(columns=["method"] + list(cols), data=[
            [n] + [r_.get(c, float("nan")) if isinstance(r_.get(c), (int, float)) else float("nan") for c in cols]
            for n, r_ in rows.items()])
        run.log({"methods": table})
        for n, r_ in rows.items():
            for c in cols:
                if isinstance(r_.get(c), (int, float)):
                    run.summary[f"{n.replace(' ', '_').replace('+', '_')}/{c}"] = r_[c]
        run.summary.update({"M_refined": len(centres), "modes_covered_refined": covered, "floor_50000": floor_big,
                            "n_grad_refine": n_grad_refine})
        run.finish()


if __name__ == "__main__":
    main()
