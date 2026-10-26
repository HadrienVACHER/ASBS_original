""" Quench of DW-4 configurations to their local energy minimum (saddle-free Newton, batched), so that the short-pair graph is read on the BASIN and not on a raw distance
    threshold. E(x) = Σ_{i<j} e(d_ij), e(d) = a (d - o)^4 + b (d - o)^2 (a = 0.9, b = -4, o = 4 in the benchmark); gradient and Hessian are analytic (the Hessian of the
    pairwise sum is assembled from the blocks e'' u u^T + (e' / d) (I - u u^T), u = (x_i - x_j) / d).

    Step: -V diag(1 / max(|λ|, λ_min)) V^T g (eigenvalue flip: a descent direction even at negative curvature; the exact zero modes, translations and the rotation, carry no
    gradient), norm capped, halved until the energy decreases. Stops when ||∇E|| < tol.
"""
import torch

from adjoint_samplers.utils.dw4_states import PAIRS

A, B, O = 0.9, -4.0, 4.0


def energy_grad_hess(x, a=A, b=B, o=O):
    N = x.shape[0]
    z = x.reshape(N, 4, 2)
    E = torch.zeros(N, dtype=x.dtype)
    g = torch.zeros(N, 4, 2, dtype=x.dtype)
    H = torch.zeros(N, 4, 2, 4, 2, dtype=x.dtype)
    eye = torch.eye(2, dtype=x.dtype)
    for i, j in PAIRS:
        r = z[:, i] - z[:, j]
        d = torch.sqrt((r ** 2).sum(-1) + 1e-8)
        u = r / d[:, None]
        s = d - o
        e0, e1, e2 = a * s ** 4 + b * s ** 2, 4 * a * s ** 3 + 2 * b * s, 12 * a * s ** 2 + 2 * b
        E += e0
        g[:, i] += e1[:, None] * u
        g[:, j] -= e1[:, None] * u
        uu = u[:, :, None] * u[:, None, :]
        blk = e2[:, None, None] * uu + (e1 / d)[:, None, None] * (eye - uu)
        H[:, i, :, i, :] += blk
        H[:, j, :, j, :] += blk
        H[:, i, :, j, :] -= blk
        H[:, j, :, i, :] -= blk
    return E, g.reshape(N, 8), H.reshape(N, 8, 8)


@torch.no_grad()
def quench(x, tol=1e-4, max_iter=400, lam_min=0.05, max_step=0.5, batch=2000):
    """ returns the minimised configurations (N, 8), the final gradient norms (N,) and the number of iterations used """
    out, gn, its = [], [], 0
    for k in range(0, len(x), batch):
        xb = x[k:k + batch].double().clone()
        for it in range(max_iter):
            E, g, H = energy_grad_hess(xb)
            gnorm = g.norm(dim=-1)
            act = gnorm >= tol
            if not act.any():
                break
            lam, V = torch.linalg.eigh(H[act])
            c = torch.einsum("nij,ni->nj", V, g[act])
            step = -torch.einsum("nij,nj->ni", V, c / lam.abs().clamp(min=lam_min))
            sn = step.norm(dim=-1, keepdim=True)
            step = step * torch.clamp(max_step / sn.clamp(min=1e-30), max=1.0)
            idx = act.nonzero().flatten()
            alpha = torch.ones(len(idx), dtype=xb.dtype)
            pending = torch.ones(len(idx), dtype=torch.bool)
            for _ in range(10):
                En, _, _ = energy_grad_hess(xb[idx] + alpha[:, None] * step)
                ok = En <= E[idx] + 1e-12
                pending = ~ok
                if not pending.any():
                    break
                alpha = torch.where(pending, alpha * 0.5, alpha)
            move = ~pending
            xb[idx[move]] = xb[idx[move]] + alpha[move, None] * step[move]
        its = max(its, it + 1)
        out.append(xb)
        gn.append(energy_grad_hess(xb)[1].norm(dim=-1))
    return torch.cat(out).float(), torch.cat(gn), its
