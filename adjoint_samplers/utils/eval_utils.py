# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from scipy.optimize import linear_sum_assignment

import PIL
from matplotlib import pyplot as plt


def get_fig_axes(ncol, nrow=1, ax_length_in=2.0):
    figsize = (ncol * ax_length_in, nrow * ax_length_in)
    fig = plt.figure(figsize=figsize)
    axes = fig.subplots(nrow, ncol)
    return fig, axes


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    # https://stackoverflow.com/a/61756899
    return PIL.Image.frombytes(
        'RGB',
        fig.canvas.get_width_height(),
        fig.canvas.tostring_rgb()
    )


# from https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8
def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
        >>> R0 = torch.tensor([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float)
        >>> B = (R0.mm(A.T)).T
        >>> t0 = torch.tensor([3., 3.])
        >>> B += t0
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
        >>> B *= torch.tensor([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()


# from https://github.com/jarridrb/DEM/blob/main/dem/models/components/distribution_distances.py
def ot(x0, x1):
    dists = torch.cdist(x0, x1)
    _, col_ind = linear_sum_assignment(dists)
    x1 = x1[col_ind]
    return x1


# from https://github.com/jarridrb/DEM/blob/main/dem/models/components/distribution_distances.py
def dist_point_clouds(x0, x1):
    M = []
    for i in range(len(x0)):
        reordered = []
        for j in range(len(x1)):
            x1_reordered = ot(x0[i], x1[j])
            reordered.append(x1_reordered)
        reordered = torch.stack(reordered)
        R, t = torch.vmap(find_rigid_alignment)(
            x0[i][None].repeat(len(x1), 1, 1), reordered
        )
        superimposed = torch.matmul(reordered, R)
        M.append(torch.cdist(x0[i].reshape(1, -1), superimposed.reshape(len(x1), -1)))
    M = torch.stack(M).squeeze()
    return M


# from https://github.com/jarridrb/DEM/blob/main/dem/utils/data_utils.py
def interatomic_dist(x, n_particles, n_spatial_dim):
    B, D = x.shape
    assert D == n_particles * n_spatial_dim

    x = x.view(B, n_particles, n_spatial_dim)

    # Compute the pairwise interatomic distances
    # removes duplicates and diagonal
    distances = x[:, None, :, :] - x[:, :, None, :]
    distances = distances[
        :,
        torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1,
    ]
    return torch.linalg.norm(distances, dim=-1)


# Vectorised version of dist_point_clouds (same quantity, no Python loop over the N x N pairs): for every pair (i, j) the optimal assignment of the particles of x1[j] to those of x0[i]
# (minimising the sum of Euclidean distances, as `ot` does via linear_sum_assignment: here the minimum over the n! permutations, n = 4 for DW-4), then the Kabsch rotation of
# find_rigid_alignment (R = V U^T of the SVD of A_c^T B_c, B superimposed as B @ R) and the Euclidean distance between the flattened clouds.
def dist_point_clouds_fast(x0, x1, block=64):
    import itertools
    n, d = x0.shape[1], x0.shape[2]
    perms = torch.tensor(list(itertools.permutations(range(n))), device=x0.device)             # (P, n)
    rows = torch.arange(n, device=x0.device)
    out = []
    for s in range(0, len(x0), block):
        A = x0[s:s + block]                                                                    # (I, n, d)
        I, J = len(A), len(x1)
        C = torch.cdist(A[:, None].expand(I, J, n, d).reshape(I * J, n, d), x1[None].expand(I, J, n, d).reshape(I * J, n, d)).reshape(I, J, n, n)     # C[i, j, k, l] = |A_ik - B_jl|
        cost = C[:, :, rows[None, :], perms].sum(-1)                                            # (I, J, P): sum_k C[i, j, k, perm_k]
        best = perms[cost.argmin(-1)]                                                          # (I, J, n): column assigned to row k
        B = torch.gather(x1[None].expand(I, J, n, d), 2, best[..., None].expand(I, J, n, d))   # reordered B[i, j] = x1[j][best[i, j]]
        Ac = A[:, None] - A[:, None].mean(2, keepdim=True)
        Bc = B - B.mean(2, keepdim=True)
        H = torch.matmul(Ac.transpose(-1, -2), Bc)                                             # (I, J, d, d)
        U, S, Vh = torch.linalg.svd(H)
        R = torch.matmul(Vh.transpose(-1, -2), U.transpose(-1, -2))                            # V U^T
        sup = torch.matmul(B, R)
        out.append((A[:, None].reshape(I, 1, n * d) - sup.reshape(I, J, n * d)).norm(dim=-1))
    return torch.cat(out)
