""" DW-4 (4 particles in 2D, pairwise double well in the distance d: E(d) = 0.9 (d - 4)^4 - 4 (d - 4)^2, wells at d = 2.51 (short) and 5.49 (long), barrier at d = 4).

    STATE of a configuration: the graph of SHORT pairs (d < d0 = 4) on the 4 particles, up to graph isomorphism. The distances already remove rotations and translations.
    There are 11 graphs on 4 vertices up to isomorphism; (number of edges, sorted degree sequence) separates all 11 of them.
"""
import itertools

import numpy as np
import torch

D0 = 4.0
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _key(adj):
    deg = [sum(1 for (i, j), a in zip(PAIRS, adj) if a and k in (i, j)) for k in range(4)]
    return (int(sum(adj)), tuple(sorted(deg)))


def _classes():
    keys = sorted({_key(a) for a in itertools.product([0, 1], repeat=6)})
    return keys


KEYS = _classes()                       # 11 classes
KEY_ID = {k: i for i, k in enumerate(KEYS)}
NAMES = {
    (0, (0, 0, 0, 0)): "empty (0 short pairs)", (1, (0, 0, 1, 1)): "1 pair", (2, (0, 1, 1, 2)): "path of 2 (3 particles)",
    (2, (1, 1, 1, 1)): "2 disjoint pairs", (3, (0, 2, 2, 2)): "triangle", (3, (1, 1, 1, 3)): "star (3 edges)", (3, (1, 1, 2, 2)): "path of 3 (4 particles)",
    (4, (2, 2, 2, 2)): "4-cycle", (4, (1, 2, 2, 3)): "triangle + pendant", (5, (2, 2, 3, 3)): "K4 minus an edge", (6, (3, 3, 3, 3)): "K4 (all 6 short)",
}
# lookup table from the 6-bit adjacency code to the class id
_TABLE = np.zeros(64, dtype=np.int64)
for code in range(64):
    adj = [(code >> b) & 1 for b in range(6)]
    _TABLE[code] = KEY_ID[_key(adj)]


def pair_distances(x, n_particles=4, spatial_dim=2):
    z = x.reshape(-1, n_particles, spatial_dim)
    return torch.stack([(z[:, i] - z[:, j]).norm(dim=-1) for i, j in PAIRS], 1)     # (B, 6)


def state_ids(x, d0=D0):
    """ class id (0..10) of every configuration of x (B, 8) """
    short = (pair_distances(x) < d0).long().cpu().numpy()
    code = (short * (1 << np.arange(6))).sum(1)
    return torch.as_tensor(_TABLE[code])


def state_freq(ids):
    return torch.bincount(ids, minlength=len(KEYS)).double() / len(ids)


def state_tv(p, q):
    return 0.5 * (p - q).abs().sum().item()
