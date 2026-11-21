""" Voronoi truth of MoS-dk (rules_dk_step5.md): 10^6 exact draws (only the first K coordinates matter for the nearest-mean cell: the means have equal norms and live on the first K axes), floor = median TV of an exact sampler at n = 10^4 (100 repetitions).
    python scripts/mosdk_truth.py D   ->  figures/mog_mos_truth_mosdk_d<D>.json """
import json, sys
import numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gmm_dk
D, K, DELTA, NU = int(sys.argv[1]), 8, 6.0, 3.0
mu = gmm_dk.means(D, K, DELTA).double().numpy()[:, :K]; w = gmm_dk.weights(K).double().numpy(); w /= w.sum()
rng = np.random.default_rng(123)


def draw(n):
    comp = rng.choice(K, size=n, p=w); z = rng.standard_normal((n, K)); chi = rng.gamma(NU / 2, 2.0 / NU, size=(n, K))
    return mu[comp] + np.sqrt((NU + 1) / NU) * z / np.sqrt(chi)


def props(x):
    return np.bincount((x @ mu.T - 0.5 * (mu ** 2).sum(1)[None]).argmax(1), minlength=K) / len(x)


cnt = np.zeros(K); N = 0
for _ in range(5):
    x = draw(200_000); cnt += props(x) * len(x); N += len(x)
p = cnt / N
floor = float(np.median([0.5 * np.abs(props(draw(10_000)) - p).sum() for _ in range(100)]))
json.dump(dict(p=p.tolist(), floor_n=floor, n_draws=N, d=D, K=K, delta=DELTA, nu=NU), open(f"figures/mog_mos_truth_mosdk_d{D}.json", "w"), indent=1)
print(D, "p", np.round(p, 4), "floor", round(floor, 4))
