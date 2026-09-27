"""Continuous modern Hopfield (softmax attention) retrieval: exponential capacity and beta."""

import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

from dam import modern
from dam.patterns import corrupt, random_patterns

from .common import COLORS, MARKERS, save, style


def nn_success_theory(d, flips, P):
    """Exact probability that the corrupted query is strictly closest to its own pattern.

    Given the query xi, each other random +/-1 pattern x_j has x_j.xi = 2B - d
    with B ~ Binomial(d, 1/2), independently across j. The true pattern scores
    s = d - 2 flips. So success = Pr(2B - d < s)^(P-1). This is the beta -> inf
    limit of the softmax update.
    """
    s = d - 2 * flips
    # 2B - d < s  <=>  B < (s + d) / 2 = d - flips
    q = binom.cdf(d - flips - 1, d, 0.5)
    return q ** (P - 1)


def modern_capacity(rep, rng, quick=False):
    rep.h("E6. Continuous modern Hopfield: exponential capacity and the role of beta")
    d, flip_frac = 64, 0.2
    flips = int(round(flip_frac * d))
    Ps = [2 ** k for k in range(4, 15 if quick else 18)]
    betas = [0.02, 0.05, 0.2, 1.0]
    K = 64 if quick else 200
    rep.p(f"d={d} with +/-1 patterns (norm M={math.sqrt(d):.0f}). The query is a stored "
          f"pattern with {flips} bits flipped ({flip_frac:.0%}), and we apply a single update "
          "f(xi) = X^T softmax(beta X xi). Reported: nearest-neighbour success (the "
          "beta→∞ limit, vs exact theory), and the mean relative error "
          "||f(xi)-x_i||/||x_i|| at each beta.")
    rep.p()
    nn_sim, nn_th = [], []
    err = {b: [] for b in betas}
    for P in Ps:
        X = random_patterns(rng, P, d)
        idx = rng.choice(P, size=min(K, P), replace=False)
        Q = corrupt(rng, X[idx], flip_frac)
        G = Q @ X.T
        own = G[np.arange(len(idx)), idx]
        G[np.arange(len(idx)), idx] = -np.inf
        nn_sim.append(float(np.mean(own > G.max(axis=1))))
        nn_th.append(float(nn_success_theory(d, flips, P)))
        for b in betas:
            out = modern.update(X, Q, b)
            e = np.linalg.norm(out - X[idx], axis=1) / math.sqrt(d)
            err[b].append(float(e.mean()))
    rows = [[P, f"{s:.3f}", f"{t:.3f}"] + [f"{err[b][k]:.3f}" for b in betas]
            for k, (P, s, t) in enumerate(zip(Ps, nn_sim, nn_th))]
    rep.table(["P", "NN success sim", "NN success theory"] + [f"rel err β={b}" for b in betas],
              rows)

    # The worst-case bound: always valid, but informative only for small corruption.
    rep.p("Retrieval bound ||f(xi)-x_i|| <= 2M eps. We check it at P=256 and beta=1 "
          "over 200 probes per corruption level. 'Informative' means the bound "
          "is below M, i.e. it certifies retrieval to within one pattern norm:")
    rep.p()
    brow = []
    X = random_patterns(rng, 256, d)
    for fl in (0, 1, 2, 4, 13):
        viol = inf = 0
        for _ in range(200):
            i = int(rng.integers(256))
            q = corrupt(rng, X[i], fl / d)[0]
            _, bnd = modern.retrieval_bound(X, i, q, 1.0)
            actual = float(np.linalg.norm(modern.update(X, q, 1.0) - X[i]))
            viol += actual > bnd + 1e-9
            inf += bnd < math.sqrt(d)
        brow.append([fl, viol, f"{inf / 200:.2f}"])
    rep.table(["flipped bits", "violations", "fraction informative"], brow)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(Ps, nn_th, color="#0b0b0b", lw=2, label="exact theory (β→∞)")
    axes[0].plot(Ps, nn_sim, MARKERS[0], color=COLORS[0], ms=7, mec="#fcfcfb", mew=1.5,
                 label="simulation")
    axes[0].set_xscale("log", base=2)
    style(axes[0], f"Nearest-pattern retrieval, d={d}, {flip_frac:.0%} bits flipped",
          "stored patterns P", "success rate")
    axes[0].legend(fontsize=8, frameon=False)
    for c, b in enumerate(betas):
        axes[1].plot(Ps, err[b], color=COLORS[c], lw=2, marker=MARKERS[c], ms=6,
                     mec="#fcfcfb", mew=1.5, label=f"β={b}")
    axes[1].set_xscale("log", base=2)
    style(axes[1], "One-step error vs inverse temperature", "stored patterns P",
          "mean ||f(ξ)-x_i|| / ||x_i||")
    axes[1].legend(fontsize=8, frameon=False)
    save(fig, "modern_retrieval.png")
    rep.p("Figure: `figures/modern_retrieval.png`")

    # Capacity vs dimension for the beta -> inf limit.
    ds = [16, 24, 32, 48, 64, 96, 128]
    rows = []
    p50s = []
    for dd in ds:
        fl = int(round(flip_frac * dd))
        q = binom.cdf(dd - fl - 1, dd, 0.5)
        p50 = 1 + math.log(0.5) / math.log(q)
        p50s.append(p50)
        rows.append([dd, fl, f"{p50:.3g}", f"{math.log2(p50) / dd:.3f}"])
    rep.p()
    rep.p("Exact theory for the load at which nearest-pattern retrieval succeeds 50% of "
          "the time, at the same corruption fraction:")
    rep.p()
    rep.table(["d", "flips", "P50", "log2(P50)/d"], rows)
    return dict(zip(ds, p50s))
