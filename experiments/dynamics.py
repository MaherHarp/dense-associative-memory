"""Attractor dynamics: the AGS retrieval transition and basins of attraction."""

import numpy as np
import matplotlib.pyplot as plt

from dam import classical, theory
from dam.dense import DenseAM
from dam.patterns import corrupt, overlap, random_patterns

from .common import COLORS, INK, MARKERS, save, style


def ags_transition(rep, rng, quick=False):
    rep.h("E4. Hebbian retrieval transition vs the AGS replica prediction")
    rep.p("Start at a stored pattern, run asynchronous dynamics to a fixed point, and "
          "record the final overlap m. The replica-symmetric T=0 theory says m jumps "
          "from about 0.97 to about 0 at alpha_c.")
    rep.p()
    a_c = theory.ags_alpha_c()
    alphas = np.round(np.arange(0.06, 0.201, 0.02 if quick else 0.01), 3)
    Ns = [400, 1000] if quick else [500, 2000]
    trials = 2 if quick else 6
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    table = {}
    for c, N in enumerate(Ns):
        ms = []
        for a in alphas:
            P = int(round(a * N))
            vals = []
            for _ in range(trials):
                X = random_patterns(rng, P, N)
                W = classical.hebbian(X)
                s, _, _ = classical.run_async(W, X[0], rng, max_sweeps=60)
                vals.append(float(overlap(s, X[0])))
            ms.append(float(np.mean(vals)))
        table[N] = ms
        ax.plot(alphas, ms, MARKERS[c], color=COLORS[c], ms=7, mec="#fcfcfb", mew=1.5,
                label=f"simulation N={N}")
    fine = np.linspace(alphas[0], alphas[-1], 300)
    ax.plot(fine, [theory.ags_overlap(a) for a in fine], color=INK, lw=2,
            label="AGS replica theory (N→∞)")
    ax.axvline(a_c, color="#8a8984", lw=1, ls="--")
    ax.text(a_c + 0.002, 0.3, f"α_c = {a_c:.4f}", color="#52514e", fontsize=9)
    style(ax, "Hebbian network: final overlap vs load", "load α = P/N",
          "final overlap m with the start pattern")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    save(fig, "ags_transition.png")
    rows = [[f"{a:.2f}", f"{theory.ags_overlap(a):.3f}"] + [f"{table[N][k]:.3f}" for N in Ns]
            for k, a in enumerate(alphas)]
    rep.table(["alpha", "AGS m"] + [f"sim m (N={N})" for N in Ns], rows)
    rep.p(f"alpha_c (computed) = {a_c:.5f}; m(alpha_c^-) = {theory.ags_overlap(a_c - 1e-6):.4f}. "
          "Above alpha_c the theory has no retrieval branch (shown as 0). The state the "
          "dynamics actually reaches there keeps a partial overlap, and finite N pushes "
          "the apparent transition to higher alpha.")
    rep.p("Figure: `figures/ags_transition.png`")


def basins(rep, rng, quick=False):
    rep.h("E5. Basins of attraction at a fixed load")
    N = 128 if quick else 256
    P = int(0.1 * N)
    trials = 12 if quick else 40
    rep.p(f"N={N}, P={P} (load 0.1). Corrupt a stored pattern by a fraction f of bits, "
          "run asynchronous dynamics, and count a success when the final overlap is "
          ">= 0.98. Each trial uses a fresh pattern set.")
    rep.p()
    models = {
        "Hebbian": lambda X: ("w", classical.hebbian(X)),
        "Pseudo-inverse (diag kept)": lambda X: ("w", classical.pseudo_inverse(X)),
        "Pseudo-inverse (zero diag)": lambda X: ("w", classical.pseudo_inverse(X, True)),
        "Dense x^3": lambda X: ("d", DenseAM(X, "poly", 3)),
    }
    fracs = np.round(np.arange(0.0, 0.46, 0.05), 2)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    rows = []
    for c, (name, build) in enumerate(models.items()):
        succ = []
        for f in fracs:
            ok = 0
            for _ in range(trials):
                X = random_patterns(rng, P, N)
                kind, m = build(X)
                s0 = corrupt(rng, X[0], f)[0]
                if kind == "w":
                    s, _, _ = classical.run_async(m, s0, rng)
                else:
                    s, _, _ = m.run_async(s0, rng)
                ok += overlap(s, X[0]) >= 0.98
            succ.append(ok / trials)
        rows.append([name] + [f"{v:.2f}" for v in succ])
        ax.plot(fracs, succ, color=COLORS[c], lw=2, marker=MARKERS[c], ms=7,
                mec="#fcfcfb", mew=1.5, label=name)
    style(ax, f"Retrieval from corrupted cues (N={N}, P={P})", "fraction of bits flipped",
          "retrieval success rate")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    save(fig, "basins.png")
    rep.table(["model"] + [f"f={f}" for f in fracs], rows)
    rep.p("Figure: `figures/basins.png`")
