"""Crosstalk theory vs simulation: one-step bit errors and error-free capacity scaling."""

import math

import numpy as np
import matplotlib.pyplot as plt

from dam import classical, theory
from dam.dense import DenseAM
from dam.patterns import random_patterns

from .common import COLORS, MARKERS, find_capacity, loglog_slope, save, stable_fraction, style


def bit_error_vs_load(rep, rng, quick=False):
    rep.h("E1. One-step bit error rate: simulation vs Gaussian crosstalk theory")
    rep.p("Probe = a stored pattern. We count bits whose single update would flip. "
          "The theory curve is Phi(-S/sqrt((P-1)V)), with the variance V computed exactly "
          "from the binomial law of the overlaps.")
    rep.p()
    configs = [(2, 256, 256), (3, 64, 64 ** 2), (4, 32, 32 ** 3)]
    K = 64 if quick else 200
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    rows = []
    for c, (n, N, scale) in enumerate(configs):
        loads = np.geomspace(0.02, 1.2, 6 if quick else 9)
        sims, ths = [], []
        for a in loads:
            P = max(2, int(round(a * scale)))
            X = random_patterns(rng, P, N)
            am = DenseAM(X, "poly", n)
            idx = rng.choice(P, size=min(K, P), replace=False)
            err = 1.0 - am.stable_bits(X[idx]).mean()
            th = theory.bit_error(N, P, "poly", n)
            sims.append(err)
            ths.append(th)
            rows.append([n, N, P, f"{P / scale:.3f}", f"{err:.4f}", f"{th:.4f}"])
        x = [max(2, int(round(a * scale))) / scale for a in loads]
        ax.plot(x, ths, color=COLORS[c], lw=2, label=f"n={n}, N={N} theory")
        ax.plot(x, np.maximum(sims, 1e-5), MARKERS[c], color=COLORS[c], ms=7,
                mec="#fcfcfb", mew=1.5, label=f"n={n} simulation")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 0.6)
    style(ax, "Bit error vs load, F(x)=x^n", "load P / N^(n-1)", "one-step bit error rate")
    ax.legend(fontsize=8, frameon=False)
    save(fig, "bit_error_vs_load.png")
    rep.table(["n", "N", "P", "P/N^(n-1)", "sim bit err", "theory"], rows)
    rep.p("Figure: `figures/bit_error_vs_load.png`")


def _dense_frac(N, P, n, kind, rng, probes):
    return stable_fraction(lambda X: DenseAM(X, kind, n), lambda m, S: m.stable_bits(S),
                           N, P, rng, probes=probes)


def capacity_scaling(rep, rng, quick=False):
    rep.h("E2. Error-free capacity scaling (50% of stored patterns fully stable)")
    rep.p("P* comes from a log-space bisection on a Monte Carlo stable fraction. "
          "The theory value assumes independent bit errors: (1 - p_err)^N = 1/2.")
    rep.p()
    probes = 128 if quick else 256
    grids = {2: [32, 64, 128, 256, 512], 3: [16, 24, 32, 48, 64], 4: [12, 16, 24, 32, 48]}
    if quick:
        grids = {2: [32, 64, 128], 3: [16, 24, 32], 4: [12, 16, 24]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rows, results = [], {}
    for c, (n, Ns) in enumerate(grids.items()):
        sim = [find_capacity(lambda P: _dense_frac(N, P, n, "poly", rng, probes)) for N in Ns]
        th = [theory.capacity(N, "poly", n) for N in Ns]
        results[n] = (Ns, sim, th)
        for N, s, t in zip(Ns, sim, th):
            rows.append([f"x^{n}", N, f"{s:.1f}", f"{t:.1f}", f"{s / t:.2f}"])
        axes[0].plot(Ns, th, color=COLORS[c], lw=2, label=f"n={n} theory")
        axes[0].plot(Ns, sim, MARKERS[c], color=COLORS[c], ms=7, mec="#fcfcfb", mew=1.5,
                     label=f"n={n} simulation")
    rep.table(["F", "N", "P* sim", "P* theory", "sim/theory"], rows)
    slope_rows = []
    for n, (Ns, sim, th) in results.items():
        slope_rows.append([n, n - 1, f"{loglog_slope(Ns, sim):.2f}", f"{loglog_slope(Ns, th):.2f}",
                           f"1/(2*{theory.double_factorial(2 * n - 3)}*ln N)"])
    rep.p("Log-log slope of P* vs N. The asymptotic exponent is n-1; the ln N factor "
          "pulls finite-size slopes below it:")
    rep.p()
    rep.table(["n", "asymptotic exponent", "sim slope", "theory slope", "prefactor ~"], slope_rows)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    style(axes[0], "Polynomial DAM capacity", "N (neurons)", "P* (patterns)")
    axes[0].legend(fontsize=8, frameon=False)

    # Exponential F: Gaussian SNR theory vs simulation.
    Ns = [8, 12, 16] if quick else [8, 10, 12, 14, 16, 18, 20]
    sim = [find_capacity(lambda P: _dense_frac(N, P, 0, "exp", rng, probes)) for N in Ns]
    gauss = [theory.capacity(N, "exp") for N in Ns]
    rate = float(np.polyfit(Ns, np.log2(sim), 1)[0])
    g_rate = float(np.polyfit(Ns, np.log2(gauss), 1)[0])
    rows = [[N, f"{s:.1f}", f"{g:.3g}", f"{s / 2 ** N:.4f}"] for N, s, g in zip(Ns, sim, gauss)]
    rep.p("Exponential F(x)=e^x:")
    rep.p()
    rep.table(["N", "P* sim", "P* Gaussian theory", "P*/2^N"], rows)
    rep.p(f"Fitted growth: sim P* ~ 2^({rate:.3f} N); Gaussian SNR theory predicts "
          f"2^({g_rate:.3f} N) (asymptotically log2(e^2/cosh 2) = "
          f"{math.log2(math.e ** 2 / math.cosh(2)):.3f}).")
    axes[1].plot(Ns, gauss, color=COLORS[1], lw=2, label="Gaussian SNR theory")
    axes[1].plot(Ns, sim, MARKERS[0], color=COLORS[0], ms=7, mec="#fcfcfb", mew=1.5,
                 label="simulation")
    axes[1].plot(Ns, [2.0 ** N for N in Ns], color="#8a8984", lw=1, ls="--",
                 label="2^N (all binary states)")
    axes[1].set_yscale("log")
    style(axes[1], "Exponential DAM capacity", "N (neurons)", "P* (patterns)")
    axes[1].legend(fontsize=8, frameon=False)
    save(fig, "capacity_scaling.png")
    rep.p("Figure: `figures/capacity_scaling.png`")
    return results, rate


def classical_rules(rep, rng, quick=False):
    rep.h("E3. Pairwise learning rules under correlated (biased) patterns")
    rep.p("Bias a means every bit is +1 with probability (1+a)/2, so any two patterns "
          "have an expected overlap a^2. P* is again the 50%-fully-stable load.")
    rep.p()
    N = 64 if quick else 128
    probes = 128 if quick else 256
    rules = {
        "Hebbian": classical.hebbian,
        "Storkey": classical.storkey,
        "Pseudo-inverse (zero diag)": lambda X: classical.pseudo_inverse(X, zero_diag=True),
    }
    biases = [0.0, 0.2, 0.4, 0.6]
    rows = []
    for name, rule in rules.items():
        row = [name]
        for a in biases:
            frac = lambda P: stable_fraction(rule, classical.stable_bits, N, P, rng,
                                             probes=probes, bias=a)
            cap = find_capacity(frac, p_max=N - 1)
            row.append("all P < N" if math.isinf(cap) else f"{cap / N:.3f}")
        rows.append(row)
    rep.table(["rule (N=%d), P*/N" % N] + [f"a={a}" for a in biases], rows)
    rep.p("Reference points: the Hebbian theory gives P*/N = "
          f"{theory.capacity(N, 'poly', 2) / N:.3f} at a=0, and Storkey's asymptotic "
          f"N/sqrt(2 ln N) gives {1 / math.sqrt(2 * math.log(N)):.3f}. The pseudo-inverse "
          "stores every P < N pattern at any bias: with the diagonal kept, W xi = xi; with "
          "it zeroed, the field is (1 - w_ii) xi and w_ii ~ P/N < 1. Its advantage is "
          "invariance to correlation, and it costs an O(P^3) solve plus non-local learning.")
