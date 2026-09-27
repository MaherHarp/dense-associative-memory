"""Shared helpers: capacity search, a markdown report, and a matplotlib style."""

import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")

# Fixed categorical order (blue, orange, aqua, yellow); markers carry identity too.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=140, facecolor="#fcfcfb")
    plt.close(fig)


class Report:
    def __init__(self):
        self.lines = []
        self.t0 = time.time()

    def h(self, text):
        self.lines += ["", f"## {text}", ""]
        print(f"\n== {text}")

    def p(self, text=""):
        self.lines.append(text)
        print(text)

    def table(self, header, rows):
        self.p("| " + " | ".join(header) + " |")
        self.p("|" + "---|" * len(header))
        for r in rows:
            self.p("| " + " | ".join(str(c) for c in r) + " |")
        self.p()

    def write(self, name):
        os.makedirs(RES_DIR, exist_ok=True)
        self.p(f"\n_Total wall time: {time.time() - self.t0:.1f} s._")
        with open(os.path.join(RES_DIR, name), "w") as f:
            f.write("# Results\n" + "\n".join(self.lines) + "\n")


def stable_fraction(make_model, stable_bits, N, P, rng, probes=256, max_k=64, bias=0.0):
    """Fraction of stored patterns whose every bit is a fixed point.

    Draws fresh pattern sets until `probes` stored patterns have been checked.
    make_model(X) builds the memory; stable_bits(model, S) returns a K x N mask.
    """
    from dam.patterns import random_patterns

    hits = total = 0
    while total < probes:
        X = random_patterns(rng, P, N, bias)
        model = make_model(X)
        k = min(P, max_k, probes - total)
        ok = stable_bits(model, X[:k]).all(axis=1)
        hits += int(ok.sum())
        total += k
    return hits / total


def find_capacity(frac_at, p_lo=2, p_max=10 ** 7, iters=7):
    """Log-space bisection for the P where frac_at(P) crosses 0.5.

    Brackets by doubling, then bisects on integers. frac_at is a noisy MC
    estimate, so the result is itself an estimate: +/- a few percent at 256 probes.
    """
    lo, hi = p_lo, p_lo
    while frac_at(hi) >= 0.5:
        if hi == p_max:
            return float("inf")  # still mostly stable at the largest admissible load
        lo, hi = hi, min(hi * 2, p_max)
    for _ in range(iters):
        mid = int(round(math.sqrt(lo * hi)))
        if mid in (lo, hi):
            break
        if frac_at(mid) >= 0.5:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def loglog_slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])
