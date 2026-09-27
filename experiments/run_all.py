"""Run every experiment and write results/results.md plus figures/*.png.

    python3 -m experiments.run_all           # full run
    python3 -m experiments.run_all --quick   # smaller grids, for smoke testing
"""

import argparse
import math

import numpy as np

from dam import theory

from .capacity import bit_error_vs_load, capacity_scaling, classical_rules
from .common import Report
from .dynamics import ags_transition, basins
from .modern_exp import modern_capacity


def cost_table(rep, poly_results, exp_rate):
    N = 64
    rep.h(f"E7. What the capacity costs: parameters and compute at N = {N}")
    rep.p("The same energy can be stored two ways. The tensor form keeps the n-body "
          "synaptic tensor (C(N, n) distinct entries). The pattern form keeps the "
          "P x N matrix of +/-1 patterns (hidden units). A full asynchronous sweep costs "
          "N * C(N-1, n-1) multiply-adds in tensor form and about N * P in pattern form. "
          "P* is the simulated value when N is on the simulated grid, otherwise the theory value.")
    rep.p()
    rows = []
    for n in (2, 3, 4):
        Ns, sim, _ = poly_results[n]
        simulated = dict(zip(Ns, sim))
        P = simulated.get(N, theory.capacity(N, "poly", n))
        bits = N * P
        tensor = math.comb(N, n)
        rows.append([f"x^{n}", f"{P:.0f}", "sim" if N in simulated else "theory",
                     f"{bits:.3g}", f"{tensor:.3g}", f"{bits / tensor:.3f}",
                     f"{N * math.comb(N - 1, n - 1):.3g}", f"{N * P:.3g}"])
    rep.table(["F", "P*", "source", "bits stored (N P*)", "tensor params",
               "bits / tensor param", "sweep MACs (tensor)", "sweep MACs (pattern)"], rows)
    rep.p("Superlinear capacity is paid for in parameters. In pattern form the memory "
          "is the patterns, stored at exactly 1 bit per parameter, and update cost grows "
          "linearly in P. In tensor form the parameter count grows as N^n / n!, so "
          "bits per parameter stays well below 1 and shrinks roughly like 1/ln N. "
          f"The exponential model grows as 2^({exp_rate:.2f} N) in the simulated range and "
          "has no finite tensor form. Its pattern form is a softmax over all stored "
          "patterns, the same cost as the continuous model in E6.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rep = Report()
    rep.p(f"Seed {args.seed}; {'quick' if args.quick else 'full'} run. All numbers are "
          "from simulation on synthetic random patterns, apart from rows labelled theory.")
    bit_error_vs_load(rep, rng, args.quick)
    poly, exp_rate = capacity_scaling(rep, rng, args.quick)
    classical_rules(rep, rng, args.quick)
    ags_transition(rep, rng, args.quick)
    basins(rep, rng, args.quick)
    modern_capacity(rep, rng, args.quick)
    cost_table(rep, poly, exp_rate)
    rep.write("results_quick.md" if args.quick else "results.md")


if __name__ == "__main__":
    main()
