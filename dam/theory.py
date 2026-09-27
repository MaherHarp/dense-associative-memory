"""Analytical baselines the simulations are compared against.

1. Signal-to-noise crosstalk theory for dense memories with separable F.
   Probe the network with a stored pattern xi^1 and ask whether bit i is stable.
   The drive toward the correct value splits into

       signal     S = F(N) - F(N-2)                      (from xi^1 itself)
       crosstalk  sum_{mu != 1} eps_mu g(h_mu),
                  g(h) = F(h+1) - F(h-1),

   with h_mu a sum of N-1 iid +/-1 values and eps_mu an independent random
   sign. Each crosstalk term has mean 0 and variance V = E[g(h)^2], which we
   compute exactly from the binomial law of h. The central limit theorem then
   gives the one-step bit error rate

       p_err = Phi(-S / sqrt((P-1) V)).

   For F = x^2 this is Phi(-sqrt((N-1)/(P-1))), the textbook Hebbian result.
   For F = x^n it scales as N^(n-1) / (2n-3)!!. That is the Krotov-Hopfield
   capacity law.

2. Amit-Gutfreund-Sompolinsky replica-symmetric solution at T = 0 for the
   Hebbian network. The retrieval overlap m = erf(y), where y is the largest
   root of

       y (sqrt(2 alpha) + (2/sqrt(pi)) exp(-y^2)) = erf(y).

   A nonzero root exists only for alpha < alpha_c ~ 0.138.
"""

import math

import numpy as np
from scipy.special import erf, ndtr, ndtri
from scipy.stats import binom


def crosstalk_moments(N: int, kind: str = "poly", n: int = 2) -> tuple[float, float]:
    """Return (log S, log V) for probing a stored pattern of an N-bit network.

    Logs keep the exponential family in range (e^N at N = 1000).
    """
    if kind == "exp":
        # g(h) = e^h (e - 1/e); E[e^{2h}] = cosh(2)^{N-1}; S = e^N (1 - e^-2).
        logS = N + math.log1p(-math.exp(-2.0))
        logV = 2.0 * math.log(math.e - 1.0 / math.e) + (N - 1) * math.log(math.cosh(2.0))
        return logS, logV
    k = np.arange(N)
    h = 2.0 * k - (N - 1)
    pmf = binom.pmf(k, N - 1, 0.5)
    # Divide by N^(n-1) before squaring to keep the moments in range.
    scale = float(N) ** (n - 1)
    g = ((h + 1.0) ** n - (h - 1.0) ** n) / scale
    V = float(np.sum(pmf * g * g))
    S = (float(N) ** n - float(N - 2) ** n) / scale
    return math.log(S), math.log(V)


def bit_error(N: int, P: int, kind: str = "poly", n: int = 2) -> float:
    """Gaussian SNR prediction of the one-step bit error rate."""
    if P <= 1:
        return 0.0
    logS, logV = crosstalk_moments(N, kind, n)
    snr = math.exp(logS - 0.5 * (logV + math.log(P - 1)))
    return float(ndtr(-snr))


def capacity(N: int, kind: str = "poly", n: int = 2, stable_frac: float = 0.5) -> float:
    """Load P* at which a stored pattern has all N bits stable w.p. stable_frac.

    Assumes independent bit errors, so we need (1 - p_err)^N = stable_frac.
    That gives p_err = 1 - stable_frac^(1/N), z = Phi^-1(p_err), and the
    closed form P* = 1 + S^2 / (V z^2).
    """
    logS, logV = crosstalk_moments(N, kind, n)
    p_err = -math.expm1(math.log(stable_frac) / N)
    z = float(ndtri(p_err))
    return 1.0 + math.exp(2.0 * logS - logV - 2.0 * math.log(abs(z)))


def double_factorial(k: int) -> int:
    return 1 if k <= 0 else k * double_factorial(k - 2)


# ---------------------------------------------------------------- AGS theory

def _ags_phi(y: np.ndarray) -> np.ndarray:
    """sqrt(2 alpha) as a function of the root y."""
    return erf(y) / y - (2.0 / math.sqrt(math.pi)) * np.exp(-y * y)


def ags_alpha_c() -> float:
    """Critical load: max_y phi(y)^2 / 2, found by dense grid plus golden refinement."""
    ys = np.linspace(0.05, 5.0, 20001)
    j = int(np.argmax(_ags_phi(ys)))
    a, b = ys[max(j - 1, 0)], ys[min(j + 1, ys.size - 1)]
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(100):
        c, d = b - gr * (b - a), a + gr * (b - a)
        if _ags_phi(np.array(c)) > _ags_phi(np.array(d)):
            b = d
        else:
            a = c
    y = 0.5 * (a + b)
    return float(_ags_phi(np.array(y)) ** 2 / 2.0)


def ags_overlap(alpha: float) -> float:
    """Retrieval overlap m(alpha) on the retrieval branch; 0 above alpha_c."""
    target = math.sqrt(2.0 * alpha)
    ys = np.linspace(0.05, 8.0, 40001)
    phi = _ags_phi(ys) - target
    # Largest root: scan from the right for the last sign change.
    idx = np.nonzero(np.diff(np.sign(phi)))[0]
    if idx.size == 0:
        return 0.0
    j = int(idx[-1])
    a, b = ys[j], ys[j + 1]
    for _ in range(80):
        c = 0.5 * (a + b)
        if (_ags_phi(np.array(c)) - target) * (_ags_phi(np.array(a)) - target) <= 0:
            b = c
        else:
            a = c
    return float(erf(0.5 * (a + b)))
