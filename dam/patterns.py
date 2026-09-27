"""Pattern generation and corruption for binary associative memories.

All states and patterns are float64 arrays of +/-1. Float storage keeps the
linear algebra simple; nothing here is memory-bound at the sizes we use.
"""

import numpy as np


def random_patterns(rng: np.random.Generator, P: int, N: int, bias: float = 0.0) -> np.ndarray:
    """P x N matrix of +/-1 patterns, each bit +1 with probability (1 + bias) / 2.

    bias = 0 gives the unbiased patterns all capacity theory assumes; bias > 0
    makes patterns correlated with each other (they share the mean direction).
    """
    if not -1.0 < bias < 1.0:
        raise ValueError("bias must lie in (-1, 1)")
    return np.where(rng.random((P, N)) < (1.0 + bias) / 2.0, 1.0, -1.0)


def corrupt(rng: np.random.Generator, X: np.ndarray, frac: float) -> np.ndarray:
    """Flip exactly round(frac * N) distinct bits in every row of X."""
    X = np.atleast_2d(X)
    K, N = X.shape
    flips = int(round(frac * N))
    out = X.copy()
    if flips == 0:
        return out
    # argsort of iid uniforms gives an independent random subset per row.
    idx = np.argsort(rng.random((K, N)), axis=1)[:, :flips]
    rows = np.arange(K)[:, None]
    out[rows, idx] *= -1.0
    return out


def overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalized overlap m = a.b / N along the last axis (1 = identical)."""
    return np.sum(a * b, axis=-1) / a.shape[-1]
