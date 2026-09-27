"""Continuous modern Hopfield network (Ramsauer et al. 2020).

Patterns are the rows of X (P x d); the state xi is in R^d.

    E(xi) = -lse(beta, X xi) + 1/2 xi.xi + const,
    lse(beta, z) = beta^-1 log sum exp(beta z)

    f(xi) = X^T softmax(beta X xi)

f is the concave-convex procedure (CCCP) step for E, so E never increases.
Its form is one head of softmax attention: query xi, keys and values X.

Retrieval bound (derived here in the same way as Ramsauer's Theorem 5).
With M = max_j ||x_j|| and separation Delta_i = x_i.x_i - max_{j!=i} x_i.x_j:

    x_j.xi - x_i.xi = (x_j - x_i).x_i + (x_j - x_i).(xi - x_i)
                    <= -Delta_i + 2 M ||xi - x_i||
    1 - p_i = sum_{j!=i} p_j <= (P-1) exp(-beta (Delta_i - 2 M ||xi - x_i||)) =: eps
    ||f(xi) - x_i|| = ||sum_{j!=i} p_j (x_j - x_i)|| <= 2 M eps
"""

import numpy as np
from scipy.special import logsumexp, softmax


def update(X: np.ndarray, xi: np.ndarray, beta: float) -> np.ndarray:
    """One retrieval step. xi can be (d,) or a (K, d) batch of queries."""
    p = softmax(beta * (np.atleast_2d(xi) @ X.T), axis=-1)
    out = p @ X
    return out[0] if xi.ndim == 1 else out


def energy(X: np.ndarray, xi: np.ndarray, beta: float) -> float:
    return float(-logsumexp(beta * (X @ xi)) / beta + 0.5 * xi @ xi)


def retrieve(X: np.ndarray, xi: np.ndarray, beta: float, max_steps: int = 50,
             tol: float = 1e-10) -> tuple[np.ndarray, int]:
    for step in range(1, max_steps + 1):
        new = update(X, xi, beta)
        if np.max(np.abs(new - xi)) < tol:
            return new, step
        xi = new
    return xi, max_steps


def separation(X: np.ndarray, i: int) -> float:
    g = X @ X[i]
    others = np.delete(g, i)
    return float(g[i] - others.max())


def retrieval_bound(X: np.ndarray, i: int, xi: np.ndarray, beta: float) -> tuple[float, float]:
    """Return (eps, 2 M eps): upper bounds on 1 - p_i and on ||f(xi) - x_i||.

    Informative only when Delta_i > 2 M ||xi - x_i||; otherwise eps >= 1 or
    grows without limit.
    """
    P = X.shape[0]
    M = float(np.max(np.linalg.norm(X, axis=1)))
    expo = -beta * (separation(X, i) - 2.0 * M * np.linalg.norm(xi - X[i]))
    eps = (P - 1) * np.exp(min(expo, 700.0))
    return eps, 2.0 * M * eps
