"""Classical (pairwise) Hopfield networks: learning rules and dynamics.

Energy E(s) = -1/2 s^T W s. For symmetric W with non-negative diagonal,
asynchronous sign updates never increase E, so they always reach a fixed
point. Synchronous updates carry no such guarantee and can end in a 2-cycle.
"""

import numpy as np


def hebbian(X: np.ndarray) -> np.ndarray:
    """Outer-product rule W = X^T X / N with zero self-coupling."""
    N = X.shape[1]
    W = X.T @ X / N
    np.fill_diagonal(W, 0.0)
    return W


def storkey(X: np.ndarray) -> np.ndarray:
    """Storkey (1997) incremental rule, which subtracts each new pattern's crosstalk.

    w_ij += (xi_i xi_j - xi_i h_ji - h_ij xi_j) / N, where
    h_ij = sum_{k != i, j} w_ik xi_k. Unlike the original, we zero the diagonal
    after every step so the Lyapunov guarantee holds.
    """
    P, N = X.shape
    W = np.zeros((N, N))
    for xi in X:
        h = W @ xi
        # H[i, j] = h_i - w_ii xi_i - w_ij xi_j  (the diagonal is kept at zero)
        H = h[:, None] - W * xi[None, :]
        W += (np.outer(xi, xi) - xi[:, None] * H.T - H * xi[None, :]) / N
        np.fill_diagonal(W, 0.0)
    return W


def pseudo_inverse(X: np.ndarray, zero_diag: bool = False) -> np.ndarray:
    """Projection rule W = X^T (X X^T)^-1 X, the orthogonal projector onto span(X).

    With the diagonal kept, every pattern satisfies W xi = xi exactly (so all
    P < N patterns are fixed points). The self-coupling w_ii then approaches
    P/N, and that self-coupling is what shrinks the basins.
    """
    C = X @ X.T
    W = X.T @ np.linalg.solve(C, X)
    W = 0.5 * (W + W.T)
    if zero_diag:
        np.fill_diagonal(W, 0.0)
    return W


def energy(W: np.ndarray, s: np.ndarray) -> float:
    return float(-0.5 * s @ W @ s)


def stable_bits(W: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Boolean K x N mask: does bit i of state k agree with sign(W s)_i?

    A zero field counts as unstable, which is the conservative choice.
    """
    H = np.atleast_2d(S) @ W.T
    return H * S > 0


def run_async(W: np.ndarray, s0: np.ndarray, rng: np.random.Generator,
              max_sweeps: int = 100) -> tuple[np.ndarray, int, bool]:
    """Random-order asynchronous updates until a full sweep changes nothing.

    Keeps the local field h = W s up to date incrementally, so each flip costs
    O(N). On a zero field the current state is kept. Returns
    (state, sweeps, converged).
    """
    s = s0.astype(float).copy()
    h = W @ s
    N = s.size
    for sweep in range(1, max_sweeps + 1):
        changed = False
        for i in rng.permutation(N):
            hi = h[i]
            if hi == 0.0:
                continue
            new = 1.0 if hi > 0.0 else -1.0
            if new != s[i]:
                h += W[i] * (new - s[i])  # W symmetric: row i == column i
                s[i] = new
                changed = True
        if not changed:
            return s, sweep, True
    return s, max_sweeps, False


def run_sync(W: np.ndarray, s0: np.ndarray, max_steps: int = 100) -> tuple[np.ndarray, int, str]:
    """Update all bits in parallel. Returns (state, steps, 'fixed' | 'cycle2' | 'none')."""
    prev = None
    s = s0.astype(float).copy()
    for step in range(1, max_steps + 1):
        h = W @ s
        new = np.where(h > 0, 1.0, np.where(h < 0, -1.0, s))
        if np.array_equal(new, s):
            return s, step, "fixed"
        if prev is not None and np.array_equal(new, prev):
            return new, step, "cycle2"
        prev, s = s, new
    return s, max_steps, "none"
