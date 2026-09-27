"""Dense associative memories (Krotov & Hopfield 2016; Demircigil et al. 2017).

Energy E(s) = -sum_mu F(m_mu), where m_mu = xi_mu . s is the unnormalized
overlap with pattern mu. Two families of F:

  * 'poly', F(x) = x^n. n = 2 is exactly the classical Hebbian network.
  * 'exp',  F(x) = exp(x). For numerical range the energy is reported as
    -logsumexp(m), a monotone transform that leaves every update decision
    unchanged.

Asynchronous update of bit i: set s_i to whichever sign gives lower energy
with every other bit held fixed:

    s_i <- sign( sum_mu F(h_mu + xi_mu,i) - F(h_mu - xi_mu,i) ),
    h_mu = m_mu - xi_mu,i s_i   (the overlap with bit i removed).

This is a coordinate descent step, so E never increases, for any F.

The model is stored in its bipartite ("hidden unit") form: the P x N pattern
matrix. The equivalent n-body synaptic tensor would need N^n entries.
"""

import numpy as np
from scipy.special import comb, logsumexp


class DenseAM:
    def __init__(self, X: np.ndarray, kind: str = "poly", n: int = 3):
        if kind not in ("poly", "exp"):
            raise ValueError("kind must be 'poly' or 'exp'")
        if kind == "poly" and n < 2:
            raise ValueError("polynomial degree must be >= 2")
        self.X = np.asarray(X, dtype=float)
        self.P, self.N = self.X.shape
        self.kind = kind
        self.n = n
        # (h+x)^n - (h-x)^n = 2 x sum_{k odd} C(n,k) h^(n-k) for x = +/-1.
        self._odd = [(float(comb(n, k, exact=True)), n - k) for k in range(1, n + 1, 2)]

    def F(self, x: np.ndarray) -> np.ndarray:
        return x ** self.n if self.kind == "poly" else np.exp(x)

    def energy(self, s: np.ndarray) -> float:
        m = self.X @ s
        if self.kind == "poly":
            return float(-np.sum(m ** self.n))
        return float(-logsumexp(m))

    def _drive(self, h: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Preference for s_i = +1 over -1, reduced over the pattern axis (last).

        h: (..., P) overlaps with bit i removed; x: (P,) column i of X.
        Positive means +1 has lower energy. For 'exp' this is a log-ratio,
        which has the same sign.
        """
        if self.kind == "poly":
            g = np.zeros_like(h)
            for c, p in self._odd:
                g += c * h ** p
            return 2.0 * np.sum(x * g, axis=-1)
        return logsumexp(h + x, axis=-1) - logsumexp(h - x, axis=-1)

    def drive_naive(self, s: np.ndarray, i: int) -> float:
        """Reference implementation: evaluate F at both candidate overlaps directly."""
        sp, sm = s.copy(), s.copy()
        sp[i], sm[i] = 1.0, -1.0
        if self.kind == "poly":
            return float(np.sum(self.F(self.X @ sp)) - np.sum(self.F(self.X @ sm)))
        return float(logsumexp(self.X @ sp) - logsumexp(self.X @ sm))

    def stable_bits(self, S: np.ndarray) -> np.ndarray:
        """Boolean K x N mask: would bit i of state k stay put under an update?

        Costs O(K N P) with O(K P) memory. A zero drive counts as unstable.
        """
        S = np.atleast_2d(S).astype(float)
        M = S @ self.X.T  # K x P
        out = np.empty(S.shape, dtype=bool)
        for i in range(self.N):
            x = self.X[:, i]
            h = M - S[:, i:i + 1] * x[None, :]
            out[:, i] = self._drive(h, x) * S[:, i] > 0
        return out

    def run_async(self, s0: np.ndarray, rng: np.random.Generator,
                  max_sweeps: int = 100) -> tuple[np.ndarray, int, bool]:
        """Random-order asynchronous descent, keeping the overlaps m = X s up to date."""
        s = s0.astype(float).copy()
        m = self.X @ s
        for sweep in range(1, max_sweeps + 1):
            changed = False
            for i in rng.permutation(self.N):
                x = self.X[:, i]
                d = float(self._drive(m - x * s[i], x))
                if d == 0.0:
                    continue
                new = 1.0 if d > 0.0 else -1.0
                if new != s[i]:
                    m += x * (new - s[i])
                    s[i] = new
                    changed = True
            if not changed:
                return s, sweep, True
        return s, max_sweeps, False
