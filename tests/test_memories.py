"""Correctness tests: equivalences, Lyapunov functions, exact fixed points, theory."""

import itertools
import math
import unittest

import numpy as np
from scipy.special import softmax

from dam import classical, modern, theory
from dam.dense import DenseAM
from dam.patterns import corrupt, overlap, random_patterns


class TestPatterns(unittest.TestCase):
    def test_corrupt_flips_exact_count(self):
        rng = np.random.default_rng(0)
        X = random_patterns(rng, 20, 50)
        Y = corrupt(rng, X, 0.2)
        np.testing.assert_array_equal((X != Y).sum(axis=1), np.full(20, 10))

    def test_bias_sets_mean(self):
        rng = np.random.default_rng(1)
        X = random_patterns(rng, 400, 400, bias=0.4)
        self.assertAlmostEqual(X.mean(), 0.4, delta=0.01)


class TestClassical(unittest.TestCase):
    def test_hebbian_equals_dense_quadratic(self):
        """F(x)=x^2 in pattern form is the Hebbian network: same decisions, same trajectory."""
        rng = np.random.default_rng(2)
        # Odd P with even N makes every field an odd integer / N, so no zero-field
        # ties, where float rounding in W could otherwise break the tie differently.
        X = random_patterns(rng, 31, 100)
        W, am = classical.hebbian(X), DenseAM(X, "poly", 2)
        S = corrupt(rng, X[:10], 0.3)
        np.testing.assert_array_equal(classical.stable_bits(W, S), am.stable_bits(S))
        for s0 in S:
            a, _, _ = classical.run_async(W, s0, np.random.default_rng(5))
            b, _, _ = am.run_async(s0, np.random.default_rng(5))
            np.testing.assert_array_equal(a, b)

    def test_async_energy_monotone_all_rules(self):
        rng = np.random.default_rng(3)
        X = random_patterns(rng, 25, 60, bias=0.3)
        for W in (classical.hebbian(X), classical.storkey(X),
                  classical.pseudo_inverse(X), classical.pseudo_inverse(X, True)):
            np.testing.assert_allclose(W, W.T, atol=1e-12)
            s = corrupt(rng, X[0], 0.4)[0]
            h = W @ s
            e = classical.energy(W, s)
            for i in rng.permutation(60).tolist() * 3:
                if h[i] != 0 and np.sign(h[i]) != s[i]:
                    h += W[i] * (-2 * s[i])
                    s[i] = -s[i]
                    e_new = classical.energy(W, s)
                    self.assertLessEqual(e_new, e + 1e-9)
                    e = e_new
            _, _, converged = classical.run_async(W, s, rng)
            self.assertTrue(converged)

    def test_pseudo_inverse_stores_every_pattern(self):
        rng = np.random.default_rng(4)
        X = random_patterns(rng, 90, 100, bias=0.5)  # heavily correlated, P close to N
        for zero_diag in (False, True):
            W = classical.pseudo_inverse(X, zero_diag)
            self.assertTrue(classical.stable_bits(W, X).all())

    def test_storkey_beats_hebbian_at_moderate_load(self):
        rng = np.random.default_rng(5)
        X = random_patterns(rng, 30, 100)
        heb = classical.stable_bits(classical.hebbian(X), X).all(axis=1).mean()
        sto = classical.stable_bits(classical.storkey(X), X).all(axis=1).mean()
        self.assertGreater(sto, heb)

    def test_sync_two_cycle_exists(self):
        """Parallel updates are not Lyapunov: an anti-ferromagnetic pair oscillates."""
        W = np.array([[0.0, 1.0], [1.0, 0.0]])
        _, _, kind = classical.run_sync(W, np.array([1.0, -1.0]))
        self.assertEqual(kind, "cycle2")
        _, _, _ = classical.run_async(W, np.array([1.0, -1.0]), np.random.default_rng(0))


class TestDense(unittest.TestCase):
    def test_incremental_drive_matches_naive(self):
        rng = np.random.default_rng(6)
        X = random_patterns(rng, 40, 20)
        S = random_patterns(rng, 5, 20)
        for kind, n in (("poly", 3), ("poly", 4), ("poly", 5), ("exp", 0)):
            am = DenseAM(X, kind, n)
            M = S @ X.T
            for s, m in zip(S, M):
                for i in range(20):
                    x = X[:, i]
                    fast = float(am._drive(m - x * s[i], x))
                    self.assertAlmostEqual(fast, am.drive_naive(s, i),
                                           delta=1e-9 * max(1.0, abs(fast)))

    def test_async_energy_monotone(self):
        rng = np.random.default_rng(7)
        X = random_patterns(rng, 200, 30)
        for kind, n in (("poly", 3), ("poly", 4), ("exp", 0)):
            am = DenseAM(X, kind, n)
            s = random_patterns(rng, 1, 30)[0]
            m = X @ s
            e = am.energy(s)
            for i in rng.permutation(30).tolist() * 4:
                x = X[:, i]
                d = float(am._drive(m - x * s[i], x))
                if d != 0 and np.sign(d) != s[i]:
                    m += x * (-2 * s[i])
                    s[i] = -s[i]
                    e_new = am.energy(s)
                    self.assertLess(e_new, e + 1e-9 * abs(e))
                    e = e_new

    def test_exponential_stores_far_more_than_n(self):
        rng = np.random.default_rng(8)
        N, P = 16, 1000  # P = 62 N; a pairwise network holds ~0.14 N = 2
        X = random_patterns(rng, P, N)
        frac = DenseAM(X, "exp").stable_bits(X[:200]).all(axis=1).mean()
        self.assertGreater(frac, 0.9)

    def test_cubic_retrieves_from_corruption_above_pairwise_limit(self):
        rng = np.random.default_rng(9)
        N, P = 100, 60  # load 0.6: far above alpha_c for the Hebbian net
        X = random_patterns(rng, P, N)
        am = DenseAM(X, "poly", 3)
        s, _, conv = am.run_async(corrupt(rng, X[0], 0.2)[0], rng)
        self.assertTrue(conv)
        self.assertEqual(overlap(s, X[0]), 1.0)


class TestTheory(unittest.TestCase):
    def test_crosstalk_variance_by_enumeration(self):
        """V = E[g(h)^2] over all 2^(N-1) overlap sign patterns, for small N."""
        for N, n in ((7, 2), (7, 3), (9, 4)):
            logS, logV = theory.crosstalk_moments(N, "poly", n)
            vals = [(sum(c) + 1.0) ** n - (sum(c) - 1.0) ** n
                    for c in itertools.product((-1.0, 1.0), repeat=N - 1)]
            V = np.mean(np.square(vals)) / float(N) ** (2 * (n - 1))
            S = (N ** n - (N - 2) ** n) / float(N) ** (n - 1)
            self.assertAlmostEqual(math.exp(logV), V, places=9)
            self.assertAlmostEqual(math.exp(logS), S, places=9)

    def test_quadratic_matches_textbook(self):
        N, P = 500, 60
        self.assertAlmostEqual(theory.bit_error(N, P, "poly", 2),
                               float(0.5 * math.erfc(math.sqrt((N - 1) / (P - 1)) / math.sqrt(2))),
                               places=12)

    def test_bit_error_matches_simulation(self):
        rng = np.random.default_rng(10)
        for n, N, P in ((2, 300, 60), (3, 60, 400)):
            X = random_patterns(rng, P, N)
            sim = 1.0 - DenseAM(X, "poly", n).stable_bits(X[:150]).mean()
            th = theory.bit_error(N, P, "poly", n)
            se = math.sqrt(th * (1 - th) / (150 * N))
            self.assertLess(abs(sim - th), 5 * se + 0.1 * th)

    def test_ags_constants(self):
        self.assertAlmostEqual(theory.ags_alpha_c(), 0.1379, delta=5e-4)
        # m(alpha) has a square-root singularity at alpha_c; probe just below it.
        self.assertAlmostEqual(theory.ags_overlap(theory.ags_alpha_c() - 1e-7), 0.967, delta=2e-3)
        self.assertEqual(theory.ags_overlap(0.15), 0.0)
        self.assertGreater(theory.ags_overlap(0.05), 0.9999)


class TestModern(unittest.TestCase):
    def test_update_is_softmax_attention(self):
        rng = np.random.default_rng(11)
        X, q, beta = rng.normal(size=(50, 8)), rng.normal(size=8), 0.7
        expected = softmax(beta * X @ q) @ X
        np.testing.assert_allclose(modern.update(X, q, beta), expected, atol=1e-12)
        Q = rng.normal(size=(4, 8))
        np.testing.assert_allclose(modern.update(X, Q, beta)[2], modern.update(X, Q[2], beta))

    def test_energy_nonincreasing(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(100, 16))
        for beta in (0.1, 1.0, 8.0):
            xi = rng.normal(size=16)
            e = modern.energy(X, xi, beta)
            for _ in range(30):
                xi = modern.update(X, xi, beta)
                e_new = modern.energy(X, xi, beta)
                self.assertLessEqual(e_new, e + 1e-10)
                e = e_new

    def test_retrieval_bound_holds(self):
        rng = np.random.default_rng(13)
        for _ in range(300):
            d, P = int(rng.integers(8, 40)), int(rng.integers(2, 300))
            X = random_patterns(rng, P, d)
            i = int(rng.integers(P))
            q = corrupt(rng, X[i], float(rng.uniform(0, 0.1)))[0]
            beta = float(rng.uniform(0.05, 3.0))
            eps, bound = modern.retrieval_bound(X, i, q, beta)
            p_i = softmax(beta * X @ q)[i]
            self.assertLessEqual(1.0 - p_i, eps + 1e-12)
            self.assertLessEqual(np.linalg.norm(modern.update(X, q, beta) - X[i]), bound + 1e-9)

    def test_low_beta_averages_high_beta_retrieves(self):
        rng = np.random.default_rng(14)
        X = random_patterns(rng, 64, 64)
        out_lo, _ = modern.retrieve(X, X[3], 1e-3)
        out_hi, _ = modern.retrieve(X, corrupt(rng, X[3], 0.2)[0], 1.0)
        np.testing.assert_allclose(out_lo, X.mean(axis=0), atol=0.05)  # metastable mean state
        np.testing.assert_allclose(out_hi, X[3], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
