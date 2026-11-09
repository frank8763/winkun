"""Deterministic estimator contracts: zero models, native steps and optimizers."""

import math
import unittest

import numpy as np

from physical_gae import physical_gae


class PhysicalGAEContracts(unittest.TestCase):
    def evaluate(self, r, v, vn, n, b, c, lam=1., gamma=.9):
        return physical_gae(
            np.asarray(r, dtype=float), np.asarray(v, dtype=float),
            np.asarray(vn, dtype=float), np.asarray(n, dtype=np.int64),
            np.asarray(b, dtype=bool), np.asarray(c, dtype=bool),
            gamma_native=gamma, lambda_native=lam,
        )

    def test_lambda_one_matches_independent_mc_with_variable_holds(self):
        # The independent oracle constructs each full discounted return forwards;
        # it does not repeat the estimator's backward delta recursion.
        rng = np.random.default_rng(20261109)
        r, v = rng.normal(size=(2, 13, 3))
        n = rng.integers(1, 101, size=r.shape)
        vn = np.vstack((v[1:], np.zeros((1, 3))))
        bootstrap = np.ones(r.shape, dtype=bool)
        bootstrap[-1] = False
        trace = bootstrap.copy()
        gamma = math.exp(-.002 / 60)
        _, targets = physical_gae(r, v, vn, n, bootstrap, trace,
                                  gamma_native=gamma, lambda_native=1.)
        expected = np.zeros_like(r)
        for first in range(len(r)):
            factor = np.ones(3)
            for k in range(first, len(r)):
                expected[first] += factor * r[k]
                factor *= gamma ** n[k]
        np.testing.assert_allclose(targets, expected, atol=2e-14, rtol=2e-14)

    def test_lambda_zero_is_one_step_td(self):
        a, target = self.evaluate([[1.], [3.]], [[2.], [4.]],
                                  [[4.], [9.]], [[2], [1]],
                                  [[True], [False]], [[True], [False]], lam=0.)
        np.testing.assert_allclose(a[:, 0], [1 + .9**2 * 4 - 2, -1])
        np.testing.assert_allclose(target[:, 0], [1 + .9**2 * 4, 3])

    def test_true_task_deadline_does_not_bootstrap(self):
        _, target = self.evaluate([[2.]], [[3.]], [[999.]], [[100]],
                                  [[False]], [[False]])
        np.testing.assert_array_equal(target, [[2.]])

    def test_rollout_cut_bootstraps_actual_final_observation(self):
        _, target = self.evaluate([[2.]], [[3.]], [[7.]], [[2]],
                                  [[True]], [[False]])
        np.testing.assert_allclose(target, [[2 + .9**2 * 7]])

    def test_reset_episode_cannot_leak_into_previous_episode(self):
        # Row 0 is a rollout cut; row 1 belongs to a reset episode. It must not
        # enter row 0's advantage even when its reward changes dramatically.
        first = self.evaluate([[2.], [10.]], [[1.], [3.]], [[7.], [0.]],
                              [[2], [1]], [[True], [False]], [[False], [False]])
        second = self.evaluate([[2.], [1e9]], [[1.], [3.]], [[7.], [0.]],
                               [[2], [1]], [[True], [False]], [[False], [False]])
        np.testing.assert_array_equal(first[0][0], second[0][0])

    def test_environment_columns_are_independent(self):
        args = ([[1., 5.], [3., -7.]], [[2., 9.], [4., 8.]],
                [[4., 8.], [0., 0.]], [[2, 4], [1, 3]],
                [[True, True], [False, False]], [[True, False], [False, False]])
        joint = self.evaluate(*args)
        for col in range(2):
            single = self.evaluate(*[np.asarray(a)[:, col:col+1] for a in args])
            for actual, expected in zip(joint, single):
                np.testing.assert_array_equal(actual[:, col:col+1], expected)

    def test_inputs_are_not_modified(self):
        args = [np.array([[1.], [2.]])] * 3 + [np.array([[1], [2]]),
                np.array([[True], [False]]), np.array([[True], [False]])]
        copies = [a.copy() for a in args]
        physical_gae(*args, gamma_native=.9, lambda_native=.8)
        for actual, expected in zip(args, copies):
            np.testing.assert_array_equal(actual, expected)

    def test_physical_decay_depends_on_elapsed_native_time(self):
        gamma, lam = math.exp(-.002 / 60), math.exp(-.002 / 4)
        a, _ = self.evaluate([[0.], [1.]], [[0.], [0.]], [[0.], [0.]],
                              [[100], [1]], [[True], [False]],
                              [[True], [False]], gamma=gamma, lam=lam)
        self.assertAlmostEqual(a[0, 0], math.exp(-.2/60 - .2/4), places=14)

    def test_invalid_masks_nonfinite_and_shapes_are_rejected(self):
        good = [np.zeros((1, 1)) for _ in range(3)] + [
            np.ones((1, 1), dtype=np.int64), np.zeros((1, 1), dtype=bool),
            np.zeros((1, 1), dtype=bool)]
        mutations = [(0, np.array([[np.nan]])), (1, np.zeros((2, 1))),
                     (3, np.array([[0]])), (3, np.array([[1.5]])),
                     (4, np.array([[0]])), (5, np.array([[True]]))]
        for index, value in mutations:
            args = list(good)
            args[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                physical_gae(*args, gamma_native=.9, lambda_native=.8)
        for gamma, lam in [(0., .8), (1.1, .8), (.9, -.1), (.9, float('nan'))]:
            with self.subTest(gamma=gamma, lam=lam), self.assertRaises(ValueError):
                physical_gae(*good, gamma_native=gamma, lambda_native=lam)


if __name__ == '__main__':
    unittest.main(verbosity=2)
