"""pass^k with the unbiased estimator from the tau-bench paper: mean over tasks of C(c,k)/C(n,k)."""

from __future__ import annotations

from math import comb

import pytest

from switchboard.sim import pass_hat_k


def test_one_task_matches_the_combinatorial_formula() -> None:
    trials = [True, True, False, True, False, True, True, False]  # n=8, c=5
    for k in range(1, 9):
        assert pass_hat_k({"t": trials}, k) == pytest.approx(comb(5, k) / comb(8, k))


def test_pass_hat_one_is_the_mean_success_rate_across_tasks() -> None:
    tasks = {"a": [True, False, False, False], "b": [True, True, True, True]}
    assert pass_hat_k(tasks, 1) == pytest.approx((0.25 + 1.0) / 2)


def test_tasks_are_averaged_with_equal_weight() -> None:
    tasks = {"a": [True, True, False, False], "b": [True] * 4, "c": [False] * 4}
    expected = (comb(2, 2) / comb(4, 2) + 1.0 + 0.0) / 3
    assert pass_hat_k(tasks, 2) == pytest.approx(expected)


def test_needs_enough_trials_and_a_positive_k() -> None:
    with pytest.raises(ValueError, match="at least 3 trials"):
        pass_hat_k({"a": [True, True]}, 3)
    with pytest.raises(ValueError, match="k must be"):
        pass_hat_k({"a": [True]}, 0)
    with pytest.raises(ValueError, match="no tasks"):
        pass_hat_k({}, 1)


def test_pass_hat_k_never_increases_with_k() -> None:
    tasks = {
        "a": [True, False, True, True, True, False],
        "b": [True, True, True, False, True, True],
    }
    values = [pass_hat_k(tasks, k) for k in range(1, 7)]
    assert values == sorted(values, reverse=True)
