"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_max, daily_mean, daily_min


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""

    test_input = np.array([[0, 0], [0, 0], [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""

    test_input = np.array([[1, 2], [3, 4], [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


@pytest.mark.parametrize(
    ("test", "expected"),
    [
        [[[4, 2, 5], [1, 6, 2], [4, 1, 9]], [4, 6, 9]],
        [[[-4, -2], [-1, 6], [4, -9]], [4, 6]],
        [[[0], [0], [0]], [0]],
    ],
    ids=["positive ints", "negative ints", "zeros"],
)
def test_daily_max(test, expected):
    """Test that max function works for an array of positive integers."""
    test_input = np.array(test)

    npt.assert_array_equal(daily_max(test_input), expected)


def test_daily_min():
    """Test that min function works for an array of positive and negative integers."""
    test_input = np.array([[4, -2, 5], [1, -6, 2], [-4, -1, 9]])
    test_result = np.array([-4, -6, 2])

    npt.assert_array_equal(daily_min(test_input), test_result)


@pytest.mark.parametrize(
    "test",
    [
        [["Hello", "There"], ["General", "Kenobi"]],
        [["Hello", None], [1.2, []]],
    ],
)
def test_daily_min_error_string(test):
    with pytest.raises(Exception):
        daily_min(test)
