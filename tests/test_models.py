"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_max, daily_mean, daily_min, patient_normalise


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
        [np.array([[1, 2], [3, 4], [5, 6]]), np.array([3, 4])],
        [np.array([[0, 0], [0, 0], [0, 0]]), np.array([0, 0])],
    ],
    ids=["positive integers", "zeros"],
)
def test_daily_mean(test, expected):
    npt.assert_array_equal(daily_mean(test), expected)


@pytest.mark.parametrize(
    ("test", "expected"),
    [
        [np.array([[1, 2], [3, 4], [5, 6]]), np.array([5, 6])],
        [np.array([[0, 0], [0, 0], [0, 0]]), np.array([0, 0])],
    ],
    ids=["positive integers", "zeros"],
)
def test_daily_max(test, expected):
    npt.assert_array_equal(daily_max(test), expected)


@pytest.mark.parametrize(
    ("test", "expected"),
    [
        [np.array([[1, 2], [3, 4], [5, 6]]), np.array([1, 2])],
        [np.array([[0, 0], [0, 0], [0, 0]]), np.array([0, 0])],
        [np.array([[+4, -2, +5], [+1, -6, +2], [-4, -1, +9]]), np.array([-4, -6, 2])],
    ],
    ids=["positive integers", "zeros", "mixed integers"],
)
def test_daily_min(test, expected):
    npt.assert_array_equal(daily_min(test), expected)


def test_daily_string_raises_error():
    with pytest.raises(TypeError):
        error_expected = daily_min([["Hello", "there"], ["General", "Kenobi"]])


@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            None,
        ),
        (
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            None,
        ),
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            None,
        ),
        (
            [[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
            None,
            ValueError("inflammation values should be non-negative"),
        ),
        (
            [4, 5, 6],
            None,
            ValueError("inflammation array should be 2-dimensional"),
        ),
        (
            "hello",
            None,
            TypeError("data input should be ndarray"),
        ),
        (
            3,
            None,
            TypeError("data input should be ndarray"),
        ),
    ],
)
def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers.
    Test with a relative and absolute tolerance of 0.01."""
    if isinstance(test, list):
        test = np.array(test)
    if expect_raises is not None:
        with pytest.raises(type(expect_raises), match=str(expect_raises)):
            patient_normalise(test)

    else:
        result = patient_normalise(test)
        npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)
