import numpy as np
from numpy import typing as npt  # noqa: F401
import pandas as pd
from typing_extensions import assert_type

from tests import check

left = pd.Series([1, 2, 3])  # left operand


def test_mul_py_scalar() -> None:
    """Test pd.Series[int] * Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j

    check(assert_type(left * b, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left * i, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left * f, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left * c, "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(assert_type(b * left, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(i * left, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(f * left, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(c * left, "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(assert_type(left.mul(b), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.mul(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.mul(f), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.mul(c), "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(assert_type(left.rmul(b), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rmul(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rmul(f), "pd.Series[float]"), pd.Series, np.floating)
    check(
        assert_type(left.rmul(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )


def test_mul_py_sequence() -> None:
    """Test pd.Series[int] * Python native sequence"""
    b, i, f, c = [True, False, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]

    check(assert_type(left * b, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left * i, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left * f, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left * c, "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(assert_type(b * left, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(i * left, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(f * left, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(c * left, "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(assert_type(left.mul(b), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.mul(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.mul(f), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.mul(c), "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(assert_type(left.rmul(b), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rmul(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rmul(f), "pd.Series[float]"), pd.Series, np.floating)
    check(
        assert_type(left.rmul(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )


def test_mul_numpy_array() -> None:
    """Test pd.Series[int] * numpy array"""
    b = np.array([True, False, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)

    check(assert_type(left * b, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left * i, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left * f, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left * c, "pd.Series[complex]"), pd.Series, np.complexfloating)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rmul__` cannot override. At runtime, they return
    # `Series`s with the correct element type.
    check(assert_type(b * left, "npt.NDArray[np.bool_]"), pd.Series, np.integer)
    check(assert_type(i * left, "npt.NDArray[np.int64]"), pd.Series, np.integer)
    check(assert_type(f * left, "npt.NDArray[np.float64]"), pd.Series, np.floating)
    check(
        assert_type(c * left, "npt.NDArray[np.complex128]"),
        pd.Series,
        np.complexfloating,
    )

    check(assert_type(left.mul(b), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.mul(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.mul(f), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.mul(c), "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(assert_type(left.rmul(b), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rmul(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rmul(f), "pd.Series[float]"), pd.Series, np.floating)
    check(
        assert_type(left.rmul(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )


def test_mul_pd_series() -> None:
    """Test pd.Series[int] * pandas series"""
    b = pd.Series([True, False, True])
    i = pd.Series([2, 3, 5])
    f = pd.Series([1.0, 2.0, 3.0])
    c = pd.Series([1.1j, 2.2j, 4.1j])

    check(assert_type(left * b, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left * i, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left * f, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left * c, "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(assert_type(b * left, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(i * left, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(f * left, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(c * left, "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(assert_type(left.mul(b), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.mul(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.mul(f), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.mul(c), "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(assert_type(left.rmul(b), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rmul(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rmul(f), "pd.Series[float]"), pd.Series, np.floating)
    check(
        assert_type(left.rmul(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
