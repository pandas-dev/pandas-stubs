import numpy as np
from numpy import typing as npt  # noqa: F401
import pandas as pd
from typing_extensions import assert_type

from tests import check

left = pd.Series([1.0, 2.0, 3.0])  # left operand


def test_add_py_scalar() -> None:
    """Test pd.Series[float] + Python native scalars"""
    i, f, c = 1, 1.0, 1j

    check(assert_type(left + i, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left + f, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left + c, "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(i + left, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(f + left, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(c + left, "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.add(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.add(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.add(c), "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.radd(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.radd(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.radd(c), "pd.Series[complex]"), pd.Series, np.complex128)


def test_add_py_sequence() -> None:
    """Test pd.Series[float] + Python native sequence"""
    i, f, c = [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]

    check(assert_type(left + i, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left + f, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left + c, "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(i + left, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(f + left, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(c + left, "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.add(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.add(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.add(c), "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.radd(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.radd(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.radd(c), "pd.Series[complex]"), pd.Series, np.complex128)


def test_add_numpy_array() -> None:
    """Test pd.Series[float] + numpy array"""
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)

    check(assert_type(left + i, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left + f, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left + c, "pd.Series[complex]"), pd.Series, np.complex128)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__radd__` cannot override. At runtime, they return
    # `Series`s with the correct element type.
    check(assert_type(i + left, "npt.NDArray[np.int64]"), pd.Series, np.float64)
    check(assert_type(f + left, "npt.NDArray[np.float64]"), pd.Series, np.float64)
    check(assert_type(c + left, "npt.NDArray[np.complex128]"), pd.Series, np.complex128)

    check(assert_type(left.add(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.add(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.add(c), "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.radd(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.radd(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.radd(c), "pd.Series[complex]"), pd.Series, np.complex128)


def test_add_pd_series() -> None:
    """Test pd.Series[float] + pandas series"""
    i = pd.Series([2, 3, 5])
    f = pd.Series([1.0, 2.0, 3.0])
    c = pd.Series([1.1j, 2.2j, 4.1j])

    check(assert_type(left + i, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left + f, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left + c, "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(i + left, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(f + left, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(c + left, "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.add(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.add(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.add(c), "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.radd(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.radd(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.radd(c), "pd.Series[complex]"), pd.Series, np.complex128)
