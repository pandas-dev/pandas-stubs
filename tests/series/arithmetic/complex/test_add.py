import numpy as np
import pandas as pd
from typing_extensions import assert_type

from tests import check

left = pd.Series([1j, 2j, 3j])  # left operand


def test_add_py_scalar() -> None:
    """Test pd.Series[complex] + Python native scalars"""
    i, f, c = 1, 1.0, 1j

    check(assert_type(left + i, "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left + f, "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left + c, "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(i + left, "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(f + left, "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(c + left, "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.add(i), "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left.add(f), "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left.add(c), "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.radd(i), "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left.radd(f), "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left.radd(c), "pd.Series[complex]"), pd.Series, np.complex128)


def test_add_py_sequence() -> None:
    """Test pd.Series[complex] + Python native sequence"""
    i, f, c = [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]

    check(assert_type(left + i, "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left + f, "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left + c, "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(i + left, "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(f + left, "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(c + left, "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.add(i), "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left.add(f), "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left.add(c), "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.radd(i), "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left.radd(f), "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left.radd(c), "pd.Series[complex]"), pd.Series, np.complex128)


def test_add_numpy_array() -> None:
    """Test pd.Series[complex] + numpy array"""
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)

    check(assert_type(left + i, "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left + f, "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left + c, "pd.Series[complex]"), pd.Series, np.complex128)

    # check(assert_type(i + l, "pd.Series[complex]"), pd.Series, np.complex128)
    # check(assert_type(f + l, "pd.Series[complex]"), pd.Series, np.complex128)
    # check(assert_type(c + l, "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.add(i), "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left.add(f), "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left.add(c), "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.radd(i), "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left.radd(f), "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left.radd(c), "pd.Series[complex]"), pd.Series, np.complex128)


def test_add_pd_series() -> None:
    """Test pd.Series[complex] + pandas series"""
    i = pd.Series([2, 3, 5])
    f = pd.Series([1.0, 2.0, 3.0])
    c = pd.Series([1.1j, 2.2j, 4.1j])

    check(assert_type(left + i, "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left + f, "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left + c, "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(i + left, "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(f + left, "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(c + left, "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.add(i), "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left.add(f), "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left.add(c), "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.radd(i), "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left.radd(f), "pd.Series[complex]"), pd.Series, np.complex128)
    check(assert_type(left.radd(c), "pd.Series[complex]"), pd.Series, np.complex128)
