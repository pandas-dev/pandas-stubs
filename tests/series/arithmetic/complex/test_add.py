import numpy as np
from numpy import typing as npt  # noqa: F401
import pandas as pd
from typing_extensions import assert_type

from tests import check

left = pd.Series([1j, 2j, 3j])  # left operand


def test_add_py_scalar() -> None:
    """Test pd.Series[complex] + Python native scalars"""
    i, f, c = 1, 1.0, 1j

    check(assert_type(left + i, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left + f, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left + c, "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(assert_type(i + left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(f + left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(c + left, "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(assert_type(left.add(i), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.add(f), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.add(c), "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(
        assert_type(left.radd(i), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.radd(f), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.radd(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )


def test_add_py_sequence() -> None:
    """Test pd.Series[complex] + Python native sequence"""
    i, f, c = [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]

    check(assert_type(left + i, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left + f, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left + c, "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(assert_type(i + left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(f + left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(c + left, "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(assert_type(left.add(i), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.add(f), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.add(c), "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(
        assert_type(left.radd(i), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.radd(f), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.radd(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )


def test_add_numpy_array() -> None:
    """Test pd.Series[complex] + numpy array"""
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)

    check(assert_type(left + i, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left + f, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left + c, "pd.Series[complex]"), pd.Series, np.complexfloating)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__radd__` cannot override. At runtime, they return
    # `Series`s with the correct element type.
    check(assert_type(i + left, "npt.NDArray[np.int64]"), pd.Series, np.complexfloating)
    check(
        assert_type(f + left, "npt.NDArray[np.float64]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(c + left, "npt.NDArray[np.complex128]"),
        pd.Series,
        np.complexfloating,
    )

    check(assert_type(left.add(i), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.add(f), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.add(c), "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(
        assert_type(left.radd(i), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.radd(f), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.radd(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )


def test_add_pd_series() -> None:
    """Test pd.Series[complex] + pandas series"""
    i = pd.Series([2, 3, 5])
    f = pd.Series([1.0, 2.0, 3.0])
    c = pd.Series([1.1j, 2.2j, 4.1j])

    check(assert_type(left + i, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left + f, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left + c, "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(assert_type(i + left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(f + left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(c + left, "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(assert_type(left.add(i), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.add(f), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.add(c), "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(
        assert_type(left.radd(i), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.radd(f), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.radd(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
