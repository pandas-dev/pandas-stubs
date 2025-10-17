import numpy as np
from numpy import typing as npt  # noqa: F401
import pandas as pd
import pytest
from typing_extensions import assert_type

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)


@pytest.fixture
def left() -> "pd.Index[int]":
    """Left operand"""
    lo = pd.Index([1, 2, 3])
    return check(assert_type(lo, "pd.Index[int]"), pd.Index, np.integer)


def test_truediv_py_scalar(left: "pd.Index[int]") -> None:
    """Test pd.Index[int] / Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j

    check(assert_type(left / b, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left / i, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left / f, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left / c, "pd.Index[complex]"), pd.Index, np.complexfloating)

    check(assert_type(b / left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(i / left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(f / left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(c / left, "pd.Index[complex]"), pd.Index, np.complexfloating)


def test_truediv_py_sequence(left: "pd.Index[int]") -> None:
    """Test pd.Index[int] / Python native sequences"""
    b, i, f, c = [True, False, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]

    check(assert_type(left / b, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left / i, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left / f, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left / c, "pd.Index[complex]"), pd.Index, np.complexfloating)

    check(assert_type(b / left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(i / left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(f / left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(c / left, "pd.Index[complex]"), pd.Index, np.complexfloating)


def test_truediv_numpy_array(left: "pd.Index[int]") -> None:
    """Test pd.Index[int] / numpy arrays"""
    b = np.array([True, False, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)

    check(assert_type(left / b, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left / i, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left / f, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left / c, "pd.Index[complex]"), pd.Index, np.complexfloating)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rtruediv__` cannot override. At runtime, they return
    # `Series` with the correct element type.
    check(assert_type(b / left, "npt.NDArray[np.float64]"), pd.Index, np.floating)
    check(assert_type(i / left, "npt.NDArray[np.float64]"), pd.Index, np.floating)
    check(assert_type(f / left, "npt.NDArray[np.float64]"), pd.Index, np.floating)
    check(
        assert_type(c / left, "npt.NDArray[np.complex128]"),
        pd.Index,
        np.complexfloating,
    )


def test_truediv_pd_scalar(left: "pd.Index[int]") -> None:
    """Test pd.Index[int] / pandas scalars"""
    s, d = pd.Timestamp(2025, 9, 24), pd.Timedelta(seconds=1)

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left / s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _01 = left / d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = s / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d / left, "pd.Index[pd.Timedelta]"), pd.Index, pd.Timedelta)


def test_truediv_pd_index(left: "pd.Index[int]") -> None:
    """Test pd.Index[int] / pandas Indexes"""
    b = pd.Index([True, False, True])
    i = pd.Index([2, 3, 5])
    f = pd.Index([1.0, 2.0, 3.0])
    c = pd.Index([1.1j, 2.2j, 4.1j])

    check(assert_type(left / b, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left / i, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left / f, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left / c, "pd.Index[complex]"), pd.Index, np.complexfloating)

    check(assert_type(b / left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(i / left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(f / left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(c / left, "pd.Index[complex]"), pd.Index, np.complexfloating)
