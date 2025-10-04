from datetime import timedelta

import numpy as np
from numpy import typing as npt  # noqa: F401
import pandas as pd
import pytest
from typing_extensions import (
    Never,
    assert_type,
)

from tests import (
    PD_LTE_23,
    TYPE_CHECKING_INVALID_USAGE,
    check,
)


@pytest.fixture
def left() -> "pd.Index[pd.Timedelta]":
    """left operand"""
    # pandas-dev/pandas#62524
    lo = pd.Index([1]) * [timedelta(seconds=1)]  # left operand
    return check(assert_type(lo, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)


def test_mul_py_scalar(left: "pd.Index[pd.Timedelta]") -> None:
    """Test pd.Index[pd.Timedelta] * Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j

    # pandas-dev/pandas#62316
    if PD_LTE_23:
        check(assert_type(left * b, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    check(assert_type(left * i, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    check(assert_type(left * f, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left * c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    if PD_LTE_23:
        check(assert_type(b * left, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    check(assert_type(i * left, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    check(assert_type(f * left, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _1 = c * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]


def test_mul_py_sequence(left: "pd.Index[pd.Timedelta]") -> None:
    """Test pd.Index[pd.Timedelta] * Python native sequences"""
    b, i, f, c = [True], [2], [1.5], [1.7j]

    # pandas-dev/pandas#62316
    if PD_LTE_23:
        check(assert_type(left * b, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    check(assert_type(left * i, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    check(assert_type(left * f, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left * c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    if PD_LTE_23:
        check(assert_type(b * left, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    check(assert_type(i * left, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    check(assert_type(f * left, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _1 = c * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]


def test_mul_numpy_array(left: "pd.Index[pd.Timedelta]") -> None:
    """Test pd.Index[pd.Timedelta] * numpy arrays"""
    b = np.array([True], np.bool_)
    i = np.array([2], np.int64)
    f = np.array([1.5], np.float64)
    c = np.array([1.7j], np.complex128)

    # pandas-dev/pandas#62316
    if PD_LTE_23:
        check(assert_type(left * b, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    check(assert_type(left * i, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    check(assert_type(left * f, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left * c, Never)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rmul__` cannot override. At runtime, they return
    # `Series` with the correct element type.
    if PD_LTE_23:
        check(assert_type(b * left, "npt.NDArray[np.bool_]"), pd.Index, timedelta)
    check(assert_type(i * left, "npt.NDArray[np.int64]"), pd.Index, timedelta)
    check(assert_type(f * left, "npt.NDArray[np.float64]"), pd.Index, timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        # We made it Never, but numpy takes over
        assert_type(c * left, "npt.NDArray[np.complex128]")


def test_mul_pd_index(left: "pd.Index[pd.Timedelta]") -> None:
    """Test pd.Index[pd.Timedelta] * pandas Indexes"""
    b = pd.Index([True])
    i = pd.Index([2])
    f = pd.Index([1.5])
    c = pd.Index([1.7j])

    # pandas-dev/pandas#62316
    if PD_LTE_23:
        check(assert_type(left * b, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    check(assert_type(left * i, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    check(assert_type(left * f, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left * c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    if PD_LTE_23:
        check(assert_type(b * left, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    check(assert_type(i * left, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    check(assert_type(f * left, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _1 = c * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
