from datetime import (
    datetime,
    timedelta,
)
from typing import Any

import numpy as np
from numpy import typing as npt  # noqa: F401
import pandas as pd
import pytest
from typing_extensions import (
    Never,
    assert_type,
)

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)
from tests._typing import np_ndarray_int64


@pytest.fixture
def left() -> "pd.Index[bool]":
    """Left operand"""
    lo = pd.Index([True, True, False])
    return check(assert_type(lo, "pd.Index[bool]"), pd.Index, np.bool_)


def test_mul_py_scalar(left: "pd.Index[bool]") -> None:
    """Test pd.Index[bool] * Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j
    s, d = datetime(2025, 10, 1), timedelta(seconds=1)

    check(assert_type(left * b, "pd.Index[bool]"), pd.Index, np.bool_)
    check(assert_type(left * i, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(left * f, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left * c, "pd.Index[complex]"), pd.Index, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        _04 = left * s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _05 = left * d  # type: ignore[type-var] # pyright: ignore[reportOperatorIssue]

    check(assert_type(b * left, "pd.Index[bool]"), pd.Index, np.bool_)
    check(assert_type(i * left, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(f * left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(c * left, "pd.Index[complex]"), pd.Index, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        _14 = s * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _15 = d * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]


def test_mul_py_sequence(left: "pd.Index[bool]") -> None:
    """Test pd.Index[bool] * Python native sequences"""
    b, i, f, c = [True, False, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]
    s = [datetime(2025, 10, d) for d in (1, 2, 3)]
    d = [timedelta(seconds=s + 1) for s in range(3)]

    check(assert_type(left * b, "pd.Index[bool]"), pd.Index, np.bool_)
    check(assert_type(left * i, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(left * f, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left * c, "pd.Index[complex]"), pd.Index, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        _04 = left * s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _05 = left * d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    check(assert_type(b * left, "pd.Index[bool]"), pd.Index, np.bool_)
    check(assert_type(i * left, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(f * left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(c * left, "pd.Index[complex]"), pd.Index, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        _14 = s * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _15 = d * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]


def test_mul_numpy_array(left: "pd.Index[bool]") -> None:
    """Test pd.Index[bool] * numpy arrays"""
    b = np.array([True, False, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)
    s = np.array([np.datetime64(f"2025-10-{d:02d}") for d in (1, 2, 3)], np.datetime64)
    d = np.array([np.timedelta64(s + 1, "s") for s in range(3)], np.timedelta64)

    check(assert_type(left * b, "pd.Index[bool]"), pd.Index, np.bool_)
    check(assert_type(left * i, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(left * f, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left * c, "pd.Index[complex]"), pd.Index, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left * s, Never)
        assert_type(left * d, Never)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rmul__` cannot override. At runtime, they return
    # `Index` with the correct element type.
    check(assert_type(b * left, "npt.NDArray[np.bool_]"), pd.Index, np.bool_)
    check(assert_type(i * left, np_ndarray_int64), pd.Index, np.integer)
    check(assert_type(f * left, "npt.NDArray[np.float64]"), pd.Index, np.floating)
    check(
        assert_type(c * left, "npt.NDArray[np.complex128]"),
        pd.Index,
        np.complexfloating,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(s * left, Any)
        assert_type(d * left, "npt.NDArray[np.timedelta64]")


def test_mul_pd_index(left: "pd.Index[bool]") -> None:
    """Test pd.Index[bool] * pandas Indexes"""
    b = pd.Index([True, False, True])
    i = pd.Index([2, 3, 5])
    f = pd.Index([1.0, 2.0, 3.0])
    c = pd.Index([1.1j, 2.2j, 4.1j])
    s = pd.Index([datetime(2025, 10, d) for d in (1, 2, 3)])
    d = pd.Index([timedelta(seconds=s + 1) for s in range(3)])

    check(assert_type(left * b, "pd.Index[bool]"), pd.Index, np.bool_)
    check(assert_type(left * i, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(left * f, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left * c, "pd.Index[complex]"), pd.Index, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        _04 = left * s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _05 = left * d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    check(assert_type(b * left, "pd.Index[bool]"), pd.Index, np.bool_)
    check(assert_type(i * left, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(f * left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(c * left, "pd.Index[complex]"), pd.Index, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        _14 = s * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _15 = d * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
