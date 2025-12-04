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
from tests._typing import (
    np_ndarray_bool,
    np_ndarray_int64,
    np_ndarray_td,
)


@pytest.fixture
def left() -> "pd.Series[int]":
    """Left operand"""
    lo = pd.Series([1, 2, 3])
    return check(assert_type(lo, "pd.Series[int]"), pd.Series, np.integer)


def test_mul_py_scalar(left: "pd.Series[int]") -> None:
    """Test pd.Series[int] * Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j
    s, d = datetime(2025, 9, 27), timedelta(seconds=1)

    check(assert_type(left * b, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left * i, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left * f, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left * c, "pd.Series[complex]"), pd.Series, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        _04 = left * s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(left * d, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(b * left, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(i * left, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(f * left, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(c * left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        _14 = s * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(d * left, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left.mul(b), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.mul(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.mul(f), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.mul(c), "pd.Series[complex]"), pd.Series, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.mul(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left.rmul(b), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rmul(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rmul(f), "pd.Series[float]"), pd.Series, np.floating)
    check(
        assert_type(left.rmul(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rmul(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)


def test_mul_py_sequence(left: "pd.Series[int]") -> None:
    """Test pd.Series[int] * Python native sequences"""
    b, i, f, c = [True, False, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]
    s = [datetime(2025, 9, d) for d in (27, 28, 29)]
    d = [timedelta(seconds=s + 1) for s in range(3)]

    check(assert_type(left * b, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left * i, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left * f, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left * c, "pd.Series[complex]"), pd.Series, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        _04 = left * s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(left * d, "pd.Series[pd.Timedelta]"), pd.Series, timedelta)

    check(assert_type(b * left, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(i * left, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(f * left, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(c * left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        _14 = s * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(d * left, "pd.Series[pd.Timedelta]"), pd.Series, timedelta)

    check(assert_type(left.mul(b), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.mul(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.mul(f), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.mul(c), "pd.Series[complex]"), pd.Series, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.mul(d), "pd.Series[pd.Timedelta]"), pd.Series, timedelta)

    check(assert_type(left.rmul(b), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rmul(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rmul(f), "pd.Series[float]"), pd.Series, np.floating)
    check(
        assert_type(left.rmul(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rmul(d), "pd.Series[pd.Timedelta]"), pd.Series, timedelta)


def test_mul_numpy_array(left: "pd.Series[int]") -> None:
    """Test pd.Series[int] * numpy arrays"""
    b = np.array([True, False, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)
    s = np.array([np.datetime64(f"2025-09-{d}") for d in (27, 28, 29)], np.datetime64)
    d = np.array([np.timedelta64(s + 1, "s") for s in range(3)], np.timedelta64)

    check(assert_type(left * b, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left * i, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left * f, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left * c, "pd.Series[complex]"), pd.Series, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left * s, Never)
    check(assert_type(left * d, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rmul__` cannot override. At runtime, they return
    # `Series` with the correct element type.
    check(assert_type(b * left, np_ndarray_bool), pd.Series, np.integer)
    check(assert_type(i * left, np_ndarray_int64), pd.Series, np.integer)
    check(assert_type(f * left, "npt.NDArray[np.float64]"), pd.Series, np.floating)
    check(
        assert_type(c * left, "npt.NDArray[np.complex128]"),
        pd.Series,
        np.complexfloating,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(s * left, Any)
    check(assert_type(d * left, np_ndarray_td), pd.Series, pd.Timedelta)

    check(assert_type(left.mul(b), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.mul(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.mul(f), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.mul(c), "pd.Series[complex]"), pd.Series, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.mul(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left.rmul(b), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rmul(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rmul(f), "pd.Series[float]"), pd.Series, np.floating)
    check(
        assert_type(left.rmul(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rmul(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)


def test_mul_pd_index(left: "pd.Series[int]") -> None:
    """Test pd.Series[int] * pandas Indexes"""
    b = pd.Index([True, False, True])
    i = pd.Index([2, 3, 5])
    f = pd.Index([1.0, 2.0, 3.0])
    c = pd.Index([1.1j, 2.2j, 4.1j])
    s = pd.Index([datetime(2025, 9, d) for d in (27, 28, 29)])
    d = pd.Index([timedelta(seconds=s + 1) for s in range(3)])

    check(assert_type(left * b, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left * i, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left * f, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left * c, "pd.Series[complex]"), pd.Series, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        _04 = left * s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(left * d, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(b * left, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(i * left, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(f * left, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(c * left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        _14 = s * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(d * left, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left.mul(b), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.mul(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.mul(f), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.mul(c), "pd.Series[complex]"), pd.Series, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.mul(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left.rmul(b), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rmul(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rmul(f), "pd.Series[float]"), pd.Series, np.floating)
    check(
        assert_type(left.rmul(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rmul(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)


def test_mul_pd_series(left: "pd.Series[int]") -> None:
    """Test pd.Series[int] * pandas Series"""
    b = pd.Series([True, False, True])
    i = pd.Series([2, 3, 5])
    f = pd.Series([1.0, 2.0, 3.0])
    c = pd.Series([1.1j, 2.2j, 4.1j])
    s = pd.Series([datetime(2025, 9, d) for d in (27, 28, 29)])
    d = pd.Series([timedelta(seconds=s + 1) for s in range(3)])

    check(assert_type(left * b, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left * i, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left * f, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left * c, "pd.Series[complex]"), pd.Series, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        _04 = left * s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(left * d, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(b * left, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(i * left, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(f * left, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(c * left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        _14 = s * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(d * left, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left.mul(b), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.mul(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.mul(f), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.mul(c), "pd.Series[complex]"), pd.Series, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.mul(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left.rmul(b), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rmul(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rmul(f), "pd.Series[float]"), pd.Series, np.floating)
    check(
        assert_type(left.rmul(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rmul(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
