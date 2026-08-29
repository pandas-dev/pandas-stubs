from datetime import timedelta
from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)


@pytest.fixture
def left() -> "pd.Series[pd.Period]":
    """Left operand"""
    lo = pd.Series([pd.Period("2025-08-20", freq="D")])  # left operand
    return check(assert_type(lo, "pd.Series[pd.Period]"), pd.Series, pd.Period)


def test_add_py_scalar(left: "pd.Series[pd.Period]") -> None:
    """Test pd.Series[pd.Period] + Python native scalars"""
    d = timedelta(days=1)
    i = 1
    p = pd.Period("2025-08-20", freq="D")

    check(assert_type(left + d, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(d + left, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.add(d), "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.radd(d), "pd.Series[pd.Period]"), pd.Series, pd.Period)

    check(assert_type(left + i, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(i + left, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.add(i), "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.radd(i), "pd.Series[pd.Period]"), pd.Series, pd.Period)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + p  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
        _1 = p + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
        left.add(p)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]
        left.radd(p)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]


def test_add_py_sequence(left: "pd.Series[pd.Period]") -> None:
    """Test pd.Series[pd.Period] + Python native sequences"""
    d = [timedelta(days=1)]
    i = [1]
    p = [pd.Period("2025-08-20", freq="D")]

    check(assert_type(left + d, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(d + left, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.add(d), "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.radd(d), "pd.Series[pd.Period]"), pd.Series, pd.Period)

    check(assert_type(left + i, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(i + left, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.add(i), "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.radd(i), "pd.Series[pd.Period]"), pd.Series, pd.Period)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + p  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
        _1 = p + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
        left.add(p)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]
        left.radd(p)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]


def test_add_numpy_scalar(left: "pd.Series[pd.Period]") -> None:
    """Test pd.Series[pd.Period] + numpy scalars"""
    d = np.timedelta64(1, "D")
    i = np.int64(1)
    s = np.datetime64("2025-08-20")

    check(assert_type(left + d, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(d + left, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.add(d), "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.radd(d), "pd.Series[pd.Period]"), pd.Series, pd.Period)

    check(assert_type(left + i, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(i + left, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.add(i), "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.radd(i), "pd.Series[pd.Period]"), pd.Series, pd.Period)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
        _1 = s + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
        left.add(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]
        left.radd(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]


def test_add_pd_scalar(left: "pd.Series[pd.Period]") -> None:
    """Test pd.Series[pd.Period] + pandas scalars"""
    d = pd.Timedelta(days=1)
    off = pd.offsets.Day(1)
    p = pd.Period("2025-08-20", freq="D")

    check(assert_type(left + d, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(d + left, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.add(d), "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.radd(d), "pd.Series[pd.Period]"), pd.Series, pd.Period)

    check(assert_type(left + off, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(off + left, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.add(off), "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.radd(off), "pd.Series[pd.Period]"), pd.Series, pd.Period)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + p  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
        _1 = p + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
        left.add(p)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]
        left.radd(p)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]


def test_add_pd_series(left: "pd.Series[pd.Period]") -> None:
    """Test pd.Series[pd.Period] + pandas Series"""
    d = pd.Series([pd.Timedelta(days=1)])
    i = pd.Series([1])
    off = pd.Series([pd.offsets.Day(1)])
    p = pd.Series([pd.Period("2025-08-20", freq="D")])

    check(assert_type(left + d, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(d + left, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.add(d), "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.radd(d), "pd.Series[pd.Period]"), pd.Series, pd.Period)

    check(assert_type(left + i, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(i + left, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.add(i), "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.radd(i), "pd.Series[pd.Period]"), pd.Series, pd.Period)

    check(assert_type(left + off, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(off + left, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.add(off), "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.radd(off), "pd.Series[pd.Period]"), pd.Series, pd.Period)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + p  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
        _1 = p + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
        left.add(p)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]
        left.radd(p)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]


def test_add_pd_index(left: "pd.Series[pd.Period]") -> None:
    """Test pd.Series[pd.Period] + pandas Index"""
    d = pd.TimedeltaIndex([pd.Timedelta(days=1)])
    i = pd.Index([1])
    off = pd.Index([pd.offsets.Day(1)])
    p = pd.PeriodIndex(["2025-08-20"], freq="D")

    check(assert_type(left + d, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(d + left, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.add(d), "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.radd(d), "pd.Series[pd.Period]"), pd.Series, pd.Period)

    check(assert_type(left + i, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(i + left, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.add(i), "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.radd(i), "pd.Series[pd.Period]"), pd.Series, pd.Period)

    check(assert_type(left + off, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(off + left, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.add(off), "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.radd(off), "pd.Series[pd.Period]"), pd.Series, pd.Period)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + p  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
        _1 = p + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
        left.add(p)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]
        left.radd(p)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]
