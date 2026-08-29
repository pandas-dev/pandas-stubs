from typing import assert_type

import pandas as pd
import pytest

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

from pandas.tseries.offsets import (
    BaseOffset,
    Day,
)


@pytest.fixture
def left() -> "pd.Series[BaseOffset]":
    """Left operand"""
    lo = pd.Series([Day(1)])  # left operand
    return check(assert_type(lo, "pd.Series[BaseOffset]"), pd.Series, Day)


def test_add_period(left: "pd.Series[BaseOffset]") -> None:
    """Test pd.Series[BaseOffset] + Period, Index[Period], Series[Period]"""
    p = pd.Period("2025-08-20", freq="D")
    pi = pd.PeriodIndex(["2025-08-20"], freq="D")
    ps = pd.Series([pd.Period("2025-08-20", freq="D")])

    check(assert_type(left + p, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(p + left, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.add(p), "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.radd(p), "pd.Series[pd.Period]"), pd.Series, pd.Period)

    check(assert_type(left + pi, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(pi + left, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.add(pi), "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.radd(pi), "pd.Series[pd.Period]"), pd.Series, pd.Period)

    check(assert_type(left + ps, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(ps + left, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.add(ps), "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(left.radd(ps), "pd.Series[pd.Period]"), pd.Series, pd.Period)


def test_add_offset(left: "pd.Series[BaseOffset]") -> None:
    """Test pd.Series[BaseOffset] + BaseOffset, Index[BaseOffset], Series[BaseOffset]"""
    off = Day(1)
    off_idx = pd.Index([Day(1)])
    off_sr = pd.Series([Day(1)])

    check(assert_type(left + off, "pd.Series[BaseOffset]"), pd.Series, Day)
    check(assert_type(off + left, "pd.Series[BaseOffset]"), pd.Series, Day)
    check(assert_type(left.add(off), "pd.Series[BaseOffset]"), pd.Series, Day)
    check(assert_type(left.radd(off), "pd.Series[BaseOffset]"), pd.Series, Day)

    check(assert_type(left + off_idx, "pd.Series[BaseOffset]"), pd.Series, Day)
    check(assert_type(off_idx + left, "pd.Series[BaseOffset]"), pd.Series, Day)
    check(assert_type(left.add(off_idx), "pd.Series[BaseOffset]"), pd.Series, Day)
    check(assert_type(left.radd(off_idx), "pd.Series[BaseOffset]"), pd.Series, Day)

    check(assert_type(left + off_sr, "pd.Series[BaseOffset]"), pd.Series, Day)
    check(assert_type(off_sr + left, "pd.Series[BaseOffset]"), pd.Series, Day)
    check(assert_type(left.add(off_sr), "pd.Series[BaseOffset]"), pd.Series, Day)
    check(assert_type(left.radd(off_sr), "pd.Series[BaseOffset]"), pd.Series, Day)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + 1  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _1 = 1 + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _2 = left + "str"  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _3 = "str" + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
