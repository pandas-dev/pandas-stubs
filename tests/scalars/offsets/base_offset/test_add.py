from datetime import (
    date,
    datetime,
    timedelta,
)
from typing import (
    Never,
    assert_type,
)

import numpy as np
import pandas as pd
from pandas.api.typing import NaTType
import pytest

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

from pandas.tseries.offsets import (
    BaseOffset,
    Day,
    MonthEnd,
)


@pytest.fixture
def left() -> BaseOffset:
    """Left operand"""
    return MonthEnd()


def test_add_py_scalar(left: BaseOffset) -> None:
    """Test BaseOffset + Python native scalars"""
    datum = date(2026, 1, 1)
    s = datetime(2026, 1, 1)
    d = timedelta(hours=2)

    check(assert_type(left + datum, pd.Timestamp), pd.Timestamp)
    check(assert_type(datum + left, pd.Timestamp), pd.Timestamp)

    check(assert_type(left + s, pd.Timestamp), pd.Timestamp)
    check(assert_type(s + left, pd.Timestamp), pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        _l_d = left + d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _d_l = d + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_add_numpy_scalar(left: BaseOffset) -> None:
    """Test BaseOffset + numpy scalars"""
    s = np.datetime64("2026-01-01")
    d = np.timedelta64(2, "h")

    check(assert_type(left + s, pd.Timestamp), pd.Timestamp)
    check(assert_type(s + left, pd.Timestamp), pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        _l_d = left + d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _d_l = d + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_add_pd_scalar(left: BaseOffset) -> None:
    """Test BaseOffset + pandas scalars"""
    s = pd.Timestamp("2026-01-01")
    d = pd.Timedelta("2h")
    day = Day()

    check(assert_type(left + s, pd.Timestamp), pd.Timestamp)
    check(assert_type(s + left, pd.Timestamp), pd.Timestamp)

    check(assert_type(left + pd.NaT, NaTType), NaTType)
    check(assert_type(pd.NaT + left, NaTType), NaTType)

    if TYPE_CHECKING_INVALID_USAGE:
        _l_d = left + d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _d_l = d + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]

        _l_day = left + day  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _day_l = day + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]

        _l_l = left + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_add_numpy_array(left: BaseOffset) -> None:
    """Test BaseOffset + numpy arrays"""
    s = np.array([datetime(2026, 1, 1)], np.object_)

    if TYPE_CHECKING_INVALID_USAGE:

        def _l_s() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(left + s, Never)

        # numpy's `__add__` wins `s + left` statically, so the reflected overload is
        # checked directly, in its own function since `Never` ends reachability.
        def _s_l() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(left.__radd__(s), Never)
