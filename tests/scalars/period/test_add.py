from datetime import (
    datetime,
    timedelta,
)
from typing import assert_type

import numpy as np
import pandas as pd
from pandas.api.typing import NaTType
import pytest

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)


@pytest.fixture
def left() -> pd.Period:
    """Left operand"""
    return pd.Period("2025-08-20", freq="D")


def test_add_py_scalar(left: pd.Period) -> None:
    """Test pd.Period + Python native scalars"""
    d = timedelta(days=1)
    i = 1
    s = datetime(2025, 8, 20)

    check(assert_type(left + d, pd.Period), pd.Period)
    check(assert_type(d + left, pd.Period), pd.Period)

    check(assert_type(left + i, pd.Period), pd.Period)
    check(assert_type(i + left, pd.Period), pd.Period)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _1 = s + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _2 = left + "str"  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _3 = "str" + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _4 = left + 1.5  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _5 = 1.5 + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_add_numpy_scalar(left: pd.Period) -> None:
    """Test pd.Period + numpy scalars"""
    d = np.timedelta64(1, "D")
    i = np.int64(1)
    s = np.datetime64("2025-08-20")

    check(assert_type(left + d, pd.Period), pd.Period)
    check(assert_type(d + left, pd.Period), pd.Period)

    check(assert_type(left + i, pd.Period), pd.Period)
    check(assert_type(i + left, pd.Period), pd.Period)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _1 = s + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_add_pd_scalar(left: pd.Period) -> None:
    """Test pd.Period + pandas scalars"""
    d = pd.Timedelta(days=1)
    off = pd.offsets.Day(1)
    p = pd.Period("2025-08-20", freq="D")
    nat = pd.NaT

    check(assert_type(left + d, pd.Period), pd.Period)
    check(assert_type(d + left, pd.Period), pd.Period)

    check(assert_type(left + off, pd.Period), pd.Period)
    check(assert_type(off + left, pd.Period), pd.Period)

    check(assert_type(left + nat, NaTType), NaTType)
    check(assert_type(nat + left, NaTType), NaTType)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + p  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _1 = p + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
