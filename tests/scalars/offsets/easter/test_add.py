import datetime as dt
from typing import assert_type

import numpy as np
import pandas as pd
from pandas.api.typing import NaTType
import pytest

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)
from tests._typing import np_ndarray_object

from pandas.tseries.offsets import Easter


@pytest.fixture
def left() -> Easter:
    """The Easter offset under test."""
    return Easter()


def test_easter_python_scalars(left: Easter) -> None:
    """Easter handles python scalars in both directions."""
    check(assert_type(left + dt.date(2026, 1, 1), pd.Timestamp), pd.Timestamp)
    check(assert_type(dt.date(2026, 1, 1) + left, pd.Timestamp), pd.Timestamp)
    check(assert_type(left + dt.datetime(2026, 1, 1), pd.Timestamp), pd.Timestamp)
    check(assert_type(dt.datetime(2026, 1, 1) + left, pd.Timestamp), pd.Timestamp)
    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + dt.timedelta(hours=2)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _1 = dt.timedelta(hours=2) + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_easter_numpy_scalars(left: Easter) -> None:
    """Easter handles numpy scalars in both directions."""
    check(assert_type(left + np.datetime64("2026-01-01"), pd.Timestamp), pd.Timestamp)
    check(assert_type(np.datetime64("2026-01-01") + left, pd.Timestamp), pd.Timestamp)
    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + np.timedelta64(2, "h")  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _1 = np.timedelta64(2, "h") + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_easter_pandas_scalars(left: Easter) -> None:
    """Easter handles pandas scalars in both directions."""
    check(assert_type(left + pd.Timestamp("2026-01-01"), pd.Timestamp), pd.Timestamp)
    check(assert_type(pd.Timestamp("2026-01-01") + left, pd.Timestamp), pd.Timestamp)
    check(assert_type(left + pd.NaT, NaTType), NaTType)
    check(assert_type(pd.NaT + left, NaTType), NaTType)
    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + pd.Timedelta("2h")  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _1 = pd.Timedelta("2h") + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_easter_numpy_arrays(left: Easter) -> None:
    """Easter supports object arrays, including empty and n-dimensional ones."""
    values: np_ndarray_object[tuple[int, ...]] = np.array(
        [dt.datetime(2026, 1, 1)], dtype=object
    )
    empty: np_ndarray_object[tuple[int, ...]] = np.array([], dtype=object)
    matrix: np_ndarray_object[tuple[int, int]] = np.array(
        [[dt.datetime(2026, 1, 1)]], dtype=object
    )
    check(
        assert_type(left + values, np_ndarray_object[tuple[int, ...]]),
        np.ndarray,
    )
    check(
        assert_type(left + empty, np_ndarray_object[tuple[int, ...]]),
        np.ndarray,
    )
    check(
        assert_type(left + matrix, np_ndarray_object[tuple[int, int]]),
        np.ndarray,
    )
    # NumPy's reflected array operator returns Any, masking offset.__radd__.
    check(values + left, np.ndarray)
    # Check the reflected contract directly instead.
    check(
        assert_type(left.__radd__(values), np_ndarray_object[tuple[int, ...]]),
        np.ndarray,
    )
