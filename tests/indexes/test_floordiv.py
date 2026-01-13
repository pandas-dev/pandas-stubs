from datetime import (
    datetime,
    timedelta,
)
from typing import Any

import numpy as np
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


@pytest.fixture
def left_i() -> pd.Index:
    """Left operand"""
    lo = pd.MultiIndex.from_arrays([[1, 2, 3]]).levels[0]
    return check(assert_type(lo, pd.Index), pd.Index, np.integer)


def test_floordiv_py_scalar(left_i: pd.Index) -> None:
    """Test pd.Index[int] // Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j
    s, d = datetime(2025, 9, 27), timedelta(seconds=1)

    check(assert_type(left_i // b, pd.Index), pd.Index, np.integer)
    check(assert_type(left_i // i, pd.Index), pd.Index, np.integer)
    check(assert_type(left_i // f, pd.Index), pd.Index, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left_i // c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _04 = left_i // s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _05 = left_i // d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]

    check(assert_type(b // left_i, pd.Index), pd.Index, np.integer)
    check(assert_type(i // left_i, pd.Index), pd.Index, np.integer)
    check(assert_type(f // left_i, pd.Index), pd.Index, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _13 = c // left_i  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _14 = s // left_i  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(d // left_i, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)


def test_floordiv_py_sequence(left_i: pd.Index) -> None:
    """Test pd.Index[int] // Python native sequences"""
    b, i, f, c = [True, True, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]
    s = [datetime(2025, 10, 27 + d) for d in range(3)]
    d = [timedelta(seconds=s) for s in range(3)]

    check(assert_type(left_i // b, pd.Index), pd.Index, np.integer)
    check(assert_type(left_i // i, pd.Index), pd.Index, np.integer)
    check(assert_type(left_i // f, pd.Index), pd.Index, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left_i // c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _04 = left_i // s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _05 = left_i // d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]

    check(assert_type(b // left_i, pd.Index), pd.Index, np.integer)
    check(assert_type(i // left_i, pd.Index), pd.Index, np.integer)
    check(assert_type(f // left_i, pd.Index), pd.Index, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _13 = c // left_i  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _14 = s // left_i  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(d // left_i, pd.Index), pd.Index, timedelta)


def test_floordiv_numpy_array(left_i: pd.Index) -> None:
    """Test pd.Index[int] // numpy arrays"""
    b = np.array([True, True, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)
    s = np.array(
        [np.datetime64(f"2025-10-{d:02d}") for d in (23, 24, 25)], np.datetime64
    )
    d = np.array([np.timedelta64(s + 1, "s") for s in range(3)], np.timedelta64)

    check(assert_type(left_i // b, pd.Index), pd.Index, np.integer)
    check(assert_type(left_i // i, pd.Index), pd.Index, np.integer)
    check(assert_type(left_i // f, pd.Index), pd.Index, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left_i // c, Never)
        assert_type(left_i // s, Never)
        assert_type(left_i // d, Never)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rfloordiv__` cannot override. At runtime, they lead to
    # errors or pd.Index.
    check(
        assert_type(b // left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Index,
        np.integer,
    )
    check(
        assert_type(i // left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Index,
        np.integer,
    )
    check(
        assert_type(f // left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Index,
        np.floating,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(c // left_i, Any)
        assert_type(s // left_i, Any)
    check(
        assert_type(d // left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.TimedeltaIndex,
        pd.Timedelta,
    )


def test_floordiv_pd_index(left_i: pd.Index) -> None:
    """Test pd.Index[int] // pandas Indexes"""
    b = pd.Index([True, True, True])
    i = pd.Index([2, 3, 5])
    f = pd.Index([1.0, 2.0, 3.0])
    c = pd.Index([1.1j, 2.2j, 4.1j])
    s = pd.Index([datetime(2025, 10, d) for d in (27, 28, 29)])
    d = pd.Index([timedelta(seconds=s + 1) for s in range(3)])

    check(assert_type(left_i // b, pd.Index), pd.Index, np.integer)
    check(assert_type(left_i // i, pd.Index), pd.Index, np.integer)
    check(assert_type(left_i // f, pd.Index), pd.Index, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left_i // c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _04 = left_i // s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _05 = left_i // d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]

    check(assert_type(b // left_i, pd.Index), pd.Index, np.integer)
    check(assert_type(i // left_i, pd.Index), pd.Index, np.integer)
    check(assert_type(f // left_i, pd.Index), pd.Index, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _13 = c // left_i  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _14 = s // left_i  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(d // left_i, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
