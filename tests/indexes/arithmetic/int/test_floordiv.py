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
def left() -> "pd.Index[int]":
    """Left operand"""
    lo = pd.Index([1, 2, 3])
    return check(assert_type(lo, "pd.Index[int]"), pd.Index, np.integer)


def test_floordiv_py_scalar(left: "pd.Index[int]") -> None:
    """Test pd.Index[int] // Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j
    s, d = datetime(2025, 9, 27), timedelta(seconds=1)

    check(assert_type(left // b, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(left // i, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(left // f, "pd.Index[float]"), pd.Index, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left // c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left // s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _05 = left // d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    check(assert_type(b // left, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(i // left, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(f // left, "pd.Index[float]"), pd.Index, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _13 = c // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _14 = s // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d // left, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)


def test_floordiv_py_sequence(left: "pd.Index[int]") -> None:
    """Test pd.Index[int] // Python native sequences"""
    b, i, f, c = [True, True, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]
    s = [datetime(2025, 10, 27 + d) for d in range(3)]
    d = [timedelta(seconds=s) for s in range(3)]

    check(assert_type(left // b, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(left // i, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(left // f, "pd.Index[float]"), pd.Index, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left // c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left // s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _05 = left // d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    check(assert_type(b // left, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(i // left, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(f // left, "pd.Index[float]"), pd.Index, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _13 = c // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _14 = s // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _15 = d // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]


def test_floordiv_numpy_array(left: "pd.Index[int]") -> None:
    """Test pd.Index[int] // numpy arrays"""
    b = np.array([True, True, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)
    s = np.array(
        [np.datetime64(f"2025-10-{d:02d}") for d in (23, 24, 25)], np.datetime64
    )
    d = np.array([np.timedelta64(s + 1, "s") for s in range(3)], np.timedelta64)

    check(assert_type(left // b, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(left // i, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(left // f, "pd.Index[float]"), pd.Index, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left // c, Never)
        assert_type(left // s, Never)
        assert_type(left // d, Never)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rfloordiv__` cannot override. At runtime, they lead to
    # errors or pd.Index.
    check(b // left, pd.Index, np.integer)
    check(i // left, pd.Index, np.integer)
    check(f // left, pd.Index, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(c // left, Any)
        assert_type(s // left, Any)
        assert_type(d // left, "np.typing.NDArray[np.int64]")


def test_floordiv_pd_index(left: "pd.Index[int]") -> None:
    """Test pd.Index[int] // pandas Indexes"""
    b = pd.Index([True, True, True])
    i = pd.Index([2, 3, 5])
    f = pd.Index([1.0, 2.0, 3.0])
    c = pd.Index([1.1j, 2.2j, 4.1j])
    s = pd.Index([datetime(2025, 10, d) for d in (27, 28, 29)])
    d = pd.Index([timedelta(seconds=s + 1) for s in range(3)])

    check(assert_type(left // b, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(left // i, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(left // f, "pd.Index[float]"), pd.Index, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left // c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left // s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _05 = left // d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    check(assert_type(b // left, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(i // left, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(f // left, "pd.Index[float]"), pd.Index, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _13 = c // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _14 = s // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d // left, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
