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
def left_i() -> pd.Series:
    """Left operand"""
    lo = pd.DataFrame({"a": [1, 2, 3]})["a"]
    return check(assert_type(lo, pd.Series), pd.Series, np.integer)


def test_floordiv_py_scalar(left_i: pd.Series) -> None:
    """Test pd.Series[int] // Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j
    s, d = datetime(2025, 9, 27), timedelta(seconds=1)

    check(assert_type(left_i // b, pd.Series), pd.Series, np.integer)
    check(assert_type(left_i // i, pd.Series), pd.Series, np.integer)
    check(assert_type(left_i // f, pd.Series), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left_i // c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left_i // s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _05 = left_i // d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    check(assert_type(b // left_i, pd.Series), pd.Series, np.integer)
    check(assert_type(i // left_i, pd.Series), pd.Series, np.integer)
    check(assert_type(f // left_i, pd.Series), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _13 = c // left_i  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _14 = s // left_i  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d // left_i, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left_i.floordiv(b), pd.Series), pd.Series, np.integer)
    check(assert_type(left_i.floordiv(i), pd.Series), pd.Series, np.integer)
    check(assert_type(left_i.floordiv(f), pd.Series), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        left_i.floordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left_i.floordiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left_i.floordiv(d)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]

    check(assert_type(left_i.rfloordiv(b), pd.Series), pd.Series, np.integer)
    check(assert_type(left_i.rfloordiv(i), pd.Series), pd.Series, np.integer)
    check(assert_type(left_i.rfloordiv(f), pd.Series), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        left_i.rfloordiv(c)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left_i.rfloordiv(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(
        assert_type(left_i.rfloordiv(d), "pd.Series[pd.Timedelta]"),
        pd.Series,
        pd.Timedelta,
    )


def test_floordiv_py_sequence(left_i: pd.Series) -> None:
    """Test pd.Series[int] // Python native sequences"""
    b, i, f, c = [True, True, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]
    s = [datetime(2025, 10, 27 + d) for d in range(3)]
    d = [timedelta(seconds=s) for s in range(3)]

    check(assert_type(left_i // b, pd.Series), pd.Series, np.integer)
    check(assert_type(left_i // i, pd.Series), pd.Series, np.integer)
    check(assert_type(left_i // f, pd.Series), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left_i // c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left_i // s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _05 = left_i // d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    check(assert_type(b // left_i, pd.Series), pd.Series, np.integer)
    check(assert_type(i // left_i, pd.Series), pd.Series, np.integer)
    check(assert_type(f // left_i, pd.Series), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _13 = c // left_i  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _14 = s // left_i  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d // left_i, pd.Series), pd.Series, timedelta)

    check(assert_type(left_i.floordiv(b), pd.Series), pd.Series, np.integer)
    check(assert_type(left_i.floordiv(i), pd.Series), pd.Series, np.integer)
    check(assert_type(left_i.floordiv(f), pd.Series), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        left_i.floordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left_i.floordiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left_i.floordiv(d)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]

    check(assert_type(left_i.rfloordiv(b), pd.Series), pd.Series, np.integer)
    check(assert_type(left_i.rfloordiv(i), pd.Series), pd.Series, np.integer)
    check(assert_type(left_i.rfloordiv(f), pd.Series), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        left_i.rfloordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left_i.rfloordiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(
        assert_type(left_i.rfloordiv(d), "pd.Series[pd.Timedelta]"),
        pd.Series,
        pd.Timedelta,
    )


def test_floordiv_numpy_array(left_i: pd.Series) -> None:
    """Test pd.Series[int] // numpy arrays"""
    b = np.array([True, True, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)
    s = np.array(
        [np.datetime64(f"2025-10-{d:02d}") for d in (23, 24, 25)], np.datetime64
    )
    d = np.array([np.timedelta64(s + 1, "s") for s in range(3)], np.timedelta64)

    check(assert_type(left_i // b, pd.Series), pd.Series, np.integer)
    check(assert_type(left_i // i, pd.Series), pd.Series, np.integer)
    check(assert_type(left_i // f, pd.Series), pd.Series, np.floating)

    def _03() -> None:  # pyright: ignore[reportUnusedFunction]
        assert_type(left_i // c, Never)

    def _04() -> None:  # pyright: ignore[reportUnusedFunction]
        assert_type(left_i // s, Never)

    def _05() -> None:  # pyright: ignore[reportUnusedFunction]
        assert_type(left_i // d, Never)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rfloordiv__` cannot override. At runtime, they lead to
    # errors or pd.Series.
    check(b // left_i, pd.Series, np.integer)
    check(i // left_i, pd.Series, np.integer)
    check(f // left_i, pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(c // left_i, Any)
        assert_type(s // left_i, Any)
    check(
        assert_type(d // left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Series,
        pd.Timedelta,
    )

    check(assert_type(left_i.floordiv(b), pd.Series), pd.Series, np.integer)
    check(assert_type(left_i.floordiv(i), pd.Series), pd.Series, np.integer)
    check(assert_type(left_i.floordiv(f), pd.Series), pd.Series, np.floating)

    def _23() -> None:  # pyright: ignore[reportUnusedFunction]
        assert_type(
            left_i.floordiv(c),  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportAssertTypeFailure,reportCallIssue]
            Never,
        )

    def _24() -> None:  # pyright: ignore[reportUnusedFunction]
        assert_type(
            left_i.floordiv(s),  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportAssertTypeFailure,reportCallIssue]
            Never,
        )

    def _25() -> None:  # pyright: ignore[reportUnusedFunction]
        assert_type(left_i.floordiv(d), Never)

    check(assert_type(left_i.rfloordiv(b), pd.Series), pd.Series, np.integer)
    check(assert_type(left_i.rfloordiv(i), pd.Series), pd.Series, np.integer)
    check(assert_type(left_i.rfloordiv(f), pd.Series), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        left_i.rfloordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left_i.rfloordiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(
        assert_type(left_i.rfloordiv(d), "pd.Series[pd.Timedelta]"),
        pd.Series,
        pd.Timedelta,
    )


def test_floordiv_pd_index(left_i: pd.Series) -> None:
    """Test pd.Series[int] // pandas Indexes"""
    b = pd.Index([True, True, True])
    i = pd.Index([2, 3, 5])
    f = pd.Index([1.0, 2.0, 3.0])
    c = pd.Index([1.1j, 2.2j, 4.1j])
    s = pd.Index([datetime(2025, 10, d) for d in (27, 28, 29)])
    d = pd.Index([timedelta(seconds=s + 1) for s in range(3)])

    check(assert_type(left_i // b, pd.Series), pd.Series, np.integer)
    check(assert_type(left_i // i, pd.Series), pd.Series, np.integer)
    check(assert_type(left_i // f, pd.Series), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left_i // c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left_i // s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    def _05() -> None:  # pyright: ignore[reportUnusedFunction]
        assert_type(left_i // d, Never)

    check(assert_type(b // left_i, pd.Series), pd.Series, np.integer)
    check(assert_type(i // left_i, pd.Series), pd.Series, np.integer)
    check(assert_type(f // left_i, pd.Series), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _13 = c // left_i  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _14 = s // left_i  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d // left_i, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left_i.floordiv(b), pd.Series), pd.Series, np.integer)
    check(assert_type(left_i.floordiv(i), pd.Series), pd.Series, np.integer)
    check(assert_type(left_i.floordiv(f), pd.Series), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        left_i.floordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left_i.floordiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]

    def _25() -> None:  # pyright: ignore[reportUnusedFunction]
        assert_type(left_i.floordiv(d), Never)

    check(assert_type(left_i.rfloordiv(b), pd.Series), pd.Series, np.integer)
    check(assert_type(left_i.rfloordiv(i), pd.Series), pd.Series, np.integer)
    check(assert_type(left_i.rfloordiv(f), pd.Series), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        left_i.rfloordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left_i.rfloordiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(
        assert_type(left_i.rfloordiv(d), "pd.Series[pd.Timedelta]"),
        pd.Series,
        pd.Timedelta,
    )


def test_floordiv_pd_series(left_i: pd.Series) -> None:
    """Test pd.Series[int] // pandas Series"""
    b = pd.Series([True, True, True])
    i = pd.Series([2, 3, 5])
    f = pd.Series([1.0, 2.0, 3.0])
    c = pd.Series([1.1j, 2.2j, 4.1j])
    s = pd.Series([datetime(2025, 10, d) for d in (27, 28, 29)])
    d = pd.Series([timedelta(seconds=s + 1) for s in range(3)])

    check(assert_type(left_i // b, pd.Series), pd.Series, np.integer)
    check(assert_type(left_i // i, pd.Series), pd.Series, np.integer)
    check(assert_type(left_i // f, pd.Series), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left_i // c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left_i // s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        # left_i // d  # This invalid one cannot be detected by static type checking

    check(assert_type(b // left_i, pd.Series), pd.Series, np.integer)
    check(assert_type(i // left_i, pd.Series), pd.Series, np.integer)
    check(assert_type(f // left_i, pd.Series), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _13 = c // left_i  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _14 = s // left_i  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d // left_i, pd.Series), pd.Series, pd.Timedelta)

    check(assert_type(left_i.floordiv(b), pd.Series), pd.Series, np.integer)
    check(assert_type(left_i.floordiv(i), pd.Series), pd.Series, np.integer)
    check(assert_type(left_i.floordiv(f), pd.Series), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        left_i.floordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left_i.floordiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        # left_i.floordiv(d)  # This invalid one cannot be detected by static type checking

    check(assert_type(left_i.rfloordiv(b), pd.Series), pd.Series, np.integer)
    check(assert_type(left_i.rfloordiv(i), pd.Series), pd.Series, np.integer)
    check(assert_type(left_i.rfloordiv(f), pd.Series), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        left_i.rfloordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left_i.rfloordiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]

    check(
        assert_type(left_i.rfloordiv(d), "pd.Series[pd.Timedelta]"),
        pd.Series,
        pd.Timedelta,
    )
