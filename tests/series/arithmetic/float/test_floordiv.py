from typing import Any

import numpy as np
import pandas as pd
from typing_extensions import (
    Never,
    assert_type,
)

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

left = pd.Series([1.2, 2.4, 3.6])  # left operand


def test_floordiv_py_scalar() -> None:
    """Test pd.Series[float] // Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left // b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left // i, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left // f, "pd.Series[float]"), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _1 = left // c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

        _2 = b // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(i // left, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(f // left, "pd.Series[float]"), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _3 = c // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

        left.floordiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.floordiv(i), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.floordiv(f), "pd.Series[float]"), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        left.floordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]

        left.rfloordiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rfloordiv(i), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.rfloordiv(f), "pd.Series[float]"), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        left.rfloordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]


def test_floordiv_py_sequence() -> None:
    """Test pd.Series[float] // Python native sequences"""
    b, i, f, c = [True, False, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left // b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left // i, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left // f, "pd.Series[float]"), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _1 = left // c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

        _2 = b // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(i // left, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(f // left, "pd.Series[float]"), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _3 = c // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

        left.floordiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.floordiv(i), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.floordiv(f), "pd.Series[float]"), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        left.floordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]

        left.rfloordiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rfloordiv(i), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.rfloordiv(f), "pd.Series[float]"), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        left.rfloordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]


def test_floordiv_numpy_array() -> None:
    """Test pd.Series[float] // numpy arrays"""
    b = np.array([True, False, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left // b, Never)
    check(assert_type(left // i, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left // f, "pd.Series[float]"), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left // c, Never)

        # `numpy` typing gives the corresponding `ndarray`s in the static type
        # checking, where our `__rfloordiv__` cannot override. At runtime, they lead to
        # errors or pd.Series.
        assert_type(b // left, Any)  # pyright: ignore[reportAssertTypeFailure]
    check(
        assert_type(i // left, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Series,
        np.floating,
    )
    check(
        assert_type(f // left, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Series,
        np.floating,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(c // left, Any)

        left.floordiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.floordiv(i), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.floordiv(f), "pd.Series[float]"), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        left.floordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]

        left.rfloordiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rfloordiv(i), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.rfloordiv(f), "pd.Series[float]"), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        left.rfloordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]


def test_floordiv_pd_index() -> None:
    """Test pd.Series[float] // pandas Indexes"""
    b = pd.Index([True, False, True])
    i = pd.Index([2, 3, 5])
    f = pd.Index([1.0, 2.0, 3.0])
    c = pd.Index([1.1j, 2.2j, 4.1j])

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left // b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left // i, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left // f, "pd.Series[float]"), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _1 = left // c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

        _2 = b // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(i // left, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(f // left, "pd.Series[float]"), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _3 = c // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

        left.floordiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.floordiv(i), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.floordiv(f), "pd.Series[float]"), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        left.floordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]

        left.rfloordiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rfloordiv(i), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.rfloordiv(f), "pd.Series[float]"), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        left.rfloordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]


def test_floordiv_pd_series() -> None:
    """Test pd.Series[float] // pandas Series"""
    b = pd.Series([True, False, True])
    i = pd.Series([2, 3, 5])
    f = pd.Series([1.0, 2.0, 3.0])
    c = pd.Series([1.1j, 2.2j, 4.1j])

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left // b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left // i, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left // f, "pd.Series[float]"), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _1 = left // c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

        _2 = b // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(i // left, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(f // left, "pd.Series[float]"), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        _3 = c // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

        left.floordiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.floordiv(i), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.floordiv(f), "pd.Series[float]"), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        left.floordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]

        left.rfloordiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rfloordiv(i), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.rfloordiv(f), "pd.Series[float]"), pd.Series, np.floating)
    if TYPE_CHECKING_INVALID_USAGE:
        left.rfloordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
