from typing import (
    Never,
    assert_type,
)

import numpy as np
from numpy import typing as npt  # noqa: F401
import pandas as pd
import pytest

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)
from tests._typing import (
    np_ndarray_bool,
    np_ndarray_int64,
)


@pytest.fixture
def left() -> "pd.Series[pd.Timedelta]":
    """Left operand"""
    lo = pd.Series([pd.Timedelta(1, "s")])  # left operand
    return check(assert_type(lo, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)


def test_mul_py_scalar(left: "pd.Series[pd.Timedelta]") -> None:
    """Test pd.Series[pd.Timedelta] * Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left * b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(
        assert_type(  # pyrefly: ignore[assert-type]
            left * i,  # pyrefly: ignore[unsupported-operation]
            "pd.Series[pd.Timedelta]",
        ),  # pyrefly: ignore [assert-type]
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(  # pyrefly: ignore[assert-type]
            left * f,  # pyrefly: ignore[unsupported-operation]
            "pd.Series[pd.Timedelta]",
        ),  # pyrefly: ignore [assert-type]
        pd.Series,
        pd.Timedelta,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left * c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(
        assert_type(  # pyrefly: ignore[assert-type]
            i * left,  # pyrefly: ignore[unsupported-operation]
            "pd.Series[pd.Timedelta]",
        ),  # pyrefly: ignore [assert-type]
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(  # pyrefly: ignore[assert-type]
            f * left,  # pyrefly: ignore[unsupported-operation]
            "pd.Series[pd.Timedelta]",
        ),  # pyrefly: ignore [assert-type]
        pd.Series,
        pd.Timedelta,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        _13 = c * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]

    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(
        assert_type(  # pyrefly: ignore[assert-type]
            left.mul(i),  # pyrefly: ignore[no-matching-overload]
            "pd.Series[pd.Timedelta]",
        ),  # pyrefly: ignore [assert-type]
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(  # pyrefly: ignore[assert-type]
            left.mul(f),  # pyrefly: ignore[no-matching-overload]
            "pd.Series[pd.Timedelta]",
        ),  # pyrefly: ignore [assert-type]
        pd.Series,
        pd.Timedelta,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]

    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(
        assert_type(  # pyrefly: ignore[assert-type]
            left.rmul(i),  # pyrefly: ignore[no-matching-overload]
            "pd.Series[pd.Timedelta]",
        ),  # pyrefly: ignore [assert-type]
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(  # pyrefly: ignore[assert-type]
            left.rmul(f),  # pyrefly: ignore[no-matching-overload]
            "pd.Series[pd.Timedelta]",
        ),  # pyrefly: ignore [assert-type]
        pd.Series,
        pd.Timedelta,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]


def test_mul_py_sequence(left: "pd.Series[pd.Timedelta]") -> None:
    """Test pd.Series[pd.Timedelta] * Python native sequences"""
    b, i, f, c = [True], [2], [1.5], [1.7j]

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left * b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(
        assert_type(  # pyrefly: ignore[assert-type]
            left * i,  # pyrefly: ignore[unsupported-operation]
            "pd.Series[pd.Timedelta]",
        ),  # pyrefly: ignore [assert-type]
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(  # pyrefly: ignore[assert-type]
            left * f,  # pyrefly: ignore[unsupported-operation]
            "pd.Series[pd.Timedelta]",
        ),  # pyrefly: ignore [assert-type]
        pd.Series,
        pd.Timedelta,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left * c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(
        assert_type(  # pyrefly: ignore[assert-type]
            i * left,  # pyrefly: ignore[unsupported-operation]
            "pd.Series[pd.Timedelta]",
        ),  # pyrefly: ignore [assert-type]
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(  # pyrefly: ignore[assert-type]
            f * left,  # pyrefly: ignore[unsupported-operation]
            "pd.Series[pd.Timedelta]",
        ),  # pyrefly: ignore [assert-type]
        pd.Series,
        pd.Timedelta,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        _13 = c * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]

    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(
        assert_type(  # pyrefly: ignore[assert-type]
            left.mul(i),  # pyrefly: ignore[no-matching-overload]
            "pd.Series[pd.Timedelta]",
        ),  # pyrefly: ignore [assert-type]
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(  # pyrefly: ignore[assert-type]
            left.mul(f),  # pyrefly: ignore[no-matching-overload]
            "pd.Series[pd.Timedelta]",
        ),  # pyrefly: ignore [assert-type]
        pd.Series,
        pd.Timedelta,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]

    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(
        assert_type(  # pyrefly: ignore[assert-type]
            left.rmul(i),  # pyrefly: ignore[no-matching-overload]
            "pd.Series[pd.Timedelta]",
        ),  # pyrefly: ignore [assert-type]
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(  # pyrefly: ignore[assert-type]
            left.rmul(f),  # pyrefly: ignore[no-matching-overload]
            "pd.Series[pd.Timedelta]",
        ),  # pyrefly: ignore [assert-type]
        pd.Series,
        pd.Timedelta,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]


def test_mul_numpy_array(left: "pd.Series[pd.Timedelta]") -> None:
    """Test pd.Series[pd.Timedelta] * numpy arrays"""
    b = np.array([True], np.bool_)
    i = np.array([2], np.int64)
    f = np.array([1.5], np.float64)
    c = np.array([1.7j], np.complex128)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left * b, Never)
    check(assert_type(left * i, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left * f, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left * c, Never)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rmul__` cannot override. At runtime, they return
    # `Series` with the correct element type.
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(b * left, np_ndarray_bool)
    check(assert_type(i * left, np_ndarray_int64), pd.Series, pd.Timedelta)
    check(assert_type(f * left, "npt.NDArray[np.float64]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        # We made it Never, but numpy takes over
        assert_type(c * left, "npt.NDArray[np.complex128]")

    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(
        assert_type(  # pyrefly: ignore[bad-argument-type]
            left.mul(i), "pd.Series[pd.Timedelta]"
        ),  # pyrefly: ignore[bad-argument-type]
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(  # pyrefly: ignore[bad-argument-type]
            left.mul(f), "pd.Series[pd.Timedelta]"
        ),  # pyrefly: ignore[bad-argument-type]
        pd.Series,
        pd.Timedelta,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(
        assert_type(  # pyrefly: ignore[bad-argument-type]
            left.rmul(i), "pd.Series[pd.Timedelta]"
        ),  # pyrefly: ignore[bad-argument-type]
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(  # pyrefly: ignore[bad-argument-type]
            left.rmul(f), "pd.Series[pd.Timedelta]"
        ),  # pyrefly: ignore[bad-argument-type]
        pd.Series,
        pd.Timedelta,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]


def test_mul_pd_index(left: "pd.Series[pd.Timedelta]") -> None:
    """Test pd.Series[pd.Timedelta] * pandas Indexes"""
    b = pd.Index([True])
    i = pd.Index([2])
    f = pd.Index([1.5])
    c = pd.Index([1.7j])

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left * b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
    check(assert_type(left * i, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left * f, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left * c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b * left  # type: ignore[operator,type-var,unused-ignore] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # mypy gives different errors for mypy and test_dist  # pyrefly: ignore[unsupported-operation]
    check(assert_type(i * left, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(f * left, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _13 = c * left  # type: ignore[operator,type-var,unused-ignore] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # mypy gives different errors for mypy and test_dist  # pyrefly: ignore[unsupported-operation]

    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]
    check(
        assert_type(  # pyrefly: ignore[bad-argument-type]
            left.mul(i), "pd.Series[pd.Timedelta]"
        ),  # pyrefly: ignore[bad-argument-type]
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(  # pyrefly: ignore[bad-argument-type]
            left.mul(f), "pd.Series[pd.Timedelta]"
        ),  # pyrefly: ignore[bad-argument-type]
        pd.Series,
        pd.Timedelta,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]

    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]
    check(
        assert_type(  # pyrefly: ignore[bad-argument-type]
            left.rmul(i), "pd.Series[pd.Timedelta]"
        ),  # pyrefly: ignore[bad-argument-type]
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(  # pyrefly: ignore[bad-argument-type]
            left.rmul(f), "pd.Series[pd.Timedelta]"
        ),  # pyrefly: ignore[bad-argument-type]
        pd.Series,
        pd.Timedelta,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]


def test_mul_pd_series(left: "pd.Series[pd.Timedelta]") -> None:
    """Test pd.Series[pd.Timedelta] * pandas Series"""
    b = pd.Series([True])
    i = pd.Series([2])
    f = pd.Series([1.5])
    c = pd.Series([1.7j])

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left * b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
    check(assert_type(left * i, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left * f, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left * c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
    check(assert_type(i * left, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(f * left, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _13 = c * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]

    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]
    check(
        assert_type(  # pyrefly: ignore[bad-argument-type]
            left.mul(i), "pd.Series[pd.Timedelta]"
        ),  # pyrefly: ignore[bad-argument-type]
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(  # pyrefly: ignore[bad-argument-type]
            left.mul(f), "pd.Series[pd.Timedelta]"
        ),  # pyrefly: ignore[bad-argument-type]
        pd.Series,
        pd.Timedelta,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]

    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]
    check(
        assert_type(  # pyrefly: ignore[bad-argument-type]
            left.rmul(i), "pd.Series[pd.Timedelta]"
        ),  # pyrefly: ignore[bad-argument-type]
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(  # pyrefly: ignore[bad-argument-type]
            left.rmul(f), "pd.Series[pd.Timedelta]"
        ),  # pyrefly: ignore[bad-argument-type]
        pd.Series,
        pd.Timedelta,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]
