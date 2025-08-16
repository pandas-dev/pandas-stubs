import numpy as np
from numpy import typing as npt  # noqa: F401
import pandas as pd
from typing_extensions import (
    Never,
    assert_type,
)

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

left = pd.Series([True, True, False])  # left operand


def test_sub_py_scalar() -> None:
    """Test pd.Series[bool] - Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left - b, Never)
    check(assert_type(left - i, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left - f, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left - c, "pd.Series[complex]"), pd.Series, np.complexfloating)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(b - left, Never)
    check(assert_type(i - left, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(f - left, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(c - left, "pd.Series[complex]"), pd.Series, np.complexfloating)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left.sub(b), Never)
    check(assert_type(left.sub(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.sub(f), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.sub(c), "pd.Series[complex]"), pd.Series, np.complexfloating)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left.rsub(b), Never)
    check(assert_type(left.rsub(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rsub(f), "pd.Series[float]"), pd.Series, np.floating)
    check(
        assert_type(left.rsub(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )


def test_sub_py_sequence() -> None:
    """Test pd.Series[bool] - Python native sequence"""
    b, i, f, c = [True, False, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left - b, Never)
    check(assert_type(left - i, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left - f, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left - c, "pd.Series[complex]"), pd.Series, np.complexfloating)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(b - left, Never)
    check(assert_type(i - left, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(f - left, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(c - left, "pd.Series[complex]"), pd.Series, np.complexfloating)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left.sub(b), Never)
    check(assert_type(left.sub(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.sub(f), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.sub(c), "pd.Series[complex]"), pd.Series, np.complexfloating)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left.rsub(b), Never)
    check(assert_type(left.rsub(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rsub(f), "pd.Series[float]"), pd.Series, np.floating)
    check(
        assert_type(left.rsub(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )


def test_sub_numpy_array() -> None:
    """Test pd.Series[bool] - numpy array"""
    b = np.array([True, False, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left - b, Never)
    check(assert_type(left - i, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left - f, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left - c, "pd.Series[complex]"), pd.Series, np.complexfloating)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rsub__` cannot override. At runtime, they return
    # `Series`s with the correct element type.
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(b - left, Never)
    check(assert_type(i - left, "npt.NDArray[np.int64]"), pd.Series, np.integer)
    check(assert_type(f - left, "npt.NDArray[np.float64]"), pd.Series, np.floating)
    check(
        assert_type(c - left, "npt.NDArray[np.complex128]"),
        pd.Series,
        np.complexfloating,
    )

    if TYPE_CHECKING_INVALID_USAGE:
        left.sub(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.sub(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.sub(f), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.sub(c), "pd.Series[complex]"), pd.Series, np.complexfloating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.rsub(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rsub(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rsub(f), "pd.Series[float]"), pd.Series, np.floating)
    check(
        assert_type(left.rsub(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )


def test_sub_pd_series() -> None:
    """Test pd.Series[bool] - pandas series"""
    b = pd.Series([True, False, True])
    i = pd.Series([2, 3, 5])
    f = pd.Series([1.0, 2.0, 3.0])
    c = pd.Series([1.1j, 2.2j, 4.1j])

    if TYPE_CHECKING_INVALID_USAGE:
        _ = left - b  # pyright: ignore[reportOperatorIssue]
    check(assert_type(left - i, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left - f, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left - c, "pd.Series[complex]"), pd.Series, np.complexfloating)

    if TYPE_CHECKING_INVALID_USAGE:
        _ = b - left  # pyright: ignore[reportOperatorIssue]
    check(assert_type(i - left, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(f - left, "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(c - left, "pd.Series[complex]"), pd.Series, np.complexfloating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.sub(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.sub(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.sub(f), "pd.Series[float]"), pd.Series, np.floating)
    check(assert_type(left.sub(c), "pd.Series[complex]"), pd.Series, np.complexfloating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.rsub(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rsub(i), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(left.rsub(f), "pd.Series[float]"), pd.Series, np.floating)
    check(
        assert_type(left.rsub(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
