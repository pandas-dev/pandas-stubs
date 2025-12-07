import numpy as np
from numpy import typing as npt  # noqa: F401
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
from tests._typing import (
    np_ndarray_bool,
    np_ndarray_int64,
)


@pytest.fixture
def left() -> pd.TimedeltaIndex:
    """Left operand"""
    lo = pd.Index([pd.Timedelta(1, "s")])  # left operand
    return check(assert_type(lo, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)


def test_mul_py_scalar(left: pd.TimedeltaIndex) -> None:
    """Test pd.TimedeltaIndex * Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left * b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left * i, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    check(assert_type(left * f, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left * c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(i * left, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    check(assert_type(f * left, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _13 = c * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]


def test_mul_py_sequence(left: pd.TimedeltaIndex) -> None:
    """Test pd.TimedeltaIndex * Python native sequences"""
    b, i, f, c = [True], [2], [1.5], [1.7j]

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left * b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left * i, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    check(assert_type(left * f, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left * c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(i * left, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    check(assert_type(f * left, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _13 = c * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]


def test_mul_numpy_array(left: pd.TimedeltaIndex) -> None:
    """Test pd.TimedeltaIndex * numpy arrays"""
    b = np.array([True], np.bool_)
    i = np.array([2], np.int64)
    f = np.array([1.5], np.float64)
    c = np.array([1.7j], np.complex128)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left * b, Never)
    check(assert_type(left * i, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    check(assert_type(left * f, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left * c, Never)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rmul__` cannot override. At runtime, they return
    # `Series` with the correct element type.
    if TYPE_CHECKING_INVALID_USAGE:
        # We made it Never, but numpy takes over
        assert_type(b * left, np_ndarray_bool)
    check(assert_type(i * left, np_ndarray_int64), pd.TimedeltaIndex, pd.Timedelta)
    check(
        assert_type(f * left, "npt.NDArray[np.float64]"),
        pd.TimedeltaIndex,
        pd.Timedelta,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(c * left, "npt.NDArray[np.complex128]")


def test_mul_pd_index(left: pd.TimedeltaIndex) -> None:
    """Test pd.TimedeltaIndex * pandas Indexes"""
    b = pd.Index([True])
    i = pd.Index([2])
    f = pd.Index([1.5])
    c = pd.Index([1.7j])

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left * b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left * i, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    check(assert_type(left * f, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left * c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(i * left, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    check(assert_type(f * left, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _13 = c * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
