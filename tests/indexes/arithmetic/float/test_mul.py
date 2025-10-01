from datetime import (
    datetime,
    timedelta,
)
from typing import Any

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


@pytest.fixture
def left() -> "pd.Index[float]":
    """left operand"""
    lo = pd.Index([1.0, 2.0, 3.0])
    return check(assert_type(lo, "pd.Index[float]"), pd.Index, np.floating)


def test_mul_py_scalar(left: "pd.Index[float]") -> None:
    """Test pd.Index[float] * Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j
    s, d = datetime(2025, 9, 30), timedelta(seconds=1)

    check(assert_type(left * b, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left * i, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left * f, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left * c, "pd.Index[complex]"), pd.Index, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        _05 = left * s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left * d, "pd.TimedeltaIndex"), pd.TimedeltaIndex, timedelta)

    check(assert_type(b * left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(i * left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(f * left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(c * left, "pd.Index[complex]"), pd.Index, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        _15 = s * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d * left, "pd.TimedeltaIndex"), pd.TimedeltaIndex, timedelta)


def test_mul_py_sequence(left: "pd.Index[float]") -> None:
    """Test pd.Index[float] * Python native sequences"""
    b, i, f, c = [True, False, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]
    s = [datetime(2025, 9, d) for d in (27, 28, 29)]
    d = [timedelta(seconds=s + 1) for s in range(3)]

    check(assert_type(left * b, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left * i, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left * f, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left * c, "pd.Index[complex]"), pd.Index, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        _05 = left * s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left * d, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)

    check(assert_type(b * left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(i * left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(f * left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(c * left, "pd.Index[complex]"), pd.Index, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        _15 = s * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d * left, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)


def test_mul_numpy_array(left: "pd.Index[float]") -> None:
    """Test pd.Index[float] * numpy arrays"""
    b = np.array([True, False, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)
    s = np.array([np.datetime64(f"2025-09-{d}") for d in (27, 28, 29)], np.datetime64)
    d = np.array([np.timedelta64(s + 1, "s") for s in range(3)], np.timedelta64)

    check(assert_type(left * b, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left * i, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left * f, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left * c, "pd.Index[complex]"), pd.Index, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left * s, Never)
    check(assert_type(left * d, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rmul__` cannot override. At runtime, they return
    # `Series` with the correct element type.
    check(assert_type(b * left, "npt.NDArray[np.bool_]"), pd.Index, np.floating)
    check(assert_type(i * left, "npt.NDArray[np.int64]"), pd.Index, np.floating)
    check(assert_type(f * left, "npt.NDArray[np.float64]"), pd.Index, np.floating)
    check(
        assert_type(c * left, "npt.NDArray[np.complex128]"),
        pd.Index,
        np.complexfloating,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(s * left, Any)
    check(
        assert_type(d * left, "npt.NDArray[np.timedelta64]"),
        pd.TimedeltaIndex,
        pd.Timedelta,
    )


def test_mul_pd_index(left: "pd.Index[float]") -> None:
    """Test pd.Index[float] * pandas Indexes"""
    b = pd.Index([True, False, True])
    i = pd.Index([2, 3, 5])
    f = pd.Index([1.0, 2.0, 3.0])
    c = pd.Index([1.1j, 2.2j, 4.1j])
    s = pd.Index([datetime(2025, 9, d) for d in (27, 28, 29)])
    d = pd.Index([timedelta(seconds=s + 1) for s in range(3)])

    check(assert_type(left * b, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left * i, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left * f, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left * c, "pd.Index[complex]"), pd.Index, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        _05 = left * s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left * d, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)

    check(assert_type(b * left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(i * left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(f * left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(c * left, "pd.Index[complex]"), pd.Index, np.complexfloating)
    if TYPE_CHECKING_INVALID_USAGE:
        _15 = s * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d * left, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
