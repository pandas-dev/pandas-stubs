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
def left() -> "pd.Index[bool]":
    """Left operand"""
    lo = pd.Index([True, False, True])
    return check(assert_type(lo, "pd.Index[bool]"), pd.Index, np.bool_)


def test_truediv_py_scalar(left: "pd.Index[bool]") -> None:
    """Test pd.Index[bool] / Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j

    if TYPE_CHECKING_INVALID_USAGE:
        # TODO: python/mypy#20061
        _00 = left / b  # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(left / i, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left / f, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left / c, "pd.Index[complex]"), pd.Index, np.complexfloating)

    if TYPE_CHECKING_INVALID_USAGE:
        # TODO: python/mypy#20061
        _10 = b / left  # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(i / left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(f / left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(c / left, "pd.Index[complex]"), pd.Index, np.complexfloating)


def test_truediv_py_sequence(left: "pd.Index[bool]") -> None:
    """Test pd.Index[bool] / Python native sequences"""
    b, i, f, c = [True, False, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]

    if TYPE_CHECKING_INVALID_USAGE:
        # TODO: python/mypy#20061
        _00 = left / b  # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(left / i, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left / f, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left / c, "pd.Index[complex]"), pd.Index, np.complexfloating)

    if TYPE_CHECKING_INVALID_USAGE:
        # TODO: python/mypy#20061
        _10 = b / left  # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(i / left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(f / left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(c / left, "pd.Index[complex]"), pd.Index, np.complexfloating)


def test_truediv_numpy_array(left: "pd.Index[bool]") -> None:
    """Test pd.Index[bool] / numpy arrays"""
    b = np.array([True, False, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left / b, Never)
    check(assert_type(left / i, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left / f, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left / c, "pd.Index[complex]"), pd.Index, np.complexfloating)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rtruediv__` cannot override. At runtime, they return
    # `Index` with the correct element type.
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(b / left, "npt.NDArray[np.float64]")
    check(assert_type(i / left, "npt.NDArray[np.float64]"), pd.Index, np.floating)
    check(assert_type(f / left, "npt.NDArray[np.float64]"), pd.Index, np.floating)
    check(
        assert_type(c / left, "npt.NDArray[np.complex128]"),
        pd.Index,
        np.complexfloating,
    )


def test_truediv_pd_index(left: "pd.Index[bool]") -> None:
    """Test pd.Index[bool] / pandas Indexes"""
    b = pd.Index([True, False, True])
    i = pd.Index([2, 3, 5])
    f = pd.Index([1.0, 2.0, 3.0])
    c = pd.Index([1.1j, 2.2j, 4.1j])

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left / b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(left / i, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left / f, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left / c, "pd.Index[complex]"), pd.Index, np.complexfloating)

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(i / left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(f / left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(c / left, "pd.Index[complex]"), pd.Index, np.complexfloating)
