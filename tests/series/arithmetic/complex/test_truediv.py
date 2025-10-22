import numpy as np
from numpy import typing as npt  # noqa: F401
import pandas as pd
import pytest
from typing_extensions import assert_type

from tests import check


@pytest.fixture
def left() -> "pd.Series[complex]":
    """Left operand"""
    lo = pd.Series([1j, 2j, 3j])
    return check(assert_type(lo, "pd.Series[complex]"), pd.Series, np.complexfloating)


def test_truediv_py_scalar(left: "pd.Series[complex]") -> None:
    """Test pd.Series[complex] / Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j

    check(assert_type(left / b, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left / i, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left / f, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left / c, "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(assert_type(b / left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(i / left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(f / left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(c / left, "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(
        assert_type(left.truediv(b), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.truediv(i), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.truediv(f), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.truediv(c), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )

    check(assert_type(left.div(b), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.div(i), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.div(f), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.div(c), "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(
        assert_type(left.rtruediv(b), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.rtruediv(i), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.rtruediv(f), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.rtruediv(c), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )

    check(
        assert_type(left.rdiv(b), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.rdiv(i), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.rdiv(f), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.rdiv(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )


def test_truediv_py_sequence(left: "pd.Series[complex]") -> None:
    """Test pd.Series[complex] / Python native sequences"""
    b, i, f, c = [True, False, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]

    check(assert_type(left / b, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left / i, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left / f, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left / c, "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(assert_type(b / left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(i / left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(f / left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(c / left, "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(
        assert_type(left.truediv(b), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.truediv(i), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.truediv(f), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.truediv(c), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )

    check(assert_type(left.div(b), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.div(i), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.div(f), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.div(c), "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(
        assert_type(left.rtruediv(b), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.rtruediv(i), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.rtruediv(f), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.rtruediv(c), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )

    check(
        assert_type(left.rdiv(b), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.rdiv(i), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.rdiv(f), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.rdiv(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )


def test_truediv_numpy_array(left: "pd.Series[complex]") -> None:
    """Test pd.Series[complex] / numpy arrays"""
    b = np.array([True, False, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)

    check(assert_type(left / b, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left / i, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left / f, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left / c, "pd.Series[complex]"), pd.Series, np.complexfloating)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rtruediv__` cannot override. At runtime, they return
    # `Series` with the correct element type.
    check(
        assert_type(b / left, "npt.NDArray[np.float64]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(i / left, "npt.NDArray[np.float64]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(f / left, "npt.NDArray[np.float64]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(c / left, "npt.NDArray[np.complex128]"),
        pd.Series,
        np.complexfloating,
    )

    check(
        assert_type(left.truediv(b), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.truediv(i), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.truediv(f), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.truediv(c), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )

    check(assert_type(left.div(b), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.div(i), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.div(f), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.div(c), "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(
        assert_type(left.rtruediv(b), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.rtruediv(i), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.rtruediv(f), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.rtruediv(c), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )

    check(
        assert_type(left.rdiv(b), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.rdiv(i), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.rdiv(f), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.rdiv(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )


def test_truediv_pd_index(left: "pd.Series[complex]") -> None:
    """Test pd.Series[complex] / pandas Indexes"""
    b = pd.Index([True, False, True])
    i = pd.Index([2, 3, 5])
    f = pd.Index([1.0, 2.0, 3.0])
    c = pd.Index([1.1j, 2.2j, 4.1j])

    check(assert_type(left / b, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left / i, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left / f, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left / c, "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(assert_type(b / left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(i / left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(f / left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(c / left, "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(
        assert_type(left.truediv(b), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.truediv(i), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.truediv(f), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.truediv(c), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )

    check(assert_type(left.div(b), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.div(i), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.div(f), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.div(c), "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(
        assert_type(left.rtruediv(b), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.rtruediv(i), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.rtruediv(f), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.rtruediv(c), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )

    check(
        assert_type(left.rdiv(b), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.rdiv(i), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.rdiv(f), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.rdiv(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )


def test_truediv_pd_series(left: "pd.Series[complex]") -> None:
    """Test pd.Series[complex] / pandas Series"""
    b = pd.Series([True, False, True])
    i = pd.Series([2, 3, 5])
    f = pd.Series([1.0, 2.0, 3.0])
    c = pd.Series([1.1j, 2.2j, 4.1j])

    check(assert_type(left / b, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left / i, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left / f, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left / c, "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(assert_type(b / left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(i / left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(f / left, "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(c / left, "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(
        assert_type(left.truediv(b), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.truediv(i), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.truediv(f), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.truediv(c), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )

    check(assert_type(left.div(b), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.div(i), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.div(f), "pd.Series[complex]"), pd.Series, np.complexfloating)
    check(assert_type(left.div(c), "pd.Series[complex]"), pd.Series, np.complexfloating)

    check(
        assert_type(left.rtruediv(b), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.rtruediv(i), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.rtruediv(f), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(
        assert_type(left.rtruediv(c), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )

    check(
        assert_type(left.rdiv(b), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.rdiv(i), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.rdiv(f), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
    check(
        assert_type(left.rdiv(c), "pd.Series[complex]"), pd.Series, np.complexfloating
    )
