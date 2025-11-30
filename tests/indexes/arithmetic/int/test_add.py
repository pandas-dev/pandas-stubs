import numpy as np
from numpy import typing as npt  # noqa: F401
import pandas as pd
from typing_extensions import assert_type

from tests import (
    check,
    np_ndarray_int64,
)

# left operand
left = pd.Index([1, 2, 3])


def test_add_py_scalar() -> None:
    """Test pd.Index[int] + Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j

    check(assert_type(left + b, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(left + i, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(left + f, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left + c, "pd.Index[complex]"), pd.Index, np.complexfloating)

    check(assert_type(b + left, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(i + left, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(f + left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(c + left, "pd.Index[complex]"), pd.Index, np.complexfloating)


def test_add_py_sequence() -> None:
    """Test pd.Index[int] + Python native sequences"""
    b, i, f, c = [True, False, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]

    check(assert_type(left + b, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(left + i, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(left + f, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left + c, "pd.Index[complex]"), pd.Index, np.complexfloating)

    check(assert_type(b + left, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(i + left, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(f + left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(c + left, "pd.Index[complex]"), pd.Index, np.complexfloating)


def test_add_numpy_array() -> None:
    """Test pd.Index[int] + numpy arrays"""
    b = np.array([True, False, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)

    check(assert_type(left + b, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(left + i, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(left + f, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left + c, "pd.Index[complex]"), pd.Index, np.complexfloating)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__radd__` cannot override. At runtime, they return
    # `Index`es with the correct element type.
    check(assert_type(b + left, "npt.NDArray[np.bool_]"), pd.Index, np.integer)
    check(assert_type(i + left, np_ndarray_int64), pd.Index, np.integer)
    check(assert_type(f + left, "npt.NDArray[np.float64]"), pd.Index, np.floating)
    check(
        assert_type(c + left, "npt.NDArray[np.complex128]"),
        pd.Index,
        np.complexfloating,
    )


def test_add_pd_index() -> None:
    """Test pd.Index[int] + pandas Indexes"""
    b = pd.Index([True, False, True])
    i = pd.Index([2, 3, 5])
    f = pd.Index([1.0, 2.0, 3.0])
    c = pd.Index([1.1j, 2.2j, 4.1j])

    check(assert_type(left + b, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(left + i, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(left + f, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(left + c, "pd.Index[complex]"), pd.Index, np.complexfloating)

    check(assert_type(b + left, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(i + left, "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(f + left, "pd.Index[float]"), pd.Index, np.floating)
    check(assert_type(c + left, "pd.Index[complex]"), pd.Index, np.complexfloating)
