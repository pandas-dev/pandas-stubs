from typing import NoReturn

import numpy as np
from numpy import typing as npt  # noqa: F401
import pandas as pd
from typing_extensions import assert_type

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

# left operand
left_i = pd.MultiIndex.from_tuples([(1,), (2,), (3,)]).levels[0]


def test_sub_i_py_scalar() -> None:
    """Test pd.Index[Any] (int) - Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j

    check(assert_type(left_i - b, pd.Index), pd.Index)
    check(assert_type(left_i - i, pd.Index), pd.Index)
    check(assert_type(left_i - f, pd.Index), pd.Index)
    check(assert_type(left_i - c, pd.Index), pd.Index)

    check(assert_type(b - left_i, pd.Index), pd.Index)
    check(assert_type(i - left_i, pd.Index), pd.Index)
    check(assert_type(f - left_i, pd.Index), pd.Index)
    check(assert_type(c - left_i, pd.Index), pd.Index)


def test_sub_i_py_sequence() -> None:
    """Test pd.Index[Any] (int) - Python native sequence"""
    b, i, f, c = [True, False, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]

    check(assert_type(left_i - b, pd.Index), pd.Index)
    check(assert_type(left_i - i, pd.Index), pd.Index)
    check(assert_type(left_i - f, pd.Index), pd.Index)
    check(assert_type(left_i - c, pd.Index), pd.Index)

    check(assert_type(b - left_i, pd.Index), pd.Index)
    check(assert_type(i - left_i, pd.Index), pd.Index)
    check(assert_type(f - left_i, pd.Index), pd.Index)
    check(assert_type(c - left_i, pd.Index), pd.Index)


def test_sub_i_numpy_array() -> None:
    """Test pd.Index[Any] (int) - numpy array"""
    b = np.array([True, False, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)

    check(assert_type(left_i - b, pd.Index), pd.Index)
    check(assert_type(left_i - i, pd.Index), pd.Index)
    check(assert_type(left_i - f, pd.Index), pd.Index)
    check(assert_type(left_i - c, pd.Index), pd.Index)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rsub__` cannot override. At runtime, they return
    # `Index`s.
    # `mypy` thinks the return types are `Any`, which is a bug.
    check(assert_type(b - left_i, NoReturn), pd.Index)  # type: ignore[assert-type]
    check(
        assert_type(i - left_i, "npt.NDArray[np.int64]"), pd.Index  # type: ignore[assert-type]
    )
    check(
        assert_type(f - left_i, "npt.NDArray[np.float64]"), pd.Index  # type: ignore[assert-type]
    )
    check(
        assert_type(c - left_i, "npt.NDArray[np.complex128]"), pd.Index  # type: ignore[assert-type]
    )


def test_sub_i_pd_index() -> None:
    """Test pd.Index[Any] (int) - pandas index"""
    a = pd.MultiIndex.from_tuples([(1,), (2,), (3,)]).levels[0]
    b = pd.Index([True, False, True])
    i = pd.Index([2, 3, 5])
    f = pd.Index([1.0, 2.0, 3.0])
    c = pd.Index([1.1j, 2.2j, 4.1j])

    check(assert_type(left_i - a, pd.Index), pd.Index)
    check(assert_type(left_i - b, pd.Index), pd.Index)
    check(assert_type(left_i - i, pd.Index), pd.Index)
    check(assert_type(left_i - f, pd.Index), pd.Index)
    check(assert_type(left_i - c, pd.Index), pd.Index)

    check(assert_type(a - left_i, pd.Index), pd.Index)
    check(assert_type(b - left_i, pd.Index), pd.Index)
    check(assert_type(i - left_i, pd.Index), pd.Index)
    check(assert_type(f - left_i, pd.Index), pd.Index)
    check(assert_type(c - left_i, pd.Index), pd.Index)


def test_sub_str_py_str() -> None:
    """Test pd.Series[Any] (int) - Python str"""
    s = "abc"

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left_i - s  # type: ignore[operator] # pyright:ignore[reportOperatorIssue]
        _1 = s - left_i  # type: ignore[operator] # pyright:ignore[reportOperatorIssue]
