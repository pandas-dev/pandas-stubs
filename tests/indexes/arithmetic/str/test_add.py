import sys
from typing import Any

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
from tests._typing import np_ndarray_int64

left = pd.Index(["1", "23", "456"])  # left operand


def test_add_py_scalar() -> None:
    """Test pd.Index[str] + Python native 'scalar's"""
    i = 4
    r0 = "right"

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + i  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left + r0, "pd.Index[str]"), pd.Index, str)

    if TYPE_CHECKING_INVALID_USAGE:
        _1 = i + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(r0 + left, "pd.Index[str]"), pd.Index, str)


def test_add_py_sequence() -> None:
    """Test pd.Index[str] + Python native sequences"""
    i = [3, 5, 8]
    r0 = ["a", "bc", "def"]
    r1 = tuple(r0)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + i  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left + r0, "pd.Index[str]"), pd.Index, str)
    check(assert_type(left + r1, "pd.Index[str]"), pd.Index, str)

    if TYPE_CHECKING_INVALID_USAGE:
        _1 = i + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(r0 + left, "pd.Index[str]"), pd.Index, str)
    check(assert_type(r1 + left, "pd.Index[str]"), pd.Index, str)


def test_add_numpy_array() -> None:
    """Test pd.Index[str] + numpy arrays"""
    i = np.array([3, 5, 8], np.int64)
    r0 = np.array(["a", "bc", "def"], np.str_)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left + i, Never)
    check(assert_type(left + r0, "pd.Index[str]"), pd.Index, str)

    # `numpy` typing gives `npt.NDArray[np.int64]` in the static type
    # checking, where our `__radd__` cannot override. At runtime, they return
    # `Index`es.
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(i + left, np_ndarray_int64)
    if sys.version_info >= (3, 11):
        # `numpy` typing gives `npt.NDArray[np.int64]` in the static type
        # checking, where our `__radd__` cannot override. At runtime, they return
        # `Index`es.
        check(assert_type(r0 + left, "npt.NDArray[np.str_]"), pd.Index, str)
    else:
        # Python 3.10 uses NumPy 2.2.6, and it has for r0 ndarray[tuple[int,...], dtype[str_]]
        # Python 3.11+ uses NumPy 2.3.2, and it has for r0 ndarray[tuple[Any,...,dtype[str_]]
        # https://github.com/pandas-dev/pandas-stubs/pull/1274#discussion_r2291498975
        check(assert_type(r0 + left, Any), pd.Index, str)


def test_add_pd_index() -> None:
    """Test pd.Index[str] + pandas Indexes"""
    i = pd.Index([3, 5, 8])
    r0 = pd.Index(["a", "bc", "def"])

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + i  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left + r0, "pd.Index[str]"), pd.Index, str)

    if TYPE_CHECKING_INVALID_USAGE:
        _1 = i + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(r0 + left, "pd.Index[str]"), pd.Index, str)
