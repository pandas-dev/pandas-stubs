import numpy as np
from numpy import typing as npt  # noqa: F401
import pandas as pd
from typing_extensions import assert_type

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

left_i = pd.DataFrame({"a": [1, 2, 3]})["a"]  # left operand


def test_mul_py_scalar() -> None:
    """Test pd.Series[Any] (int) * Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j

    check(assert_type(left_i * b, pd.Series), pd.Series)
    check(assert_type(left_i * i, pd.Series), pd.Series)
    check(assert_type(left_i * f, pd.Series), pd.Series)
    check(assert_type(left_i * c, pd.Series), pd.Series)

    check(assert_type(b * left_i, pd.Series), pd.Series)
    check(assert_type(i * left_i, pd.Series), pd.Series)
    check(assert_type(f * left_i, pd.Series), pd.Series)
    check(assert_type(c * left_i, pd.Series), pd.Series)

    check(assert_type(left_i.mul(b), pd.Series), pd.Series)
    check(assert_type(left_i.mul(i), pd.Series), pd.Series)
    check(assert_type(left_i.mul(f), pd.Series), pd.Series)
    check(assert_type(left_i.mul(c), pd.Series), pd.Series)

    check(assert_type(left_i.rmul(b), pd.Series), pd.Series)
    check(assert_type(left_i.rmul(i), pd.Series), pd.Series)
    check(assert_type(left_i.rmul(f), pd.Series), pd.Series)
    check(assert_type(left_i.rmul(c), pd.Series), pd.Series)


def test_mul_py_sequence() -> None:
    """Test pd.Series[Any] (int) * Python native sequence"""
    b, i, f, c = [True, False, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]

    check(assert_type(left_i * b, pd.Series), pd.Series)
    check(assert_type(left_i * i, pd.Series), pd.Series)
    check(assert_type(left_i * f, pd.Series), pd.Series)
    check(assert_type(left_i * c, pd.Series), pd.Series)

    check(assert_type(b * left_i, pd.Series), pd.Series)
    check(assert_type(i * left_i, pd.Series), pd.Series)
    check(assert_type(f * left_i, pd.Series), pd.Series)
    check(assert_type(c * left_i, pd.Series), pd.Series)

    check(assert_type(left_i.mul(b), pd.Series), pd.Series)
    check(assert_type(left_i.mul(i), pd.Series), pd.Series)
    check(assert_type(left_i.mul(f), pd.Series), pd.Series)
    check(assert_type(left_i.mul(c), pd.Series), pd.Series)

    check(assert_type(left_i.rmul(b), pd.Series), pd.Series)
    check(assert_type(left_i.rmul(i), pd.Series), pd.Series)
    check(assert_type(left_i.rmul(f), pd.Series), pd.Series)
    check(assert_type(left_i.rmul(c), pd.Series), pd.Series)


def test_mul_numpy_array() -> None:
    """Test pd.Series[Any] (int) * numpy array"""
    b = np.array([True, False, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)

    check(assert_type(left_i * b, pd.Series), pd.Series)
    check(assert_type(left_i * i, pd.Series), pd.Series)
    check(assert_type(left_i * f, pd.Series), pd.Series)
    check(assert_type(left_i * c, pd.Series), pd.Series)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rmul__` cannot override. At runtime, they return
    # `Series`s.
    # `mypy` thinks the return types are `Any`, which is a bug.
    check(
        assert_type(b * left_i, "npt.NDArray[np.bool_]"), pd.Series  # type: ignore[assert-type]
    )
    check(
        assert_type(i * left_i, "npt.NDArray[np.int64]"), pd.Series  # type: ignore[assert-type]
    )
    check(
        assert_type(f * left_i, "npt.NDArray[np.float64]"), pd.Series  # type: ignore[assert-type]
    )
    check(
        assert_type(c * left_i, "npt.NDArray[np.complex128]"), pd.Series  # type: ignore[assert-type]
    )

    check(assert_type(left_i.mul(b), pd.Series), pd.Series)
    check(assert_type(left_i.mul(i), pd.Series), pd.Series)
    check(assert_type(left_i.mul(f), pd.Series), pd.Series)
    check(assert_type(left_i.mul(c), pd.Series), pd.Series)

    check(assert_type(left_i.rmul(b), pd.Series), pd.Series)
    check(assert_type(left_i.rmul(i), pd.Series), pd.Series)
    check(assert_type(left_i.rmul(f), pd.Series), pd.Series)
    check(assert_type(left_i.rmul(c), pd.Series), pd.Series)


def test_mul_pd_series() -> None:
    """Test pd.Series[Any] (int) * pandas series"""
    a = pd.DataFrame({"a": [1, 2, 3]})["a"]
    b = pd.Series([True, False, True])
    i = pd.Series([2, 3, 5])
    f = pd.Series([1.0, 2.0, 3.0])
    c = pd.Series([1.1j, 2.2j, 4.1j])

    check(assert_type(left_i * a, pd.Series), pd.Series)
    check(assert_type(left_i * b, pd.Series), pd.Series)
    check(assert_type(left_i * i, pd.Series), pd.Series)
    check(assert_type(left_i * f, pd.Series), pd.Series)
    check(assert_type(left_i * c, pd.Series), pd.Series)

    check(assert_type(a * left_i, pd.Series), pd.Series)
    check(assert_type(b * left_i, pd.Series), pd.Series)
    check(assert_type(i * left_i, pd.Series), pd.Series)
    check(assert_type(f * left_i, pd.Series), pd.Series)
    check(assert_type(c * left_i, pd.Series), pd.Series)

    check(assert_type(left_i.mul(a), pd.Series), pd.Series)
    check(assert_type(left_i.mul(b), pd.Series), pd.Series)
    check(assert_type(left_i.mul(i), pd.Series), pd.Series)
    check(assert_type(left_i.mul(f), pd.Series), pd.Series)
    check(assert_type(left_i.mul(c), pd.Series), pd.Series)

    check(assert_type(left_i.rmul(a), pd.Series), pd.Series)
    check(assert_type(left_i.rmul(b), pd.Series), pd.Series)
    check(assert_type(left_i.rmul(i), pd.Series), pd.Series)
    check(assert_type(left_i.rmul(f), pd.Series), pd.Series)
    check(assert_type(left_i.rmul(c), pd.Series), pd.Series)


def test_mul_str_py_str() -> None:
    """Test pd.Series[Any] (int) * Python str"""
    s = "abc"

    if TYPE_CHECKING_INVALID_USAGE:
        left_i * s  # type: ignore[operator] # pyright:ignore[reportOperatorIssue]
        s * left_i  # type: ignore[operator] # pyright:ignore[reportOperatorIssue]
        left_i.mul(s)  # type: ignore[type-var] # pyright: ignore[reportArgumentType,reportCallIssue]
        left_i.rmul(s)  # type: ignore[type-var] # pyright: ignore[reportArgumentType,reportCallIssue]
