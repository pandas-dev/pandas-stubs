import numpy as np
from numpy import typing as npt  # noqa: F401
import pandas as pd
from typing_extensions import assert_type

from tests import check

left = pd.DataFrame({"a": [1, 2, 3]})["a"]  # left operand


def test_add_py_scalar() -> None:
    """Test pd.Series[Any] + Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j

    check(assert_type(left + b, pd.Series), pd.Series)
    check(assert_type(left + i, pd.Series), pd.Series)
    check(assert_type(left + f, pd.Series), pd.Series)
    check(assert_type(left + c, pd.Series), pd.Series)

    check(assert_type(b + left, pd.Series), pd.Series)
    check(assert_type(i + left, pd.Series), pd.Series)
    check(assert_type(f + left, pd.Series), pd.Series)
    check(assert_type(c + left, pd.Series), pd.Series)

    check(assert_type(left.add(b), pd.Series), pd.Series)
    check(assert_type(left.add(i), pd.Series), pd.Series)
    check(assert_type(left.add(f), pd.Series), pd.Series)
    check(assert_type(left.add(c), pd.Series), pd.Series)

    check(assert_type(left.radd(b), pd.Series), pd.Series)
    check(assert_type(left.radd(i), pd.Series), pd.Series)
    check(assert_type(left.radd(f), pd.Series), pd.Series)
    check(assert_type(left.radd(c), pd.Series), pd.Series)


def test_add_py_sequence() -> None:
    """Test pd.Series[Any] + Python native sequence"""
    b, i, f, c = [True, False, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]

    check(assert_type(left + b, pd.Series), pd.Series)
    check(assert_type(left + i, pd.Series), pd.Series)
    check(assert_type(left + f, pd.Series), pd.Series)
    check(assert_type(left + c, pd.Series), pd.Series)

    check(assert_type(b + left, pd.Series), pd.Series)
    check(assert_type(i + left, pd.Series), pd.Series)
    check(assert_type(f + left, pd.Series), pd.Series)
    check(assert_type(c + left, pd.Series), pd.Series)

    check(assert_type(left.add(b), pd.Series), pd.Series)
    check(assert_type(left.add(i), pd.Series), pd.Series)
    check(assert_type(left.add(f), pd.Series), pd.Series)
    check(assert_type(left.add(c), pd.Series), pd.Series)

    check(assert_type(left.radd(b), pd.Series), pd.Series)
    check(assert_type(left.radd(i), pd.Series), pd.Series)
    check(assert_type(left.radd(f), pd.Series), pd.Series)
    check(assert_type(left.radd(c), pd.Series), pd.Series)


def test_add_numpy_array() -> None:
    """Test pd.Series[Any] + numpy array"""
    b = np.array([True, False, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)

    check(assert_type(left + b, pd.Series), pd.Series)
    check(assert_type(left + i, pd.Series), pd.Series)
    check(assert_type(left + f, pd.Series), pd.Series)
    check(assert_type(left + c, pd.Series), pd.Series)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__radd__` cannot override. At runtime, they return
    # `Series`s.
    # `mypy` thinks the return types are `Any`, which is a bug.
    check(
        assert_type(b + left, "npt.NDArray[np.bool_]"), pd.Series  # type: ignore[assert-type]
    )
    check(
        assert_type(i + left, "npt.NDArray[np.int64]"), pd.Series  # type: ignore[assert-type]
    )
    check(
        assert_type(f + left, "npt.NDArray[np.float64]"), pd.Series  # type: ignore[assert-type]
    )
    check(
        assert_type(c + left, "npt.NDArray[np.complex128]"), pd.Series  # type: ignore[assert-type]
    )

    check(assert_type(left.add(b), pd.Series), pd.Series)
    check(assert_type(left.add(i), pd.Series), pd.Series)
    check(assert_type(left.add(f), pd.Series), pd.Series)
    check(assert_type(left.add(c), pd.Series), pd.Series)

    check(assert_type(left.radd(b), pd.Series), pd.Series)
    check(assert_type(left.radd(i), pd.Series), pd.Series)
    check(assert_type(left.radd(f), pd.Series), pd.Series)
    check(assert_type(left.radd(c), pd.Series), pd.Series)


def test_add_pd_series() -> None:
    """Test pd.Series[Any] + pandas series"""
    b = pd.Series([True, False, True])
    i = pd.Series([2, 3, 5])
    f = pd.Series([1.0, 2.0, 3.0])
    c = pd.Series([1.1j, 2.2j, 4.1j])

    check(assert_type(left + b, pd.Series), pd.Series)
    check(assert_type(left + i, pd.Series), pd.Series)
    check(assert_type(left + f, pd.Series), pd.Series)
    check(assert_type(left + c, pd.Series), pd.Series)

    check(assert_type(b + left, pd.Series), pd.Series)
    check(assert_type(i + left, pd.Series), pd.Series)
    check(assert_type(f + left, pd.Series), pd.Series)
    check(assert_type(c + left, pd.Series), pd.Series)

    check(assert_type(left.add(b), pd.Series), pd.Series)
    check(assert_type(left.add(i), pd.Series), pd.Series)
    check(assert_type(left.add(f), pd.Series), pd.Series)
    check(assert_type(left.add(c), pd.Series), pd.Series)

    check(assert_type(left.radd(b), pd.Series), pd.Series)
    check(assert_type(left.radd(i), pd.Series), pd.Series)
    check(assert_type(left.radd(f), pd.Series), pd.Series)
    check(assert_type(left.radd(c), pd.Series), pd.Series)
