from typing import (
    Any,
    cast,
)

import numpy as np
import pandas as pd
from typing_extensions import (
    Never,
    assert_type,
)

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

# left operands
left_i = pd.DataFrame({"a": [1, 2, 3]})["a"]
left_str = pd.DataFrame({"a": ["1", "2", "3_"]})["a"]


def test_add_i_py_scalar() -> None:
    """Test pd.Series[Any] (int) + Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j

    check(assert_type(left_i + b, pd.Series), pd.Series)
    check(assert_type(left_i + i, pd.Series), pd.Series)
    check(assert_type(left_i + f, pd.Series), pd.Series)
    check(assert_type(left_i + c, pd.Series), pd.Series)

    check(assert_type(b + left_i, pd.Series), pd.Series)
    check(assert_type(i + left_i, pd.Series), pd.Series)
    check(assert_type(f + left_i, pd.Series), pd.Series)
    check(assert_type(c + left_i, pd.Series), pd.Series)

    check(assert_type(left_i.add(b), pd.Series), pd.Series)
    check(assert_type(left_i.add(i), pd.Series), pd.Series)
    check(assert_type(left_i.add(f), pd.Series), pd.Series)
    check(assert_type(left_i.add(c), pd.Series), pd.Series)

    check(assert_type(left_i.radd(b), pd.Series), pd.Series)
    check(assert_type(left_i.radd(i), pd.Series), pd.Series)
    check(assert_type(left_i.radd(f), pd.Series), pd.Series)
    check(assert_type(left_i.radd(c), pd.Series), pd.Series)


def test_add_i_py_sequence() -> None:
    """Test pd.Series[Any] (int) + Python native sequence"""
    # mypy believe it is already list[Any], whereas pyright gives list[bool | int | complex]
    a = cast(list[Any], [True, 3, 5j])  # type: ignore[redundant-cast]
    b = [True, False, True]
    i = [2, 3, 5]
    f = [1.0, 2.0, 3.0]
    c = [1j, 1j, 4j]

    check(assert_type(left_i + a, pd.Series), pd.Series)
    check(assert_type(left_i + b, pd.Series), pd.Series)
    check(assert_type(left_i + i, pd.Series), pd.Series)
    check(assert_type(left_i + f, pd.Series), pd.Series)
    check(assert_type(left_i + c, pd.Series), pd.Series)

    check(assert_type(a + left_i, pd.Series), pd.Series)
    check(assert_type(b + left_i, pd.Series), pd.Series)
    check(assert_type(i + left_i, pd.Series), pd.Series)
    check(assert_type(f + left_i, pd.Series), pd.Series)
    check(assert_type(c + left_i, pd.Series), pd.Series)

    check(assert_type(left_i.add(a), pd.Series), pd.Series)
    check(assert_type(left_i.add(b), pd.Series), pd.Series)
    check(assert_type(left_i.add(i), pd.Series), pd.Series)
    check(assert_type(left_i.add(f), pd.Series), pd.Series)
    check(assert_type(left_i.add(c), pd.Series), pd.Series)

    check(assert_type(left_i.radd(a), pd.Series), pd.Series)
    check(assert_type(left_i.radd(b), pd.Series), pd.Series)
    check(assert_type(left_i.radd(i), pd.Series), pd.Series)
    check(assert_type(left_i.radd(f), pd.Series), pd.Series)
    check(assert_type(left_i.radd(c), pd.Series), pd.Series)


def test_add_i_numpy_array() -> None:
    """Test pd.Series[Any] (int) + numpy array"""
    b = np.array([True, False, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)

    check(assert_type(left_i + b, pd.Series), pd.Series)
    check(assert_type(left_i + i, pd.Series), pd.Series)
    check(assert_type(left_i + f, pd.Series), pd.Series)
    check(assert_type(left_i + c, pd.Series), pd.Series)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__radd__` cannot override. At runtime, they return
    # `Series`.
    # microsoft/pyright#10924
    check(
        assert_type(b + left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Series,
    )
    check(
        assert_type(i + left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Series,
    )
    check(
        assert_type(f + left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Series,
    )
    check(
        assert_type(c + left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Series,
    )

    check(assert_type(left_i.add(b), pd.Series), pd.Series)
    check(assert_type(left_i.add(i), pd.Series), pd.Series)
    check(assert_type(left_i.add(f), pd.Series), pd.Series)
    check(assert_type(left_i.add(c), pd.Series), pd.Series)

    check(assert_type(left_i.radd(b), pd.Series), pd.Series)
    check(assert_type(left_i.radd(i), pd.Series), pd.Series)
    check(assert_type(left_i.radd(f), pd.Series), pd.Series)
    check(assert_type(left_i.radd(c), pd.Series), pd.Series)


def test_add_i_pd_series() -> None:
    """Test pd.Series[Any] (int) + pandas series"""
    a = pd.DataFrame({"a": [1, 2, 3]})["a"]
    b = pd.Series([True, False, True])
    i = pd.Series([2, 3, 5])
    f = pd.Series([1.0, 2.0, 3.0])
    c = pd.Series([1.1j, 2.2j, 4.1j])

    check(assert_type(left_i + a, pd.Series), pd.Series)
    check(assert_type(left_i + b, pd.Series), pd.Series)
    check(assert_type(left_i + i, pd.Series), pd.Series)
    check(assert_type(left_i + f, pd.Series), pd.Series)
    check(assert_type(left_i + c, pd.Series), pd.Series)

    check(assert_type(a + left_i, pd.Series), pd.Series)
    check(assert_type(b + left_i, pd.Series), pd.Series)
    check(assert_type(i + left_i, pd.Series), pd.Series)
    check(assert_type(f + left_i, pd.Series), pd.Series)
    check(assert_type(c + left_i, pd.Series), pd.Series)

    check(assert_type(left_i.add(a), pd.Series), pd.Series)
    check(assert_type(left_i.add(b), pd.Series), pd.Series)
    check(assert_type(left_i.add(i), pd.Series), pd.Series)
    check(assert_type(left_i.add(f), pd.Series), pd.Series)
    check(assert_type(left_i.add(c), pd.Series), pd.Series)

    check(assert_type(left_i.radd(a), pd.Series), pd.Series)
    check(assert_type(left_i.radd(b), pd.Series), pd.Series)
    check(assert_type(left_i.radd(i), pd.Series), pd.Series)
    check(assert_type(left_i.radd(f), pd.Series), pd.Series)
    check(assert_type(left_i.radd(c), pd.Series), pd.Series)


def test_add_i_pd_index() -> None:
    """Test pd.Series[Any] (int) + pandas index"""
    a = pd.MultiIndex.from_tuples([(1,), (2,), (3,)]).levels[0]
    b = pd.Index([True, False, True])
    i = pd.Index([2, 3, 5])
    f = pd.Index([1.0, 2.0, 3.0])
    c = pd.Index([1.1j, 2.2j, 4.1j])

    check(assert_type(left_i + a, pd.Series), pd.Series)
    check(assert_type(left_i + b, pd.Series), pd.Series)
    check(assert_type(left_i + i, pd.Series), pd.Series)
    check(assert_type(left_i + f, pd.Series), pd.Series)
    check(assert_type(left_i + c, pd.Series), pd.Series)

    check(assert_type(a + left_i, pd.Series), pd.Series)
    check(assert_type(b + left_i, pd.Series), pd.Series)
    check(assert_type(i + left_i, pd.Series), pd.Series)
    check(assert_type(f + left_i, pd.Series), pd.Series)
    check(assert_type(c + left_i, pd.Series), pd.Series)

    check(assert_type(left_i.add(a), pd.Series), pd.Series)
    check(assert_type(left_i.add(b), pd.Series), pd.Series)
    check(assert_type(left_i.add(i), pd.Series), pd.Series)
    check(assert_type(left_i.add(f), pd.Series), pd.Series)
    check(assert_type(left_i.add(c), pd.Series), pd.Series)

    check(assert_type(left_i.radd(a), pd.Series), pd.Series)
    check(assert_type(left_i.radd(b), pd.Series), pd.Series)
    check(assert_type(left_i.radd(i), pd.Series), pd.Series)
    check(assert_type(left_i.radd(f), pd.Series), pd.Series)
    check(assert_type(left_i.radd(c), pd.Series), pd.Series)


def test_add_i_py_str() -> None:
    """Test pd.Series[Any] (int) + Python str"""
    s = "abc"

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left_i + s, Never)
        assert_type(s + left_i, Never)

        def _type_checking_enabler_0() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(left_i.add(s), Never)

        def _type_checking_enabler_1() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(left_i.radd(s), Never)
