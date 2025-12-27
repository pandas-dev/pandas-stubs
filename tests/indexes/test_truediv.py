from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typing_extensions import assert_type

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)


@pytest.fixture
def left_i() -> pd.Index:
    """Left operand"""
    lo = pd.MultiIndex.from_arrays([[1, 2, 3]]).levels[0]
    return check(assert_type(lo, pd.Index), pd.Index, np.integer)


def test_truediv_py_scalar(left_i: pd.Index) -> None:
    """Test pd.Index[Any] (int) / Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j

    check(assert_type(left_i / b, pd.Index), pd.Index)
    check(assert_type(left_i / i, pd.Index), pd.Index)
    check(assert_type(left_i / f, pd.Index), pd.Index)
    check(assert_type(left_i / c, pd.Index), pd.Index)

    check(assert_type(b / left_i, pd.Index), pd.Index)
    check(assert_type(i / left_i, pd.Index), pd.Index)
    check(assert_type(f / left_i, pd.Index), pd.Index)
    check(assert_type(c / left_i, pd.Index), pd.Index)


def test_truediv_py_sequence(left_i: pd.Index) -> None:
    """Test pd.Index[Any] (int) / Python native sequences"""
    b, i, f, c = [True, False, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]

    check(assert_type(left_i / b, pd.Index), pd.Index)
    check(assert_type(left_i / i, pd.Index), pd.Index)
    check(assert_type(left_i / f, pd.Index), pd.Index)
    check(assert_type(left_i / c, pd.Index), pd.Index)

    check(assert_type(b / left_i, pd.Index), pd.Index)
    check(assert_type(i / left_i, pd.Index), pd.Index)
    check(assert_type(f / left_i, pd.Index), pd.Index)
    check(assert_type(c / left_i, pd.Index), pd.Index)


def test_truediv_numpy_array(left_i: pd.Index) -> None:
    """Test pd.Index[Any] (int) / numpy arrays"""
    b = np.array([True, False, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex64)

    check(assert_type(left_i / b, pd.Index), pd.Index)
    check(assert_type(left_i / i, pd.Index), pd.Index)
    check(assert_type(left_i / f, pd.Index), pd.Index)
    check(assert_type(left_i / c, pd.Index), pd.Index)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rtruediv__` cannot override. At runtime, they return
    # `Index`.
    # microsoft/pyright#10924
    check(
        assert_type(b / left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Index,
    )
    check(
        assert_type(i / left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Index,
    )
    check(
        assert_type(f / left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Index,
    )
    check(
        assert_type(c / left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Index,
    )


def test_truediv_pd_index(left_i: pd.Index) -> None:
    """Test pd.Index[Any] (int) / pandas Indexes"""
    a = pd.MultiIndex.from_tuples([(1,), (2,), (3,)]).levels[0]
    b = pd.Index([True, False, True])
    i = pd.Index([2, 3, 5])
    f = pd.Index([1.0, 2.0, 3.0])
    c = pd.Index([1.1j, 2.2j, 4.1j])

    check(assert_type(left_i / a, pd.Index), pd.Index)
    check(assert_type(left_i / b, pd.Index), pd.Index)
    check(assert_type(left_i / i, pd.Index), pd.Index)
    check(assert_type(left_i / f, pd.Index), pd.Index)
    check(assert_type(left_i / c, pd.Index), pd.Index)

    check(assert_type(a / left_i, pd.Index), pd.Index)
    check(assert_type(b / left_i, pd.Index), pd.Index)
    check(assert_type(i / left_i, pd.Index), pd.Index)
    check(assert_type(f / left_i, pd.Index), pd.Index)
    check(assert_type(c / left_i, pd.Index), pd.Index)


def test_truediv_paths(tmp_path: Path) -> None:
    """Test pd.Index of paths / path object.

    Also GH 682."""
    fpath = Path("a.png")
    folders, fpaths = pd.Index([tmp_path, tmp_path]), pd.Index([fpath, fpath])

    check(assert_type(folders / fpath, pd.Index), pd.Index, Path)

    check(assert_type(tmp_path / fpaths, pd.Index), pd.Index, Path)


def test_truediv_path(tmp_path: Path) -> None:
    """Test pd.Index / path object.

    Also GH 682."""
    fnames = pd.Index(["a.png", "b.gz", "c.txt"])

    check(assert_type(fnames / tmp_path, pd.Index), pd.Index, Path)
    check(assert_type(tmp_path / fnames, pd.Index), pd.Index, Path)


def test_truediv_str_py_str(left_i: pd.Index) -> None:
    """Test pd.Index[Any] (int) / Python str"""
    s = "abc"

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left_i / s  # type: ignore[operator] # pyright:ignore[reportOperatorIssue,reportUnknownVariableType]
        _1 = s / left_i  # type: ignore[operator] # pyright:ignore[reportOperatorIssue,reportUnknownVariableType]
