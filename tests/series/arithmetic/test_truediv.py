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
def left_i() -> pd.Series:
    """Left operand"""
    lo = pd.DataFrame({"a": [1, 2, 3]})["a"]
    return check(assert_type(lo, pd.Series), pd.Series, np.integer)


def test_truediv_py_scalar(left_i: pd.Series) -> None:
    """Test pd.Series[Any] (int) / Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j

    check(assert_type(left_i / b, pd.Series), pd.Series)
    check(assert_type(left_i / i, pd.Series), pd.Series)
    check(assert_type(left_i / f, pd.Series), pd.Series)
    check(assert_type(left_i / c, pd.Series), pd.Series)

    check(assert_type(b / left_i, pd.Series), pd.Series)
    check(assert_type(i / left_i, pd.Series), pd.Series)
    check(assert_type(f / left_i, pd.Series), pd.Series)
    check(assert_type(c / left_i, pd.Series), pd.Series)

    check(assert_type(left_i.truediv(b), pd.Series), pd.Series)
    check(assert_type(left_i.truediv(i), pd.Series), pd.Series)
    check(assert_type(left_i.truediv(f), pd.Series), pd.Series)
    check(assert_type(left_i.truediv(c), pd.Series), pd.Series)

    check(assert_type(left_i.div(b), pd.Series), pd.Series)
    check(assert_type(left_i.div(i), pd.Series), pd.Series)
    check(assert_type(left_i.div(f), pd.Series), pd.Series)
    check(assert_type(left_i.div(c), pd.Series), pd.Series)

    check(assert_type(left_i.rtruediv(b), pd.Series), pd.Series)
    check(assert_type(left_i.rtruediv(i), pd.Series), pd.Series)
    check(assert_type(left_i.rtruediv(f), pd.Series), pd.Series)
    check(assert_type(left_i.rtruediv(c), pd.Series), pd.Series)

    check(assert_type(left_i.rdiv(b), pd.Series), pd.Series)
    check(assert_type(left_i.rdiv(i), pd.Series), pd.Series)
    check(assert_type(left_i.rdiv(f), pd.Series), pd.Series)
    check(assert_type(left_i.rdiv(c), pd.Series), pd.Series)


def test_truediv_py_sequence(left_i: pd.Series) -> None:
    """Test pd.Series[Any] (int) / Python native sequences"""
    b, i, f, c = [True, False, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]

    check(assert_type(left_i / b, pd.Series), pd.Series)
    check(assert_type(left_i / i, pd.Series), pd.Series)
    check(assert_type(left_i / f, pd.Series), pd.Series)
    check(assert_type(left_i / c, pd.Series), pd.Series)

    check(assert_type(b / left_i, pd.Series), pd.Series)
    check(assert_type(i / left_i, pd.Series), pd.Series)
    check(assert_type(f / left_i, pd.Series), pd.Series)
    check(assert_type(c / left_i, pd.Series), pd.Series)

    check(assert_type(left_i.truediv(b), pd.Series), pd.Series)
    check(assert_type(left_i.truediv(i), pd.Series), pd.Series)
    check(assert_type(left_i.truediv(f), pd.Series), pd.Series)
    check(assert_type(left_i.truediv(c), pd.Series), pd.Series)

    check(assert_type(left_i.div(b), pd.Series), pd.Series)
    check(assert_type(left_i.div(i), pd.Series), pd.Series)
    check(assert_type(left_i.div(f), pd.Series), pd.Series)
    check(assert_type(left_i.div(c), pd.Series), pd.Series)

    check(assert_type(left_i.rtruediv(b), pd.Series), pd.Series)
    check(assert_type(left_i.rtruediv(i), pd.Series), pd.Series)
    check(assert_type(left_i.rtruediv(f), pd.Series), pd.Series)
    check(assert_type(left_i.rtruediv(c), pd.Series), pd.Series)

    check(assert_type(left_i.rdiv(b), pd.Series), pd.Series)
    check(assert_type(left_i.rdiv(i), pd.Series), pd.Series)
    check(assert_type(left_i.rdiv(f), pd.Series), pd.Series)
    check(assert_type(left_i.rdiv(c), pd.Series), pd.Series)


def test_truediv_numpy_array(left_i: pd.Series) -> None:
    """Test pd.Series[Any] (int) / numpy arrays"""
    b = np.array([True, False, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex64)

    check(assert_type(left_i / b, pd.Series), pd.Series)
    check(assert_type(left_i / i, pd.Series), pd.Series)
    check(assert_type(left_i / f, pd.Series), pd.Series)
    check(assert_type(left_i / c, pd.Series), pd.Series)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rtruediv__` cannot override. At runtime, they return
    # `Series`.
    # microsoft/pyright#10924
    check(
        assert_type(b / left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Series,
    )
    check(
        assert_type(i / left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Series,
    )
    check(
        assert_type(f / left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Series,
    )
    check(
        assert_type(c / left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Series,
    )

    check(assert_type(left_i.truediv(b), pd.Series), pd.Series)
    check(assert_type(left_i.truediv(i), pd.Series), pd.Series)
    check(assert_type(left_i.truediv(f), pd.Series), pd.Series)
    check(assert_type(left_i.truediv(c), pd.Series), pd.Series)

    check(assert_type(left_i.div(b), pd.Series), pd.Series)
    check(assert_type(left_i.div(i), pd.Series), pd.Series)
    check(assert_type(left_i.div(f), pd.Series), pd.Series)
    check(assert_type(left_i.div(c), pd.Series), pd.Series)

    check(assert_type(left_i.rtruediv(b), pd.Series), pd.Series)
    check(assert_type(left_i.rtruediv(i), pd.Series), pd.Series)
    check(assert_type(left_i.rtruediv(f), pd.Series), pd.Series)
    check(assert_type(left_i.rtruediv(c), pd.Series), pd.Series)

    check(assert_type(left_i.rdiv(b), pd.Series), pd.Series)
    check(assert_type(left_i.rdiv(i), pd.Series), pd.Series)
    check(assert_type(left_i.rdiv(f), pd.Series), pd.Series)
    check(assert_type(left_i.rdiv(c), pd.Series), pd.Series)


def test_truediv_pd_index(left_i: pd.Series) -> None:
    """Test pd.Series[Any] (int) / pandas Indexes"""
    a = pd.MultiIndex.from_tuples([(1,), (2,), (3,)]).levels[0]
    b = pd.Index([True, False, True])
    i = pd.Index([2, 3, 5])
    f = pd.Index([1.0, 2.0, 3.0])
    c = pd.Index([1.1j, 2.2j, 4.1j])

    check(assert_type(left_i / a, pd.Series), pd.Series)
    check(assert_type(left_i / b, pd.Series), pd.Series)
    check(assert_type(left_i / i, pd.Series), pd.Series)
    check(assert_type(left_i / f, pd.Series), pd.Series)
    check(assert_type(left_i / c, pd.Series), pd.Series)

    check(assert_type(a / left_i, pd.Series), pd.Series)
    check(assert_type(b / left_i, pd.Series), pd.Series)
    check(assert_type(i / left_i, pd.Series), pd.Series)
    check(assert_type(f / left_i, pd.Series), pd.Series)
    check(assert_type(c / left_i, pd.Series), pd.Series)

    check(assert_type(left_i.truediv(a), pd.Series), pd.Series)
    check(assert_type(left_i.truediv(b), pd.Series), pd.Series)
    check(assert_type(left_i.truediv(i), pd.Series), pd.Series)
    check(assert_type(left_i.truediv(f), pd.Series), pd.Series)
    check(assert_type(left_i.truediv(c), pd.Series), pd.Series)

    check(assert_type(left_i.div(a), pd.Series), pd.Series)
    check(assert_type(left_i.div(b), pd.Series), pd.Series)
    check(assert_type(left_i.div(i), pd.Series), pd.Series)
    check(assert_type(left_i.div(f), pd.Series), pd.Series)
    check(assert_type(left_i.div(c), pd.Series), pd.Series)

    check(assert_type(left_i.rtruediv(a), pd.Series), pd.Series)
    check(assert_type(left_i.rtruediv(b), pd.Series), pd.Series)
    check(assert_type(left_i.rtruediv(i), pd.Series), pd.Series)
    check(assert_type(left_i.rtruediv(f), pd.Series), pd.Series)
    check(assert_type(left_i.rtruediv(c), pd.Series), pd.Series)

    check(assert_type(left_i.rdiv(a), pd.Series), pd.Series)
    check(assert_type(left_i.rdiv(b), pd.Series), pd.Series)
    check(assert_type(left_i.rdiv(i), pd.Series), pd.Series)
    check(assert_type(left_i.rdiv(f), pd.Series), pd.Series)
    check(assert_type(left_i.rdiv(c), pd.Series), pd.Series)


def test_truediv_pd_series(left_i: pd.Series) -> None:
    """Test pd.Series[Any] (int) / pandas Series"""
    a = pd.DataFrame({"a": [1, 2, 3]})["a"]
    b = pd.Series([True, False, True])
    i = pd.Series([2, 3, 5])
    f = pd.Series([1.0, 2.0, 3.0])
    c = pd.Series([1.1j, 2.2j, 4.1j])

    check(assert_type(left_i / a, pd.Series), pd.Series)
    check(assert_type(left_i / b, pd.Series), pd.Series)
    check(assert_type(left_i / i, pd.Series), pd.Series)
    check(assert_type(left_i / f, pd.Series), pd.Series)
    check(assert_type(left_i / c, pd.Series), pd.Series)

    check(assert_type(a / left_i, pd.Series), pd.Series)
    check(assert_type(b / left_i, pd.Series), pd.Series)
    check(assert_type(i / left_i, pd.Series), pd.Series)
    check(assert_type(f / left_i, pd.Series), pd.Series)
    check(assert_type(c / left_i, pd.Series), pd.Series)

    check(assert_type(left_i.truediv(a), pd.Series), pd.Series)
    check(assert_type(left_i.truediv(b), pd.Series), pd.Series)
    check(assert_type(left_i.truediv(i), pd.Series), pd.Series)
    check(assert_type(left_i.truediv(f), pd.Series), pd.Series)
    check(assert_type(left_i.truediv(c), pd.Series), pd.Series)

    check(assert_type(left_i.div(a), pd.Series), pd.Series)
    check(assert_type(left_i.div(b), pd.Series), pd.Series)
    check(assert_type(left_i.div(i), pd.Series), pd.Series)
    check(assert_type(left_i.div(f), pd.Series), pd.Series)
    check(assert_type(left_i.div(c), pd.Series), pd.Series)

    check(assert_type(left_i.rtruediv(a), pd.Series), pd.Series)
    check(assert_type(left_i.rtruediv(b), pd.Series), pd.Series)
    check(assert_type(left_i.rtruediv(i), pd.Series), pd.Series)
    check(assert_type(left_i.rtruediv(f), pd.Series), pd.Series)
    check(assert_type(left_i.rtruediv(c), pd.Series), pd.Series)

    check(assert_type(left_i.rdiv(a), pd.Series), pd.Series)
    check(assert_type(left_i.rdiv(b), pd.Series), pd.Series)
    check(assert_type(left_i.rdiv(i), pd.Series), pd.Series)
    check(assert_type(left_i.rdiv(f), pd.Series), pd.Series)
    check(assert_type(left_i.rdiv(c), pd.Series), pd.Series)


def test_truediv_paths(tmp_path: Path) -> None:
    """Test pd.Series of paths / path object.

    Also GH 682."""
    fpath = Path("a.png")
    folders, fpaths = pd.Series([tmp_path, tmp_path]), pd.Series([fpath, fpath])

    check(assert_type(folders / fpath, pd.Series), pd.Series, Path)
    check(assert_type(folders.truediv(fpath), pd.Series), pd.Series, Path)
    check(assert_type(folders.div(fpath), pd.Series), pd.Series, Path)

    # mypy thinks it's `Path`, in contrast to Series.__rtruediv__(self, other: Path) -> Series: ...
    check(assert_type(tmp_path / fpaths, pd.Series), pd.Series, Path)  # type: ignore[assert-type]
    check(assert_type(fpaths.rtruediv(tmp_path), pd.Series), pd.Series, Path)
    check(assert_type(fpaths.rdiv(tmp_path), pd.Series), pd.Series, Path)


def test_truediv_path(tmp_path: Path) -> None:
    """Test pd.Series / path object.

    Also GH 682."""
    fnames = pd.Series(["a.png", "b.gz", "c.txt"])

    check(assert_type(fnames / tmp_path, pd.Series), pd.Series, Path)
    check(assert_type(tmp_path / fnames, pd.Series), pd.Series, Path)

    check(assert_type(fnames.truediv(tmp_path), pd.Series), pd.Series, Path)
    check(assert_type(fnames.div(tmp_path), pd.Series), pd.Series, Path)

    check(assert_type(fnames.rtruediv(tmp_path), pd.Series), pd.Series, Path)
    check(assert_type(fnames.rdiv(tmp_path), pd.Series), pd.Series, Path)


def test_truediv_str_py_str(left_i: pd.Series) -> None:
    """Test pd.Series[Any] (int) / Python str"""
    s = "abc"

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left_i / s  # type: ignore[operator] # pyright:ignore[reportOperatorIssue]
        _1 = s / left_i  # type: ignore[operator] # pyright:ignore[reportOperatorIssue]

        left_i.truediv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left_i.div(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]

        left_i.rtruediv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left_i.rdiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
