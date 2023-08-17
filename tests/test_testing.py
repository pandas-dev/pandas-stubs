from __future__ import annotations

import os.path

import pandas as pd
from pandas._testing import ensure_clean
from pandas.testing import (
    assert_frame_equal,
    assert_series_equal,
)
from typing_extensions import assert_type

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)


def test_types_assert_series_equal() -> None:
    s1 = pd.Series([0, 1, 1, 0])
    s2 = pd.Series([0, 1, 1, 0])
    assert_series_equal(left=s1, right=s2)
    assert_series_equal(
        s1,
        s2,
        check_freq=False,
        check_categorical=True,
        check_flags=True,
        check_datetimelike_compat=True,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        assert_series_equal(  # type: ignore[call-overload] # pyright: ignore[reportGeneralTypeIssues]
            s1,
            s2,
            check_dtype=True,
            check_less_precise=True,
            check_names=True,
        )
    assert_series_equal(s1, s2, check_like=True)
    # GH 417
    assert_series_equal(s1, s2, check_index=False)


def test_assert_frame_equal():
    df1 = pd.DataFrame({"x": [1, 2, 3]})
    df2 = pd.DataFrame({"x": [1, 2, 3]})
    # GH 56
    assert_frame_equal(df1, df2, check_index_type=False)


def test_ensure_clean():
    with ensure_clean() as path:
        check(assert_type(path, str), str)
        pd.DataFrame({"x": [1, 2, 3]}).to_csv(path)
    assert not os.path.exists(path)
