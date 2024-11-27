"""Test module for classes in pandas.api.typing."""

from typing import (
    TYPE_CHECKING,
    Literal,
    assert_type,
)

import numpy as np
import pandas as pd
from pandas.api.typing import (
    DataFrameGroupBy,
    DatetimeIndexResamplerGroupby,
    FrozenList,
    NAType,
    SeriesGroupBy,
)

from pandas._typing import Scalar

from tests import check

if TYPE_CHECKING:
    from pandas.core.groupby.groupby import _ResamplerGroupBy  # noqa: F401


def test_dataframegroupby():
    df = pd.DataFrame({"a": [1, 2, 3]})
    check(
        assert_type(df.groupby("a"), DataFrameGroupBy[Scalar, Literal[True]]),
        DataFrameGroupBy,
    )


def test_seriesgroupby():
    sr: pd.Series[int] = pd.Series([1, 2, 3], index=pd.Index(["a", "b", "a"]))
    check(
        assert_type(sr.groupby("a"), SeriesGroupBy[int, Scalar]),
        SeriesGroupBy,
    )


def test_frozenlist() -> None:
    flst = FrozenList([1, 2, 3])
    check(assert_type(flst, FrozenList), FrozenList)


def tests_datetimeindexersamplergroupby() -> None:
    idx = pd.date_range("1999-1-1", periods=365, freq="D")
    df = pd.DataFrame(
        np.random.standard_normal((365, 2)), index=idx, columns=["col1", "col2"]
    )
    gb_df = df.groupby("col2")
    check(
        assert_type(gb_df.resample("ME"), "_ResamplerGroupBy[pd.DataFrame]"),
        DatetimeIndexResamplerGroupby,
        pd.DataFrame,
    )


def test_natype() -> None:
    i64dt = pd.Int64Dtype()
    check(assert_type(i64dt.na_value, NAType), NAType)
