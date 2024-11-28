"""Test module for classes in pandas.api.typing."""

import numpy as np
import pandas as pd
from pandas.api.typing import (
    DataFrameGroupBy,
    DatetimeIndexResamplerGroupby,
    NAType,
    SeriesGroupBy,
)
from typing_extensions import assert_type

from tests import check


def test_dataframegroupby():
    df = pd.DataFrame({"a": [1, 2, 3]})
    check(
        assert_type(df.groupby("a"), DataFrameGroupBy),
        DataFrameGroupBy,
    )


def test_seriesgroupby():
    sr: pd.Series[int] = pd.Series([1, 2, 3], index=pd.Index(["a", "b", "a"]))
    check(
        assert_type(sr.groupby(level=0), SeriesGroupBy),
        SeriesGroupBy,
    )


def tests_datetimeindexersamplergroupby() -> None:
    idx = pd.date_range("1999-1-1", periods=365, freq="D")
    df = pd.DataFrame(
        np.random.standard_normal((365, 2)), index=idx, columns=["col1", "col2"]
    )
    gb_df = df.groupby("col2")
    check(
        assert_type(gb_df.resample("ME"), DatetimeIndexResamplerGroupby),
        DatetimeIndexResamplerGroupby,
        pd.DataFrame,
    )


def test_natype() -> None:
    i64dt = pd.Int64Dtype()
    check(assert_type(i64dt.na_value, NAType), NAType)
