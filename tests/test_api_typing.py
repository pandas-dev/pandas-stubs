"""Test module for classes in pandas.api.typing."""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas._testing import ensure_clean
from pandas.api.typing import (
    DataFrameGroupBy,
    DatetimeIndexResamplerGroupby,
    Expanding,
    ExpandingGroupby,
    ExponentialMovingWindow,
    ExponentialMovingWindowGroupby,
    JsonReader,
    NaTType,
    NAType,
    PeriodIndexResamplerGroupby,
    Resampler,
    Rolling,
    RollingGroupby,
    SeriesGroupBy,
    TimedeltaIndexResamplerGroupby,
    TimeGrouper,
    Window,
)
import pytest
from typing_extensions import assert_type

from tests import check

from pandas.io.json._json import read_json
from pandas.io.stata import StataReader


def test_dataframegroupby():
    df = pd.DataFrame({"a": [1, 2, 3]})
    group = df.groupby("a")

    def f1(gb: DataFrameGroupBy):
        check(gb, DataFrameGroupBy)

    f1(group)


def test_seriesgroupby():
    sr = pd.Series([1, 2, 3], index=pd.Index(["a", "b", "a"]))

    def f1(gb: SeriesGroupBy):
        check(gb, SeriesGroupBy)

    f1(sr.groupby(level=0))


def tests_datetimeindexersamplergroupby() -> None:
    idx = pd.date_range("1999-1-1", periods=365, freq="D")
    df = pd.DataFrame(
        np.random.standard_normal((365, 2)), index=idx, columns=["col1", "col2"]
    )
    gb_df = df.groupby("col2")
    check(gb_df.resample("ME"), DatetimeIndexResamplerGroupby)


def test_timedeltaindexresamplergroupby() -> None:
    idx = pd.TimedeltaIndex(["0 days", "1 days", "2 days", "3 days", "4 days"])
    df = pd.DataFrame(
        np.random.standard_normal((5, 2)), index=idx, columns=["col1", "col2"]
    )
    gb_df = df.groupby("col2")
    check(gb_df.resample("1D"), TimedeltaIndexResamplerGroupby)


def test_natype() -> None:
    i64dt = pd.Int64Dtype()
    check(assert_type(i64dt.na_value, NAType), NAType)


def test_nattype() -> None:
    td = pd.Timedelta("1 day")
    as_nat = pd.NaT
    check(assert_type(td + as_nat, NaTType), NaTType)


def test_expanding() -> None:
    df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
    check(df.expanding(), Expanding)


def test_expanding_groubpy() -> None:
    df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
    check(df.groupby("B").expanding(), ExpandingGroupby)


def test_ewm() -> None:
    df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
    check(df.ewm(2), ExponentialMovingWindow)


def test_ewm_groubpy() -> None:
    df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
    check(df.groupby("B").ewm(2), ExponentialMovingWindowGroupby)


def test_json_reader() -> None:
    df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})

    with ensure_clean() as path:
        check(assert_type(df.to_json(path), None), type(None))
        json_reader = read_json(path, chunksize=1, lines=True)
        check(json_reader, JsonReader)
        json_reader.close()


@pytest.mark.skip("Resampling with a PeriodIndex is deprecated.")
def test_periodindexresamplergroupby() -> None:
    idx = pd.period_range("2020-01-28 09:00", periods=4, freq="D")
    df = pd.DataFrame(data=4 * [range(2)], index=idx, columns=["a", "b"])
    check(
        df.groupby("a").resample("3min"),
        PeriodIndexResamplerGroupby,
    )


def test_resampler() -> None:
    s = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("20130101", periods=5, freq="s"))
    check(s.resample("3min"), Resampler)


def test_rolling() -> None:
    df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
    check(df.rolling(2), Rolling)


def test_rolling_groupby() -> None:
    df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
    check(df.groupby("B").rolling(2), RollingGroupby)


def test_timegrouper() -> None:
    check(pd.Grouper(key="Publish date", freq="1W"), TimeGrouper)


def test_window() -> None:
    ser = pd.Series([0, 1, 5, 2, 8])
    check(ser.rolling(2, win_type="gaussian"), Window)


def test_statereader(tmp_path: Path) -> None:
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["col_1", "col_2"])
    time_stamp = pd.Timestamp(2000, 2, 29, 14, 21)
    variable_labels = {"col_1": "This is an example"}
    path = tmp_path / "file"
    df.to_stata(
        path, time_stamp=time_stamp, variable_labels=variable_labels, version=None
    )
    with StataReader(path) as reader:
        check(reader, StataReader)
