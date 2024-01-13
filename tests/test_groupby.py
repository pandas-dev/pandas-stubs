from __future__ import annotations

from collections.abc import Iterator
import datetime as dt
from typing import (
    TYPE_CHECKING,
    Literal,
    Union,
    cast,
)

import numpy as np
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Series,
    Timedelta,
    date_range,
)
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.resample import (
    DatetimeIndexResamplerGroupby,
    Resampler,
)
from pandas.core.window import (
    ExpandingGroupby,
    ExponentialMovingWindowGroupby,
    RollingGroupby,
)
from typing_extensions import assert_type

from tests import (
    PD_LTE_21,
    TYPE_CHECKING_INVALID_USAGE,
    check,
    pytest_warns_bounded,
)

if TYPE_CHECKING:
    from pandas.core.groupby.groupby import _ResamplerGroupBy  # noqa: F401

DR = date_range("1999-1-1", periods=365, freq="D")
DF_ = DataFrame(np.random.standard_normal((365, 1)), index=DR)
BY = Series(np.random.choice([1, 2], 365), index=DR)
S = DF_.iloc[:, 0]
DF = DataFrame({"col1": S, "col2": S, "col3": BY})
GB_DF = DF.groupby("col3")
GB_S = cast("SeriesGroupBy[float, int]", GB_DF.col1)

M = "M" if PD_LTE_21 else "ME"


def test_frame_groupby_resample() -> None:
    # basic
    check(
        assert_type(GB_DF.resample(M), "_ResamplerGroupBy[DataFrame]"),
        DatetimeIndexResamplerGroupby,
        DataFrame,
    )
    check(
        assert_type(GB_DF.resample(Timedelta(days=30)), "_ResamplerGroupBy[DataFrame]"),
        DatetimeIndexResamplerGroupby,
        DataFrame,
    )
    check(
        assert_type(
            GB_DF.resample(dt.timedelta(days=30)), "_ResamplerGroupBy[DataFrame]"
        ),
        DatetimeIndexResamplerGroupby,
        DataFrame,
    )

    # props
    check(assert_type(GB_DF.resample(M).obj, DataFrame), DataFrame)
    check(assert_type(GB_DF.resample(M).ax, Index), DatetimeIndex)

    # agg funcs
    check(assert_type(GB_DF.resample(M).sum(), DataFrame), DataFrame)
    check(assert_type(GB_DF.resample(M).prod(), DataFrame), DataFrame)
    check(assert_type(GB_DF.resample(M).min(), DataFrame), DataFrame)
    check(assert_type(GB_DF.resample(M).max(), DataFrame), DataFrame)
    check(assert_type(GB_DF.resample(M).first(), DataFrame), DataFrame)
    check(assert_type(GB_DF.resample(M).last(), DataFrame), DataFrame)
    check(assert_type(GB_DF.resample(M).mean(), DataFrame), DataFrame)
    check(assert_type(GB_DF.resample(M).sum(), DataFrame), DataFrame)
    check(assert_type(GB_DF.resample(M).median(), DataFrame), DataFrame)
    check(assert_type(GB_DF.resample(M).ohlc(), DataFrame), DataFrame)
    check(assert_type(GB_DF.resample(M).nunique(), DataFrame), DataFrame)

    # quantile
    check(assert_type(GB_DF.resample(M).quantile(0.5), DataFrame), DataFrame)
    check(assert_type(GB_DF.resample(M).quantile([0.5, 0.7]), DataFrame), DataFrame)
    check(
        assert_type(GB_DF.resample(M).quantile(np.array([0.5, 0.7])), DataFrame),
        DataFrame,
    )

    # std / var
    check(assert_type(GB_DF.resample(M).std(), DataFrame), DataFrame)
    check(assert_type(GB_DF.resample(M).var(2), DataFrame), DataFrame)

    # size / count
    check(assert_type(GB_DF.resample(M).size(), "Series[int]"), Series, np.integer)
    check(assert_type(GB_DF.resample(M).count(), DataFrame), DataFrame)

    # filling
    check(assert_type(GB_DF.resample(M).ffill(), DataFrame), DataFrame)
    check(assert_type(GB_DF.resample(M).nearest(), DataFrame), DataFrame)
    check(assert_type(GB_DF.resample(M).bfill(), DataFrame), DataFrame)

    # fillna (deprecated)
    if TYPE_CHECKING_INVALID_USAGE:
        GB_DF.resample(M).fillna("ffill")  # type: ignore[operator] # pyright: ignore

    # aggregate / apply
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        lower="2.0.99",
    ):
        check(assert_type(GB_DF.resample(M).aggregate(np.sum), DataFrame), DataFrame)
        check(assert_type(GB_DF.resample(M).agg(np.sum), DataFrame), DataFrame)
        check(assert_type(GB_DF.resample(M).apply(np.sum), DataFrame), DataFrame)
        check(
            assert_type(GB_DF.resample(M).aggregate([np.sum, np.mean]), DataFrame),
            DataFrame,
        )
        check(
            assert_type(GB_DF.resample(M).aggregate(["sum", np.mean]), DataFrame),
            DataFrame,
        )
        check(
            assert_type(
                GB_DF.resample(M).aggregate({"col1": "sum", "col2": np.mean}),
                DataFrame,
            ),
            DataFrame,
        )
        check(
            assert_type(
                GB_DF.resample(M).aggregate(
                    {"col1": ["sum", np.mean], "col2": np.mean}
                ),
                DataFrame,
            ),
            DataFrame,
        )

    def f(val: DataFrame) -> Series:
        return val.mean()

    check(assert_type(GB_DF.resample(M).aggregate(f), DataFrame), DataFrame)

    # aggregate combinations
    def df2frame(val: DataFrame) -> DataFrame:
        return DataFrame(val)

    def df2series(val: DataFrame) -> Series:
        return val.mean()

    def df2scalar(val: DataFrame) -> float:
        return float(val.mean().mean())

    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        lower="2.0.99",
    ):
        check(GB_DF.resample(M).aggregate(np.sum), DataFrame)
        check(GB_DF.resample(M).aggregate([np.mean]), DataFrame)
        check(GB_DF.resample(M).aggregate(["sum", np.mean]), DataFrame)
        check(GB_DF.resample(M).aggregate({"col1": np.sum}), DataFrame)
        check(
            GB_DF.resample(M).aggregate({"col1": np.sum, "col2": np.mean}),
            DataFrame,
        )
        check(
            GB_DF.resample(M).aggregate({"col1": [np.sum], "col2": ["sum", np.mean]}),
            DataFrame,
        )
        check(
            GB_DF.resample(M).aggregate({"col1": np.sum, "col2": ["sum", np.mean]}),
            DataFrame,
        )
        check(
            GB_DF.resample(M).aggregate({"col1": "sum", "col2": [np.mean]}),
            DataFrame,
        )
    check(GB_DF.resample(M).aggregate("sum"), DataFrame)
    check(GB_DF.resample(M).aggregate(df2frame), DataFrame)
    check(GB_DF.resample(M).aggregate(df2series), DataFrame)
    check(GB_DF.resample(M).aggregate(df2scalar), DataFrame)

    # asfreq
    check(assert_type(GB_DF.resample(M).asfreq(-1.0), DataFrame), DataFrame)

    # getattr
    check(
        assert_type(GB_DF.resample(M).col1, "_ResamplerGroupBy[DataFrame]"),
        DatetimeIndexResamplerGroupby,
    )

    # getitem
    check(
        assert_type(GB_DF.resample(M)["col1"], "_ResamplerGroupBy[DataFrame]"),
        DatetimeIndexResamplerGroupby,
    )
    check(
        assert_type(
            GB_DF.resample(M)[["col1", "col2"]], "_ResamplerGroupBy[DataFrame]"
        ),
        DatetimeIndexResamplerGroupby,
    )

    # interpolate
    check(assert_type(GB_DF.resample(M).interpolate(), DataFrame), DataFrame)
    check(
        assert_type(GB_DF.resample(M).interpolate(method="linear"), DataFrame),
        DataFrame,
    )
    check(assert_type(GB_DF.resample(M).interpolate(inplace=True), None), type(None))

    # pipe
    def g(val: Resampler[DataFrame]) -> DataFrame:
        assert isinstance(val, Resampler)
        return val.mean()

    check(assert_type(GB_DF.resample(M).pipe(g), DataFrame), DataFrame)

    def h(val: Resampler[DataFrame]) -> Series:
        assert isinstance(val, Resampler)
        return val.mean().mean()

    check(assert_type(GB_DF.resample(M).pipe(h), Series), Series)

    def i(val: Resampler[DataFrame]) -> float:
        assert isinstance(val, Resampler)
        return float(val.mean().mean().mean())

    check(assert_type(GB_DF.resample(M).pipe(i), float), float)

    # transform
    def j(val: Series) -> Series:
        return -1 * val

    check(assert_type(GB_DF.resample(M).transform(j), DataFrame), DataFrame)


def test_series_groupby_resample() -> None:
    # basic
    check(
        assert_type(GB_S.resample(M), "_ResamplerGroupBy[Series[float]]"),
        DatetimeIndexResamplerGroupby,
        Series,
    )

    # props
    check(assert_type(GB_S.resample(M).obj, "Series[float]"), Series, float)
    check(assert_type(GB_S.resample(M).ax, Index), DatetimeIndex)

    # agg funcs
    check(assert_type(GB_S.resample(M).sum(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample(M).prod(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample(M).min(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample(M).max(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample(M).first(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample(M).last(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample(M).mean(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample(M).sum(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample(M).median(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample(M).ohlc(), DataFrame), DataFrame)
    check(assert_type(GB_S.resample(M).nunique(), "Series[int]"), Series, np.integer)

    # quantile
    check(assert_type(GB_S.resample(M).quantile(0.5), "Series[float]"), Series, float)
    check(
        assert_type(GB_S.resample(M).quantile([0.5, 0.7]), "Series[float]"),
        Series,
        float,
    )
    check(
        assert_type(GB_S.resample(M).quantile(np.array([0.5, 0.7])), "Series[float]"),
        Series,
    )

    # std / var
    check(assert_type(GB_S.resample(M).std(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample(M).var(2), "Series[float]"), Series, float)

    # size / count
    check(assert_type(GB_S.resample(M).size(), "Series[int]"), Series, np.integer)
    check(assert_type(GB_S.resample(M).count(), "Series[int]"), Series, np.integer)

    # filling
    check(assert_type(GB_S.resample(M).ffill(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample(M).nearest(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample(M).bfill(), "Series[float]"), Series, float)

    # fillna (deprecated)
    if TYPE_CHECKING_INVALID_USAGE:
        GB_S.resample(M).fillna("ffill")  # type: ignore[operator] # pyright: ignore

    # aggregate
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        lower="2.0.99",
    ):
        check(
            assert_type(GB_S.resample(M).aggregate(np.sum), Union[DataFrame, Series]),
            Series,
        )
        check(
            assert_type(GB_S.resample(M).agg(np.sum), Union[DataFrame, Series]), Series
        )
        check(
            assert_type(GB_S.resample(M).apply(np.sum), Union[DataFrame, Series]),
            Series,
        )
        check(
            assert_type(
                GB_S.resample(M).aggregate([np.sum, np.mean]), Union[DataFrame, Series]
            ),
            DataFrame,
        )
        check(
            assert_type(
                GB_S.resample(M).aggregate(["sum", np.mean]), Union[DataFrame, Series]
            ),
            DataFrame,
        )
        check(
            assert_type(
                GB_S.resample(M).aggregate({"col1": "sum", "col2": np.mean}),
                Union[DataFrame, Series],
            ),
            DataFrame,
        )

    def f(val: Series) -> float:
        return val.mean()

    check(assert_type(GB_S.resample(M).aggregate(f), Union[DataFrame, Series]), Series)

    # asfreq
    check(assert_type(GB_S.resample(M).asfreq(-1.0), "Series[float]"), Series, float)

    # interpolate
    check(assert_type(GB_S.resample(M).interpolate(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample(M).interpolate(inplace=True), None), type(None))

    # pipe
    def g(val: Resampler[Series]) -> float:
        assert isinstance(val, Resampler)
        return float(val.mean().mean())

    check(assert_type(GB_S.resample(M).pipe(g), float), float)

    # transform
    def h(val: Series) -> Series:
        return -1 * val

    check(assert_type(GB_S.resample(M).transform(h), Series), Series)

    # aggregate combinations
    def s2series(val: Series) -> Series:
        return Series(val)

    def s2scalar(val: Series) -> float:
        return float(val.mean())

    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        lower="2.0.99",
    ):
        check(GB_S.resample(M).aggregate(np.sum), Series)
        check(GB_S.resample(M).aggregate([np.mean]), DataFrame)
        check(GB_S.resample(M).aggregate(["sum", np.mean]), DataFrame)
        check(GB_S.resample(M).aggregate({"sum": np.sum}), DataFrame)
        check(GB_S.resample(M).aggregate({"sum": np.sum, "mean": np.mean}), DataFrame)
    check(GB_S.resample(M).aggregate("sum"), Series)
    check(GB_S.resample(M).aggregate(s2series), Series)
    check(GB_S.resample(M).aggregate(s2scalar), Series)


def test_frame_groupby_rolling() -> None:
    # basic
    check(
        assert_type(GB_DF.rolling(1), "RollingGroupby[DataFrame]"),
        RollingGroupby,
        DataFrame,
    )

    # props
    check(assert_type(GB_DF.rolling(1).obj, DataFrame), DataFrame)
    check(assert_type(GB_DF.rolling(1).on, Union[str, Index, None]), type(None))
    check(assert_type(GB_DF.rolling(1).method, Literal["single", "table"]), str)
    check(assert_type(GB_DF.rolling(1).axis, int), int)

    # agg funcs
    check(assert_type(GB_DF.rolling(1).sum(), DataFrame), DataFrame)
    check(assert_type(GB_DF.rolling(1).min(), DataFrame), DataFrame)
    check(assert_type(GB_DF.rolling(1).max(), DataFrame), DataFrame)
    check(assert_type(GB_DF.rolling(1).mean(), DataFrame), DataFrame)
    check(assert_type(GB_DF.rolling(1).sum(), DataFrame), DataFrame)
    check(assert_type(GB_DF.rolling(1).median(), DataFrame), DataFrame)

    # quantile / std / var / count
    check(assert_type(GB_DF.rolling(1).quantile(0.5), DataFrame), DataFrame)
    check(assert_type(GB_DF.rolling(1).std(), DataFrame), DataFrame)
    check(assert_type(GB_DF.rolling(1).var(2), DataFrame), DataFrame)
    check(assert_type(GB_DF.rolling(1).count(), DataFrame), DataFrame)

    # aggregate / apply
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        lower="2.0.99",
    ):
        check(assert_type(GB_DF.rolling(1).aggregate(np.sum), DataFrame), DataFrame)
        check(assert_type(GB_DF.rolling(1).agg(np.sum), DataFrame), DataFrame)
        check(assert_type(GB_DF.rolling(1).apply(np.sum), DataFrame), DataFrame)
        check(
            assert_type(GB_DF.rolling(1).aggregate([np.sum, np.mean]), DataFrame),
            DataFrame,
        )
        check(
            assert_type(GB_DF.rolling(1).aggregate(["sum", np.mean]), DataFrame),
            DataFrame,
        )
        check(
            assert_type(
                GB_DF.rolling(1).aggregate({"col1": "sum", "col2": np.mean}),
                DataFrame,
            ),
            DataFrame,
        )
        check(
            assert_type(
                GB_DF.rolling(1).aggregate({"col1": ["sum", np.mean], "col2": np.mean}),
                DataFrame,
            ),
            DataFrame,
        )

    def f(val: DataFrame) -> Series:
        return val.mean()

    check(assert_type(GB_DF.rolling(1).aggregate(f), DataFrame), DataFrame)

    # aggregate combinations
    def df2series(val: DataFrame) -> Series:
        assert isinstance(val, Series)
        return val.mean()

    def df2scalar(val: DataFrame) -> float:
        return float(val.mean().mean())

    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        lower="2.0.99",
    ):
        check(GB_DF.rolling(1).aggregate(np.sum), DataFrame)
        check(GB_DF.rolling(1).aggregate([np.mean]), DataFrame)
        check(GB_DF.rolling(1).aggregate(["sum", np.mean]), DataFrame)
        check(GB_DF.rolling(1).aggregate({"col1": np.sum}), DataFrame)
        check(
            GB_DF.rolling(1).aggregate({"col1": np.sum, "col2": np.mean}),
            DataFrame,
        )
        check(
            GB_DF.rolling(1).aggregate({"col1": [np.sum], "col2": ["sum", np.mean]}),
            DataFrame,
        )
        check(
            GB_DF.rolling(1).aggregate({"col1": np.sum, "col2": ["sum", np.mean]}),
            DataFrame,
        )
        check(
            GB_DF.rolling(1).aggregate({"col1": "sum", "col2": [np.mean]}),
            DataFrame,
        )
    check(GB_DF.rolling(1).aggregate("sum"), DataFrame)
    check(GB_DF.rolling(1).aggregate(df2series), DataFrame)
    check(GB_DF.rolling(1).aggregate(df2scalar), DataFrame)

    # getattr
    check(
        assert_type(GB_DF.rolling(1).col1, "RollingGroupby[DataFrame]"),
        RollingGroupby,
    )

    # getitem
    check(
        assert_type(GB_DF.rolling(1)["col1"], "RollingGroupby[DataFrame]"),
        RollingGroupby,
    )
    check(
        assert_type(GB_DF.rolling(1)[["col1", "col2"]], "RollingGroupby[DataFrame]"),
        RollingGroupby,
    )

    # iter
    iterator = iter(GB_DF.rolling(1))
    check(assert_type(iterator, Iterator[DataFrame]), Iterator)
    check(assert_type(next(iterator), DataFrame), DataFrame)
    check(assert_type(list(GB_DF.rolling(1)), list[DataFrame]), list, DataFrame)


def test_series_groupby_rolling() -> None:
    # basic
    check(
        assert_type(GB_S.rolling(1), "RollingGroupby[Series[float]]"),
        RollingGroupby,
        Series,
    )

    # props
    check(assert_type(GB_S.rolling(1).obj, "Series[float]"), Series, float)

    # agg funcs
    check(assert_type(GB_S.rolling(1).sum(), "Series[float]"), Series, float)
    check(assert_type(GB_S.rolling(1).min(), "Series[float]"), Series, float)
    check(assert_type(GB_S.rolling(1).max(), "Series[float]"), Series, float)
    check(assert_type(GB_S.rolling(1).mean(), "Series[float]"), Series, float)
    check(assert_type(GB_S.rolling(1).sum(), "Series[float]"), Series, float)
    check(assert_type(GB_S.rolling(1).median(), "Series[float]"), Series, float)

    # quantile / std / var / count
    check(assert_type(GB_S.rolling(1).quantile(0.5), "Series[float]"), Series, float)
    check(assert_type(GB_S.rolling(1).std(), "Series[float]"), Series, float)
    check(assert_type(GB_S.rolling(1).var(2), "Series[float]"), Series, float)
    check(assert_type(GB_S.rolling(1).count(), "Series[float]"), Series, float)

    # aggregate
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        lower="2.0.99",
    ):
        check(assert_type(GB_S.rolling(1).aggregate("sum"), Series), Series)
        check(assert_type(GB_S.rolling(1).aggregate(np.sum), Series), Series)
        check(assert_type(GB_S.rolling(1).agg(np.sum), Series), Series)
        check(
            assert_type(GB_S.rolling(1).apply(np.sum), "Series[float]"), Series, float
        )
        check(
            assert_type(GB_S.rolling(1).aggregate([np.sum, np.mean]), DataFrame),
            DataFrame,
        )
        check(
            assert_type(GB_S.rolling(1).aggregate(["sum", np.mean]), DataFrame),
            DataFrame,
        )
        check(
            assert_type(
                GB_S.rolling(1).aggregate({"col1": "sum", "col2": np.mean}), DataFrame
            ),
            DataFrame,
        )

    def f(val: Series) -> float:
        return val.mean()

    check(assert_type(GB_S.rolling(1).aggregate(f), Series), Series)

    def s2scalar(val: Series) -> float:
        return float(val.mean())

    check(assert_type(GB_S.rolling(1).aggregate(s2scalar), Series), Series)

    # iter
    iterator = iter(GB_S.rolling(1))
    check(assert_type(iterator, "Iterator[Series[float]]"), Iterator)
    check(assert_type(next(iterator), "Series[float]"), Series, float)
    check(assert_type(list(GB_S.rolling(1)), "list[Series[float]]"), list, Series)


def test_frame_groupby_expanding() -> None:
    # basic
    check(
        assert_type(GB_DF.expanding(1), "ExpandingGroupby[DataFrame]"),
        ExpandingGroupby,
        DataFrame,
    )

    # props
    check(assert_type(GB_DF.expanding(1).obj, DataFrame), DataFrame)
    check(assert_type(GB_DF.expanding(1).on, Union[str, Index, None]), type(None))
    check(assert_type(GB_DF.expanding(1).method, Literal["single", "table"]), str)
    check(assert_type(GB_DF.expanding(1).axis, int), int)

    # agg funcs
    check(assert_type(GB_DF.expanding(1).sum(), DataFrame), DataFrame)
    check(assert_type(GB_DF.expanding(1).min(), DataFrame), DataFrame)
    check(assert_type(GB_DF.expanding(1).max(), DataFrame), DataFrame)
    check(assert_type(GB_DF.expanding(1).mean(), DataFrame), DataFrame)
    check(assert_type(GB_DF.expanding(1).sum(), DataFrame), DataFrame)
    check(assert_type(GB_DF.expanding(1).median(), DataFrame), DataFrame)

    # quantile / std / var / count
    check(assert_type(GB_DF.expanding(1).quantile(0.5), DataFrame), DataFrame)
    check(assert_type(GB_DF.expanding(1).std(), DataFrame), DataFrame)
    check(assert_type(GB_DF.expanding(1).var(2), DataFrame), DataFrame)
    check(assert_type(GB_DF.expanding(1).count(), DataFrame), DataFrame)

    # aggregate / apply
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        lower="2.0.99",
    ):
        check(assert_type(GB_DF.expanding(1).aggregate(np.sum), DataFrame), DataFrame)
        check(assert_type(GB_DF.expanding(1).agg(np.sum), DataFrame), DataFrame)
        check(assert_type(GB_DF.expanding(1).apply(np.sum), DataFrame), DataFrame)
        check(
            assert_type(GB_DF.expanding(1).aggregate([np.sum, np.mean]), DataFrame),
            DataFrame,
        )
        check(
            assert_type(GB_DF.expanding(1).aggregate(["sum", np.mean]), DataFrame),
            DataFrame,
        )
        check(
            assert_type(
                GB_DF.expanding(1).aggregate({"col1": "sum", "col2": np.mean}),
                DataFrame,
            ),
            DataFrame,
        )
        check(
            assert_type(
                GB_DF.expanding(1).aggregate(
                    {"col1": ["sum", np.mean], "col2": np.mean}
                ),
                DataFrame,
            ),
            DataFrame,
        )

    def f(val: DataFrame) -> Series:
        return val.mean()

    check(assert_type(GB_DF.expanding(1).aggregate(f), DataFrame), DataFrame)

    # aggregate combinations
    def df2series(val: DataFrame) -> Series:
        assert isinstance(val, Series)
        return val.mean()

    def df2scalar(val: DataFrame) -> float:
        return float(val.mean().mean())

    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        lower="2.0.99",
    ):
        check(GB_DF.expanding(1).aggregate(np.sum), DataFrame)
        check(GB_DF.expanding(1).aggregate([np.mean]), DataFrame)
        check(GB_DF.expanding(1).aggregate(["sum", np.mean]), DataFrame)
        check(GB_DF.expanding(1).aggregate({"col1": np.sum}), DataFrame)
        check(
            GB_DF.expanding(1).aggregate({"col1": np.sum, "col2": np.mean}),
            DataFrame,
        )
        check(
            GB_DF.expanding(1).aggregate({"col1": [np.sum], "col2": ["sum", np.mean]}),
            DataFrame,
        )
        check(
            GB_DF.expanding(1).aggregate({"col1": np.sum, "col2": ["sum", np.mean]}),
            DataFrame,
        )
        check(
            GB_DF.expanding(1).aggregate({"col1": "sum", "col2": [np.mean]}),
            DataFrame,
        )
    check(GB_DF.expanding(1).aggregate("sum"), DataFrame)
    check(GB_DF.expanding(1).aggregate(df2series), DataFrame)
    check(GB_DF.expanding(1).aggregate(df2scalar), DataFrame)

    # getattr
    check(
        assert_type(GB_DF.expanding(1).col1, "ExpandingGroupby[DataFrame]"),
        ExpandingGroupby,
    )

    # getitem
    check(
        assert_type(GB_DF.expanding(1)["col1"], "ExpandingGroupby[DataFrame]"),
        ExpandingGroupby,
    )
    check(
        assert_type(
            GB_DF.expanding(1)[["col1", "col2"]], "ExpandingGroupby[DataFrame]"
        ),
        ExpandingGroupby,
    )

    # iter
    iterator = iter(GB_DF.expanding(1))
    check(assert_type(iterator, Iterator[DataFrame]), Iterator)
    check(assert_type(next(iterator), DataFrame), DataFrame)
    check(assert_type(list(GB_DF.expanding(1)), list[DataFrame]), list, DataFrame)


def test_series_groupby_expanding() -> None:
    # basic
    check(
        assert_type(GB_S.expanding(1), "ExpandingGroupby[Series[float]]"),
        ExpandingGroupby,
        Series,
    )

    # props
    check(assert_type(GB_S.expanding(1).obj, "Series[float]"), Series, float)

    # agg funcs
    check(assert_type(GB_S.expanding(1).sum(), "Series[float]"), Series, float)
    check(assert_type(GB_S.expanding(1).min(), "Series[float]"), Series, float)
    check(assert_type(GB_S.expanding(1).max(), "Series[float]"), Series, float)
    check(assert_type(GB_S.expanding(1).mean(), "Series[float]"), Series, float)
    check(assert_type(GB_S.expanding(1).sum(), "Series[float]"), Series, float)
    check(assert_type(GB_S.expanding(1).median(), "Series[float]"), Series, float)

    # quantile / std / var / count
    check(assert_type(GB_S.expanding(1).quantile(0.5), "Series[float]"), Series, float)
    check(assert_type(GB_S.expanding(1).std(), "Series[float]"), Series, float)
    check(assert_type(GB_S.expanding(1).var(2), "Series[float]"), Series, float)
    check(assert_type(GB_S.expanding(1).count(), "Series[float]"), Series, float)

    # aggregate
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        lower="2.0.99",
    ):
        check(assert_type(GB_S.expanding(1).aggregate("sum"), Series), Series)
        check(assert_type(GB_S.expanding(1).aggregate(np.sum), Series), Series)
        check(assert_type(GB_S.expanding(1).agg(np.sum), Series), Series)
        check(
            assert_type(GB_S.expanding(1).apply(np.sum), "Series[float]"), Series, float
        )
        check(
            assert_type(GB_S.expanding(1).aggregate([np.sum, np.mean]), DataFrame),
            DataFrame,
        )
        check(
            assert_type(GB_S.expanding(1).aggregate(["sum", np.mean]), DataFrame),
            DataFrame,
        )
        check(
            assert_type(
                GB_S.expanding(1).aggregate({"col1": "sum", "col2": np.mean}), DataFrame
            ),
            DataFrame,
        )

    def f(val: Series) -> float:
        return val.mean()

    check(assert_type(GB_S.expanding(1).aggregate(f), Series), Series)

    def s2scalar(val: Series) -> float:
        return float(val.mean())

    check(assert_type(GB_S.expanding(1).aggregate(s2scalar), Series), Series)

    # iter
    iterator = iter(GB_S.expanding(1))
    check(assert_type(iterator, "Iterator[Series[float]]"), Iterator)
    check(assert_type(next(iterator), "Series[float]"), Series, float)
    check(assert_type(list(GB_S.expanding(1)), "list[Series[float]]"), list, Series)


def test_frame_groupby_ewm() -> None:
    # basic
    check(
        assert_type(GB_DF.ewm(1), "ExponentialMovingWindowGroupby[DataFrame]"),
        ExponentialMovingWindowGroupby,
        DataFrame,
    )

    # props
    check(assert_type(GB_DF.ewm(1).obj, DataFrame), DataFrame)
    check(assert_type(GB_DF.ewm(1).on, Union[str, Index, None]), type(None))
    check(assert_type(GB_DF.ewm(1).method, Literal["single", "table"]), str)
    check(assert_type(GB_DF.ewm(1).axis, int), int)

    # agg funcs
    check(assert_type(GB_DF.ewm(1).sum(), DataFrame), DataFrame)
    check(assert_type(GB_DF.ewm(1).mean(), DataFrame), DataFrame)
    check(assert_type(GB_DF.ewm(1).sum(), DataFrame), DataFrame)

    # std / var
    check(assert_type(GB_DF.ewm(1).std(), DataFrame), DataFrame)
    check(assert_type(GB_DF.ewm(1).var(), DataFrame), DataFrame)

    # aggregate
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        lower="2.0.99",
    ):
        check(assert_type(GB_DF.ewm(1).aggregate(np.sum), DataFrame), DataFrame)
        check(assert_type(GB_DF.ewm(1).agg(np.sum), DataFrame), DataFrame)
        check(
            assert_type(GB_DF.ewm(1).aggregate([np.sum, np.mean]), DataFrame),
            DataFrame,
        )
        check(
            assert_type(GB_DF.ewm(1).aggregate(["sum", np.mean]), DataFrame),
            DataFrame,
        )
        check(
            assert_type(
                GB_DF.ewm(1).aggregate({"col1": "sum", "col2": np.mean}),
                DataFrame,
            ),
            DataFrame,
        )
        check(
            assert_type(
                GB_DF.ewm(1).aggregate({"col1": ["sum", np.mean], "col2": np.mean}),
                DataFrame,
            ),
            DataFrame,
        )

    # aggregate combinations
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        lower="2.0.99",
    ):
        check(GB_DF.ewm(1).aggregate(np.sum), DataFrame)
        check(GB_DF.ewm(1).aggregate([np.mean]), DataFrame)
        check(GB_DF.ewm(1).aggregate(["sum", np.mean]), DataFrame)
        check(GB_DF.ewm(1).aggregate({"col1": np.sum}), DataFrame)
        check(
            GB_DF.ewm(1).aggregate({"col1": np.sum, "col2": np.mean}),
            DataFrame,
        )
        check(
            GB_DF.ewm(1).aggregate({"col1": [np.sum], "col2": ["sum", np.mean]}),
            DataFrame,
        )
        check(
            GB_DF.ewm(1).aggregate({"col1": np.sum, "col2": ["sum", np.mean]}),
            DataFrame,
        )
        check(
            GB_DF.ewm(1).aggregate({"col1": "sum", "col2": [np.mean]}),
            DataFrame,
        )
    check(GB_DF.ewm(1).aggregate("sum"), DataFrame)

    # getattr
    check(
        assert_type(GB_DF.ewm(1).col1, "ExponentialMovingWindowGroupby[DataFrame]"),
        ExponentialMovingWindowGroupby,
    )

    # getitem
    check(
        assert_type(GB_DF.ewm(1)["col1"], "ExponentialMovingWindowGroupby[DataFrame]"),
        ExponentialMovingWindowGroupby,
    )
    check(
        assert_type(
            GB_DF.ewm(1)[["col1", "col2"]], "ExponentialMovingWindowGroupby[DataFrame]"
        ),
        ExponentialMovingWindowGroupby,
    )

    # iter
    iterator = iter(GB_DF.ewm(1))
    check(assert_type(iterator, Iterator[DataFrame]), Iterator)
    check(assert_type(next(iterator), DataFrame), DataFrame)
    check(assert_type(list(GB_DF.ewm(1)), list[DataFrame]), list, DataFrame)


def test_series_groupby_ewm() -> None:
    # basic
    check(
        assert_type(GB_S.ewm(1), "ExponentialMovingWindowGroupby[Series[float]]"),
        ExponentialMovingWindowGroupby,
        Series,
    )

    # props
    check(assert_type(GB_S.ewm(1).obj, "Series[float]"), Series, float)

    # agg funcs
    check(assert_type(GB_S.ewm(1).sum(), "Series[float]"), Series, float)
    check(assert_type(GB_S.ewm(1).mean(), "Series[float]"), Series, float)
    check(assert_type(GB_S.ewm(1).sum(), "Series[float]"), Series, float)

    # std / var
    check(assert_type(GB_S.ewm(1).std(), "Series[float]"), Series, float)
    check(assert_type(GB_S.ewm(1).var(), "Series[float]"), Series, float)

    # aggregate
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        lower="2.0.99",
    ):
        check(assert_type(GB_S.ewm(1).aggregate("sum"), Series), Series)
        check(assert_type(GB_S.ewm(1).aggregate(np.sum), Series), Series)
        check(assert_type(GB_S.ewm(1).agg(np.sum), Series), Series)
        check(
            assert_type(GB_S.ewm(1).aggregate([np.sum, np.mean]), DataFrame),
            DataFrame,
        )
        check(
            assert_type(GB_S.ewm(1).aggregate(["sum", np.mean]), DataFrame),
            DataFrame,
        )
        check(
            assert_type(
                GB_S.ewm(1).aggregate({"col1": "sum", "col2": np.mean}), DataFrame
            ),
            DataFrame,
        )

    # iter
    iterator = iter(GB_S.ewm(1))
    check(assert_type(iterator, "Iterator[Series[float]]"), Iterator)
    check(assert_type(next(iterator), "Series[float]"), Series, float)
    check(assert_type(list(GB_S.ewm(1)), "list[Series[float]]"), list, Series)
