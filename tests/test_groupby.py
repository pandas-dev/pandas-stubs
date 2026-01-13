from __future__ import annotations

from collections.abc import Iterator
import datetime as dt
from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
)

import numpy as np
from pandas import (
    DataFrame,
    Index,
    Series,
    Timedelta,
    date_range,
)
from pandas.core.groupby.generic import (
    DataFrameGroupBy,
    SeriesGroupBy,
)
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
    PD_LTE_23,
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


def test_frame_groupby_resample() -> None:
    # basic
    check(
        assert_type(GB_DF.resample("ME"), "_ResamplerGroupBy[DataFrame]"),
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

    # agg funcs
    with pytest_warns_bounded(
        FutureWarning,
        "DataFrameGroupBy.(apply|resample) operated on the grouping columns",
        lower="2.2.99",
        upper="2.99",
    ):
        check(assert_type(GB_DF.resample("ME").sum(), DataFrame), DataFrame)
        check(assert_type(GB_DF.resample("ME").prod(), DataFrame), DataFrame)
        check(assert_type(GB_DF.resample("ME").min(), DataFrame), DataFrame)
        check(assert_type(GB_DF.resample("ME").max(), DataFrame), DataFrame)
        check(assert_type(GB_DF.resample("ME").first(), DataFrame), DataFrame)
        check(assert_type(GB_DF.resample("ME").last(), DataFrame), DataFrame)
        check(assert_type(GB_DF.resample("ME").mean(), DataFrame), DataFrame)
        check(assert_type(GB_DF.resample("ME").sum(), DataFrame), DataFrame)
        check(assert_type(GB_DF.resample("ME").median(), DataFrame), DataFrame)
        check(assert_type(GB_DF.resample("ME").ohlc(), DataFrame), DataFrame)
        check(assert_type(GB_DF.resample("ME").nunique(), DataFrame), DataFrame)

        # quantile
        check(assert_type(GB_DF.resample("ME").quantile(0.5), DataFrame), DataFrame)
        check(
            assert_type(GB_DF.resample("ME").quantile([0.5, 0.7]), DataFrame), DataFrame
        )
        check(
            assert_type(GB_DF.resample("ME").quantile(np.array([0.5, 0.7])), DataFrame),
            DataFrame,
        )

        # std / var
        check(assert_type(GB_DF.resample("ME").std(), DataFrame), DataFrame)
        check(assert_type(GB_DF.resample("ME").var(2), DataFrame), DataFrame)

        # size / count
        check(
            assert_type(GB_DF.resample("ME").size(), "Series[int]"), Series, np.integer
        )
        check(assert_type(GB_DF.resample("ME").count(), DataFrame), DataFrame)

        # filling
        check(assert_type(GB_DF.resample("ME").ffill(), DataFrame), DataFrame)
        check(assert_type(GB_DF.resample("ME").nearest(), DataFrame), DataFrame)
        check(assert_type(GB_DF.resample("ME").bfill(), DataFrame), DataFrame)

    # fillna (deprecated)
    if TYPE_CHECKING_INVALID_USAGE:
        GB_DF.resample("ME").fillna("ffill")  # type: ignore[operator] # pyright: ignore

    # aggregate / apply
    with pytest_warns_bounded(
        FutureWarning,
        "DataFrameGroupBy.(apply|resample) operated on the grouping columns",
        lower="2.2.99",
        upper="2.99",
    ):
        with pytest_warns_bounded(
            FutureWarning,
            r"The provided callable <function (sum|mean) .*> is currently using ",
            upper="2.99",
        ):
            check(
                assert_type(GB_DF.resample("ME").aggregate(np.sum), DataFrame),
                DataFrame,
            )
            check(assert_type(GB_DF.resample("ME").agg(np.sum), DataFrame), DataFrame)
            check(assert_type(GB_DF.resample("ME").apply(np.sum), DataFrame), DataFrame)
            check(
                assert_type(
                    GB_DF.resample("ME").aggregate([np.sum, np.mean]), DataFrame
                ),
                DataFrame,
            )
            check(
                assert_type(
                    GB_DF.resample("ME").aggregate(["sum", np.mean]), DataFrame
                ),
                DataFrame,
            )
            check(
                assert_type(
                    GB_DF.resample("ME").aggregate({"col1": "sum", "col2": np.mean}),
                    DataFrame,
                ),
                DataFrame,
            )
            check(
                assert_type(
                    GB_DF.resample("ME").aggregate(
                        {"col1": ["sum", np.mean], "col2": np.mean}
                    ),
                    DataFrame,
                ),
                DataFrame,
            )

    def f(val: DataFrame) -> Series:
        return val.mean()

    with pytest_warns_bounded(
        FutureWarning,
        "DataFrameGroupBy.(apply|resample) operated on the grouping columns",
        lower="2.2.99",
        upper="2.99",
    ):
        check(assert_type(GB_DF.resample("ME").aggregate(f), DataFrame), DataFrame)

    # aggregate combinations
    def df2frame(val: DataFrame) -> DataFrame:
        return DataFrame(val)

    def df2series(val: DataFrame) -> Series:
        return val.mean()

    def df2scalar(val: DataFrame) -> float:
        return float(val.mean().mean())

    with pytest_warns_bounded(
        FutureWarning,
        "DataFrameGroupBy.(apply|resample) operated on the grouping columns",
        lower="2.2.99",
        upper="2.99",
    ):
        with pytest_warns_bounded(
            FutureWarning,
            r"The provided callable <function (sum|mean) .*> is currently using ",
            upper="2.99",
        ):
            check(GB_DF.resample("ME").aggregate(np.sum), DataFrame)
            check(GB_DF.resample("ME").aggregate([np.mean]), DataFrame)
            check(GB_DF.resample("ME").aggregate(["sum", np.mean]), DataFrame)
            check(GB_DF.resample("ME").aggregate({"col1": np.sum}), DataFrame)
            check(
                GB_DF.resample("ME").aggregate({"col1": np.sum, "col2": np.mean}),
                DataFrame,
            )
            check(
                GB_DF.resample("ME").aggregate(
                    {"col1": [np.sum], "col2": ["sum", np.mean]}
                ),
                DataFrame,
            )
            check(
                GB_DF.resample("ME").aggregate(
                    {"col1": np.sum, "col2": ["sum", np.mean]}
                ),
                DataFrame,
            )
            check(
                GB_DF.resample("ME").aggregate({"col1": "sum", "col2": [np.mean]}),
                DataFrame,
            )
        check(GB_DF.resample("ME").aggregate("sum"), DataFrame)
        check(GB_DF.resample("ME").aggregate(df2frame), DataFrame)
        check(GB_DF.resample("ME").aggregate(df2series), DataFrame)
        check(GB_DF.resample("ME").aggregate(df2scalar), DataFrame)

        # asfreq
        check(assert_type(GB_DF.resample("ME").asfreq(-1.0), DataFrame), DataFrame)

        # getattr
        check(
            assert_type(GB_DF.resample("ME").col1, "_ResamplerGroupBy[DataFrame]"),
            DatetimeIndexResamplerGroupby,
        )

        # getitem
        check(
            assert_type(GB_DF.resample("ME")["col1"], "_ResamplerGroupBy[DataFrame]"),
            DatetimeIndexResamplerGroupby,
        )
        check(
            assert_type(
                GB_DF.resample("ME")[["col1", "col2"]], "_ResamplerGroupBy[DataFrame]"
            ),
            DatetimeIndexResamplerGroupby,
        )

        # interpolate
        if PD_LTE_23:
            check(assert_type(GB_DF.resample("ME").interpolate(), DataFrame), DataFrame)
            check(
                assert_type(
                    GB_DF.resample("ME").interpolate(method="linear"), DataFrame
                ),
                DataFrame,
            )

        def resample_interpolate(x: DataFrame) -> DataFrame:
            return x.resample("ME").interpolate()

        check(assert_type(GB_DF.apply(resample_interpolate), DataFrame), DataFrame)

        def resample_interpolate_linear(x: DataFrame) -> DataFrame:
            return x.resample("ME").interpolate(method="linear")

        check(
            assert_type(
                GB_DF.apply(
                    resample_interpolate_linear,
                ),
                DataFrame,
            ),
            DataFrame,
        )

        # pipe
        def g(val: Resampler[DataFrame]) -> DataFrame:
            assert isinstance(val, Resampler)
            return val.mean()

        check(assert_type(GB_DF.resample("ME").pipe(g), DataFrame), DataFrame)

        def h(val: Resampler[DataFrame]) -> Series:
            assert isinstance(val, Resampler)
            return val.mean().mean()

        check(assert_type(GB_DF.resample("ME").pipe(h), Series), Series)

        def i(val: Resampler[DataFrame]) -> float:
            assert isinstance(val, Resampler)
            return float(val.mean().mean().mean())

        check(assert_type(GB_DF.resample("ME").pipe(i), float), float)

        # transform
        def j(val: Series) -> Series:
            return -1 * val

        check(assert_type(GB_DF.resample("ME").transform(j), DataFrame), DataFrame)


def test_series_groupby_resample() -> None:
    # basic
    check(
        assert_type(GB_S.resample("ME"), "_ResamplerGroupBy[Series[float]]"),
        DatetimeIndexResamplerGroupby,
        Series,
    )

    # agg funcs
    check(assert_type(GB_S.resample("ME").sum(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample("ME").prod(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample("ME").min(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample("ME").max(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample("ME").first(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample("ME").last(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample("ME").mean(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample("ME").sum(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample("ME").median(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample("ME").ohlc(), DataFrame), DataFrame)
    check(assert_type(GB_S.resample("ME").nunique(), "Series[int]"), Series, np.integer)

    # quantile
    check(
        assert_type(GB_S.resample("ME").quantile(0.5), "Series[float]"), Series, float
    )
    check(
        assert_type(GB_S.resample("ME").quantile([0.5, 0.7]), "Series[float]"),
        Series,
        float,
    )
    check(
        assert_type(
            GB_S.resample("ME").quantile(np.array([0.5, 0.7])), "Series[float]"
        ),
        Series,
    )

    # std / var
    check(assert_type(GB_S.resample("ME").std(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample("ME").var(2), "Series[float]"), Series, float)

    # size / count
    check(assert_type(GB_S.resample("ME").size(), "Series[int]"), Series, np.integer)
    check(assert_type(GB_S.resample("ME").count(), "Series[int]"), Series, np.integer)

    # filling
    check(assert_type(GB_S.resample("ME").ffill(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample("ME").nearest(), "Series[float]"), Series, float)
    check(assert_type(GB_S.resample("ME").bfill(), "Series[float]"), Series, float)

    # fillna (deprecated)
    if TYPE_CHECKING_INVALID_USAGE:
        GB_S.resample("ME").fillna("ffill")  # type: ignore[operator] # pyright: ignore

    # aggregate
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        upper="2.99",
    ):
        check(
            assert_type(GB_S.resample("ME").aggregate(np.sum), DataFrame | Series),
            Series,
        )
        check(
            assert_type(GB_S.resample("ME").agg(np.sum), DataFrame | Series),
            Series,
        )
        check(
            assert_type(GB_S.resample("ME").apply(np.sum), DataFrame | Series),
            Series,
        )
        check(
            assert_type(
                GB_S.resample("ME").aggregate([np.sum, np.mean]),
                DataFrame | Series,
            ),
            DataFrame,
        )
        check(
            assert_type(
                GB_S.resample("ME").aggregate(["sum", np.mean]),
                DataFrame | Series,
            ),
            DataFrame,
        )
        check(
            assert_type(
                GB_S.resample("ME").aggregate({"col1": "sum", "col2": np.mean}),
                DataFrame | Series,
            ),
            DataFrame,
        )

    def f(val: Series) -> float:
        return val.mean()

    check(assert_type(GB_S.resample("ME").aggregate(f), DataFrame | Series), Series)

    # asfreq
    check(assert_type(GB_S.resample("ME").asfreq(-1.0), "Series[float]"), Series, float)

    # interpolate
    if PD_LTE_23:
        check(
            assert_type(GB_S.resample("ME").interpolate(), "Series[float]"),
            Series,
            float,
        )
    else:
        check(
            assert_type(
                GB_S.apply(lambda x: x.resample("ME").interpolate()), "Series[float]"
            ),
            Series,
            float,
        )

    # pipe
    def g(val: Resampler[Series]) -> float:
        assert isinstance(val, Resampler)
        return float(val.mean().mean())

    check(assert_type(GB_S.resample("ME").pipe(g), float), float)

    # transform
    def h(val: Series) -> Series:
        return -1 * val

    check(assert_type(GB_S.resample("ME").transform(h), Series), Series)

    # aggregate combinations
    def s2series(val: Series) -> Series:
        return Series(val)

    def s2scalar(val: Series) -> float:
        return float(val.mean())

    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        upper="2.99",
    ):
        check(GB_S.resample("ME").aggregate(np.sum), Series)
        check(GB_S.resample("ME").aggregate([np.mean]), DataFrame)
        check(GB_S.resample("ME").aggregate(["sum", np.mean]), DataFrame)
        check(GB_S.resample("ME").aggregate({"sum": np.sum}), DataFrame)
        check(
            GB_S.resample("ME").aggregate({"sum": np.sum, "mean": np.mean}), DataFrame
        )
    check(GB_S.resample("ME").aggregate("sum"), Series)
    check(GB_S.resample("ME").aggregate(s2series), Series)
    check(GB_S.resample("ME").aggregate(s2scalar), Series)


def test_frame_groupby_rolling() -> None:
    # basic
    check(
        assert_type(GB_DF.rolling(1), "RollingGroupby[DataFrame]"),
        RollingGroupby,
        DataFrame,
    )

    # props
    check(assert_type(GB_DF.rolling(1).on, str | Index | None), type(None))
    check(assert_type(GB_DF.rolling(1).method, Literal["single", "table"]), str)
    if PD_LTE_23:
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
        upper="2.99",
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
        assert isinstance(val, Series)  # type: ignore[unreachable]
        return val.mean()  # type: ignore[unreachable]

    def df2scalar(val: DataFrame) -> float:
        return float(val.mean().mean())

    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        upper="2.99",
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
        upper="2.99",
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
    check(assert_type(GB_DF.expanding(1).on, str | Index | None), type(None))
    check(assert_type(GB_DF.expanding(1).method, Literal["single", "table"]), str)
    if PD_LTE_23:
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
        upper="2.99",
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
        assert isinstance(val, Series)  # type: ignore[unreachable]
        return val.mean()  # type: ignore[unreachable]

    def df2scalar(val: DataFrame) -> float:
        return float(val.mean().mean())

    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        upper="2.99",
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
        upper="2.99",
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
    check(assert_type(GB_DF.ewm(1).on, str | Index | None), type(None))
    check(assert_type(GB_DF.ewm(1).method, Literal["single", "table"]), str)
    if PD_LTE_23:
        check(assert_type(GB_DF.ewm(1).axis, int), int)

    # agg funcs
    check(assert_type(GB_DF.ewm(1).sum(), DataFrame), DataFrame)
    check(assert_type(GB_DF.ewm(1).mean(), DataFrame), DataFrame)
    check(assert_type(GB_DF.ewm(1).sum(), DataFrame), DataFrame)

    # std / var
    check(assert_type(GB_DF.ewm(1).std(), DataFrame), DataFrame)
    check(assert_type(GB_DF.ewm(1).var(), DataFrame), DataFrame)

    # aggregate
    if PD_LTE_23:
        with pytest_warns_bounded(
            FutureWarning,
            r"The provided callable <function (sum|mean) .*> is currently using ",
            upper="2.99",
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
            upper="2.99",
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
        upper="2.99",
    ):
        check(assert_type(GB_S.ewm(1).aggregate("sum"), Series), Series)
        if PD_LTE_23:
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


def test_engine() -> None:
    if TYPE_CHECKING_INVALID_USAGE:
        # See issue #810
        DataFrameGroupBy().aggregate(
            "size",
            "some",
            "args",
            engine=0,  # type: ignore[call-overload] # pyright: ignore
            engine_kwargs="not valid",  # pyright: ignore
            other_kwarg="",
        )
    GB_DF.aggregate("size", engine="cython", engine_kwargs={})


def test_groupby_getitem() -> None:
    df = DataFrame(np.random.random((3, 4)), columns=["a", "b", "c", "d"])
    check(assert_type(df.groupby("a")["b"].sum(), Series), Series, float)
    check(assert_type(df.groupby("a")[["b", "c"]].sum(), DataFrame), DataFrame)


def test_series_value_counts() -> None:
    df = DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]})
    check(
        assert_type(df.groupby("a")["b"].value_counts(), "Series[int]"),
        Series,
        np.int64,
    )
    check(
        assert_type(df.groupby("a")["b"].value_counts(bins=[3, 5, 7]), "Series[int]"),
        Series,
        np.int64,
    )
    check(
        assert_type(df.groupby("a")["b"].value_counts(normalize=True), "Series[float]"),
        Series,
        np.float64,
    )
    check(
        assert_type(
            df.groupby("a")["b"].value_counts(bins=(3, 5, 7), normalize=True),
            "Series[float]",
        ),
        Series,
        np.float64,
    )


def test_dataframe_value_counts() -> None:
    df = DataFrame({"a": [1, 1, 2], "b": [4, 5, 6], "c": [5, 5, 2]})
    check(
        assert_type(df.groupby("a")[["b", "c"]].value_counts(), "Series[int]"),
        Series,
        np.int64,
    )


def test_dataframe_apply_kwargs() -> None:
    # GH 1266
    df = DataFrame({"group": ["A", "A", "B", "B", "C"], "value": [10, 15, 10, 25, 30]})

    def add_constant_to_mean(group: DataFrame, constant: int) -> DataFrame:
        mean_val = group["value"].mean()
        group["adjusted"] = mean_val + constant
        return group

    check(
        assert_type(
            df.groupby("group", group_keys=False)[["group", "value"]].apply(
                add_constant_to_mean, constant=5
            ),
            DataFrame,
        ),
        DataFrame,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        df.groupby("group", group_keys=False)[["group", "value"]].apply(
            add_constant_to_mean,
            constant="5",  # type: ignore[call-overload] # pyright: ignore[reportCallIssue, reportArgumentType]
        )


def test_frame_groupby_aggregate() -> None:
    """Test DataFrame.groupby.aggregate (GH1339)."""
    df = DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        }
    )

    dico = {"a": ("a", "mean")}

    check(assert_type(df.groupby("b").agg(a=("a", "mean")), DataFrame), DataFrame)
    check(assert_type(df.groupby("b").agg(**dico), DataFrame), DataFrame)
