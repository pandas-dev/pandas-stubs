from collections.abc import (
    Generator,
    Hashable,
)
from typing import Union

import numpy as np
import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Series,
    date_range,
)
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.resample import Resampler
from typing_extensions import assert_type

from pandas._typing import Scalar

from tests import (
    check,
    pytest_warns_bounded,
)

DR = date_range("1999-1-1", periods=365, freq="D")
DF_ = DataFrame(np.random.standard_normal((365, 1)), index=DR)
S = DF_.iloc[:, 0]
DF = DataFrame({"col1": S, "col2": S})

_AggRetType = Union[DataFrame, Series]


def test_props() -> None:
    check(assert_type(DF.resample("ME").obj, DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").ax, Index), DatetimeIndex)


def test_iter() -> None:
    assert_type(
        iter(DF.resample("ME")), Generator[tuple[Hashable, DataFrame], None, None]
    )
    for v in DF.resample("ME"):
        check(assert_type(v, tuple[Hashable, DataFrame]), tuple)


def test_agg_funcs() -> None:
    check(assert_type(DF.resample("ME").sum(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").prod(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").min(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").max(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").first(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").last(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").mean(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").sum(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").median(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").ohlc(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").nunique(), DataFrame), DataFrame)


def test_quantile() -> None:
    check(assert_type(DF.resample("ME").quantile(0.5), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").quantile([0.5, 0.7]), DataFrame), DataFrame)
    check(
        assert_type(DF.resample("ME").quantile(np.array([0.5, 0.7])), DataFrame),
        DataFrame,
    )


def test_std_var() -> None:
    check(assert_type(DF.resample("ME").std(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").var(2), DataFrame), DataFrame)


def test_size_count() -> None:
    check(assert_type(DF.resample("ME").size(), Series), Series)
    check(assert_type(DF.resample("ME").count(), DataFrame), DataFrame)


def test_filling() -> None:
    check(assert_type(DF.resample("ME").ffill(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").nearest(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").bfill(), DataFrame), DataFrame)


def test_fillna() -> None:
    with pytest_warns_bounded(
        FutureWarning,
        "DatetimeIndexResampler.fillna is deprecated ",
        lower="2.0.99",
    ):
        check(assert_type(DF.resample("ME").fillna("pad"), DataFrame), DataFrame)
        check(assert_type(DF.resample("ME").fillna("backfill"), DataFrame), DataFrame)
        check(assert_type(DF.resample("ME").fillna("ffill"), DataFrame), DataFrame)
        check(assert_type(DF.resample("ME").fillna("bfill"), DataFrame), DataFrame)
        check(
            assert_type(DF.resample("ME").fillna("nearest", limit=2), DataFrame),
            DataFrame,
        )


def test_aggregate() -> None:
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        lower="2.0.99",
    ):
        check(assert_type(DF.resample("ME").aggregate(np.sum), _AggRetType), DataFrame)
        check(assert_type(DF.resample("ME").agg(np.sum), _AggRetType), DataFrame)
        check(assert_type(DF.resample("ME").apply(np.sum), _AggRetType), DataFrame)
        check(
            assert_type(DF.resample("ME").aggregate([np.sum, np.mean]), _AggRetType),
            DataFrame,
        )
        check(
            assert_type(DF.resample("ME").aggregate(["sum", np.mean]), _AggRetType),
            DataFrame,
        )
        check(
            assert_type(
                DF.resample("ME").aggregate({"col1": "sum", "col2": np.mean}),
                _AggRetType,
            ),
            DataFrame,
        )
        check(
            assert_type(
                DF.resample("ME").aggregate(
                    {"col1": ["sum", np.mean], "col2": np.mean}
                ),
                _AggRetType,
            ),
            DataFrame,
        )

    def f(val: DataFrame) -> Series:
        return val.mean()

    check(assert_type(DF.resample("ME").aggregate(f), _AggRetType), DataFrame)


def test_asfreq() -> None:
    check(assert_type(DF.resample("ME").asfreq(-1.0), DataFrame), DataFrame)


def test_getattr() -> None:
    check(assert_type(DF.resample("ME").col1, SeriesGroupBy), SeriesGroupBy)


def test_interpolate() -> None:
    check(assert_type(DF.resample("ME").interpolate(), DataFrame), DataFrame)
    check(
        assert_type(DF.resample("ME").interpolate(method="time"), DataFrame), DataFrame
    )


def test_interpolate_inplace() -> None:
    check(assert_type(DF.resample("ME").interpolate(inplace=True), None), type(None))


def test_pipe() -> None:
    def f(val: DataFrame) -> DataFrame:
        return DataFrame(val)

    check(assert_type(DF.resample("ME").pipe(f), DataFrame), DataFrame)

    def g(val: DataFrame) -> Series:
        return val.mean()

    check(assert_type(DF.resample("ME").pipe(g), DataFrame), DataFrame)

    def h(val: DataFrame) -> float:
        return val.mean().mean()

    check(assert_type(DF.resample("ME").pipe(h), Series), Series)


def test_transform() -> None:
    def f(val: Series) -> Series:
        return -1 * val

    check(assert_type(DF.resample("ME").transform(f), DataFrame), DataFrame)


def test_props_series() -> None:
    check(assert_type(S.resample("ME").obj, Series), Series)
    check(assert_type(S.resample("ME").ax, Index), DatetimeIndex)


def test_iter_series() -> None:
    for v in S.resample("ME"):
        check(assert_type(v, tuple[Hashable, Series]), tuple)


def test_agg_funcs_series() -> None:
    check(assert_type(S.resample("ME").sum(), Series), Series)
    check(assert_type(S.resample("ME").prod(), Series), Series)
    check(assert_type(S.resample("ME").min(), Series), Series)
    check(assert_type(S.resample("ME").max(), Series), Series)
    check(assert_type(S.resample("ME").first(), Series), Series)
    check(assert_type(S.resample("ME").last(), Series), Series)
    check(assert_type(S.resample("ME").mean(), Series), Series)
    check(assert_type(S.resample("ME").sum(), Series), Series)
    check(assert_type(S.resample("ME").median(), Series), Series)
    check(assert_type(S.resample("ME").ohlc(), DataFrame), DataFrame)
    check(assert_type(S.resample("ME").nunique(), Series), Series)


def test_quantile_series() -> None:
    check(assert_type(S.resample("ME").quantile(0.5), Series), Series)
    check(assert_type(S.resample("ME").quantile([0.5, 0.7]), Series), Series)
    check(
        assert_type(S.resample("ME").quantile(np.array([0.5, 0.7])), Series),
        Series,
    )


def test_std_var_series() -> None:
    check(assert_type(S.resample("ME").std(), Series), Series)
    check(assert_type(S.resample("ME").var(2), Series), Series)


def test_size_count_series() -> None:
    check(assert_type(S.resample("ME").size(), Series), Series)
    check(assert_type(S.resample("ME").count(), Series), Series)


def test_filling_series() -> None:
    check(assert_type(S.resample("ME").ffill(), Series), Series)
    check(assert_type(S.resample("ME").nearest(), Series), Series)
    check(assert_type(S.resample("ME").bfill(), Series), Series)


def test_fillna_series() -> None:
    with pytest_warns_bounded(
        FutureWarning,
        "DatetimeIndexResampler.fillna is deprecated ",
        lower="2.0.99",
    ):
        check(assert_type(S.resample("ME").fillna("pad"), Series), Series)
        check(assert_type(S.resample("ME").fillna("backfill"), Series), Series)
        check(assert_type(S.resample("ME").fillna("ffill"), Series), Series)
        check(assert_type(S.resample("ME").fillna("bfill"), Series), Series)
        check(assert_type(S.resample("ME").fillna("nearest", limit=2), Series), Series)


def test_aggregate_series() -> None:
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        lower="2.0.99",
    ):
        check(assert_type(S.resample("ME").aggregate(np.sum), _AggRetType), Series)
        check(assert_type(S.resample("ME").agg(np.sum), _AggRetType), Series)
        check(assert_type(S.resample("ME").apply(np.sum), _AggRetType), Series)
        check(
            assert_type(S.resample("ME").aggregate([np.sum, np.mean]), _AggRetType),
            DataFrame,
        )
        check(
            assert_type(S.resample("ME").aggregate(["sum", np.mean]), _AggRetType),
            DataFrame,
        )
        check(
            assert_type(
                S.resample("ME").aggregate({"col1": "sum", "col2": np.mean}),
                _AggRetType,
            ),
            DataFrame,
        )

    def f(val: Series) -> float:
        return val.mean()

    check(assert_type(S.resample("ME").aggregate(f), _AggRetType), Series)


def test_asfreq_series() -> None:
    check(assert_type(S.resample("ME").asfreq(-1.0), Series), Series)


def test_interpolate_series() -> None:
    check(assert_type(S.resample("ME").interpolate(), Series), Series)
    check(assert_type(S.resample("ME").interpolate(method="time"), Series), Series)


def test_interpolate_inplace_series() -> None:
    check(assert_type(S.resample("ME").interpolate(inplace=True), None), type(None))


def test_pipe_series() -> None:
    def f(val: Series) -> Series:
        return Series(val)

    check(assert_type(S.resample("ME").pipe(f), Series), Series)

    def g(val: Resampler) -> float:
        return float(val.mean().mean())

    check(assert_type(S.resample("ME").pipe(g), Scalar), float)

    def h(val: Series) -> DataFrame:
        return DataFrame({0: val, 1: val})

    check(assert_type(S.resample("ME").pipe(h), DataFrame), DataFrame)


def test_transform_series() -> None:
    def f(val: Series) -> Series:
        return -1 * val

    check(assert_type(S.resample("ME").transform(f), Series), Series)


def test_aggregate_series_combinations() -> None:
    def s2series(val: Series) -> Series:
        return pd.Series(val)

    def s2scalar(val: Series) -> float:
        return float(val.mean())

    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        lower="2.0.99",
    ):
        check(S.resample("ME").aggregate(np.sum), Series)
        check(S.resample("ME").aggregate([np.mean]), DataFrame)
        check(S.resample("ME").aggregate(["sum", np.mean]), DataFrame)
        check(S.resample("ME").aggregate({"sum": np.sum}), DataFrame)
        check(S.resample("ME").aggregate({"sum": np.sum, "mean": np.mean}), DataFrame)
    check(S.resample("ME").aggregate("sum"), Series)
    check(S.resample("ME").aggregate(s2series), Series)
    check(S.resample("ME").aggregate(s2scalar), Series)


def test_aggregate_frame_combinations() -> None:
    def df2frame(val: DataFrame) -> DataFrame:
        return pd.DataFrame(val)

    def df2series(val: DataFrame) -> Series:
        return val.mean()

    def df2scalar(val: DataFrame) -> float:
        return float(val.mean().mean())

    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        lower="2.0.99",
    ):
        check(DF.resample("ME").aggregate(np.sum), DataFrame)
        check(DF.resample("ME").aggregate([np.mean]), DataFrame)
        check(DF.resample("ME").aggregate(["sum", np.mean]), DataFrame)
        check(DF.resample("ME").aggregate({"col1": np.sum}), DataFrame)
        check(DF.resample("ME").aggregate({"col1": np.sum, "col2": np.mean}), DataFrame)
        check(
            DF.resample("ME").aggregate({"col1": [np.sum], "col2": ["sum", np.mean]}),
            DataFrame,
        )
        check(
            DF.resample("ME").aggregate({"col1": np.sum, "col2": ["sum", np.mean]}),
            DataFrame,
        )
        check(
            DF.resample("ME").aggregate({"col1": "sum", "col2": [np.mean]}), DataFrame
        )

    check(DF.resample("ME").aggregate("sum"), DataFrame)
    check(DF.resample("ME").aggregate(df2frame), DataFrame)
    check(DF.resample("ME").aggregate(df2series), DataFrame)
    check(DF.resample("ME").aggregate(df2scalar), DataFrame)
