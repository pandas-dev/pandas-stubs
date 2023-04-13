from typing import (
    Generator,
    Hashable,
    Tuple,
    Union,
)

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

from tests import check

DR = date_range("1999-1-1", periods=365, freq="D")
DF_ = DataFrame(np.random.standard_normal((365, 1)), index=DR)
S = DF_.iloc[:, 0]
DF = DataFrame({"col1": S, "col2": S})

_AggRetType = Union[DataFrame, Series]


def test_props() -> None:
    check(assert_type(DF.resample("m").obj, DataFrame), DataFrame)
    check(assert_type(DF.resample("m").ax, Index), DatetimeIndex)


def test_iter() -> None:
    assert_type(
        iter(DF.resample("m")), Generator[Tuple[Hashable, DataFrame], None, None]
    )
    for v in DF.resample("m"):
        check(assert_type(v, Tuple[Hashable, DataFrame]), tuple)


def test_agg_funcs() -> None:
    check(assert_type(DF.resample("m").sum(), DataFrame), DataFrame)
    check(assert_type(DF.resample("m").prod(), DataFrame), DataFrame)
    check(assert_type(DF.resample("m").min(), DataFrame), DataFrame)
    check(assert_type(DF.resample("m").max(), DataFrame), DataFrame)
    check(assert_type(DF.resample("m").first(), DataFrame), DataFrame)
    check(assert_type(DF.resample("m").last(), DataFrame), DataFrame)
    check(assert_type(DF.resample("m").mean(), DataFrame), DataFrame)
    check(assert_type(DF.resample("m").sum(), DataFrame), DataFrame)
    check(assert_type(DF.resample("m").median(), DataFrame), DataFrame)
    check(assert_type(DF.resample("m").ohlc(), DataFrame), DataFrame)
    check(assert_type(DF.resample("m").nunique(), DataFrame), DataFrame)


def test_quantile() -> None:
    check(assert_type(DF.resample("m").quantile(0.5), DataFrame), DataFrame)
    check(assert_type(DF.resample("m").quantile([0.5, 0.7]), DataFrame), DataFrame)
    check(
        assert_type(DF.resample("m").quantile(np.array([0.5, 0.7])), DataFrame),
        DataFrame,
    )


def test_std_var() -> None:
    check(assert_type(DF.resample("m").std(), DataFrame), DataFrame)
    check(assert_type(DF.resample("m").var(2), DataFrame), DataFrame)


def test_size_count() -> None:
    check(assert_type(DF.resample("m").size(), Series), Series)
    check(assert_type(DF.resample("m").count(), DataFrame), DataFrame)


def test_filling() -> None:
    check(assert_type(DF.resample("m").ffill(), DataFrame), DataFrame)
    check(assert_type(DF.resample("m").nearest(), DataFrame), DataFrame)
    check(assert_type(DF.resample("m").bfill(), DataFrame), DataFrame)


def test_fillna() -> None:
    check(assert_type(DF.resample("m").fillna("pad"), DataFrame), DataFrame)
    check(assert_type(DF.resample("m").fillna("backfill"), DataFrame), DataFrame)
    check(assert_type(DF.resample("m").fillna("ffill"), DataFrame), DataFrame)
    check(assert_type(DF.resample("m").fillna("bfill"), DataFrame), DataFrame)
    check(
        assert_type(DF.resample("m").fillna("nearest", limit=2), DataFrame), DataFrame
    )


def test_aggregate() -> None:
    check(assert_type(DF.resample("m").aggregate(np.sum), _AggRetType), DataFrame)
    check(assert_type(DF.resample("m").agg(np.sum), _AggRetType), DataFrame)
    check(assert_type(DF.resample("m").apply(np.sum), _AggRetType), DataFrame)
    check(
        assert_type(DF.resample("m").aggregate([np.sum, np.mean]), _AggRetType),
        DataFrame,
    )
    check(
        assert_type(DF.resample("m").aggregate(["sum", np.mean]), _AggRetType),
        DataFrame,
    )
    check(
        assert_type(
            DF.resample("m").aggregate({"col1": "sum", "col2": np.mean}),
            _AggRetType,
        ),
        DataFrame,
    )
    check(
        assert_type(
            DF.resample("m").aggregate({"col1": ["sum", np.mean], "col2": np.mean}),
            _AggRetType,
        ),
        DataFrame,
    )

    def f(val: DataFrame) -> Series:
        return val.mean()

    check(assert_type(DF.resample("m").aggregate(f), _AggRetType), DataFrame)


def test_asfreq() -> None:
    check(assert_type(DF.resample("m").asfreq(-1.0), DataFrame), DataFrame)


def test_getattr() -> None:
    check(assert_type(DF.resample("m").col1, SeriesGroupBy), SeriesGroupBy)


def test_interpolate() -> None:
    check(assert_type(DF.resample("m").interpolate(), DataFrame), DataFrame)
    check(
        assert_type(DF.resample("m").interpolate(method="time"), DataFrame), DataFrame
    )


def test_interpolate_inplace() -> None:
    check(assert_type(DF.resample("m").interpolate(inplace=True), None), type(None))


def test_pipe() -> None:
    def f(val: DataFrame) -> DataFrame:
        return DataFrame(val)

    check(assert_type(DF.resample("m").pipe(f), DataFrame), DataFrame)

    def g(val: DataFrame) -> Series:
        return val.mean()

    check(assert_type(DF.resample("m").pipe(g), DataFrame), DataFrame)

    def h(val: DataFrame) -> float:
        return val.mean().mean()

    check(assert_type(DF.resample("m").pipe(h), Series), Series)


def test_transform() -> None:
    def f(val: Series) -> Series:
        return -1 * val

    check(assert_type(DF.resample("m").transform(f), DataFrame), DataFrame)


def test_props_series() -> None:
    check(assert_type(S.resample("m").obj, Series), Series)
    check(assert_type(S.resample("m").ax, Index), DatetimeIndex)


def test_iter_series() -> None:
    for v in S.resample("m"):
        check(assert_type(v, Tuple[Hashable, Series]), tuple)


def test_agg_funcs_series() -> None:
    check(assert_type(S.resample("m").sum(), Series), Series)
    check(assert_type(S.resample("m").prod(), Series), Series)
    check(assert_type(S.resample("m").min(), Series), Series)
    check(assert_type(S.resample("m").max(), Series), Series)
    check(assert_type(S.resample("m").first(), Series), Series)
    check(assert_type(S.resample("m").last(), Series), Series)
    check(assert_type(S.resample("m").mean(), Series), Series)
    check(assert_type(S.resample("m").sum(), Series), Series)
    check(assert_type(S.resample("m").median(), Series), Series)
    check(assert_type(S.resample("m").ohlc(), DataFrame), DataFrame)
    check(assert_type(S.resample("m").nunique(), Series), Series)


def test_quantile_series() -> None:
    check(assert_type(S.resample("m").quantile(0.5), Series), Series)
    check(assert_type(S.resample("m").quantile([0.5, 0.7]), Series), Series)
    check(
        assert_type(S.resample("m").quantile(np.array([0.5, 0.7])), Series),
        Series,
    )


def test_std_var_series() -> None:
    check(assert_type(S.resample("m").std(), Series), Series)
    check(assert_type(S.resample("m").var(2), Series), Series)


def test_size_count_series() -> None:
    check(assert_type(S.resample("m").size(), Series), Series)
    check(assert_type(S.resample("m").count(), Series), Series)


def test_filling_series() -> None:
    check(assert_type(S.resample("m").ffill(), Series), Series)
    check(assert_type(S.resample("m").nearest(), Series), Series)
    check(assert_type(S.resample("m").bfill(), Series), Series)


def test_fillna_series() -> None:
    check(assert_type(S.resample("m").fillna("pad"), Series), Series)
    check(assert_type(S.resample("m").fillna("backfill"), Series), Series)
    check(assert_type(S.resample("m").fillna("ffill"), Series), Series)
    check(assert_type(S.resample("m").fillna("bfill"), Series), Series)
    check(assert_type(S.resample("m").fillna("nearest", limit=2), Series), Series)


def test_aggregate_series() -> None:
    check(assert_type(S.resample("m").aggregate(np.sum), _AggRetType), Series)
    check(assert_type(S.resample("m").agg(np.sum), _AggRetType), Series)
    check(assert_type(S.resample("m").apply(np.sum), _AggRetType), Series)
    check(
        assert_type(S.resample("m").aggregate([np.sum, np.mean]), _AggRetType),
        DataFrame,
    )
    check(
        assert_type(S.resample("m").aggregate(["sum", np.mean]), _AggRetType),
        DataFrame,
    )
    check(
        assert_type(
            S.resample("m").aggregate({"col1": "sum", "col2": np.mean}),
            _AggRetType,
        ),
        DataFrame,
    )

    def f(val: Series) -> float:
        return val.mean()

    check(assert_type(S.resample("m").aggregate(f), _AggRetType), Series)


def test_asfreq_series() -> None:
    check(assert_type(S.resample("m").asfreq(-1.0), Series), Series)


def test_interpolate_series() -> None:
    check(assert_type(S.resample("m").interpolate(), Series), Series)
    check(assert_type(S.resample("m").interpolate(method="time"), Series), Series)


def test_interpolate_inplace_series() -> None:
    check(assert_type(S.resample("m").interpolate(inplace=True), None), type(None))


def test_pipe_series() -> None:
    def f(val: Series) -> Series:
        return Series(val)

    check(assert_type(S.resample("m").pipe(f), Series), Series)

    def g(val: Resampler) -> float:
        return float(val.mean().mean())

    check(assert_type(S.resample("m").pipe(g), Scalar), float)

    def h(val: Series) -> DataFrame:
        return DataFrame({0: val, 1: val})

    check(assert_type(S.resample("m").pipe(h), DataFrame), DataFrame)


def test_transform_series() -> None:
    def f(val: Series) -> Series:
        return -1 * val

    check(assert_type(S.resample("m").transform(f), Series), Series)


def test_aggregate_series_combinations() -> None:
    def s2series(val: Series) -> Series:
        return pd.Series(val)

    def s2scalar(val: Series) -> float:
        return float(val.mean())

    check(S.resample("m").aggregate(np.sum), Series)
    check(S.resample("m").aggregate("sum"), Series)
    check(S.resample("m").aggregate(s2series), Series)
    check(S.resample("m").aggregate(s2scalar), Series)
    check(S.resample("m").aggregate([np.mean]), DataFrame)
    check(S.resample("m").aggregate(["sum", np.mean]), DataFrame)
    check(S.resample("m").aggregate({"sum": np.sum}), DataFrame)
    check(S.resample("m").aggregate({"sum": np.sum, "mean": np.mean}), DataFrame)


def test_aggregate_frame_combinations() -> None:
    def df2frame(val: DataFrame) -> DataFrame:
        return pd.DataFrame(val)

    def df2series(val: DataFrame) -> Series:
        return val.mean()

    def df2scalar(val: DataFrame) -> float:
        return float(val.mean().mean())

    check(DF.resample("m").aggregate(np.sum), DataFrame)
    check(DF.resample("m").aggregate("sum"), DataFrame)
    check(DF.resample("m").aggregate(df2frame), DataFrame)
    check(DF.resample("m").aggregate(df2series), DataFrame)
    check(DF.resample("m").aggregate(df2scalar), DataFrame)
    check(DF.resample("m").aggregate([np.mean]), DataFrame)
    check(DF.resample("m").aggregate(["sum", np.mean]), DataFrame)
    check(DF.resample("m").aggregate({"col1": np.sum}), DataFrame)
    check(DF.resample("m").aggregate({"col1": np.sum, "col2": np.mean}), DataFrame)
    check(
        DF.resample("m").aggregate({"col1": [np.sum], "col2": ["sum", np.mean]}),
        DataFrame,
    )
    check(
        DF.resample("m").aggregate({"col1": np.sum, "col2": ["sum", np.mean]}),
        DataFrame,
    )
    check(DF.resample("m").aggregate({"col1": "sum", "col2": [np.mean]}), DataFrame)
