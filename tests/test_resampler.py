from typing import (
    Generator,
    Hashable,
    Tuple,
    Union,
)

import numpy as np
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

_AggRetType = Union[DataFrame, Series, Scalar]
_PipeRetType = Union[_AggRetType, Resampler]


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


def test_quantile():
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
            DF.resample("m").aggregate({"col1": sum, "col2": np.mean}), _AggRetType
        ),
        DataFrame,
    )
    check(
        assert_type(
            DF.resample("m").aggregate({"col1": [sum, np.mean], "col2": np.mean}),
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

    check(assert_type(DF.resample("m").pipe(f), _PipeRetType), DataFrame)

    def g(val: DataFrame) -> Series:
        return val.mean()

    check(assert_type(DF.resample("m").pipe(g), _PipeRetType), DataFrame)


def test_transform() -> None:
    def f(val: Series) -> Series:
        return -1 * val

    check(assert_type(DF.resample("m").transform(f), DataFrame), DataFrame)
