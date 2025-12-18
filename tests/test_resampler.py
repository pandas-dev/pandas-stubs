# pyright: reportMissingTypeArgument=false
from collections.abc import (
    Hashable,
    Iterator,
)
from typing import TypeAlias

import numpy as np
import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
)
from pandas.core.groupby.generic import (
    DataFrameGroupBy,
    SeriesGroupBy,
)
from pandas.core.resample import DatetimeIndexResampler
from typing_extensions import assert_type

from tests import (
    PD_LTE_23,
    TYPE_CHECKING_INVALID_USAGE,
    check,
    pytest_warns_bounded,
)

if not PD_LTE_23:
    from pandas.errors import Pandas4Warning  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue,reportRedeclaration] # isort: skip
else:
    Pandas4Warning: TypeAlias = FutureWarning  # type: ignore[no-redef]

DR = date_range("1999-1-1", periods=365, freq="D")
DF_ = DataFrame(np.random.standard_normal((365, 1)), index=DR)
S = DF_.iloc[:, 0]
DF = DataFrame({"col1": S, "col2": S})


_AggRetType = DataFrame | Series


def test_iter() -> None:
    assert_type(iter(DF.resample("ME")), Iterator[tuple[Hashable, DataFrame]])
    for v in DF.resample("ME"):
        check(assert_type(v, tuple[Hashable, DataFrame]), tuple)


def test_agg_funcs() -> None:
    check(assert_type(DF.resample("ME").sum(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").prod(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").min(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").max(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").first(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").first(skipna=False), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").last(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").last(skipna=False), DataFrame), DataFrame)
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
    check(assert_type(DF.resample("ME").size(), "Series[int]"), Series, np.integer)
    check(assert_type(DF.resample("ME").count(), DataFrame), DataFrame)


def test_filling() -> None:
    check(assert_type(DF.resample("ME").ffill(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").nearest(), DataFrame), DataFrame)
    check(assert_type(DF.resample("ME").bfill(), DataFrame), DataFrame)


def test_fillna() -> None:
    # deprecated (and removed from stub)
    if TYPE_CHECKING_INVALID_USAGE:
        DF.resample("ME").fillna("pad")  # type: ignore[operator] # pyright: ignore


def test_aggregate() -> None:
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        upper="2.3.99",
    ):
        check(assert_type(DF.resample("ME").aggregate(np.sum), DataFrame), DataFrame)
        check(assert_type(DF.resample("ME").agg(np.sum), DataFrame), DataFrame)
        check(assert_type(DF.resample("ME").apply(np.sum), DataFrame), DataFrame)
        check(
            assert_type(DF.resample("ME").aggregate([np.sum, np.mean]), DataFrame),
            DataFrame,
        )
        check(
            assert_type(DF.resample("ME").aggregate(["sum", np.mean]), DataFrame),
            DataFrame,
        )
        check(
            assert_type(
                DF.resample("ME").aggregate({"col1": "sum", "col2": np.mean}),
                DataFrame,
            ),
            DataFrame,
        )
        check(
            assert_type(
                DF.resample("ME").aggregate(
                    {"col1": ["sum", np.mean], "col2": np.mean}
                ),
                DataFrame,
            ),
            DataFrame,
        )

    def f(val: DataFrame) -> Series:
        return val.mean()

    check(assert_type(DF.resample("ME").aggregate(f), DataFrame), DataFrame)


def test_asfreq() -> None:
    check(assert_type(DF.resample("ME").asfreq(-1.0), DataFrame), DataFrame)


def test_getattr() -> None:
    check(assert_type(DF.resample("ME").col1, SeriesGroupBy), SeriesGroupBy)


def test_interpolate() -> None:
    check(assert_type(DF.resample("ME").interpolate(), DataFrame), DataFrame)
    check(
        assert_type(DF.resample("ME").interpolate(method="time"), DataFrame),
        DataFrame,
    )


# TODO: remove the whole test function when the warning and ValueError in pandas-dev/pandas#62847 are removed
def test_interpolate_inplace() -> None:
    with pytest_warns_bounded(
        Pandas4Warning,
        r"The 'inplace' keyword in DatetimeIndexResampler.interpolate is deprecated and will be removed in a future version. resample\(...\).interpolate is never inplace.",
        lower="2.99",
    ):
        check(
            assert_type(DF.resample("ME").interpolate(inplace=False), DataFrame),
            DataFrame,
        )
    if TYPE_CHECKING_INVALID_USAGE:
        DF.resample("ME").interpolate(inplace=True)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]


def test_pipe() -> None:
    def f(val: "DatetimeIndexResampler[DataFrame]") -> DataFrame:
        assert isinstance(val, DatetimeIndexResampler)
        return DataFrame(val)

    check(assert_type(DF.resample("ME").pipe(f), DataFrame), DataFrame)

    def g(val: "DatetimeIndexResampler[DataFrame]") -> DataFrame:
        assert isinstance(val, DatetimeIndexResampler)
        return val.mean()

    check(assert_type(DF.resample("ME").pipe(g), DataFrame), DataFrame)

    def h(val: "DatetimeIndexResampler[DataFrame]") -> Series:
        assert isinstance(val, DatetimeIndexResampler)
        return val.mean().mean()

    check(assert_type(DF.resample("ME").pipe(h), Series), Series)

    def i(val: "DatetimeIndexResampler[DataFrame]") -> float:
        assert isinstance(val, DatetimeIndexResampler)
        return float(val.mean().mean().mean())

    check(assert_type(DF.resample("ME").pipe(i), float), float)

    def j(
        res: "DatetimeIndexResampler[DataFrame]",
        pos: int,
        /,
        arg1: list[float],
        arg2: str,
        *,
        kw: tuple[int],
    ) -> DataFrame:
        assert isinstance(res, DatetimeIndexResampler)
        return res.obj  # type: ignore[return-value]  # pyright: ignore[reportReturnType]

    check(
        assert_type(DF.resample("ME").pipe(j, 1, [1.0], arg2="hi", kw=(1,)), DataFrame),
        DataFrame,
    )

    if TYPE_CHECKING_INVALID_USAGE:
        DF.resample("ME").pipe(
            j,
            "a",  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
            [1.0, 2.0],
            arg2="hi",
            kw=(1,),
        )
        DF.resample("ME").pipe(
            j,
            1,
            [1.0, "b"],  # type: ignore[list-item] # pyright: ignore[reportArgumentType,reportCallIssue]
            arg2="hi",
            kw=(1,),
        )
        DF.resample("ME").pipe(
            j,
            1,
            [1.0],
            arg2=11,  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
            kw=(1,),
        )
        DF.resample("ME").pipe(
            j,
            1,
            [1.0],
            arg2="hi",
            kw=(1, 2),  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        )
        DF.resample("ME").pipe(  # type: ignore[call-arg]
            j,
            1,
            [1.0],
            arg3="hi",  # pyright: ignore[reportCallIssue]
            kw=(1,),
        )
        DF.resample("ME").pipe(  # type: ignore[call-overload]
            j,
            1,
            [1.0],
            11,
            (1,),  # pyright: ignore[reportCallIssue]
        )
        DF.resample("ME").pipe(  # type: ignore[call-overload]
            j,
            pos=1,  # pyright: ignore[reportCallIssue]
            arg1=[1.0],
            arg2=11,
            kw=(1,),
        )

    def k(x: int, t: "DatetimeIndexResampler[DataFrame]") -> DataFrame:
        assert isinstance(x, int)
        return t.obj  # type: ignore[return-value] # pyright: ignore[reportReturnType]

    check(assert_type(DF.resample("ME").pipe((k, "t"), 1), DataFrame), DataFrame)

    if TYPE_CHECKING_INVALID_USAGE:
        DF.resample("ME").pipe(  # pyright: ignore[reportCallIssue]
            (k, 1),  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
            1,
        )


def test_transform() -> None:
    def f(val: Series) -> Series:
        return -1 * val

    check(assert_type(DF.resample("ME").transform(f), DataFrame), DataFrame)


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
    check(assert_type(S.resample("ME").nunique(), "Series[int]"), Series, np.integer)


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
    check(assert_type(S.resample("ME").size(), "Series[int]"), Series, np.integer)
    check(assert_type(S.resample("ME").count(), "Series[int]"), Series, np.integer)


def test_filling_series() -> None:
    check(assert_type(S.resample("ME").ffill(), Series), Series)
    check(assert_type(S.resample("ME").nearest(), Series), Series)
    check(assert_type(S.resample("ME").bfill(), Series), Series)


def test_fillna_series() -> None:
    # deprecated (and removed from stub)
    if TYPE_CHECKING_INVALID_USAGE:
        S.resample("ME").fillna("pad")  # type: ignore[operator] # pyright: ignore


def test_aggregate_series() -> None:
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <function (sum|mean) .*> is currently using ",
        upper="2.3.99",
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


def test_pipe_series() -> None:
    def f(val: "DatetimeIndexResampler[Series]") -> Series:
        assert isinstance(val, DatetimeIndexResampler)
        return Series(val)

    check(assert_type(S.resample("ME").pipe(f), Series), Series)

    def g(val: "DatetimeIndexResampler[Series]") -> float:
        assert isinstance(val, DatetimeIndexResampler)
        return float(val.mean().mean())

    check(assert_type(S.resample("ME").pipe(g), float), float)

    def h(val: "DatetimeIndexResampler[Series]") -> DataFrame:
        assert isinstance(val, DatetimeIndexResampler)
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
        upper="2.3.99",
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
        upper="2.3.99",
    ):
        check(DF.resample("ME").aggregate(np.sum), DataFrame)
        check(DF.resample("ME").aggregate([np.mean]), DataFrame)
        check(DF.resample("ME").aggregate(["sum", np.mean]), DataFrame)
        check(DF.resample("ME").aggregate({"col1": np.sum}), DataFrame)
        check(
            DF.resample("ME").aggregate({"col1": np.sum, "col2": np.mean}),
            DataFrame,
        )
        check(
            DF.resample("ME").aggregate({"col1": [np.sum], "col2": ["sum", np.mean]}),
            DataFrame,
        )
        check(
            DF.resample("ME").aggregate({"col1": np.sum, "col2": ["sum", np.mean]}),
            DataFrame,
        )
        check(
            DF.resample("ME").aggregate({"col1": "sum", "col2": [np.mean]}),
            DataFrame,
        )

    check(DF.resample("ME").aggregate("sum"), DataFrame)
    check(DF.resample("ME").aggregate(df2frame), DataFrame)
    check(DF.resample("ME").aggregate(df2series), DataFrame)
    check(DF.resample("ME").aggregate(df2scalar), DataFrame)


def test_getitem() -> None:
    check(assert_type(DF.resample("ME")["col1"], SeriesGroupBy), SeriesGroupBy)
    check(
        assert_type(DF.resample("ME")[["col1", "col2"]], DataFrameGroupBy),
        DataFrameGroupBy,
    )
