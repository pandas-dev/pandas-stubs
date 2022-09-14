import numpy as np
from pandas import (
    DataFrame,
    Series,
    date_range,
)
from pandas.core.window import (
    Rolling,
    Window,
)
from typing_extensions import assert_type

from tests import check

IDX = date_range("1/1/2000", periods=700, freq="D")
S = Series(np.random.standard_normal(700))
DF = DataFrame({"col1": S, "col2": S})


def test_rolling_basic() -> None:
    check(assert_type(DF.rolling(win_type="gaussian"), Window), Window)
    check(assert_type(DF.rolling(10, min_periods=10), Rolling), Rolling)


def test_rolling_basic_math() -> None:
    check(assert_type(DF.rolling(10, min_periods=10).count(), DataFrame), DataFrame)
    check(assert_type(DF.rolling(10, min_periods=10).sum(), DataFrame), DataFrame)
    check(assert_type(DF.rolling(10, min_periods=10).mean(), DataFrame), DataFrame)
    check(assert_type(DF.rolling(10, min_periods=10).median(), DataFrame), DataFrame)
    check(assert_type(DF.rolling(10, min_periods=10).var(), DataFrame), DataFrame)
    check(assert_type(DF.rolling(10, min_periods=10).std(), DataFrame), DataFrame)
    check(assert_type(DF.rolling(10, min_periods=10).min(), DataFrame), DataFrame)
    check(assert_type(DF.rolling(10, min_periods=10).max(), DataFrame), DataFrame)
    check(assert_type(DF.rolling(10, min_periods=10).corr(), DataFrame), DataFrame)
    check(assert_type(DF.rolling(10, min_periods=10).cov(), DataFrame), DataFrame)
    check(assert_type(DF.rolling(10, min_periods=10).skew(), DataFrame), DataFrame)
    check(assert_type(DF.rolling(10, min_periods=10).kurt(), DataFrame), DataFrame)
    check(assert_type(DF.rolling(10, min_periods=10).sem(), DataFrame), DataFrame)
    check(assert_type(DF.rolling(10).quantile(0.5), DataFrame), DataFrame)
    check(assert_type(DF.rolling(10).rank("average"), DataFrame), DataFrame)
    check(assert_type(DF.rolling(10).rank("min"), DataFrame), DataFrame)
    check(assert_type(DF.rolling(10).rank("max"), DataFrame), DataFrame)


def test_rolling_apply() -> None:
    check(assert_type(DF.rolling(10).apply(np.mean), DataFrame), DataFrame)

    def _mean(df: DataFrame) -> Series:
        return df.mean()

    check(assert_type(DF.rolling(10).apply(_mean), DataFrame), DataFrame)

    def _mean2(df: DataFrame) -> np.ndarray:
        return np.mean(df, axis=0)

    check(assert_type(DF.rolling(10).apply(_mean2, raw=True), DataFrame), DataFrame)

    def _mean4(df: DataFrame) -> float:
        return float(np.mean(df))

    check(assert_type(DF.rolling(10).apply(_mean4, raw=True), DataFrame), DataFrame)


def test_rolling_aggregate() -> None:
    check(assert_type(DF.rolling(10).aggregate(np.mean), DataFrame), DataFrame)
    check(
        assert_type(DF.rolling(10).aggregate(["mean", np.mean]), DataFrame), DataFrame
    )
    check(
        assert_type(
            DF.rolling(10).aggregate({"col1": "mean", "col2": np.mean}), DataFrame
        ),
        DataFrame,
    )
    check(assert_type(DF.rolling(10).agg("sum"), DataFrame), DataFrame)


def test_rolling_basic_math_series() -> None:
    check(assert_type(S.rolling(10, min_periods=10).count(), Series), Series)
    check(assert_type(S.rolling(10, min_periods=10).sum(), Series), Series)
    check(assert_type(S.rolling(10, min_periods=10).mean(), Series), Series)
    check(assert_type(S.rolling(10, min_periods=10).median(), Series), Series)
    check(assert_type(S.rolling(10, min_periods=10).var(), Series), Series)
    check(assert_type(S.rolling(10, min_periods=10).std(), Series), Series)
    check(assert_type(S.rolling(10, min_periods=10).min(), Series), Series)
    check(assert_type(S.rolling(10, min_periods=10).max(), Series), Series)
    check(assert_type(S.rolling(10, min_periods=10).corr(), Series), Series)
    check(assert_type(S.rolling(10, min_periods=10).cov(), Series), Series)
    check(assert_type(S.rolling(10, min_periods=10).skew(), Series), Series)
    check(assert_type(S.rolling(10, min_periods=10).kurt(), Series), Series)
    check(assert_type(S.rolling(10, min_periods=10).sem(), Series), Series)
    check(assert_type(S.rolling(10).quantile(0.5), Series), Series)
    check(assert_type(S.rolling(10).rank("average"), Series), Series)
    check(assert_type(S.rolling(10).rank("min"), Series), Series)
    check(assert_type(S.rolling(10).rank("max"), Series), Series)


def test_rolling_apply_series() -> None:
    check(assert_type(S.rolling(10).apply(np.mean), Series), Series)

    def _mean(df: Series) -> float:
        return df.mean()

    check(assert_type(S.rolling(10).apply(_mean), Series), Series)

    def _mean2(df: Series) -> np.ndarray:
        return np.mean(df, axis=0)

    check(assert_type(S.rolling(10).apply(_mean2, raw=True), Series), Series)


def test_rolling_aggregate_series() -> None:
    check(assert_type(S.rolling(10).aggregate(np.mean), Series), Series)
    # TODO: Make sure this works
    check(assert_type(S.rolling(10).aggregate(["mean", np.mean]), Series), DataFrame)
    check(
        assert_type(S.rolling(10).aggregate({"col1": "mean", "col2": np.mean}), Series),
        DataFrame,
    )
    check(assert_type(S.rolling(10).agg("sum"), Series), Series)


def test_expanding_basic_math() -> None:
    check(assert_type(DF.expanding(10).count(), DataFrame), DataFrame)
    check(assert_type(DF.expanding(10).sum(), DataFrame), DataFrame)
    check(assert_type(DF.expanding(10).mean(), DataFrame), DataFrame)
    check(assert_type(DF.expanding(10).median(), DataFrame), DataFrame)
    check(assert_type(DF.expanding(10).var(), DataFrame), DataFrame)
    check(assert_type(DF.expanding(10).std(), DataFrame), DataFrame)
    check(assert_type(DF.expanding(10).min(), DataFrame), DataFrame)
    check(assert_type(DF.expanding(10).max(), DataFrame), DataFrame)
    check(assert_type(DF.expanding(10).corr(), DataFrame), DataFrame)
    check(assert_type(DF.expanding(10).cov(), DataFrame), DataFrame)
    check(assert_type(DF.expanding(10).skew(), DataFrame), DataFrame)
    check(assert_type(DF.expanding(10).kurt(), DataFrame), DataFrame)
    check(assert_type(DF.expanding(10).sem(), DataFrame), DataFrame)
    check(assert_type(DF.expanding(10).quantile(0.5), DataFrame), DataFrame)
    check(assert_type(DF.expanding(10).rank("average"), DataFrame), DataFrame)
    check(assert_type(DF.expanding(10).rank("min"), DataFrame), DataFrame)
    check(assert_type(DF.expanding(10).rank("max"), DataFrame), DataFrame)


def test_expanding_apply() -> None:
    check(assert_type(DF.expanding(10).apply(np.mean), DataFrame), DataFrame)

    def _mean(df: DataFrame) -> Series:
        return df.mean()

    check(assert_type(DF.expanding(10).apply(_mean), DataFrame), DataFrame)

    def _mean2(df: DataFrame) -> np.ndarray:
        return np.mean(df, axis=0)

    check(assert_type(DF.expanding(10).apply(_mean2, raw=True), DataFrame), DataFrame)

    def _mean4(df: DataFrame) -> float:
        return float(np.mean(df))

    check(assert_type(DF.expanding(10).apply(_mean4, raw=True), DataFrame), DataFrame)


def test_expanding_aggregate() -> None:
    check(assert_type(DF.expanding(10).aggregate(np.mean), DataFrame), DataFrame)
    check(
        assert_type(DF.expanding(10).aggregate(["mean", np.mean]), DataFrame), DataFrame
    )
    check(
        assert_type(
            DF.expanding(10).aggregate({"col1": "mean", "col2": np.mean}), DataFrame
        ),
        DataFrame,
    )
    check(assert_type(DF.expanding(10).agg("sum"), DataFrame), DataFrame)


def test_expanding_basic_math_series() -> None:
    check(assert_type(S.expanding(10).count(), Series), Series)
    check(assert_type(S.expanding(10).sum(), Series), Series)
    check(assert_type(S.expanding(10).mean(), Series), Series)
    check(assert_type(S.expanding(10).median(), Series), Series)
    check(assert_type(S.expanding(10).var(), Series), Series)
    check(assert_type(S.expanding(10).std(), Series), Series)
    check(assert_type(S.expanding(10).min(), Series), Series)
    check(assert_type(S.expanding(10).max(), Series), Series)
    check(assert_type(S.expanding(10).corr(), Series), Series)
    check(assert_type(S.expanding(10).cov(), Series), Series)
    check(assert_type(S.expanding(10).skew(), Series), Series)
    check(assert_type(S.expanding(10).kurt(), Series), Series)
    check(assert_type(S.expanding(10).sem(), Series), Series)
    check(assert_type(S.expanding(10).quantile(0.5), Series), Series)
    check(assert_type(S.expanding(10).rank("average"), Series), Series)
    check(assert_type(S.expanding(10).rank("min"), Series), Series)
    check(assert_type(S.expanding(10).rank("max"), Series), Series)


def test_expanding_apply_series() -> None:
    check(assert_type(S.expanding(10).apply(np.mean), Series), Series)

    def _mean(df: Series) -> float:
        return df.mean()

    check(assert_type(S.expanding(10).apply(_mean), Series), Series)

    def _mean2(df: Series) -> np.ndarray:
        return np.mean(df, axis=0)

    check(assert_type(S.expanding(10).apply(_mean2, raw=True), Series), Series)


def test_expanding_aggregate_series() -> None:
    check(assert_type(S.expanding(10).aggregate(np.mean), Series), Series)
    # TODO: Make sure this works
    check(assert_type(S.expanding(10).aggregate(["mean", np.mean]), Series), DataFrame)
    check(
        assert_type(
            S.expanding(10).aggregate({"col1": "mean", "col2": np.mean}), Series
        ),
        DataFrame,
    )
    check(assert_type(S.expanding(10).agg("sum"), Series), Series)


def test_ewm_basic_math() -> None:
    check(assert_type(DF.ewm(span=10).sum(), DataFrame), DataFrame)
    check(assert_type(DF.ewm(span=10).mean(), DataFrame), DataFrame)
    check(assert_type(DF.ewm(span=10).var(), DataFrame), DataFrame)
    check(assert_type(DF.ewm(span=10).std(), DataFrame), DataFrame)
    check(assert_type(DF.ewm(span=10).corr(), DataFrame), DataFrame)
    check(assert_type(DF.ewm(span=10).cov(), DataFrame), DataFrame)


def test_ewm_aggregate() -> None:
    check(assert_type(DF.ewm(span=10).aggregate(np.mean), DataFrame), DataFrame)
    check(
        assert_type(DF.ewm(span=10).aggregate(["mean", np.mean]), DataFrame), DataFrame
    )
    check(
        assert_type(
            DF.ewm(span=10).aggregate({"col1": "mean", "col2": np.mean}), DataFrame
        ),
        DataFrame,
    )
    check(assert_type(DF.ewm(span=10).agg("sum"), DataFrame), DataFrame)


def test_ewm_basic_math_series() -> None:
    check(assert_type(S.ewm(span=10).sum(), Series), Series)
    check(assert_type(S.ewm(span=10).mean(), Series), Series)
    check(assert_type(S.ewm(span=10).var(), Series), Series)
    check(assert_type(S.ewm(span=10).std(), Series), Series)
    check(assert_type(S.ewm(span=10).corr(), Series), Series)
    check(assert_type(S.ewm(span=10).cov(), Series), Series)


def test_ewm_aggregate_series() -> None:
    check(assert_type(S.ewm(span=10).aggregate(np.mean), Series), Series)
    # TODO: Make sure this works
    check(assert_type(S.ewm(span=10).aggregate(["mean", np.mean]), Series), DataFrame)
    check(
        assert_type(
            S.ewm(span=10).aggregate({"col1": "mean", "col2": np.mean}), Series
        ),
        DataFrame,
    )
    check(assert_type(S.ewm(span=10).agg("sum"), Series), Series)
