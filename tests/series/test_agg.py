import numpy as np
import pandas as pd
from typing_extensions import (
    Never,
    assert_type,
)

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
    pytest_warns_bounded,
)


def test_agg_any_float() -> None:
    series = pd.DataFrame({"A": [1.0, float("nan"), 2.0]})["A"]
    check(assert_type(series.mean(), float), np.float64)
    check(assert_type(series.median(), float), np.float64)
    check(assert_type(series.std(), float), np.float64)
    check(assert_type(series.var(), float), np.float64)


def test_agg_bool() -> None:
    series = pd.Series([True, False, True])
    check(assert_type(series.mean(), float), np.float64)
    check(assert_type(series.median(), float), np.float64)
    check(assert_type(series.std(), float), np.float64)
    check(assert_type(series.var(), float), np.float64)


def test_agg_int() -> None:
    series = pd.Series([3, 1, 2])
    check(assert_type(series.mean(), float), np.float64)
    check(assert_type(series.median(), float), np.float64)
    check(assert_type(series.std(), float), np.float64)
    check(assert_type(series.var(), float), np.float64)


def test_agg_float() -> None:
    series = pd.Series([3.0, float("nan"), 2.0])
    check(assert_type(series.mean(), float), np.float64)
    check(assert_type(series.median(), float), np.float64)
    check(assert_type(series.std(), float), np.float64)
    check(assert_type(series.var(), float), np.float64)


def test_agg_complex() -> None:
    series = pd.Series([3j, 3 + 4j, 2j])
    check(assert_type(series, "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(series.mean(), complex), np.complex128)
    with pytest_warns_bounded(
        np.exceptions.ComplexWarning,
        r"Casting complex values to real discards the imaginary part",
    ):
        check(assert_type(series.median(), float), np.float64)
    with (
        pytest_warns_bounded(
            np.exceptions.ComplexWarning,
            r"Casting complex values to real discards the imaginary part",
        ),
        pytest_warns_bounded(
            RuntimeWarning, r"invalid value encountered in sqrt", upper="2.3.99"
        ),
    ):
        check(assert_type(series.std(), np.float64), np.float64)
    with pytest_warns_bounded(
        np.exceptions.ComplexWarning,
        r"Casting complex values to real discards the imaginary part",
        upper="2.3.99",
    ):
        check(assert_type(series.var(), float), np.float64)


def test_agg_str() -> None:
    series = pd.Series(["1", "a", "🐼"])
    if TYPE_CHECKING_INVALID_USAGE:
        series.mean()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue]
        series.median()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue]
        series.std()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue]
        series.var()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue]


def test_agg_ts() -> None:
    series = pd.Series(pd.to_datetime(["2025-09-18", "2025-09-18", "2025-09-18"]))
    check(assert_type(series, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    check(assert_type(series.mean(), pd.Timestamp), pd.Timestamp)
    check(assert_type(series.median(), pd.Timestamp), pd.Timestamp)
    check(assert_type(series.std(), pd.Timedelta), pd.Timedelta)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(series.var(), Never)


def test_agg_td() -> None:
    series = pd.Series(pd.to_timedelta(["1 days", "2 days", "3 days"]))
    check(assert_type(series, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(series.mean(), pd.Timedelta), pd.Timedelta)
    check(assert_type(series.median(), pd.Timedelta), pd.Timedelta)
    check(assert_type(series.std(), pd.Timedelta), pd.Timedelta)

    if TYPE_CHECKING_INVALID_USAGE:

        def _0() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(series.var(), Never)
