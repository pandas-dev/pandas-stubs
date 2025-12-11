from __future__ import annotations

from collections.abc import (
    Callable,
    Hashable,
    Iterator,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
import pandas as pd
from typing_extensions import assert_type

from pandas._typing import Scalar

from tests import (
    PD_LTE_23,
    check,
    pytest_warns_bounded,
)

if TYPE_CHECKING:
    from pandas._typing import S1


def test_types_groupby_as_index() -> None:
    """Test type of groupby.size method depending on `as_index`."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    check(
        assert_type(
            df.groupby("a", as_index=False).size(),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.groupby("a", as_index=True).size(),
            "pd.Series[int]",
        ),
        pd.Series,
    )
    check(
        assert_type(
            df.groupby("a").size(),
            "pd.Series[int]",
        ),
        pd.Series,
    )


def test_types_groupby_as_index_list() -> None:
    """Test type of groupby.size method depending on list of grouper GH1045."""
    df = pd.DataFrame({"a": [1, 1, 2], "b": [2, 3, 2]})
    check(
        assert_type(
            df.groupby(["a", "b"], as_index=False).size(),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.groupby(["a", "b"], as_index=True).size(),
            "pd.Series[int]",
        ),
        pd.Series,
    )
    check(
        assert_type(
            df.groupby(["a", "b"]).size(),
            "pd.Series[int]",
        ),
        pd.Series,
    )


def test_types_groupby_as_index_value_counts() -> None:
    """Test type of groupby.value_counts method depending on `as_index`."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    check(
        assert_type(
            df.groupby("a", as_index=False).value_counts(),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.groupby("a", as_index=True).value_counts(),
            "pd.Series[int]",
        ),
        pd.Series,
    )


def test_types_groupby_size() -> None:
    """Test for GH886."""
    data = [
        {"date": "2023-12-01", "val": 12},
        {"date": "2023-12-02", "val": 2},
        {"date": "2023-12-03", "val": 1},
        {"date": "2023-12-03", "val": 10},
    ]

    df = pd.DataFrame(data)
    groupby = df.groupby("date")
    size = groupby.size()
    frame = size.to_frame()
    check(assert_type(frame.reset_index(), pd.DataFrame), pd.DataFrame)


def test_types_groupby() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5], "col3": [0, 1, 0]})
    df.index.name = "ind"
    df.groupby(by="col1")
    df.groupby(level="ind")
    df.groupby(by="col1", sort=False, as_index=True)
    df.groupby(by=["col1", "col2"])
    # GH 284
    df.groupby(df["col1"] > 2)
    df.groupby([df["col1"] > 2, df["col2"] % 2 == 1])
    df.groupby(lambda x: x)
    df.groupby([lambda x: x % 2, lambda x: x % 3])
    df.groupby(np.array([1, 0, 1]))
    df.groupby([np.array([1, 0, 0]), np.array([0, 0, 1])])
    df.groupby({1: 1, 2: 2, 3: 3})
    df.groupby([{1: 1, 2: 1, 3: 2}, {1: 1, 2: 2, 3: 2}])
    df.groupby(df.index)
    df.groupby([pd.Index([1, 0, 0]), pd.Index([0, 0, 1])])
    df.groupby(pd.Grouper(level=0))
    df.groupby([pd.Grouper(level=0), pd.Grouper(key="col1")])

    check(assert_type(df.groupby(by="col1").agg("sum"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.groupby(level="ind").aggregate("sum"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.groupby(by="col1", sort=False, as_index=True).transform(
                lambda x: x.max()
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(df.groupby(by=["col1", "col2"]).count(), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(
            df.groupby(by=["col1", "col2"]).filter(lambda x: x["col1"] > 0),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(df.groupby(by=["col1", "col2"]).nunique(), pd.DataFrame),
        pd.DataFrame,
    )
    with pytest_warns_bounded(
        FutureWarning,
        "(The provided callable <built-in function sum> is currently using|The behavior of DataFrame.sum with)",
        upper="2.3.99",
    ):
        with pytest_warns_bounded(
            FutureWarning,
            "DataFrameGroupBy.apply operated on the grouping columns",
            upper="2.3.99",
        ):
            if PD_LTE_23:
                check(
                    assert_type(df.groupby(by="col1").apply(sum), pd.DataFrame),
                    pd.DataFrame,
                )
    check(assert_type(df.groupby("col1").transform("sum"), pd.DataFrame), pd.DataFrame)
    s1 = df.set_index("col1")["col2"]
    check(assert_type(s1, pd.Series), pd.Series)
    check(assert_type(s1.groupby("col1").transform("sum"), pd.Series), pd.Series)


def test_types_groupby_methods() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5], "col3": [0, 1, 0]})
    check(assert_type(df.groupby("col1").sum(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.groupby("col1").prod(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.groupby("col1").sample(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.groupby("col1").count(), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.groupby("col1").value_counts(normalize=False), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )
    check(
        assert_type(
            df.groupby("col1").value_counts(subset=None, normalize=True),
            "pd.Series[float]",
        ),
        pd.Series,
        float,
    )
    check(assert_type(df.groupby("col1").idxmax(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.groupby("col1").idxmin(), pd.DataFrame), pd.DataFrame)


def test_types_groupby_agg() -> None:
    df = pd.DataFrame(
        data={"col1": [1, 1, 2], "col2": [3, 4, 5], "col3": [0, 1, 0], 0: [-1, -1, -1]}
    )
    check(assert_type(df.groupby("col1").agg("min"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.groupby("col1").agg(["min", "max"]), pd.DataFrame), pd.DataFrame
    )
    agg_dict1 = {"col2": "min", "col3": "max", 0: "sum"}
    check(assert_type(df.groupby("col1").agg(agg_dict1), pd.DataFrame), pd.DataFrame)

    def wrapped_min(x: pd.Series[S1]) -> S1:
        return x.min()

    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <built-in function (min|max)> is currently using",
        upper="2.3.99",
    ):
        check(assert_type(df.groupby("col1")["col3"].agg(min), pd.Series), pd.Series)
        check(
            assert_type(df.groupby("col1")["col3"].agg([min, max]), pd.DataFrame),
            pd.DataFrame,
        )
        check(assert_type(df.groupby("col1").agg(min), pd.DataFrame), pd.DataFrame)
        check(
            assert_type(df.groupby("col1").agg([min, max]), pd.DataFrame), pd.DataFrame
        )
        agg_dict2 = {"col2": min, "col3": max, 0: min}
        check(
            assert_type(df.groupby("col1").agg(agg_dict2), pd.DataFrame), pd.DataFrame
        )

        # Here, MyPy infers dict[object, object], so it must be explicitly annotated
        agg_dict3: dict[str | int, str | Callable[..., Any]] = {
            "col2": min,
            "col3": "max",
            0: wrapped_min,
        }
        check(
            assert_type(df.groupby("col1").agg(agg_dict3), pd.DataFrame), pd.DataFrame
        )
    agg_dict4 = {"col2": "sum"}
    check(assert_type(df.groupby("col1").agg(agg_dict4), pd.DataFrame), pd.DataFrame)
    agg_dict5 = {0: "sum"}
    check(assert_type(df.groupby("col1").agg(agg_dict5), pd.DataFrame), pd.DataFrame)
    named_agg = pd.NamedAgg(column="col2", aggfunc="max")
    check(
        assert_type(df.groupby("col1").agg(new_col=named_agg), pd.DataFrame),
        pd.DataFrame,
    )
    # GH#187
    cols: list[str] = ["col1", "col2"]
    check(assert_type(df.groupby(by=cols).sum(), pd.DataFrame), pd.DataFrame)

    cols_opt: list[str | None] = ["col1", "col2"]
    check(assert_type(df.groupby(by=cols_opt).sum(), pd.DataFrame), pd.DataFrame)

    cols_mixed: list[str | int] = ["col1", 0]
    check(assert_type(df.groupby(by=cols_mixed).sum(), pd.DataFrame), pd.DataFrame)
    # GH 736
    check(assert_type(df.groupby(by="col1").aggregate("size"), pd.Series), pd.Series)
    check(assert_type(df.groupby(by="col1").agg("size"), pd.Series), pd.Series)


# This was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
def test_types_group_by_with_dropna_keyword() -> None:
    df = pd.DataFrame(
        data={"col1": [1, 1, 2, 1], "col2": [2, None, 1, 2], "col3": [3, 4, 3, 2]}
    )
    check(
        assert_type(df.groupby(by="col2", dropna=True).sum(), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.groupby(by="col2", dropna=False).sum(), pd.DataFrame),
        pd.DataFrame,
    )
    check(assert_type(df.groupby(by="col2").sum(), pd.DataFrame), pd.DataFrame)


def test_types_groupby_any() -> None:
    df = pd.DataFrame(
        data={
            "col1": [1, 1, 2],
            "col2": [True, False, False],
            "col3": [False, False, False],
        }
    )
    check(assert_type(df.groupby("col1").any(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.groupby("col1").all(), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.groupby("col1")["col2"].any(), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(df.groupby("col1")["col2"].any(), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )


def test_types_groupby_iter() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5]})
    series_groupby = pd.Series([True, True, False], dtype=bool)
    first_group = next(iter(df.groupby(series_groupby)))
    check(
        assert_type(first_group[0], bool),
        bool,
    )
    check(
        assert_type(first_group[1], pd.DataFrame),
        pd.DataFrame,
    )


def test_types_groupby_level() -> None:
    # GH 836
    data = {
        "col1": [0, 0, 0],
        "col2": [0, 1, 0],
        "col3": [1, 2, 3],
        "col4": [1, 2, 3],
    }
    df = pd.DataFrame(data=data).set_index(["col1", "col2", "col3"])
    check(
        assert_type(df.groupby(level=["col1", "col2"]).sum(), pd.DataFrame),
        pd.DataFrame,
    )


def test_groupby_series_methods() -> None:
    df = pd.DataFrame({"x": [1, 2, 2, 3, 3], "y": [10, 20, 30, 40, 50]})
    gb = df.groupby("x")["y"]
    check(assert_type(gb.describe(), pd.DataFrame), pd.DataFrame)
    check(assert_type(gb.count().loc[2], int), np.integer)
    check(assert_type(gb.pct_change(), pd.Series), pd.Series)
    check(assert_type(gb.bfill(), pd.Series), pd.Series)
    check(assert_type(gb.cummax(), pd.Series), pd.Series)
    check(assert_type(gb.cummin(), pd.Series), pd.Series)
    check(assert_type(gb.cumprod(), pd.Series), pd.Series)
    check(assert_type(gb.cumsum(), pd.Series), pd.Series)
    check(assert_type(gb.ffill(), pd.Series), pd.Series)
    check(assert_type(gb.first(), pd.Series), pd.Series)
    check(assert_type(gb.head(), pd.Series), pd.Series)
    check(assert_type(gb.last(), pd.Series), pd.Series)
    check(assert_type(gb.max(), pd.Series), pd.Series)
    check(assert_type(gb.mean(), pd.Series), pd.Series)
    check(assert_type(gb.median(), pd.Series), pd.Series)
    check(assert_type(gb.min(), pd.Series), pd.Series)
    check(assert_type(gb.nlargest(), pd.Series), pd.Series)
    check(assert_type(gb.nsmallest(), pd.Series), pd.Series)
    check(assert_type(gb.nth(0), pd.DataFrame | pd.Series), pd.Series)
    check(assert_type(gb.nth[0, 1, 2], pd.DataFrame | pd.Series), pd.Series)
    check(assert_type(gb.nth((0, 1, 2)), pd.DataFrame | pd.Series), pd.Series)


def test_groupby_index() -> None:
    # GH 42
    df = pd.DataFrame(
        data={"col1": [1, 1, 2], "col2": [3, 4, 5], "col3": [0, 1, 0]}
    ).set_index("col1")
    check(assert_type(df.groupby(df.index).min(), pd.DataFrame), pd.DataFrame)


def test_groupby_result() -> None:
    # GH 142
    df = pd.DataFrame({"a": [0, 1, 2], "b": [4, 5, 6], "c": [7, 8, 9]})
    iterator = df.groupby(["a", "b"]).__iter__()
    assert_type(iterator, Iterator[tuple[tuple[Hashable, ...], pd.DataFrame]])
    index, value = next(iterator)
    assert_type((index, value), tuple[tuple[Hashable, ...], pd.DataFrame])

    if PD_LTE_23:
        check(assert_type(index, tuple[Hashable, ...]), tuple, np.integer)
    else:
        check(assert_type(index, tuple[Hashable, ...]), tuple, int)

    check(assert_type(value, pd.DataFrame), pd.DataFrame)

    iterator2 = df.groupby("a").__iter__()
    assert_type(iterator2, Iterator[tuple[Scalar, pd.DataFrame]])
    index2, value2 = next(iterator2)
    assert_type((index2, value2), tuple[Scalar, pd.DataFrame])

    check(assert_type(index2, Scalar), int)
    check(assert_type(value2, pd.DataFrame), pd.DataFrame)

    # GH 674
    # grouping by pd.MultiIndex should always resolve to a tuple as well
    multi_index = pd.MultiIndex.from_frame(df[["a", "b"]])
    iterator3 = df.groupby(multi_index).__iter__()
    assert_type(iterator3, Iterator[tuple[tuple[Hashable, ...], pd.DataFrame]])
    index3, value3 = next(iterator3)
    assert_type((index3, value3), tuple[tuple[Hashable, ...], pd.DataFrame])

    check(assert_type(index3, tuple[Hashable, ...]), tuple, int)
    check(assert_type(value3, pd.DataFrame), pd.DataFrame)

    # Want to make sure these cases are differentiated
    for (_k1, _k2), _g in df.groupby(["a", "b"]):
        pass

    for _kk, _g in df.groupby("a"):
        pass

    for (_k1, _k2), _g in df.groupby(multi_index):
        pass


def test_groupby_result_for_scalar_indexes() -> None:
    # GH 674
    dates = pd.date_range("2020-01-01", "2020-12-31")
    df = pd.DataFrame({"date": dates, "days": 1})
    period_index = pd.PeriodIndex(df.date, freq="M")
    iterator = df.groupby(period_index).__iter__()
    assert_type(iterator, Iterator[tuple[pd.Period, pd.DataFrame]])
    index, value = next(iterator)
    assert_type((index, value), tuple[pd.Period, pd.DataFrame])

    check(assert_type(index, pd.Period), pd.Period)
    check(assert_type(value, pd.DataFrame), pd.DataFrame)

    dt_index = pd.DatetimeIndex(dates)
    iterator2 = df.groupby(dt_index).__iter__()
    assert_type(iterator2, Iterator[tuple[pd.Timestamp, pd.DataFrame]])
    index2, value2 = next(iterator2)
    assert_type((index2, value2), tuple[pd.Timestamp, pd.DataFrame])

    check(assert_type(index2, pd.Timestamp), pd.Timestamp)
    check(assert_type(value2, pd.DataFrame), pd.DataFrame)

    tdelta_index = pd.TimedeltaIndex(dates - pd.Timestamp("2020-01-01"))
    iterator3 = df.groupby(tdelta_index).__iter__()
    assert_type(iterator3, Iterator[tuple[pd.Timedelta, pd.DataFrame]])
    index3, value3 = next(iterator3)
    assert_type((index3, value3), tuple[pd.Timedelta, pd.DataFrame])

    check(assert_type(index3, pd.Timedelta), pd.Timedelta)
    check(assert_type(value3, pd.DataFrame), pd.DataFrame)

    intervals: list[pd.Interval[pd.Timestamp]] = [
        pd.Interval(date, date + pd.DateOffset(days=1), closed="left") for date in dates
    ]
    interval_index = pd.IntervalIndex(intervals)
    assert_type(interval_index, "pd.IntervalIndex[pd.Interval[pd.Timestamp]]")
    iterator4 = df.groupby(interval_index).__iter__()
    assert_type(iterator4, Iterator[tuple["pd.Interval[pd.Timestamp]", pd.DataFrame]])
    index4, value4 = next(iterator4)
    assert_type((index4, value4), tuple["pd.Interval[pd.Timestamp]", pd.DataFrame])

    check(assert_type(index4, "pd.Interval[pd.Timestamp]"), pd.Interval)
    check(assert_type(value4, pd.DataFrame), pd.DataFrame)

    for _p, _g in df.groupby(period_index):
        pass

    for _dt, _g in df.groupby(dt_index):
        pass

    for _tdelta, _g in df.groupby(tdelta_index):
        pass

    for _interval, _g in df.groupby(interval_index):
        pass


def test_groupby_result_for_ambiguous_indexes() -> None:
    # GH 674
    df = pd.DataFrame({"a": [0, 1, 2], "b": [4, 5, 6], "c": [7, 8, 9]})
    # this will use pd.Index which is ambiguous
    iterator = df.groupby(df.index).__iter__()
    assert_type(iterator, Iterator[tuple[Any, pd.DataFrame]])
    index, value = next(iterator)
    assert_type((index, value), tuple[Any, pd.DataFrame])

    check(assert_type(index, Any), int)
    check(assert_type(value, pd.DataFrame), pd.DataFrame)

    # categorical indexes are also ambiguous

    # https://github.com/pandas-dev/pandas/issues/54054 needs to be fixed
    with pytest_warns_bounded(
        FutureWarning,
        "The default of observed=False is deprecated",
        upper="2.3.99",
    ):
        categorical_index = pd.CategoricalIndex(df.a)
        iterator2 = df.groupby(categorical_index).__iter__()
        assert_type(iterator2, Iterator[tuple[Any, pd.DataFrame]])
        index2, value2 = next(iterator2)
        assert_type((index2, value2), tuple[Any, pd.DataFrame])

        check(assert_type(index2, Any), int)
        check(assert_type(value2, pd.DataFrame), pd.DataFrame)


def test_groupby_apply() -> None:
    # GH 167
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    def sum_mean(x: pd.DataFrame) -> float:
        return x.sum().mean()

    with pytest_warns_bounded(
        FutureWarning,
        "DataFrameGroupBy.apply operated on the grouping columns.",
        lower="2.2.99",
        upper="2.99",
    ):
        check(
            assert_type(df.groupby("col1").apply(sum_mean), pd.Series),
            pd.Series,
        )

    lfunc: Callable[[pd.DataFrame], float] = lambda x: x.sum().mean()
    with pytest_warns_bounded(
        FutureWarning,
        "DataFrameGroupBy.apply operated on the grouping columns.",
        lower="2.2.99",
        upper="2.99",
    ):
        check(assert_type(df.groupby("col1").apply(lfunc), pd.Series), pd.Series)

    def sum_to_list(x: pd.DataFrame) -> list[Any]:
        return x.sum().tolist()

    with pytest_warns_bounded(
        FutureWarning,
        "DataFrameGroupBy.apply operated on the grouping columns.",
        lower="2.2.99",
        upper="2.99",
    ):
        check(assert_type(df.groupby("col1").apply(sum_to_list), pd.Series), pd.Series)

    def sum_to_series(x: pd.DataFrame) -> pd.Series:
        return x.sum()

    with pytest_warns_bounded(
        FutureWarning,
        "DataFrameGroupBy.apply operated on the grouping columns.",
        lower="2.2.99",
        upper="2.99",
    ):
        check(
            assert_type(df.groupby("col1").apply(sum_to_series), pd.DataFrame),
            pd.DataFrame,
        )

    def sample_to_df(x: pd.DataFrame) -> pd.DataFrame:
        return x.sample()

    with pytest_warns_bounded(
        FutureWarning,
        "DataFrameGroupBy.apply operated on the grouping columns.",
        lower="2.2.99",
        upper="2.99",
    ):
        check(
            assert_type(
                df.groupby("col1", group_keys=False).apply(sample_to_df), pd.DataFrame
            ),
            pd.DataFrame,
        )


def test_series_groupby_and_value_counts() -> None:
    df = pd.DataFrame(
        {
            "Animal": ["Falcon", "Falcon", "Parrot", "Parrot"],
            "Max Speed": [380, 370, 24, 26],
        }
    )
    c1 = df.groupby("Animal")["Max Speed"].value_counts()
    c2 = df.groupby("Animal")["Max Speed"].value_counts(normalize=True)
    check(assert_type(c1, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(c2, "pd.Series[float]"), pd.Series, float)


def test_groupby_and_transform() -> None:
    df = pd.DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar"],
            "B": ["one", "one", "two", "three", "two", "two"],
            "C": [1, 5, 5, 2, 5, 5],
            "D": [2.0, 5.0, 8.0, 1.0, 2.0, 9.0],
        }
    )
    ser = pd.Series(
        [390.0, 350.0, 30.0, 20.0],
        index=["Falcon", "Falcon", "Parrot", "Parrot"],
        name="Max Speed",
    )
    grouped = df.groupby("A")[["C", "D"]]
    grouped1 = ser.groupby(ser > 100)
    c1 = grouped.transform("sum")
    c2 = grouped.transform(lambda x: (x - x.mean()) / x.std())
    c3 = grouped1.transform("cumsum")
    c4 = grouped1.transform(lambda x: x.max() - x.min())
    check(assert_type(c1, pd.DataFrame), pd.DataFrame)
    check(assert_type(c2, pd.DataFrame), pd.DataFrame)
    check(assert_type(c3, pd.Series), pd.Series)
    check(assert_type(c4, pd.Series), pd.Series)


def test_getattr_and_dataframe_groupby() -> None:
    df = pd.DataFrame(
        data={"col1": [1, 1, 2], "col2": [3, 4, 5], "col3": [0, 1, 0], 0: [-1, -1, -1]}
    )
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <built-in function (min|max)> is currently using",
        upper="2.3.99",
    ):
        check(assert_type(df.groupby("col1").col3.agg(min), pd.Series), pd.Series)
        check(
            assert_type(df.groupby("col1").col3.agg([min, max]), pd.DataFrame),
            pd.DataFrame,
        )
