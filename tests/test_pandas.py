from __future__ import annotations

import datetime as dt
import random
from typing import (
    TYPE_CHECKING,
    Any,
    Union,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import Grouper
from pandas.api.extensions import ExtensionArray
import pytest
from typing_extensions import assert_type

from pandas._libs.missing import NAType
from pandas._libs.tslibs import NaTType
from pandas._typing import Scalar

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)


def test_types_to_datetime() -> None:
    df = pd.DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]})
    r1: pd.Series = pd.to_datetime(df)

    r2: pd.Series = pd.to_datetime(df, unit="s", origin="unix")
    r3: pd.Series = pd.to_datetime(
        df, unit="ns", dayfirst=True, utc=None, format="%M:%D", exact=False
    )
    r4: pd.DatetimeIndex = pd.to_datetime(
        [1, 2], unit="D", origin=pd.Timestamp("01/01/2000")
    )
    r5: pd.DatetimeIndex = pd.to_datetime([1, 2], unit="D", origin=3)
    r6: pd.DatetimeIndex = pd.to_datetime(["2022-01-03", "2022-02-22"])
    r7: pd.DatetimeIndex = pd.to_datetime(pd.Index(["2022-01-03", "2022-02-22"]))
    r8: pd.Series = pd.to_datetime(
        {"year": [2015, 2016], "month": [2, 3], "day": [4, 5]}
    )


def test_types_concat() -> None:
    s: pd.Series = pd.Series([0, 1, -10])
    s2: pd.Series = pd.Series([7, -5, 10])

    check(assert_type(pd.concat([s, s2]), pd.Series), pd.Series)
    check(assert_type(pd.concat([s, s2], axis=1), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(pd.concat([s, s2], keys=["first", "second"], sort=True), pd.Series),
        pd.Series,
    )
    check(
        assert_type(
            pd.concat([s, s2], keys=["first", "second"], names=["source", "row"]),
            pd.Series,
        ),
        pd.Series,
    )

    # Depends on the axis
    check(
        assert_type(pd.concat({"a": s, "b": s2}), pd.Series),
        pd.Series,
    )
    check(
        assert_type(pd.concat({"a": s, "b": s2}, axis=1), pd.DataFrame),
        pd.DataFrame,
    )
    check(assert_type(pd.concat({1: s, 2: s2}), pd.Series), pd.Series)
    check(assert_type(pd.concat({1: s, 2: s2}, axis=1), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(pd.concat({1: s, None: s2}), pd.Series),
        pd.Series,
    )
    check(
        assert_type(
            pd.concat({1: s, None: s2}, axis=1),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )

    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df2 = pd.DataFrame(data={"col1": [10, 20], "col2": [30, 40]})

    check(assert_type(pd.concat([df, df2]), pd.DataFrame), pd.DataFrame)
    check(assert_type(pd.concat([df, df2], axis=1), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(
            pd.concat([df, df2], keys=["first", "second"], sort=True), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.concat([df, df2], keys=["first", "second"], names=["source", "row"]),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )

    check(
        assert_type(
            pd.concat(
                {"a": pd.DataFrame([1, 2, 3]), "b": pd.DataFrame([4, 5, 6])}, axis=1
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.concat({"a": pd.Series([1, 2, 3]), "b": pd.Series([4, 5, 6])}, axis=1),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )

    check(assert_type(pd.concat({"a": df, "b": df2}), pd.DataFrame), pd.DataFrame)
    check(assert_type(pd.concat({1: df, 2: df2}), pd.DataFrame), pd.DataFrame)
    check(assert_type(pd.concat({1: df, None: df2}), pd.DataFrame), pd.DataFrame)

    check(
        assert_type(
            pd.concat(map(lambda x: s2, ["some_value", 3]), axis=1), pd.DataFrame
        ),
        pd.DataFrame,
    )
    adict = {"a": df, 2: df2}
    check(assert_type(pd.concat(adict), pd.DataFrame), pd.DataFrame)


def test_concat_args() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df2 = pd.DataFrame(data={"col1": [10, 20], "col2": [30, 40]}, index=[2, 3])

    check(
        assert_type(pd.concat([df, df2], keys=["df1", "df2"]), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.concat([df, df2], keys=["df1", "df2"], names=["one"]), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.concat([df, df2], keys=["df1", "df2"], names=[pd.Timedelta(1, "D")]),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.concat(
                [df, df2], keys=[("df1", "ff"), (pd.Timestamp(2000, 1, 1), "gg")]
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )

    df_dict = {"df1": df, "df2": df2}
    check(
        assert_type(
            pd.concat(df_dict.values(), keys=df_dict.keys()),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )

    check(
        assert_type(pd.concat([df, df2], ignore_index=True), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(pd.concat([df, df2], verify_integrity=True), pd.DataFrame),
        pd.DataFrame,
    )
    check(assert_type(pd.concat([df, df2], sort=True), pd.DataFrame), pd.DataFrame)
    check(assert_type(pd.concat([df, df2], copy=True), pd.DataFrame), pd.DataFrame)
    check(assert_type(pd.concat([df, df2], join="inner"), pd.DataFrame), pd.DataFrame)
    check(assert_type(pd.concat([df, df2], join="outer"), pd.DataFrame), pd.DataFrame)
    check(assert_type(pd.concat([df, df2], axis=0), pd.DataFrame), pd.DataFrame)
    check(assert_type(pd.concat([df, df2], axis=1), pd.DataFrame), pd.DataFrame)
    check(assert_type(pd.concat([df, df2], axis="index"), pd.DataFrame), pd.DataFrame)
    check(assert_type(pd.concat([df, df2], axis="columns"), pd.DataFrame), pd.DataFrame)

    df = pd.DataFrame(np.random.randn(1, 3))
    df2 = pd.DataFrame(np.random.randn(1, 4))

    levels = [["foo", "baz"], ["one", "two"]]
    names = ["first", "second"]
    check(
        assert_type(
            pd.concat(
                [df, df2, df, df2],
                keys=[("foo", "one"), ("foo", "two"), ("baz", "one"), ("baz", "two")],
                levels=levels,
                names=names,
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )

    check(
        assert_type(
            pd.concat(
                [df, df2, df, df2],
                keys=[("foo", "one"), ("foo", "two"), ("baz", "one"), ("baz", "two")],
                levels=levels,
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )


def test_types_json_normalize() -> None:
    data1: list[dict[str, Any]] = [
        {"id": 1, "name": {"first": "Coleen", "last": "Volk"}},
        {"name": {"given": "Mose", "family": "Regner"}},
        {"id": 2, "name": "Faye Raker"},
    ]
    df1: pd.DataFrame = pd.json_normalize(data=data1)
    df2: pd.DataFrame = pd.json_normalize(data=data1, max_level=0, sep=";")
    df3: pd.DataFrame = pd.json_normalize(
        data=data1, meta_prefix="id", record_prefix="name", errors="raise"
    )
    df4: pd.DataFrame = pd.json_normalize(data=data1, record_path=None, meta="id")
    data2: dict[str, Any] = {"name": {"given": "Mose", "family": "Regner"}}
    df5: pd.DataFrame = pd.json_normalize(data=data2)


def test_isna() -> None:
    # https://github.com/pandas-dev/pandas-stubs/issues/264
    s1 = pd.Series([1, np.nan, 3.2])
    check(assert_type(pd.isna(s1), "pd.Series[bool]"), pd.Series, np.bool_)

    s2 = pd.Series([1, 3.2])
    check(assert_type(pd.notna(s2), "pd.Series[bool]"), pd.Series, np.bool_)

    df1 = pd.DataFrame({"a": [1, 2, 1, 2], "b": [1, 1, 2, np.nan]})
    check(assert_type(pd.isna(df1), "pd.DataFrame"), pd.DataFrame)

    idx1 = pd.Index([1, 2, np.nan])
    check(assert_type(pd.isna(idx1), npt.NDArray[np.bool_]), np.ndarray, np.bool_)

    idx2 = pd.Index([1, 2])
    check(assert_type(pd.notna(idx2), npt.NDArray[np.bool_]), np.ndarray, np.bool_)

    assert check(assert_type(pd.isna(pd.NA), bool), bool)
    assert not check(assert_type(pd.notna(pd.NA), bool), bool)

    assert check(assert_type(pd.isna(pd.NaT), bool), bool)
    assert not check(assert_type(pd.notna(pd.NaT), bool), bool)

    assert check(assert_type(pd.isna(None), bool), bool)
    assert not check(assert_type(pd.notna(None), bool), bool)

    assert not check(assert_type(pd.isna(2.5), bool), bool)
    assert check(assert_type(pd.notna(2.5), bool), bool)

    # Checks for datetime, timedelta, np.datetime64 and np.timedelta64
    py_dt = dt.datetime.now()
    assert check(assert_type(pd.notna(py_dt), bool), bool)
    assert not check(assert_type(pd.isna(py_dt), bool), bool)

    py_td = dt.datetime.now() - py_dt
    assert check(assert_type(pd.notna(py_td), bool), bool)
    assert not check(assert_type(pd.isna(py_td), bool), bool)

    np_dt = np.datetime64(py_dt)
    assert check(assert_type(pd.notna(np_dt), bool), bool)
    assert not check(assert_type(pd.isna(np_dt), bool), bool)

    np_td = np.timedelta64(py_td)
    assert check(assert_type(pd.notna(np_td), bool), bool)
    assert not check(assert_type(pd.isna(np_td), bool), bool)

    np_nat = np.timedelta64("NaT")
    assert check(assert_type(pd.isna(np_nat), bool), bool)
    assert not check(assert_type(pd.notna(np_nat), bool), bool)

    # Check TypeGuard type narrowing functionality
    # TODO: Due to limitations in TypeGuard spec, the true annotations are not always viable
    # and as a result the type narrowing does not always work as it intuitively should
    # There is a proposal being floated for a StrictTypeGuard that will have more rigid narrowing semantics
    # In the test cases below, a commented out assertion will be included to document the optimal test result
    nullable1: str | None | NAType | NaTType = random.choice(
        ["value", None, pd.NA, pd.NaT]
    )
    if pd.notna(nullable1):
        check(assert_type(nullable1, str), str)
    if not pd.isna(nullable1):
        # check(assert_type(nullable1, str), str)  # TODO: Desired result (see comments above)
        check(assert_type(nullable1, Union[str, NaTType, NAType, None]), str)
    if pd.isna(nullable1):
        assert_type(nullable1, Union[NaTType, NAType, None])
    if not pd.notna(nullable1):
        # assert_type(nullable1, Union[NaTType, NAType, None])  # TODO: Desired result (see comments above)
        assert_type(nullable1, Union[str, NaTType, NAType, None])

    nullable2: int | None = random.choice([2, None])
    if pd.notna(nullable2):
        check(assert_type(nullable2, int), int)
    if not pd.isna(nullable2):
        # check(assert_type(nullable2, int), int)  # TODO: Desired result (see comments above)
        check(assert_type(nullable2, Union[int, None]), int)
    if pd.isna(nullable2):
        # check(assert_type(nullable2, None), type(None))  # TODO: Desired result (see comments above)
        check(assert_type(nullable2, Union[NaTType, NAType, None]), type(None))
    if not pd.notna(nullable2):
        # check(assert_type(nullable2, None), type(None))  # TODO: Desired result (see comments above)
        # TODO: MyPy and Pyright produce conflicting results:
        # assert_type(nullable2, Union[int, None])  # MyPy result
        # assert_type(
        #     nullable2, Union[int, NaTType, NAType, None]
        # )  # Pyright result
        pass

    nullable3: bool | None | NAType = random.choice([True, None, pd.NA])
    if pd.notna(nullable3):
        check(assert_type(nullable3, bool), bool)
    if not pd.isna(nullable3):
        # check(assert_type(nullable3, bool), bool)  # TODO: Desired result (see comments above)
        check(assert_type(nullable3, Union[bool, NAType, None]), bool)
    if pd.isna(nullable3):
        # assert_type(nullable3, Union[NAType, None])  # TODO: Desired result (see comments above)
        assert_type(nullable3, Union[NaTType, NAType, None])
    if not pd.notna(nullable3):
        # assert_type(nullable3, Union[NAType, None])  # TODO: Desired result (see comments above)
        # TODO: MyPy and Pyright produce conflicting results:
        # assert_type(nullable3, Union[bool, NAType, None])  # Mypy result
        # assert_type(
        #     nullable3, Union[bool, NaTType, NAType, None]
        # )  # Pyright result
        pass


# GH 55
def test_read_xml() -> None:
    if TYPE_CHECKING:  # Skip running pytest
        check(
            assert_type(
                pd.read_xml(
                    "path/to/file", xpath=".//row", stylesheet="path/to/stylesheet"
                ),
                pd.DataFrame,
            ),
            pd.DataFrame,
        )


def test_unique() -> None:
    # Taken from the docs
    check(
        assert_type(
            pd.unique(pd.Series([2, 1, 3, 3])), Union[np.ndarray, ExtensionArray]
        ),
        np.ndarray,
    )

    check(
        assert_type(
            pd.unique(pd.Series([2] + [1] * 5)), Union[np.ndarray, ExtensionArray]
        ),
        np.ndarray,
    )

    check(
        assert_type(
            pd.unique(pd.Series([pd.Timestamp("20160101"), pd.Timestamp("20160101")])),
            Union[np.ndarray, ExtensionArray],
        ),
        np.ndarray,
    )

    check(
        assert_type(
            pd.unique(
                pd.Series(
                    [
                        pd.Timestamp("20160101", tz="US/Eastern"),
                        pd.Timestamp("20160101", tz="US/Eastern"),
                    ]
                )
            ),
            Union[np.ndarray, ExtensionArray],
        ),
        pd.arrays.DatetimeArray,
    )
    check(
        assert_type(
            pd.unique(
                pd.Index(
                    [
                        pd.Timestamp("20160101", tz="US/Eastern"),
                        pd.Timestamp("20160101", tz="US/Eastern"),
                    ]
                )
            ),
            np.ndarray,
        ),
        pd.DatetimeIndex,
    )

    check(assert_type(pd.unique(list("baabc")), np.ndarray), np.ndarray)

    check(
        assert_type(
            pd.unique(pd.Series(pd.Categorical(list("baabc")))),
            Union[np.ndarray, ExtensionArray],
        ),
        pd.Categorical,
    )
    check(
        assert_type(
            pd.unique(pd.Series(pd.Categorical(list("baabc"), categories=list("abc")))),
            Union[np.ndarray, ExtensionArray],
        ),
        pd.Categorical,
    )
    check(
        assert_type(
            pd.unique(
                pd.Series(
                    pd.Categorical(list("baabc"), categories=list("abc"), ordered=True)
                )
            ),
            Union[np.ndarray, ExtensionArray],
        ),
        pd.Categorical,
    )
    check(
        assert_type(
            pd.unique([("a", "b"), ("b", "a"), ("a", "c"), ("b", "a")]), np.ndarray
        ),
        np.ndarray,
    )
    check(
        assert_type(pd.unique(pd.Index(["a", "b", "c", "a"])), np.ndarray),
        np.ndarray,
    )
    check(
        assert_type(pd.unique(pd.RangeIndex(0, 10)), np.ndarray),
        np.ndarray,
    )
    check(
        assert_type(pd.unique(pd.Categorical(["a", "b", "c", "a"])), pd.Categorical),
        pd.Categorical,
    )
    check(
        assert_type(
            pd.unique(pd.period_range("2001Q1", periods=10, freq="D")),
            pd.PeriodIndex,
        ),
        pd.PeriodIndex,
    )
    check(
        assert_type(
            pd.unique(pd.timedelta_range(start="1 day", periods=4)),
            np.ndarray,
        ),
        np.ndarray,
    )


# GH 200
def test_crosstab() -> None:
    df = pd.DataFrame({"a": [1, 2, 1, 2], "b": [1, 1, 2, 2]})
    check(
        assert_type(
            pd.crosstab(
                index=df["a"],
                columns=df["b"],
                margins=True,
                dropna=False,
                normalize="columns",
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )


def test_arrow_dtype() -> None:
    pytest.importorskip("pyarrow")

    import pyarrow as pa

    check(
        assert_type(
            pd.ArrowDtype(pa.timestamp("s", tz="America/New_York")), pd.ArrowDtype
        ),
        pd.ArrowDtype,
    )


def test_hashing():
    a = np.array([1, 2, 3])
    check(assert_type(pd.util.hash_array(a), npt.NDArray[np.uint64]), np.ndarray)
    check(
        assert_type(
            pd.util.hash_array(a, encoding="latin1", hash_key="1", categorize=True),
            npt.NDArray[np.uint64],
        ),
        np.ndarray,
    )

    b = pd.Series(a)
    c = pd.DataFrame({"a": a, "b": a})
    d = pd.Index(b)
    check(assert_type(pd.util.hash_pandas_object(b), pd.Series), pd.Series)
    check(assert_type(pd.util.hash_pandas_object(c), pd.Series), pd.Series)
    check(assert_type(pd.util.hash_pandas_object(d), pd.Series), pd.Series)
    check(
        assert_type(
            pd.util.hash_pandas_object(
                d, index=True, encoding="latin1", hash_key="apple", categorize=True
            ),
            pd.Series,
        ),
        pd.Series,
    )


def test_eval():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    check(
        assert_type(
            pd.eval("double_age = df.age * 2", target=df),
            Union[npt.NDArray, Scalar, pd.DataFrame, pd.Series, None],
        ),
        pd.DataFrame,
    )


def test_to_numeric_scalar() -> None:
    check(assert_type(pd.to_numeric(1), float), int)
    check(assert_type(pd.to_numeric("1.2"), float), float)
    check(assert_type(pd.to_numeric("blerg", errors="coerce"), float), float)
    check(assert_type(pd.to_numeric("blerg", errors="ignore"), Scalar), str)
    check(assert_type(pd.to_numeric(1, downcast="signed"), float), int)
    check(assert_type(pd.to_numeric(1, downcast="unsigned"), float), int)
    check(assert_type(pd.to_numeric(1, downcast="float"), float), int)
    check(assert_type(pd.to_numeric(1, downcast="integer"), float), int)


def test_to_numeric_array_like() -> None:
    check(
        assert_type(
            pd.to_numeric([1, 2, 3]),
            npt.NDArray,
        ),
        np.ndarray,
    )
    check(
        assert_type(
            pd.to_numeric([1.0, 2.0, 3.0]),
            npt.NDArray,
        ),
        np.ndarray,
    )
    check(
        assert_type(
            pd.to_numeric([1.0, 2.0, "3.0"]),
            npt.NDArray,
        ),
        np.ndarray,
    )
    check(
        assert_type(
            pd.to_numeric(np.array([1.0, 2.0, "3.0"], dtype=object)),
            npt.NDArray,
        ),
        np.ndarray,
    )
    check(
        assert_type(
            pd.to_numeric([1.0, 2.0, "blerg"], errors="coerce"),
            npt.NDArray,
        ),
        np.ndarray,
    )
    check(
        assert_type(pd.to_numeric([1.0, 2.0, "blerg"], errors="ignore"), npt.NDArray),
        np.ndarray,
    )
    check(
        assert_type(
            pd.to_numeric((1.0, 2.0, 3.0)),
            npt.NDArray,
        ),
        np.ndarray,
    )
    check(
        assert_type(pd.to_numeric([1, 2, 3], downcast="unsigned"), npt.NDArray),
        np.ndarray,
    )


def test_to_numeric_array_series() -> None:
    check(
        assert_type(
            pd.to_numeric(pd.Series([1, 2, 3])),
            pd.Series,
        ),
        pd.Series,
    )
    check(
        assert_type(
            pd.to_numeric(pd.Series([1, 2, "blerg"]), errors="coerce"),
            pd.Series,
        ),
        pd.Series,
    )
    check(
        assert_type(
            pd.to_numeric(pd.Series([1, 2, "blerg"]), errors="ignore"), pd.Series
        ),
        pd.Series,
    )
    check(
        assert_type(pd.to_numeric(pd.Series([1, 2, 3]), downcast="signed"), pd.Series),
        pd.Series,
    )
    check(
        assert_type(
            pd.to_numeric(pd.Series([1, 2, 3]), downcast="unsigned"), pd.Series
        ),
        pd.Series,
    )
    check(
        assert_type(pd.to_numeric(pd.Series([1, 2, 3]), downcast="integer"), pd.Series),
        pd.Series,
    )
    check(
        assert_type(pd.to_numeric(pd.Series([1, 2, 3]), downcast="float"), pd.Series),
        pd.Series,
    )


def test_wide_to_long():
    df = pd.DataFrame(
        {
            "A1970": {0: "a", 1: "b", 2: "c"},
            "A1980": {0: "d", 1: "e", 2: "f"},
            "B1970": {0: 2.5, 1: 1.2, 2: 0.7},
            "B1980": {0: 3.2, 1: 1.3, 2: 0.1},
            "X": dict(zip(range(3), np.random.randn(3))),
        }
    )
    df["id"] = df.index
    df["id2"] = df.index + 1
    check(
        assert_type(pd.wide_to_long(df, ["A", "B"], i="id", j="year"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.wide_to_long(df, ["A", "B"], i=["id", "id2"], j="year"), pd.DataFrame
        ),
        pd.DataFrame,
    )


def test_melt():
    df = pd.DataFrame(
        {
            "A": {0: "a", 1: "b", 2: "c"},
            "B": {0: 1, 1: 3, 2: 5},
            "C": {0: 2, 1: 4, 2: 6},
            "D": {0: 3, 1: 6, 2: 9},
            "E": {0: 3, 1: 6, 2: 9},
        }
    )
    check(
        assert_type(
            pd.melt(df, id_vars=["A"], value_vars=["B"], ignore_index=False),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.melt(df, id_vars=["A"], value_vars=["B"], value_name=("F",)),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    df.columns = pd.MultiIndex.from_arrays([list("ABCDE"), list("FGHIJ")])
    check(
        assert_type(
            pd.melt(
                df, id_vars=["A"], value_vars=["B"], ignore_index=False, col_level=0
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )


def test_lreshape() -> None:
    data = pd.DataFrame(
        {
            "hr1": [514, 573],
            "hr2": [545, 526],
            "team": ["Red Sox", "Yankees"],
            "year1": [2007, 2007],
            "year2": [2008, 2008],
        }
    )
    check(
        assert_type(
            pd.lreshape(
                data, {"year": ["year1", "year2"], "hr": ["hr1", "hr2"]}, dropna=True
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    data2 = pd.DataFrame(
        {
            "hr1": [514, 573],
            ("hr2",): [545, 526],
            "team": ["Red Sox", "Yankees"],
            ("year1",): [2007, 2007],
            "year2": [2008, 2008],
        }
    )
    from typing import Hashable

    groups: dict[Hashable, list[Hashable]] = {
        ("year",): [("year1",), "year2"],
        ("hr",): ["hr1", ("hr2",)],
    }
    check(
        assert_type(
            pd.lreshape(data2, groups=groups),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )


def test_factorize() -> None:
    codes, uniques = pd.factorize(["b", "b", "a", "c", "b"])
    check(assert_type(codes, np.ndarray), np.ndarray)
    check(assert_type(uniques, np.ndarray), np.ndarray)

    codes, uniques = pd.factorize(np.recarray((1,), dtype=[("x", int)]))
    check(assert_type(codes, np.ndarray), np.ndarray)
    check(assert_type(uniques, np.ndarray), np.ndarray)

    codes, cat_uniques = pd.factorize(pd.Categorical(["b", "b", "a", "c", "b"]))
    check(assert_type(codes, np.ndarray), np.ndarray)
    check(assert_type(cat_uniques, pd.Categorical), pd.Categorical)

    codes, idx_uniques = pd.factorize(pd.Index(["b", "b", "a", "c", "b"]))
    check(assert_type(codes, np.ndarray), np.ndarray)
    check(assert_type(idx_uniques, pd.Index), pd.Index)

    codes, idx_uniques = pd.factorize(pd.Series(["b", "b", "a", "c", "b"]))
    check(assert_type(codes, np.ndarray), np.ndarray)
    check(assert_type(idx_uniques, pd.Index), pd.Index)

    codes, uniques = pd.factorize("bbacb")
    check(assert_type(codes, np.ndarray), np.ndarray)
    check(assert_type(uniques, np.ndarray), np.ndarray)

    codes, uniques = pd.factorize(
        ["b", "b", "a", "c", "b"], use_na_sentinel=True, size_hint=10
    )
    check(assert_type(codes, np.ndarray), np.ndarray)
    check(assert_type(uniques, np.ndarray), np.ndarray)


def test_index_unqiue() -> None:
    ci = pd.CategoricalIndex(["a", "b", "a", "c"])
    dti = pd.DatetimeIndex([pd.Timestamp(2000, 1, 1)])

    i = pd.Index(["a", "b", "c", "a"])

    pi = pd.period_range("2000Q1", periods=2, freq="Q")
    ri = pd.RangeIndex(0, 10)

    tdi = pd.timedelta_range("1 day", "10 days", periods=10)
    mi = pd.MultiIndex.from_product([["a", "b"], ["apple", "banana"]])
    interval_i = pd.interval_range(1, 10, periods=10)

    check(assert_type(pd.unique(ci), pd.CategoricalIndex), pd.CategoricalIndex)
    check(assert_type(pd.unique(dti), np.ndarray), np.ndarray)
    check(assert_type(pd.unique(i), np.ndarray), np.ndarray)
    check(assert_type(pd.unique(pi), pd.PeriodIndex), pd.PeriodIndex)
    check(assert_type(pd.unique(ri), np.ndarray), np.ndarray)
    check(assert_type(pd.unique(tdi), np.ndarray), np.ndarray)
    check(assert_type(pd.unique(mi), np.ndarray), np.ndarray)
    check(
        assert_type(pd.unique(interval_i), "pd.IntervalIndex[pd.Interval[int]]"),
        pd.IntervalIndex,
    )


def test_cut() -> None:
    intval_idx = pd.interval_range(0, 10, 4)
    a = pd.cut([1, 2, 3, 4, 5, 6, 7, 8], 4, precision=1, duplicates="drop")
    b = pd.cut([1, 2, 3, 4, 5, 6, 7, 8], 4, labels=False, duplicates="raise")
    c = pd.cut([1, 2, 3, 4, 5, 6, 7, 8], 4, labels=["1", "2", "3", "4"])
    check(assert_type(a, pd.Categorical), pd.Categorical)
    check(assert_type(b, npt.NDArray[np.intp]), np.ndarray)
    check(assert_type(c, pd.Categorical), pd.Categorical)

    d0, d1 = pd.cut([1, 2, 3, 4, 5, 6, 7, 8], 4, retbins=True)
    e0, e1 = pd.cut([1, 2, 3, 4, 5, 6, 7, 8], 4, labels=False, retbins=True)
    f0, f1 = pd.cut(
        [1, 2, 3, 4, 5, 6, 7, 8], 4, labels=["1", "2", "3", "4"], retbins=True
    )
    check(assert_type(d0, pd.Categorical), pd.Categorical)
    check(assert_type(d1, npt.NDArray), np.ndarray)
    check(assert_type(e0, npt.NDArray[np.intp]), np.ndarray)
    check(assert_type(e1, npt.NDArray), np.ndarray)
    check(assert_type(f0, pd.Categorical), pd.Categorical)
    check(assert_type(f1, npt.NDArray), np.ndarray)

    g = pd.cut(pd.Series([1, 2, 3, 4, 5, 6, 7, 8]), 4, precision=1, duplicates="drop")
    h = pd.cut(pd.Series([1, 2, 3, 4, 5, 6, 7, 8]), 4, labels=False, duplicates="raise")
    i = pd.cut(pd.Series([1, 2, 3, 4, 5, 6, 7, 8]), 4, labels=["1", "2", "3", "4"])
    check(assert_type(g, pd.Series), pd.Series)
    check(assert_type(h, pd.Series), pd.Series)
    check(assert_type(i, pd.Series), pd.Series)

    j0, j1 = pd.cut(
        pd.Series([1, 2, 3, 4, 5, 6, 7, 8]),
        4,
        precision=1,
        duplicates="drop",
        retbins=True,
    )
    k0, k1 = pd.cut(
        pd.Series([1, 2, 3, 4, 5, 6, 7, 8]),
        4,
        labels=False,
        duplicates="raise",
        retbins=True,
    )
    l0, l1 = pd.cut(
        pd.Series([1, 2, 3, 4, 5, 6, 7, 8]),
        4,
        labels=["1", "2", "3", "4"],
        retbins=True,
    )
    m0, m1 = pd.cut(
        pd.Series([1, 2, 3, 4, 5, 6, 7, 8]),
        intval_idx,
        retbins=True,
    )
    check(assert_type(j0, pd.Series), pd.Series)
    check(assert_type(j1, npt.NDArray), np.ndarray)
    check(assert_type(k0, pd.Series), pd.Series)
    check(assert_type(k1, npt.NDArray), np.ndarray)
    check(assert_type(l0, pd.Series), pd.Series)
    check(assert_type(l1, npt.NDArray), np.ndarray)
    check(assert_type(m0, pd.Series), pd.Series)
    check(assert_type(m1, pd.IntervalIndex), pd.IntervalIndex)

    n0, n1 = pd.cut([1, 2, 3, 4, 5, 6, 7, 8], intval_idx, retbins=True)
    check(assert_type(n0, pd.Categorical), pd.Categorical)
    check(assert_type(n1, pd.IntervalIndex), pd.IntervalIndex)

    s1 = pd.Series(data=pd.date_range("1/1/2020", periods=300))
    check(
        assert_type(
            pd.cut(s1, bins=[np.datetime64("2020-01-03"), np.datetime64("2020-09-01")]),
            "pd.Series[pd.CategoricalDtype]",
        ),
        pd.Series,
    )
    check(
        assert_type(
            pd.cut(s1, bins=10),
            "pd.Series[pd.CategoricalDtype]",
        ),
        pd.Series,
        pd.Interval,
    )
    s0r, s1r = pd.cut(s1, bins=10, retbins=True)
    check(assert_type(s0r, pd.Series), pd.Series, pd.Interval)
    check(assert_type(s1r, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)
    s0rlf, s1rlf = pd.cut(s1, bins=10, labels=False, retbins=True)
    check(assert_type(s0rlf, pd.Series), pd.Series, np.int64)
    check(assert_type(s1rlf, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)
    s0rls, s1rls = pd.cut(s1, bins=4, labels=["1", "2", "3", "4"], retbins=True)
    check(assert_type(s0rls, pd.Series), pd.Series, str)
    check(assert_type(s1rls, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)


def test_qcut() -> None:
    val_list = [random.random() for _ in range(20)]
    val_arr = np.array(val_list)
    val_series = pd.Series(val_list)
    val_idx = pd.Index(val_list)

    check(
        assert_type(
            pd.qcut(val_list, 4, precision=2, duplicates="raise"), pd.Categorical
        ),
        pd.Categorical,
    )
    check(
        assert_type(
            pd.qcut(val_arr, 4, precision=2, duplicates="drop"), pd.Categorical
        ),
        pd.Categorical,
    )
    check(
        assert_type(
            pd.qcut(val_idx, 4, precision=2, duplicates="drop"), pd.Categorical
        ),
        pd.Categorical,
    )
    check(
        assert_type(pd.qcut(val_series, 4, precision=2, duplicates="raise"), pd.Series),
        pd.Series,
    )

    a0, a1 = pd.qcut(val_list, 4, retbins=True)
    b0, b1 = pd.qcut(val_arr, 4, retbins=True)
    c0, c1 = pd.qcut(val_idx, 4, retbins=True)
    d0, d1 = pd.qcut(val_series, 4, retbins=True)
    check(assert_type(a0, pd.Categorical), pd.Categorical)
    check(assert_type(b0, pd.Categorical), pd.Categorical)
    check(assert_type(c0, pd.Categorical), pd.Categorical)
    check(assert_type(d0, pd.Series), pd.Series)

    check(assert_type(a1, npt.NDArray[np.float_]), np.ndarray)
    check(assert_type(b1, npt.NDArray[np.float_]), np.ndarray)
    check(assert_type(c1, npt.NDArray[np.float_]), np.ndarray)
    check(assert_type(d1, npt.NDArray[np.float_]), np.ndarray)

    e0, e1 = pd.qcut(val_list, [0.25, 0.5, 0.75], retbins=True)
    f0, f1 = pd.qcut(val_arr, np.array([0.25, 0.5, 0.75]), retbins=True)
    g0, g1 = pd.qcut(val_idx, 4, retbins=True, labels=False)
    h0, h1 = pd.qcut(val_series, 4, retbins=True, labels=False)
    i0, i1 = pd.qcut(val_list, [0.25, 0.5, 0.75], retbins=True, labels=False)
    j0, j1 = pd.qcut(val_arr, np.array([0.25, 0.5, 0.75]), retbins=True, labels=False)

    check(assert_type(e0, pd.Categorical), pd.Categorical)
    check(assert_type(f0, pd.Categorical), pd.Categorical)
    check(assert_type(g0, npt.NDArray[np.intp]), np.ndarray)
    check(assert_type(h0, pd.Series), pd.Series)
    check(assert_type(i0, npt.NDArray[np.intp]), np.ndarray)
    check(assert_type(j0, npt.NDArray[np.intp]), np.ndarray)

    check(assert_type(e1, npt.NDArray[np.float_]), np.ndarray)
    check(assert_type(f1, npt.NDArray[np.float_]), np.ndarray)
    check(assert_type(g1, npt.NDArray[np.float_]), np.ndarray)
    check(assert_type(h1, npt.NDArray[np.float_]), np.ndarray)
    check(assert_type(i1, npt.NDArray[np.float_]), np.ndarray)
    check(assert_type(j1, npt.NDArray[np.float_]), np.ndarray)


def test_merge() -> None:
    ls = pd.Series([1, 2, 3, 4], index=[1, 2, 3, 4], name="left")
    rs = pd.Series([3, 4, 5, 6], index=[3, 4, 5, 6], name="right")
    lf = pd.DataFrame(pd.Series([1, 2, 3, 4], index=[1, 2, 3, 4], name="left"))
    rf = pd.DataFrame(pd.Series([3, 4, 5, 6], index=[3, 4, 5, 6], name="right"))

    check(
        assert_type(pd.merge(ls, rs, left_on="left", right_on="right"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge(ls, rs, how="left", left_on="left", right_on="right"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge(ls, rs, how="right", left_on="left", right_on="right"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge(ls, rs, how="outer", left_on="left", right_on="right"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge(ls, rs, how="inner", left_on="left", right_on="right"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    # TODO: When cross don't need on??
    check(assert_type(pd.merge(ls, rs, how="cross"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(
            pd.merge(ls, rs, how="inner", left_index=True, right_index=True),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge(
                ls,
                rs,
                how="inner",
                left_index=True,
                right_index=True,
                sort=True,
                copy=True,
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge(
                ls,
                rs,
                how="inner",
                left_index=True,
                right_index=True,
                suffixes=["_1", "_2"],
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge(
                ls,
                rs,
                how="inner",
                left_index=True,
                right_index=True,
                suffixes=["_1", None],
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge(
                ls,
                rs,
                how="inner",
                left_index=True,
                right_index=True,
                suffixes=("_1", None),
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge(
                ls,
                rs,
                how="inner",
                left_index=True,
                right_index=True,
                suffixes=(None, "_2"),
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge(
                ls,
                rs,
                how="inner",
                left_index=True,
                right_index=True,
                suffixes=("_1", "_2"),
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge(
                ls, rs, how="inner", left_index=True, right_index=True, indicator=True
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(pd.merge(lf, rs, left_on="left", right_on="right"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(pd.merge(ls, rf, left_on="left", right_on="right"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(pd.merge(lf, rf, left_on="left", right_on="right"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge(lf, rf, left_on=["left"], right_on=["right"]), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(pd.merge(lf, rf, left_index=True, right_index=True), pd.DataFrame),
        pd.DataFrame,
    )


def test_merge_ordered() -> None:
    ls = pd.Series([1, 2, 3, 4], index=[1, 2, 3, 4], name="left")
    rs = pd.Series([3, 4, 5, 6], index=[3, 4, 5, 6], name="right")
    lf = pd.DataFrame(
        [[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]],
        index=[1, 2, 3, 4],
        columns=["a", "b", "c"],
    )
    rf = pd.DataFrame(pd.Series([3, 4, 5, 6], index=[3, 4, 5, 6], name="b"))

    check(
        assert_type(
            pd.merge_ordered(ls, rs, left_on="left", right_on="right"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge_ordered(ls, rf, left_on="left", right_on="b"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge_ordered(lf, rs, left_on="a", right_on="right"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(assert_type(pd.merge_ordered(lf, rf, on="b"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(pd.merge_ordered(lf, rf, left_on="a", right_on="b"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge_ordered(lf, rf, left_on="b", right_on="b", how="outer"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge_ordered(lf, rf, left_on=["b"], right_on=["b"], how="outer"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge_ordered(lf, rf, left_on="b", right_on="b", how="inner"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge_ordered(lf, rf, left_on="b", right_on="b", how="left"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge_ordered(lf, rf, left_on="b", right_on="b", how="right"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(pd.merge_ordered(lf, rf, left_by="a"), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(
            pd.merge_ordered(lf, rf, left_by=["a", "c"], fill_method="ffill"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge_ordered(lf, rf, on="b", suffixes=["_1", None]), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge_ordered(lf, rf, on="b", suffixes=("_1", None)), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge_ordered(lf, rf, on="b", suffixes=(None, "_2")), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge_ordered(lf, rf, on="b", suffixes=("_1", "_2")), pd.DataFrame
        ),
        pd.DataFrame,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        pd.merge_ordered(  # type: ignore[call-overload]
            ls,
            rs,
            left_on="left",
            right_on="right",
            left_by="left",  # pyright: ignore[reportGeneralTypeIssues]
            right_by="right",  # pyright: ignore[reportGeneralTypeIssues]
        )
        pd.merge_ordered(  # type: ignore[call-overload]
            ls,
            rf,  # pyright: ignore[reportGeneralTypeIssues]
            left_on="left",
            right_on="b",
            left_by="left",  # pyright: ignore[reportGeneralTypeIssues]
            right_by="b",  # pyright: ignore[reportGeneralTypeIssues]
        )
        pd.merge_ordered(  # type: ignore[call-overload]
            lf,
            rs,
            left_on="a",
            right_on="right",
            left_by="a",  # pyright: ignore[reportGeneralTypeIssues]
            right_by="right",  # pyright: ignore[reportGeneralTypeIssues]
        )


def test_merge_asof() -> None:
    ls = pd.Series([1, 2, 3, 4], index=[1, 2, 3, 4], name="left")
    rs = pd.Series([3, 4, 5, 6], index=[3, 4, 5, 6], name="right")
    lf = pd.DataFrame(
        [[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]],
        index=[1, 2, 3, 4],
        columns=["a", "b", "c"],
    )
    rf = pd.DataFrame(
        [[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]],
        index=[1, 2, 3, 4],
        columns=["a", "b", "d"],
    )

    check(
        assert_type(
            pd.merge_asof(ls, rs, left_on="left", right_on="right"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge_asof(ls, rs, left_index=True, right_index=True), pd.DataFrame
        ),
        pd.DataFrame,
    )

    check(assert_type(pd.merge_asof(lf, rf, on="a"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(pd.merge_asof(lf, rf, left_on="a", right_on="b"), pd.DataFrame),
        pd.DataFrame,
    )

    check(
        assert_type(pd.merge_asof(lf, rf, on="a", by="b"), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(
            pd.merge_asof(lf, rf, left_on="c", right_on="d", by=["a", "b"]),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge_asof(lf, rf, on="a", left_by=["c"], right_by=["d"]), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge_asof(lf, rf, on="a", left_by=["b", "c"], right_by=["b", "d"]),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(pd.merge_asof(lf, rf, on="a", suffixes=["_1", None]), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(pd.merge_asof(lf, rf, on="a", suffixes=("_1", None)), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(pd.merge_asof(lf, rf, on="a", suffixes=("_1", "_2")), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(pd.merge_asof(lf, rf, on="a", suffixes=(None, "_2")), pd.DataFrame),
        pd.DataFrame,
    )

    quotes = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2016-05-25 13:30:00.023"),
                pd.Timestamp("2016-05-25 13:30:00.023"),
                pd.Timestamp("2016-05-25 13:30:00.030"),
                pd.Timestamp("2016-05-25 13:30:00.041"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
                pd.Timestamp("2016-05-25 13:30:00.049"),
                pd.Timestamp("2016-05-25 13:30:00.072"),
                pd.Timestamp("2016-05-25 13:30:00.075"),
            ],
            "ticker": ["GOOG", "MSFT", "MSFT", "MSFT", "GOOG", "AAPL", "GOOG", "MSFT"],
            "bid": [720.50, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
            "ask": [720.93, 51.96, 51.98, 52.00, 720.93, 98.01, 720.88, 52.03],
        }
    )
    trades = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2016-05-25 13:30:00.023"),
                pd.Timestamp("2016-05-25 13:30:00.038"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
            ],
            "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
            "price": [51.95, 51.95, 720.77, 720.92, 98.0],
            "quantity": [75, 155, 100, 100, 100],
        }
    )

    check(
        assert_type(
            pd.merge_asof(
                trades, quotes, on="time", by="ticker", tolerance=pd.Timedelta("10ms")
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge_asof(
                trades,
                quotes,
                on="time",
                by="ticker",
                tolerance=pd.Timedelta("10ms"),
                allow_exact_matches=False,
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge_asof(
                trades,
                quotes,
                on="time",
                by="ticker",
                tolerance=pd.Timedelta("10ms"),
                direction="backward",
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge_asof(
                trades,
                quotes,
                on="time",
                by="ticker",
                tolerance=pd.Timedelta("10ms"),
                direction="forward",
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge_asof(
                trades,
                quotes,
                on="time",
                by="ticker",
                tolerance=pd.Timedelta("10ms"),
                direction="nearest",
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )


def test_crosstab_args() -> None:
    a = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
    b: list = [4, 5, 6, 3, 4, 3, 5, 6, 5, 5]
    c = [1, 3, 2, 3, 1, 2, 3, 1, 3, 2]
    check(assert_type(pd.crosstab(a, b), pd.DataFrame), pd.DataFrame)
    check(assert_type(pd.crosstab(a, [b, c]), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(pd.crosstab(np.array(a), np.array(b)), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(pd.crosstab(np.array(a), [np.array(b), np.array(c)]), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(pd.crosstab(pd.Series(a), pd.Series(b)), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(pd.crosstab(pd.Index(a), pd.Index(b)), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(pd.crosstab(pd.Categorical(a), pd.Categorical(b)), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.crosstab(pd.Series(a), [pd.Series(b), pd.Series(c)]), pd.DataFrame
        ),
        pd.DataFrame,
    )
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    check(
        assert_type(pd.crosstab(a, b, values=values, aggfunc=np.sum), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(pd.crosstab(a, b, values=values, aggfunc="sum"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.crosstab(a, b, values=pd.Index(values), aggfunc=np.sum), pd.DataFrame
        ),
        pd.DataFrame,
    )

    check(
        assert_type(
            pd.crosstab(a, b, values=np.array(values), aggfunc="var"), pd.DataFrame
        ),
        pd.DataFrame,
    )

    check(
        assert_type(
            pd.crosstab(a, b, values=pd.Series(values), aggfunc=np.sum), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(pd.crosstab(a, b, values=values, aggfunc=np.mean), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(pd.crosstab(a, b, values=values, aggfunc="mean"), pd.DataFrame),
        pd.DataFrame,
    )

    def m(x: pd.Series) -> float:
        return x.sum() / len(x)

    check(
        assert_type(pd.crosstab(a, b, values=values, aggfunc=m), pd.DataFrame),
        pd.DataFrame,
    )

    def m2(x: pd.Series) -> int:
        return int(x.sum())

    check(
        assert_type(pd.crosstab(a, b, values=values, aggfunc=m2), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.crosstab(a, b, margins=True, margins_name="something"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(pd.crosstab(a, b, margins=True, dropna=True), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(pd.crosstab(a, b, colnames=["a"], rownames=["b"]), pd.DataFrame),
        pd.DataFrame,
    )
    rownames: list[tuple] = [("b", 1)]
    colnames: list[tuple] = [("a",)]
    check(
        assert_type(
            pd.crosstab(a, b, colnames=colnames, rownames=rownames),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(assert_type(pd.crosstab(a, b, normalize=0), pd.DataFrame), pd.DataFrame)
    check(assert_type(pd.crosstab(a, b, normalize=1), pd.DataFrame), pd.DataFrame)
    check(assert_type(pd.crosstab(a, b, normalize="all"), pd.DataFrame), pd.DataFrame)
    check(assert_type(pd.crosstab(a, b, normalize="index"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(pd.crosstab(a, b, normalize="columns"), pd.DataFrame), pd.DataFrame
    )


def test_pivot_extended() -> None:
    df = pd.DataFrame(
        data={
            "col1": ["first", "second", "third", "fourth"],
            "col2": [50, 70, 56, 111],
            "col3": ["A", "B", "B", "A"],
            "col4": [100, 102, 500, 600],
            ("col5",): ["E", "F", "G", "H"],
            ("col6", 6): ["apple", "banana", "cherry", "date"],
            ("col7", "other"): ["apple", "banana", "cherry", "date"],
            dt.date(2000, 1, 1): ["E", "F", "G", "H"],
            dt.datetime(2001, 1, 1, 12): ["E", "F", "G", "H"],
            dt.timedelta(7): ["E", "F", "G", "H"],
            True: ["E", "F", "G", "H"],
            9: ["E", "F", "G", "H"],
            10.0: ["E", "F", "G", "H"],
            (11.0 + 1j): ["E", "F", "G", "H"],
            pd.Timestamp(2002, 1, 1): ["E", "F", "G", "H"],
            pd.Timedelta(1, "D"): ["E", "F", "G", "H"],
        }
    )
    check(
        assert_type(
            pd.pivot(df, index="col1", columns="col3", values="col2"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot(df, index=[("col5",)], columns=[("col6", 6)], values="col2"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot(
                df, index=[("col5",)], columns=[("col6", 6)], values=[("col7", "other")]
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot(df, index=dt.date(2000, 1, 1), columns="col3", values="col2"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot(
                df, index=dt.datetime(2001, 1, 1, 12), columns="col3", values="col2"
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot(df, index=dt.timedelta(7), columns="col3", values="col2"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot(df, index=True, columns="col3", values="col2"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(pd.pivot(df, index=9, columns="col3", values="col2"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot(df, index=10.0, columns="col3", values="col2"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot(df, index=(11.0 + 1j), columns="col3", values="col2"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot(df, index=pd.Timestamp(2002, 1, 1), columns="col3", values="col2"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot(df, index=pd.Timedelta(1, "D"), columns="col3", values="col2"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )


def test_pivot_table() -> None:
    df = pd.DataFrame(
        {
            "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
            "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
            "C": [
                "small",
                "large",
                "large",
                "small",
                "small",
                "large",
                "small",
                "small",
                "large",
            ],
            "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
            "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
            ("col5",): ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
            ("col6", 6): [
                "one",
                "one",
                "one",
                "two",
                "two",
                "one",
                "one",
                "two",
                "two",
            ],
            (7, "seven"): [
                "small",
                "large",
                "large",
                "small",
                "small",
                "large",
                "small",
                "small",
                "large",
            ],
        }
    )
    check(
        assert_type(
            pd.pivot_table(
                df, values="D", index=["A", "B"], columns=["C"], aggfunc=np.sum
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot_table(
                df,
                values="D",
                index=pd.Series(["A", "B"]),
                columns=["C"],
                aggfunc="sum",
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot_table(
                df, values="D", index=["A", "B"], columns="C", aggfunc="mean"
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot_table(
                df,
                values="D",
                index=["A", "B"],
                columns=[(7, "seven")],
                aggfunc=np.sum,
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot_table(
                df,
                values="D",
                index=[("col5",), ("col6", 6)],
                columns=[(7, "seven")],
                aggfunc=np.sum,
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot_table(
                df, values="D", index=["A", "B"], columns=["C"], aggfunc="sum"
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )

    def f(x: pd.Series) -> float:
        return x.sum()

    check(
        assert_type(
            pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"], aggfunc=f),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )

    def g(x: pd.Series) -> int:
        return int(np.round(x.sum()))

    check(
        assert_type(
            pd.pivot_table(
                df,
                values=["D", "E"],
                index=["A", "B"],
                columns=["C"],
                aggfunc={"D": f, "E": g},
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot_table(
                df, values="D", index=["A", "B"], columns=["C"], aggfunc={"D": np.sum}
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot_table(
                df, values="D", index=["A", "B"], columns=["C"], aggfunc={"D": "sum"}
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot_table(
                df,
                values="D",
                index=["A", "B"],
                columns=["C"],
                aggfunc=[f, np.sum, "sum"],
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot_table(
                df,
                values="D",
                index=["A", "B"],
                columns=["C"],
                aggfunc=np.sum,
                margins=True,
                margins_name="Total",
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot_table(
                df,
                values="D",
                index=["A", "B"],
                columns=["C"],
                aggfunc=np.sum,
                dropna=True,
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot_table(
                df,
                values="D",
                index=["A", "B"],
                columns=["C"],
                aggfunc=np.sum,
                dropna=True,
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot_table(
                df,
                values="D",
                index=["A", "B"],
                columns=["C"],
                aggfunc=np.sum,
                observed=True,
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot_table(
                df,
                values="D",
                index=["A", "B"],
                columns=["C"],
                aggfunc=np.sum,
                sort=False,
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )

    idx = pd.DatetimeIndex(
        ["2011-01-01", "2011-02-01", "2011-01-02", "2011-01-01", "2011-01-02"]
    )
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "dt": pd.date_range("2011-01-01", freq="D", periods=5),
        },
        index=idx,
    )
    check(
        assert_type(
            pd.pivot_table(
                df, index=pd.Index(idx.month), columns=Grouper(key="dt", freq="M")
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot_table(
                df, index=np.array(idx.month), columns=Grouper(key="dt", freq="M")
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot_table(
                df, index=Grouper(key="dt", freq="M"), columns=pd.Index(idx.month)
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot_table(
                df, index=Grouper(key="dt", freq="M"), columns=np.array(idx.month)
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.pivot_table(
                df, index=Grouper(freq="A"), columns=Grouper(key="dt", freq="M")
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )


def test_argmin_and_argmax_return() -> None:
    df = pd.DataFrame({"a": [-1, 0, 1], "b": [1, 2, 3]})
    i = df.a.abs().argmin()
    i1 = df.a.abs().argmax()
    check(assert_type(i, np.int64), np.int64)
    check(assert_type(i1, np.int64), np.int64)
