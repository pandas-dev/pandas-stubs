from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Union,
)

import numpy as np
from numpy import typing as npt
import pandas as pd
from pandas.api.extensions import ExtensionArray
from typing_extensions import assert_type

from tests import check


def test_types_to_datetime() -> None:
    df = pd.DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]})
    r1: pd.Series = pd.to_datetime(df)
    r2: pd.Series = pd.to_datetime(
        df, unit="s", origin="unix", infer_datetime_format=True
    )
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
    rs1: pd.Series | pd.DataFrame = pd.concat({"a": s, "b": s2})
    rs1a: pd.Series | pd.DataFrame = pd.concat({"a": s, "b": s2}, axis=1)
    rs2: pd.Series | pd.DataFrame = pd.concat({1: s, 2: s2})
    rs2a: pd.Series | pd.DataFrame = pd.concat({1: s, 2: s2}, axis=1)
    rs3: pd.Series | pd.DataFrame = pd.concat({1: s, None: s2})
    rs3a: pd.Series | pd.DataFrame = pd.concat({1: s, None: s2}, axis=1)

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

    result: pd.DataFrame = pd.concat(
        {"a": pd.DataFrame([1, 2, 3]), "b": pd.DataFrame([4, 5, 6])}, axis=1
    )
    result2: pd.DataFrame | pd.Series = pd.concat(
        {"a": pd.Series([1, 2, 3]), "b": pd.Series([4, 5, 6])}, axis=1
    )

    rdf1: pd.DataFrame = pd.concat({"a": df, "b": df2})
    rdf2: pd.DataFrame = pd.concat({1: df, 2: df2})
    rdf3: pd.DataFrame = pd.concat({1: df, None: df2})

    rdf4: pd.DataFrame = pd.concat(map(lambda x: s2, ["some_value", 3]), axis=1)
    adict = {"a": df, 2: df2}
    rdict: pd.DataFrame = pd.concat(adict)


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
    check(assert_type(pd.isna(s1), "pd.Series[bool]"), pd.Series, bool)

    s2 = pd.Series([1, 3.2])
    check(assert_type(pd.notna(s2), "pd.Series[bool]"), pd.Series, bool)

    df1 = pd.DataFrame({"a": [1, 2, 1, 2], "b": [1, 1, 2, np.nan]})
    check(assert_type(pd.isna(df1), "pd.DataFrame"), pd.DataFrame)

    idx1 = pd.Index([1, 2, np.nan])
    check(assert_type(pd.isna(idx1), npt.NDArray[np.bool_]), np.ndarray, np.bool_)

    idx2 = pd.Index([1, 2])
    check(assert_type(pd.notna(idx2), npt.NDArray[np.bool_]), np.ndarray, np.bool_)

    assert check(assert_type(pd.isna(pd.NA), Literal[True]), bool)
    assert not check(assert_type(pd.notna(pd.NA), Literal[False]), bool)

    assert check(assert_type(pd.isna(pd.NaT), Literal[True]), bool)
    assert not check(assert_type(pd.notna(pd.NaT), Literal[False]), bool)

    assert check(assert_type(pd.isna(None), Literal[True]), bool)
    assert not check(assert_type(pd.notna(None), Literal[False]), bool)

    check(assert_type(pd.isna(2.5), bool), bool)
    check(assert_type(pd.notna(2.5), bool), bool)


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
            pd.Index,
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
