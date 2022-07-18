from __future__ import annotations

# flake8: noqa: F841
import tempfile
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Union,
)

import numpy as np
import pandas as pd
import pytest
from typing_extensions import assert_type

from tests import check

from pandas.io.parsers import TextFileReader


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


def test_types_read_csv() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    csv_df: str = df.to_csv()

    with tempfile.NamedTemporaryFile(delete=False) as file:
        df.to_csv(file.name)
        file.close()
        df2: pd.DataFrame = pd.read_csv(file.name)
        with pytest.warns(FutureWarning, match="The squeeze argument"):
            df3: pd.DataFrame = pd.read_csv(file.name, sep="a", squeeze=False)
        with pytest.warns(FutureWarning, match="The prefix argument has been"):
            df4: pd.DataFrame = pd.read_csv(
                file.name,
                header=None,
                prefix="b",
                mangle_dupe_cols=True,
                keep_default_na=False,
            )
        df5: pd.DataFrame = pd.read_csv(
            file.name, engine="python", true_values=[0, 1, 3], na_filter=False
        )
        df6: pd.DataFrame = pd.read_csv(
            file.name,
            skiprows=lambda x: x in [0, 2],
            skip_blank_lines=True,
            dayfirst=False,
        )
        df7: pd.DataFrame = pd.read_csv(file.name, nrows=2)
        df8: pd.DataFrame = pd.read_csv(file.name, dtype={"a": float, "b": int})
        df9: pd.DataFrame = pd.read_csv(file.name, usecols=["col1"])
        df10: pd.DataFrame = pd.read_csv(file.name, usecols={"col1"})
        df11: pd.DataFrame = pd.read_csv(file.name, usecols=[0])
        df12: pd.DataFrame = pd.read_csv(file.name, usecols=np.array([0]))
        df13: pd.DataFrame = pd.read_csv(file.name, usecols=("col1",))
        df14: pd.DataFrame = pd.read_csv(file.name, usecols=pd.Series(data=["col1"]))

        tfr1: TextFileReader = pd.read_csv(
            file.name, nrows=2, iterator=True, chunksize=3
        )
        tfr2: TextFileReader = pd.read_csv(file.name, nrows=2, chunksize=1)
        tfr3: TextFileReader = pd.read_csv(
            file.name, nrows=2, iterator=False, chunksize=1
        )
        tfr4: TextFileReader = pd.read_csv(file.name, nrows=2, iterator=True)


def test_isna() -> None:
    s = pd.Series([1, np.nan, 3.2])
    check(assert_type(pd.isna(s), "pd.Series[bool]"), pd.Series, bool)
    b: bool = pd.isna(np.nan)
    ar: np.ndarray = pd.isna(s.to_list())
    check(assert_type(pd.notna(s), "pd.Series[bool]"), pd.Series, bool)
    b2: bool = pd.notna(np.nan)
    ar2: np.ndarray = pd.notna(s.to_list())


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
