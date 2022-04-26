# flake8: noqa: F841
import tempfile
from typing import Any, Dict, List, Union

from pandas.io.parsers import TextFileReader

import numpy as np
import pandas as pd

from . import check_series_result, check_dataframe_result


def test_types_to_datetime() -> None:
    df = pd.DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]})
    r1: pd.Series = pd.to_datetime(df)
    r2: pd.Series = pd.to_datetime(df, unit="s", origin="unix", infer_datetime_format=True)
    r3: pd.Series = pd.to_datetime(df, unit="ns", dayfirst=True, utc=None, format="%M:%D", exact=False)
    r4: pd.DatetimeIndex = pd.to_datetime([1, 2], unit="D", origin=pd.Timestamp("01/01/2000"))
    r5: pd.DatetimeIndex = pd.to_datetime([1, 2], unit="D", origin=3)
    r6: pd.DatetimeIndex = pd.to_datetime(["2022-01-03", "2022-02-22"])
    r7: pd.DatetimeIndex = pd.to_datetime(pd.Index(["2022-01-03", "2022-02-22"]))
    r8: pd.Series = pd.to_datetime({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]})


def test_types_concat() -> None:
    s = pd.Series([0, 1, -10])
    s2 = pd.Series([7, -5, 10])

    check_series_result(pd.concat([s, s2]))
    check_dataframe_result(pd.concat([s, s2], axis=1))
    check_series_result(pd.concat([s, s2], keys=["first", "second"], sort=True))
    check_series_result(pd.concat([s, s2], keys=["first", "second"], names=["source", "row"]))

    # Depends on the axis
    rs1: Union[pd.Series, pd.DataFrame] = pd.concat({"a": s, "b": s2})
    rs1a: Union[pd.Series, pd.DataFrame] = pd.concat({"a": s, "b": s2}, axis=1)
    rs2: Union[pd.Series, pd.DataFrame] = pd.concat({1: s, 2: s2})
    rs2a: Union[pd.Series, pd.DataFrame] = pd.concat({1: s, 2: s2}, axis=1)
    rs3: Union[pd.Series, pd.DataFrame] = pd.concat({1: s, None: s2})
    rs3a: Union[pd.Series, pd.DataFrame] = pd.concat({1: s, None: s2}, axis=1)

    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df2 = pd.DataFrame(data={"col1": [10, 20], "col2": [30, 40]})

    check_dataframe_result(pd.concat([df, df2]))
    check_dataframe_result(pd.concat([df, df2], axis=1))
    check_dataframe_result(pd.concat([df, df2], keys=["first", "second"], sort=True))
    check_dataframe_result(pd.concat([df, df2], keys=["first", "second"], names=["source", "row"]))

    result: pd.DataFrame = pd.concat({"a": pd.DataFrame([1, 2, 3]), "b": pd.DataFrame([4, 5, 6])}, axis=1)
    result2: Union[pd.DataFrame, pd.Series] = pd.concat({"a": pd.Series([1, 2, 3]), "b": pd.Series([4, 5, 6])}, axis=1)

    rdf1: pd.DataFrame = pd.concat({"a": df, "b": df2})
    rdf2: pd.DataFrame = pd.concat({1: df, 2: df2})
    rdf3: pd.DataFrame = pd.concat({1: df, None: df2})

    rdf4: pd.DataFrame = pd.concat(list(map(lambda x: s2, ["some_value", 3])), axis=1)
    adict = {"a": df, 2: df2}
    rdict: pd.DataFrame = pd.concat(adict)


def test_types_json_normalize() -> None:
    data1: List[Dict[str, Any]] = [
        {"id": 1, "name": {"first": "Coleen", "last": "Volk"}},
        {"name": {"given": "Mose", "family": "Regner"}},
        {"id": 2, "name": "Faye Raker"},
    ]
    df1: pd.DataFrame = pd.json_normalize(data=data1)
    df2: pd.DataFrame = pd.json_normalize(data=data1, max_level=0, sep=";")
    df3: pd.DataFrame = pd.json_normalize(data=data1, meta_prefix="id", record_prefix="name", errors="raise")
    df4: pd.DataFrame = pd.json_normalize(data=data1, record_path=None, meta="id")
    data2: Dict[str, Any] = {"name": {"given": "Mose", "family": "Regner"}}
    df5: pd.DataFrame = pd.json_normalize(data=data2)


def test_types_read_csv() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    csv_df: str = df.to_csv()

    with tempfile.NamedTemporaryFile(delete=False) as file:
        df.to_csv(file.name)
        file.close()
        df2: pd.DataFrame = pd.read_csv(file.name)
        df3: pd.DataFrame = pd.read_csv(file.name, sep="a", squeeze=False)
        df4: pd.DataFrame = pd.read_csv(file.name, header=None, prefix="b", mangle_dupe_cols=True, keep_default_na=False)
        df5: pd.DataFrame = pd.read_csv(file.name, engine="python", true_values=[0, 1, 3], na_filter=False)
        df6: pd.DataFrame = pd.read_csv(file.name, skiprows=lambda x: x in [0, 2], skip_blank_lines=True, dayfirst=False)
        df7: pd.DataFrame = pd.read_csv(file.name, nrows=2)
        df8: pd.DataFrame = pd.read_csv(file.name, dtype={"a": float, "b": int})

        tfr1: TextFileReader = pd.read_csv(file.name, nrows=2, iterator=True, chunksize=3)
        tfr2: TextFileReader = pd.read_csv(file.name, nrows=2, chunksize=1)
        tfr3: TextFileReader = pd.read_csv(file.name, nrows=2, iterator=False, chunksize=1)
        tfr4: TextFileReader = pd.read_csv(file.name, nrows=2, iterator=True)


def test_isna() -> None:
    s = pd.Series([1, np.nan, 3.2])
    check_series_result(pd.isna(s), bool)
    b: bool = pd.isna(np.nan)
    ar: np.ndarray = pd.isna(list(s))
    check_series_result(pd.notna(s), bool)
    b2: bool = pd.notna(np.nan)
    ar2: np.ndarray = pd.notna(list(s))
