import io
import os.path
import pathlib
from pathlib import Path
import sqlite3
from typing import (
    TYPE_CHECKING,
    Generator,
    List,
    Union,
)

from packaging.version import parse
import pandas as pd
from pandas import (
    DataFrame,
    HDFStore,
    Series,
    __version__,
    read_clipboard,
    read_feather,
    read_hdf,
    read_html,
    read_json,
    read_orc,
    read_parquet,
    read_sas,
    read_spss,
    read_sql,
    read_sql_query,
    read_sql_table,
    read_stata,
    read_xml,
)
from pandas._testing import ensure_clean
import pytest
from typing_extensions import assert_type

from tests import check

from pandas.io.clipboard import PyperclipException
from pandas.io.json._json import JsonReader
from pandas.io.parsers import TextFileReader
from pandas.io.pytables import (
    TableIterator,
    Term,
)
from pandas.io.sas.sas7bdat import SAS7BDATReader
from pandas.io.sas.sas_xport import XportReader
from pandas.io.stata import StataReader

DF = DataFrame({"a": [1, 2, 3], "b": [0.0, 0.0, 0.0]})
CWD = os.path.split(os.path.abspath(__file__))[0]

PD_LT_15 = parse(__version__) < parse("1.5.0")


@pytest.mark.skipif(PD_LT_15, reason="pandas 1.5.0 or later required")
def test_orc():
    with ensure_clean() as path:
        check(assert_type(DF.to_orc(path), None), type(None))
        check(assert_type(read_orc(path), DataFrame), DataFrame)


@pytest.mark.skipif(PD_LT_15, reason="pandas 1.5.0 or later required")
def test_orc_path():
    with ensure_clean() as path:
        pathlib_path = Path(path)
        check(assert_type(DF.to_orc(pathlib_path), None), type(None))
        check(assert_type(read_orc(pathlib_path), DataFrame), DataFrame)


@pytest.mark.skipif(PD_LT_15, reason="pandas 1.5.0 or later required")
def test_orc_buffer():
    with ensure_clean() as path:
        file_w = open(path, "wb")
        check(assert_type(DF.to_orc(file_w), None), type(None))
        file_w.close()

        file_r = open(path, "rb")
        check(assert_type(read_orc(file_r), DataFrame), DataFrame)
        file_r.close()


@pytest.mark.skipif(PD_LT_15, reason="pandas 1.5.0 or later required")
def test_orc_columns():
    with ensure_clean() as path:
        check(assert_type(DF.to_orc(path, index=False), None), type(None))
        check(assert_type(read_orc(path, columns=["a"]), DataFrame), DataFrame)


@pytest.mark.skipif(PD_LT_15, reason="pandas 1.5.0 or later required")
def test_orc_bytes():
    check(assert_type(DF.to_orc(index=False), bytes), bytes)


def test_xml():
    with ensure_clean() as path:
        check(assert_type(DF.to_xml(path), None), type(None))
        check(assert_type(read_xml(path), DataFrame), DataFrame)
        with open(path, "rb") as f:
            check(assert_type(read_xml(f), DataFrame), DataFrame)


def test_xml_str():
    with ensure_clean() as path:
        check(assert_type(DF.to_xml(), str), str)
        out: str = DF.to_xml()
        check(assert_type(read_xml(io.StringIO(out)), DataFrame), DataFrame)


def test_read_stata_df():
    with ensure_clean() as path:
        DF.to_stata(path)
        check(assert_type(read_stata(path), pd.DataFrame), pd.DataFrame)


def test_read_stata_iterator_positional():
    with ensure_clean() as path:
        str_path = str(path)
        DF.to_stata(str_path)
        check(
            assert_type(
                read_stata(
                    str_path, False, False, None, False, False, None, False, 2, True
                ),
                StataReader,
            ),
            StataReader,
        )


def test_read_stata_iterator():
    with ensure_clean() as path:
        str_path = str(path)
        DF.to_stata(str_path)
        check(
            assert_type(read_stata(str_path, iterator=True), StataReader), StataReader
        )


def test_clipboard():
    try:
        DF.to_clipboard()
    except PyperclipException:
        pytest.skip("clipboard not available for testing")
    check(assert_type(read_clipboard(), DataFrame), DataFrame)
    check(assert_type(read_clipboard(iterator=False), DataFrame), DataFrame)
    check(assert_type(read_clipboard(chunksize=None), DataFrame), DataFrame)


def test_clipboard_iterator():
    try:
        DF.to_clipboard()
    except PyperclipException:
        pytest.skip("clipboard not available for testing")
    check(assert_type(read_clipboard(iterator=True), TextFileReader), TextFileReader)
    check(
        assert_type(read_clipboard(iterator=True, chunksize=None), TextFileReader),
        TextFileReader,
    )
    check(assert_type(read_clipboard(chunksize=1), TextFileReader), TextFileReader)
    check(
        assert_type(read_clipboard(iterator=False, chunksize=1), TextFileReader),
        TextFileReader,
    )


def test_sas_bdat() -> None:
    path = pathlib.Path(CWD, "data", "airline.sas7bdat")
    check(assert_type(read_sas(path), DataFrame), DataFrame)
    with check(
        assert_type(read_sas(path, iterator=True), Union[SAS7BDATReader, XportReader]),
        SAS7BDATReader,
    ):
        pass
    with check(
        assert_type(read_sas(path, iterator=True, format="sas7bdat"), SAS7BDATReader),
        SAS7BDATReader,
    ):
        pass
    with check(
        assert_type(read_sas(path, chunksize=1), Union[SAS7BDATReader, XportReader]),
        SAS7BDATReader,
    ):
        pass
    with check(
        assert_type(read_sas(path, chunksize=1, format="sas7bdat"), SAS7BDATReader),
        SAS7BDATReader,
    ):
        pass


def test_sas_xport() -> None:
    path = pathlib.Path(CWD, "data", "SSHSV1_A.xpt")
    check(assert_type(read_sas(path), DataFrame), DataFrame)
    with check(
        assert_type(read_sas(path, iterator=True), Union[SAS7BDATReader, XportReader]),
        XportReader,
    ):
        pass
    with check(
        assert_type(read_sas(path, iterator=True, format="xport"), XportReader),
        XportReader,
    ):
        pass
    with check(
        assert_type(read_sas(path, chunksize=1), Union[SAS7BDATReader, XportReader]),
        XportReader,
    ):
        pass
    with check(
        assert_type(read_sas(path, chunksize=1, format="xport"), XportReader),
        XportReader,
    ):
        pass


def test_hdf():
    with ensure_clean() as path:
        check(assert_type(DF.to_hdf(path, "df"), None), type(None))
        check(assert_type(read_hdf(path), Union[DataFrame, Series]), DataFrame)


def test_hdfstore():
    with ensure_clean() as path:
        store = HDFStore(path, model="w")
        check(assert_type(store, HDFStore), HDFStore)
        check(assert_type(store.put("df", DF, "table"), None), type(None))
        check(assert_type(store.append("df2", DF, "table"), None), type(None))
        check(assert_type(store.keys(), List[str]), list)
        check(assert_type(store.info(), str), str)
        check(
            assert_type(store.select("df", start=0, stop=1), Union[DataFrame, Series]),
            DataFrame,
        )
        check(
            assert_type(store.select("df", where="index>=1"), Union[DataFrame, Series]),
            DataFrame,
        )
        check(
            assert_type(
                store.select("df", where=Term("index>=1")),
                Union[DataFrame, Series],
            ),
            DataFrame,
        )
        check(
            assert_type(
                store.select("df", where=[Term("index>=1")]),
                Union[DataFrame, Series],
            ),
            DataFrame,
        )
        check(assert_type(store.get("df"), Union[DataFrame, Series]), DataFrame)
        check(assert_type(store.close(), None), type(None))

        store = HDFStore(path, model="r")
        check(
            assert_type(read_hdf(store, "df"), Union[DataFrame, Series]),
            DataFrame,
        )
        store.close()


def test_read_hdf_iterator():
    with ensure_clean() as path:
        check(assert_type(DF.to_hdf(path, "df", format="table"), None), type(None))
        ti = read_hdf(path, chunksize=1)
        check(assert_type(ti, TableIterator), TableIterator)
        ti.close()

        ti = read_hdf(path, "df", iterator=True)
        check(assert_type(ti, TableIterator), TableIterator)
        for _ in ti:
            pass
        ti.close()


def test_hdf_context_manager():
    with ensure_clean() as path:
        check(assert_type(DF.to_hdf(path, "df", format="table"), None), type(None))
        with HDFStore(path, mode="r") as store:
            check(assert_type(store.is_open, bool), bool)
            check(assert_type(store.get("df"), Union[DataFrame, Series]), DataFrame)


def test_hdf_series():
    s = DF["a"]
    with ensure_clean() as path:
        check(assert_type(s.to_hdf(path, "s"), None), type(None))
        check(assert_type(read_hdf(path, "s"), Union[DataFrame, Series]), Series)


def test_spss():
    path = Path(CWD, "data", "labelled-num.sav")
    check(assert_type(read_spss(path, convert_categoricals=True), DataFrame), DataFrame)
    check(assert_type(read_spss(str(path), usecols=["VAR00002"]), DataFrame), DataFrame)


def test_json():
    with ensure_clean() as path:
        check(assert_type(DF.to_json(path), None), type(None))
        check(assert_type(read_json(path), DataFrame), DataFrame)
    json_str = DF.to_json()
    check(assert_type(json_str, str), str)
    bin_json = io.StringIO(json_str)
    check(assert_type(read_json(bin_json), DataFrame), DataFrame)


def test_json_series():
    s = DF["a"]
    with ensure_clean() as path:
        check(assert_type(s.to_json(path), None), type(None))
        check(assert_type(read_json(path, typ="series"), Series), Series)
    check(assert_type(DF.to_json(), str), str)


def test_json_chunk():
    with ensure_clean() as path:
        check(assert_type(DF.to_json(path), None), type(None))
        json_reader = read_json(path, chunksize=1, lines=True)
        check(assert_type(json_reader, "JsonReader[DataFrame]"), JsonReader)
        for sub_df in json_reader:
            check(assert_type(sub_df, DataFrame), DataFrame)
    check(assert_type(DF.to_json(), str), str)


def test_parquet():
    with ensure_clean() as path:
        check(assert_type(DF.to_parquet(path), None), type(None))
        check(assert_type(read_parquet(path), DataFrame), DataFrame)
    check(assert_type(DF.to_parquet(), bytes), bytes)


def test_parquet_options():
    with ensure_clean(".parquet") as path:
        check(
            assert_type(DF.to_parquet(path, compression=None, index=True), None),
            type(None),
        )
        check(assert_type(read_parquet(path), DataFrame), DataFrame)


def test_feather():
    with ensure_clean() as path:
        check(assert_type(DF.to_feather(path), None), type(None))
        check(assert_type(read_feather(path), DataFrame), DataFrame)
        check(assert_type(read_feather(path, columns=["a"]), DataFrame), DataFrame)
    bio = io.BytesIO()
    check(assert_type(DF.to_feather(bio), None), type(None))
    bio.seek(0)
    check(assert_type(read_feather(bio), DataFrame), DataFrame)


def test_to_string():
    check(assert_type(DF.to_string(), str), str)
    with ensure_clean() as path:
        check(assert_type(DF.to_string(path), None), type(None))
        check(assert_type(DF.to_string(pathlib.Path(path)), None), type(None))
        with open(path, "wt") as df_string:
            check(assert_type(DF.to_string(df_string), None), type(None))
        sio = io.StringIO()
        check(assert_type(DF.to_string(sio), None), type(None))


def test_read_sql():
    with ensure_clean() as path:
        con = sqlite3.connect(path)
        check(assert_type(DF.to_sql("test", con=con), Union[int, None]), int)
        check(
            assert_type(read_sql("select * from test", con=con), DataFrame), DataFrame
        )
        con.close()


def test_read_sql_generator():
    with ensure_clean() as path:
        con = sqlite3.connect(path)
        check(assert_type(DF.to_sql("test", con=con), Union[int, None]), int)

        check(
            assert_type(
                read_sql("select * from test", con=con, chunksize=1),
                Generator[DataFrame, None, None],
            ),
            Generator,
        )
        con.close()


def test_read_sql_table():
    if TYPE_CHECKING:
        # sqlite3 doesn't support read_table, which is required for this function
        # Could only run in pytest if SQLAlchemy was installed
        with ensure_clean() as path:
            con = sqlite3.connect(path)
            assert_type(DF.to_sql("test", con=con), Union[int, None])
            assert_type(read_sql_table("test", con=con), DataFrame)
            assert_type(
                read_sql_table("test", con=con, chunksize=1),
                Generator[DataFrame, None, None],
            )
            con.close()


def test_read_sql_query():
    with ensure_clean() as path:
        con = sqlite3.connect(path)
        check(assert_type(DF.to_sql("test", con=con), Union[int, None]), int)
        check(
            assert_type(
                read_sql_query("select * from test", con=con, index_col="index"),
                DataFrame,
            ),
            DataFrame,
        )
        con.close()


def test_read_sql_query_generator():
    with ensure_clean() as path:
        con = sqlite3.connect(path)
        check(assert_type(DF.to_sql("test", con=con), Union[int, None]), int)

        check(
            assert_type(
                read_sql_query("select * from test", con=con, chunksize=1),
                Generator[DataFrame, None, None],
            ),
            Generator,
        )
        con.close()


def test_read_html():
    check(assert_type(DF.to_html(), str), str)
    with ensure_clean() as path:
        check(assert_type(DF.to_html(path), None), type(None))
        check(assert_type(read_html(path), List[DataFrame]), list)
