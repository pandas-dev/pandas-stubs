from collections import defaultdict
import csv
import io
import os.path
import pathlib
from pathlib import Path
import sqlite3
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Literal,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import (
    DataFrame,
    HDFStore,
    Series,
    errors,
    read_clipboard,
    read_csv,
    read_excel,
    read_feather,
    read_fwf,
    read_hdf,
    read_html,
    read_json,
    read_orc,
    read_parquet,
    read_pickle,
    read_sas,
    read_spss,
    read_sql,
    read_sql_query,
    read_sql_table,
    read_stata,
    read_table,
    read_xml,
)
from pandas._testing import ensure_clean
import pytest
import sqlalchemy
import sqlalchemy.ext.declarative
import sqlalchemy.orm
import sqlalchemy.orm.decl_api
from typing_extensions import assert_type

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    WINDOWS,
    check,
)

from pandas.io.api import to_pickle
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


@pytest.mark.skipif(WINDOWS, reason="ORC not available on windows")
def test_orc():
    with ensure_clean() as path:
        check(assert_type(DF.to_orc(path), None), type(None))
        check(assert_type(read_orc(path), DataFrame), DataFrame)


@pytest.mark.skipif(WINDOWS, reason="ORC not available on windows")
def test_orc_path():
    with ensure_clean() as path:
        pathlib_path = Path(path)
        check(assert_type(DF.to_orc(pathlib_path), None), type(None))
        check(assert_type(read_orc(pathlib_path), DataFrame), DataFrame)


@pytest.mark.skipif(WINDOWS, reason="ORC not available on windows")
def test_orc_buffer():
    with ensure_clean() as path:
        with open(path, "wb") as file_w:
            check(assert_type(DF.to_orc(file_w), None), type(None))

        with open(path, "rb") as file_r:
            check(assert_type(read_orc(file_r), DataFrame), DataFrame)


@pytest.mark.skipif(WINDOWS, reason="ORC not available on windows")
def test_orc_columns():
    with ensure_clean() as path:
        check(assert_type(DF.to_orc(path, index=False), None), type(None))
        check(assert_type(read_orc(path, columns=["a"]), DataFrame), DataFrame)


@pytest.mark.skipif(WINDOWS, reason="ORC not available on windows")
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


def test_pickle():
    with ensure_clean() as path:
        check(assert_type(DF.to_pickle(path), None), type(None))
        check(assert_type(read_pickle(path), Any), DataFrame)

    with ensure_clean() as path:
        check(assert_type(to_pickle(DF, path), None), type(None))
        check(assert_type(read_pickle(path), Any), DataFrame)


def test_pickle_file_handle():
    with ensure_clean() as path:
        check(assert_type(DF.to_pickle(path), None), type(None))
        file = open(path, "rb")
        check(assert_type(read_pickle(file), Any), DataFrame)
        file.close()


def test_pickle_path():
    with ensure_clean() as path:
        check(assert_type(DF.to_pickle(path), None), type(None))
        check(assert_type(read_pickle(Path(path)), Any), DataFrame)


def test_pickle_protocol():
    with ensure_clean() as path:
        DF.to_pickle(path, protocol=3)
        check(assert_type(read_pickle(path), Any), DataFrame)


def test_pickle_compression():
    with ensure_clean() as path:
        DF.to_pickle(path, compression="gzip")
        check(
            assert_type(read_pickle(path, compression="gzip"), Any),
            DataFrame,
        )

        check(
            assert_type(read_pickle(path, compression="gzip"), Any),
            DataFrame,
        )


def test_pickle_storage_options():
    with ensure_clean() as path:
        DF.to_pickle(path, storage_options={})

        check(
            assert_type(read_pickle(path, storage_options={}), Any),
            DataFrame,
        )


def test_to_pickle_series():
    s: Series = DF["a"]
    with ensure_clean() as path:
        check(assert_type(s.to_pickle(path), None), type(None))
        check(assert_type(read_pickle(path), Any), Series)


def test_read_stata_df():
    with ensure_clean() as path:
        DF.to_stata(path)
        check(assert_type(read_stata(path), pd.DataFrame), pd.DataFrame)


def test_read_stata_iterator():
    with ensure_clean() as path:
        str_path = str(path)
        DF.to_stata(str_path)
        check(
            assert_type(read_stata(str_path, iterator=True), StataReader), StataReader
        )
        reader = read_stata(str_path, chunksize=1)
        check(assert_type(reader, StataReader), StataReader)


def _true_if_b(s: str) -> bool:
    return s == "b"


def _true_if_greater_than_0(i: int) -> bool:
    return i > 0


def _true_if_first_param_is_head(t: Tuple[str, int]) -> bool:
    return t[0] == "head"


def test_clipboard():
    try:
        DF.to_clipboard()
    except errors.PyperclipException:
        pytest.skip("clipboard not available for testing")
    check(assert_type(read_clipboard(), DataFrame), DataFrame)
    check(assert_type(read_clipboard(iterator=False), DataFrame), DataFrame)
    check(assert_type(read_clipboard(chunksize=None), DataFrame), DataFrame)
    check(
        assert_type(read_clipboard(dtype=defaultdict(lambda: "f8")), DataFrame),
        DataFrame,
    )
    check(assert_type(read_clipboard(names=None), DataFrame), DataFrame)
    check(
        assert_type(read_clipboard(names=("first", "second"), header=0), DataFrame),
        DataFrame,
    )
    check(assert_type(read_clipboard(names=range(2), header=0), DataFrame), DataFrame)
    check(assert_type(read_clipboard(names=(1, "two"), header=0), DataFrame), DataFrame)
    check(
        assert_type(
            read_clipboard(names=(("first", 1), ("last", 2)), header=0), DataFrame
        ),
        DataFrame,
    )
    check(
        assert_type(read_clipboard(usecols=None), DataFrame),
        DataFrame,
    )
    check(
        assert_type(read_clipboard(usecols=["a"]), DataFrame),
        DataFrame,
    )
    check(
        assert_type(read_clipboard(usecols=(0,)), DataFrame),
        DataFrame,
    )
    check(
        assert_type(read_clipboard(usecols=range(1)), DataFrame),
        DataFrame,
    )
    check(
        assert_type(read_clipboard(usecols=_true_if_b), DataFrame),
        DataFrame,
    )
    check(
        assert_type(
            read_clipboard(
                names=[1, 2], usecols=_true_if_greater_than_0, header=0, index_col=0
            ),
            DataFrame,
        ),
        DataFrame,
    )
    check(
        assert_type(
            read_clipboard(
                names=(("head", 1), ("tail", 2)),
                usecols=_true_if_first_param_is_head,
                header=0,
                index_col=0,
            ),
            DataFrame,
        ),
        DataFrame,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        pd.read_clipboard(names="abcd")  # type: ignore[call-overload] # pyright: ignore[reportGeneralTypeIssues]
        pd.read_clipboard(usecols="abcd")  # type: ignore[arg-type] # pyright: ignore[reportGeneralTypeIssues]


def test_clipboard_iterator():
    try:
        DF.to_clipboard()
    except errors.PyperclipException:
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
        for key in store:
            check(assert_type(key, str), str)
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
    check(
        assert_type(
            read_json(s.to_json(orient=None), typ="series", orient=None), Series
        ),
        Series,
    )
    check(
        assert_type(
            read_json(s.to_json(orient="split"), typ="series", orient="split"), Series
        ),
        Series,
    )
    check(
        assert_type(
            read_json(s.to_json(orient="records"), typ="series", orient="records"),
            Series,
        ),
        Series,
    )
    check(
        assert_type(
            read_json(s.to_json(orient="index"), typ="series", orient="index"), Series
        ),
        Series,
    )
    check(
        assert_type(
            read_json(s.to_json(orient="table"), typ="series", orient="table"), Series
        ),
        Series,
    )


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
        check(assert_type(DF.to_parquet(), bytes), bytes)
        check(assert_type(read_parquet(path), DataFrame), DataFrame)


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
    with io.BytesIO() as bio:
        check(assert_type(DF.to_feather(bio), None), type(None))
        bio.seek(0)
        check(assert_type(read_feather(bio), DataFrame), DataFrame)


def test_read_csv():
    with ensure_clean() as path:
        check(assert_type(DF.to_csv(path), None), type(None))
        check(assert_type(read_csv(path), DataFrame), DataFrame)
        with open(path) as csv_file:
            check(assert_type(read_csv(csv_file), DataFrame), DataFrame)
        with open(path) as csv_file:
            sio = io.StringIO(csv_file.read())
            check(assert_type(read_csv(sio), DataFrame), DataFrame)
        check(assert_type(read_csv(path, iterator=False), DataFrame), DataFrame)
        check(assert_type(read_csv(path, chunksize=None), DataFrame), DataFrame)
        check(
            assert_type(read_csv(path, dtype=defaultdict(lambda: "f8")), DataFrame),
            DataFrame,
        )


def test_read_csv_iterator():
    with ensure_clean() as path:
        check(assert_type(DF.to_csv(path), None), type(None))
        tfr = read_csv(path, iterator=True)
        check(assert_type(tfr, TextFileReader), TextFileReader)
        tfr.close()
        tfr2 = read_csv(pathlib.Path(path), chunksize=1)
        check(
            assert_type(tfr2, TextFileReader),
            TextFileReader,
        )
        tfr2.close()


def _true_if_col1(s: str) -> bool:
    return s == "col1"


def test_types_read_csv() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    csv_df: str = df.to_csv()

    with ensure_clean() as path:
        df.to_csv(path)
        df2: pd.DataFrame = pd.read_csv(path)
        df3: pd.DataFrame = pd.read_csv(path, sep="a")
        df4: pd.DataFrame = pd.read_csv(
            path,
            header=None,
        )
        df5: pd.DataFrame = pd.read_csv(
            path, engine="python", true_values=["no", "No", "NO"], na_filter=False
        )
        df6: pd.DataFrame = pd.read_csv(
            path,
            skiprows=lambda x: x in [0, 2],
            skip_blank_lines=True,
            dayfirst=False,
        )
        df7: pd.DataFrame = pd.read_csv(path, nrows=2)
        df8: pd.DataFrame = pd.read_csv(path, dtype={"a": float, "b": int})
        df9: pd.DataFrame = pd.read_csv(path, usecols=["col1"])
        df10: pd.DataFrame = pd.read_csv(path, usecols=[0])
        df11: pd.DataFrame = pd.read_csv(path, usecols=np.array([0]))
        df12: pd.DataFrame = pd.read_csv(path, usecols=("col1",))
        df13: pd.DataFrame = pd.read_csv(path, usecols=pd.Series(data=["col1"]))
        df14: pd.DataFrame = pd.read_csv(path, converters=None)
        df15: pd.DataFrame = pd.read_csv(path, names=("first", "second"), header=0)
        df16: pd.DataFrame = pd.read_csv(path, names=range(2), header=0)
        df17: pd.DataFrame = pd.read_csv(path, names=(1, "two"), header=0)
        df18: pd.DataFrame = pd.read_csv(
            path,
            names=(
                (
                    "first",
                    1,
                ),
                ("last", 2),
            ),
            header=0,
        )
        df19: pd.DataFrame = pd.read_csv(path, usecols=None)
        df20: pd.DataFrame = pd.read_csv(path, usecols=["col1"])
        df21: pd.DataFrame = pd.read_csv(path, usecols=(0,))
        df22: pd.DataFrame = pd.read_csv(path, usecols=range(1))
        df23: pd.DataFrame = pd.read_csv(path, usecols=_true_if_col1)
        df24: pd.DataFrame = pd.read_csv(
            path, names=[1, 2], usecols=_true_if_greater_than_0, header=0, index_col=0
        )
        df25: pd.DataFrame = pd.read_csv(
            path,
            names=(("head", 1), ("tail", 2)),
            usecols=_true_if_first_param_is_head,
            header=0,
            index_col=0,
        )

        if TYPE_CHECKING_INVALID_USAGE:
            pd.read_csv(path, names="abcd")  # type: ignore[call-overload] # pyright: ignore[reportGeneralTypeIssues]
            pd.read_csv(path, usecols="abcd")  # type: ignore[arg-type] # pyright: ignore[reportGeneralTypeIssues]

        tfr1: TextFileReader = pd.read_csv(path, nrows=2, iterator=True, chunksize=3)
        tfr1.close()

        tfr2: TextFileReader = pd.read_csv(path, nrows=2, chunksize=1)
        tfr2.close()

        tfr3: TextFileReader = pd.read_csv(path, nrows=2, iterator=False, chunksize=1)
        tfr3.close()

        tfr4: TextFileReader = pd.read_csv(path, nrows=2, iterator=True)
        tfr4.close()

    df_dates = pd.DataFrame(data={"col1": ["2023-03-15", "2023-04-20"]})

    with ensure_clean() as path:
        df_dates.to_csv(path)

        df26: pd.DataFrame = pd.read_csv(
            path, parse_dates=["col1"], date_format="%Y-%m-%d"
        )
        df27: pd.DataFrame = pd.read_csv(
            path, parse_dates=["col1"], date_format={"col1": "%Y-%m-%d"}
        )
        df28: pd.DataFrame = pd.read_csv(
            path, parse_dates=["col1"], date_format={1: "%Y-%m-%d"}
        )


def test_read_table():
    with ensure_clean() as path:
        check(assert_type(DF.to_csv(path, sep="\t"), None), type(None))
        check(assert_type(read_table(path), DataFrame), DataFrame)
        check(assert_type(read_table(path, iterator=False), DataFrame), DataFrame)
        check(assert_type(read_table(path, chunksize=None), DataFrame), DataFrame)
        check(
            assert_type(read_table(path, dtype=defaultdict(lambda: "f8")), DataFrame),
            DataFrame,
        )
        check(
            assert_type(
                read_table(path, names=("first", "second"), header=0), DataFrame
            ),
            DataFrame,
        )
        check(
            assert_type(read_table(path, names=range(2), header=0), DataFrame),
            DataFrame,
        )
        check(
            assert_type(read_table(path, names=(1, "two"), header=0), DataFrame),
            DataFrame,
        )
        check(
            assert_type(
                read_table(
                    path,
                    names=(
                        (
                            "first",
                            1,
                        ),
                        ("last", 2),
                    ),
                    header=0,
                ),
                DataFrame,
            ),
            DataFrame,
        )
        check(
            assert_type(
                read_table(path, usecols=None),
                DataFrame,
            ),
            DataFrame,
        )
        check(
            assert_type(
                read_table(path, usecols=["a"]),
                DataFrame,
            ),
            DataFrame,
        )
        check(
            assert_type(
                read_table(path, usecols=(0,)),
                DataFrame,
            ),
            DataFrame,
        )
        check(
            assert_type(
                read_table(path, usecols=range(1)),
                DataFrame,
            ),
            DataFrame,
        )
        check(
            assert_type(
                read_table(path, usecols=_true_if_b),
                DataFrame,
            ),
            DataFrame,
        )
        check(
            assert_type(
                read_table(
                    path,
                    names=[1, 2],
                    usecols=_true_if_greater_than_0,
                    header=0,
                    index_col=0,
                ),
                DataFrame,
            ),
            DataFrame,
        )
        check(
            assert_type(
                read_table(
                    path,
                    names=(("head", 1), ("tail", 2)),
                    usecols=_true_if_first_param_is_head,
                    header=0,
                    index_col=0,
                ),
                DataFrame,
            ),
            DataFrame,
        )
        if TYPE_CHECKING_INVALID_USAGE:
            pd.read_table(path, names="abcd")  # type: ignore[call-overload] # pyright: ignore[reportGeneralTypeIssues]
            pd.read_table(path, usecols="abcd")  # type: ignore[arg-type] # pyright: ignore[reportGeneralTypeIssues]


def test_read_table_iterator():
    with ensure_clean() as path:
        check(assert_type(DF.to_csv(path, sep="\t"), None), type(None))
        tfr = read_table(path, iterator=True)
        check(assert_type(tfr, TextFileReader), TextFileReader)
        tfr.close()
        tfr2 = read_table(path, chunksize=1)
        check(assert_type(tfr2, TextFileReader), TextFileReader)
        tfr2.close()


def test_types_read_table():
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})

    with ensure_clean() as path:
        df.to_csv(path)

        df2: pd.DataFrame = pd.read_table(path, sep=",", converters=None)


def test_btest_read_fwf():
    with ensure_clean() as path:
        DF.to_string(path, index=False)
        check(assert_type(read_fwf(path), DataFrame), DataFrame)
        check(assert_type(read_fwf(pathlib.Path(path)), DataFrame), DataFrame)

        with open(path) as fwf_file:
            check(
                assert_type(read_fwf(fwf_file), DataFrame),
                DataFrame,
            )
        with open(path) as fwf_file:
            sio = io.StringIO(fwf_file.read())
            check(assert_type(read_fwf(sio), DataFrame), DataFrame)
        with open(path, "rb") as fwf_file:
            bio = io.BytesIO(fwf_file.read())
            check(assert_type(read_fwf(bio), DataFrame), DataFrame)
        with read_fwf(path, iterator=True) as fwf_iterator:
            check(assert_type(fwf_iterator, TextFileReader), TextFileReader)
        with read_fwf(path, chunksize=1) as fwf_iterator2:
            check(assert_type(fwf_iterator2, TextFileReader), TextFileReader)


def test_text_file_reader():
    with ensure_clean() as path:
        DF.to_string(path, index=False)
        tfr = TextFileReader(path, engine="python")
        check(assert_type(tfr, TextFileReader), TextFileReader)
        check(assert_type(tfr.read(1), DataFrame), DataFrame)
        check(assert_type(tfr.close(), None), type(None))
        with TextFileReader(path, engine="python") as tfr:
            check(assert_type(tfr.read(1), DataFrame), DataFrame)
        with TextFileReader(path, engine="python") as tfr:
            check(assert_type(tfr.__next__(), DataFrame), DataFrame)
            df_iter: DataFrame
            for df_iter in tfr:
                check(df_iter, DataFrame)


def test_to_csv_series():
    s: Series
    s = DF.iloc[:, 0]
    check(assert_type(s.to_csv(), str), str)
    with ensure_clean() as path:
        check(assert_type(s.to_csv(path), None), type(None))


def test_read_excel() -> None:
    with ensure_clean(".xlsx") as path:
        # https://github.com/pandas-dev/pandas-stubs/pull/33
        check(
            assert_type(pd.DataFrame({"A": [1, 2, 3]}).to_excel(path), None), type(None)
        )
        check(assert_type(pd.read_excel(path), pd.DataFrame), pd.DataFrame)
        check(
            assert_type(pd.read_excel(path, sheet_name="Sheet1"), pd.DataFrame),
            pd.DataFrame,
        )
        check(
            assert_type(
                pd.read_excel(path, sheet_name=["Sheet1"]),
                Dict[Union[int, str], pd.DataFrame],
            ),
            dict,
        )
        # GH 98
        check(
            assert_type(pd.read_excel(path, sheet_name=0), pd.DataFrame), pd.DataFrame
        )
        check(
            assert_type(
                pd.read_excel(path, sheet_name=[0]), Dict[Union[int, str], pd.DataFrame]
            ),
            dict,
        )
        check(
            assert_type(
                pd.read_excel(path, sheet_name=[0, "Sheet1"]),
                Dict[Union[int, str], pd.DataFrame],
            ),
            dict,
        )
        check(
            assert_type(
                pd.read_excel(path, sheet_name=None),
                Dict[Union[int, str], pd.DataFrame],
            ),
            dict,
        )
        check(
            assert_type(pd.read_excel(path, names=("test",), header=0), pd.DataFrame),
            pd.DataFrame,
        )
        check(
            assert_type(pd.read_excel(path, names=(1,), header=0), pd.DataFrame),
            pd.DataFrame,
        )
        check(
            assert_type(
                pd.read_excel(path, names=(("higher", "lower"),), header=0),
                pd.DataFrame,
            ),
            pd.DataFrame,
        ),
        check(
            assert_type(pd.read_excel(path, names=range(1), header=0), pd.DataFrame),
            pd.DataFrame,
        )
        check(
            assert_type(pd.read_excel(path, usecols=None), pd.DataFrame),
            pd.DataFrame,
        )
        check(
            assert_type(pd.read_excel(path, usecols=["A"]), pd.DataFrame),
            pd.DataFrame,
        )
        check(
            assert_type(pd.read_excel(path, usecols=(0,)), pd.DataFrame),
            pd.DataFrame,
        )
        check(
            assert_type(pd.read_excel(path, usecols=range(1)), pd.DataFrame),
            pd.DataFrame,
        )
        check(
            assert_type(pd.read_excel(path, usecols=_true_if_b), pd.DataFrame),
            pd.DataFrame,
        )
        check(
            assert_type(
                pd.read_excel(
                    path,
                    names=[1, 2],
                    usecols=_true_if_greater_than_0,
                    header=0,
                    index_col=0,
                ),
                pd.DataFrame,
            ),
            pd.DataFrame,
        )
        check(
            assert_type(
                pd.read_excel(
                    path,
                    names=(("head", 1), ("tail", 2)),
                    usecols=_true_if_first_param_is_head,
                    header=0,
                    index_col=0,
                ),
                pd.DataFrame,
            ),
            pd.DataFrame,
        )
        check(
            assert_type(
                pd.read_excel(
                    path,
                    usecols="A",
                ),
                pd.DataFrame,
            ),
            pd.DataFrame,
        )
        if TYPE_CHECKING_INVALID_USAGE:
            pd.read_excel(path, names="abcd")  # type: ignore[call-overload] # pyright: ignore[reportGeneralTypeIssues]


def test_read_excel_io_types() -> None:
    # GH 195
    df = pd.DataFrame([[1, 2], [8, 9]], columns=["A", "B"])
    with ensure_clean(".xlsx") as path:
        as_str: str = path
        df.to_excel(path)

        check(assert_type(pd.read_excel(as_str), pd.DataFrame), pd.DataFrame)

        as_path = Path(as_str)
        check(assert_type(pd.read_excel(as_path), pd.DataFrame), pd.DataFrame)

        with as_path.open("rb") as as_file:
            check(assert_type(pd.read_excel(as_file), pd.DataFrame), pd.DataFrame)

        as_bytes = as_path.read_bytes()
        check(assert_type(pd.read_excel(as_bytes), pd.DataFrame), pd.DataFrame)


def test_read_excel_basic():
    with ensure_clean(".xlsx") as path:
        check(assert_type(DF.to_excel(path), None), type(None))
        check(assert_type(read_excel(path), DataFrame), DataFrame)
        check(assert_type(read_excel(path, sheet_name="Sheet1"), DataFrame), DataFrame)
        check(assert_type(read_excel(path, sheet_name=0), DataFrame), DataFrame)


def test_read_excel_list():
    with ensure_clean(".xlsx") as path:
        check(assert_type(DF.to_excel(path), None), type(None))
        check(
            assert_type(
                read_excel(path, sheet_name=["Sheet1"]),
                Dict[Union[str, int], DataFrame],
            ),
            dict,
        )
        check(
            assert_type(
                read_excel(path, sheet_name=[0]), Dict[Union[str, int], DataFrame]
            ),
            dict,
        )


def test_read_excel_dtypes():
    # GH 440
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [10.0, 20.0, 30.3]})
    with ensure_clean(".xlsx") as path:
        check(assert_type(df.to_excel(path), None), type(None))
        dtypes = {"a": np.int64, "b": str, "c": np.float64}
        check(assert_type(read_excel(path, dtype=dtypes), pd.DataFrame), pd.DataFrame)


def test_excel_writer():
    with ensure_clean(".xlsx") as path:
        with pd.ExcelWriter(path) as ew:
            check(assert_type(ew, pd.ExcelWriter), pd.ExcelWriter)
            DF.to_excel(ew, sheet_name="A")
        check(assert_type(read_excel(path, sheet_name="A"), DataFrame), DataFrame)
        check(assert_type(read_excel(path), DataFrame), DataFrame)
        ef = pd.ExcelFile(path)
        check(assert_type(ef, pd.ExcelFile), pd.ExcelFile)
        check(assert_type(read_excel(ef, sheet_name="A"), DataFrame), DataFrame)
        check(assert_type(read_excel(ef), DataFrame), DataFrame)
        check(assert_type(ef.parse(sheet_name=0), DataFrame), DataFrame)
        check(
            assert_type(ef.parse(sheet_name=[0]), Dict[Union[str, int], DataFrame]),
            dict,
        )
        check(assert_type(ef.close(), None), type(None))


def test_excel_writer_engine():
    with ensure_clean(".xlsx") as path:
        with pd.ExcelWriter(path, engine="auto") as ew:
            check(assert_type(ew, pd.ExcelWriter), pd.ExcelWriter)
            DF.to_excel(ew, sheet_name="A")

    with ensure_clean(".xlsx") as path:
        with pd.ExcelWriter(path, engine="openpyxl") as ew:
            check(assert_type(ew, pd.ExcelWriter), pd.ExcelWriter)
            DF.to_excel(ew, sheet_name="A")
            check(
                assert_type(ew.engine, Literal["openpyxl", "odf", "xlsxwriter"]),
                str,
            )

    with ensure_clean(".ods") as path:
        with pd.ExcelWriter(path, engine="odf") as ew:
            check(assert_type(ew, pd.ExcelWriter), pd.ExcelWriter)
            DF.to_excel(ew, sheet_name="A")
            check(
                assert_type(ew.engine, Literal["openpyxl", "odf", "xlsxwriter"]),
                str,
            )

    with ensure_clean(".xlsx") as path:
        with pd.ExcelWriter(path, engine="xlsxwriter") as ew:
            check(assert_type(ew, pd.ExcelWriter), pd.ExcelWriter)
            DF.to_excel(ew, sheet_name="A")
            check(
                assert_type(ew.engine, Literal["openpyxl", "odf", "xlsxwriter"]),
                str,
            )


def test_excel_writer_append_mode():
    with ensure_clean(".xlsx") as path:
        with pd.ExcelWriter(path, mode="w") as ew:
            DF.to_excel(ew, sheet_name="A")
        with pd.ExcelWriter(path, mode="a", engine="openpyxl") as ew:
            DF.to_excel(ew, sheet_name="B")


def test_to_string():
    check(assert_type(DF.to_string(), str), str)
    with ensure_clean() as path:
        check(assert_type(DF.to_string(path), None), type(None))
        check(assert_type(DF.to_string(pathlib.Path(path)), None), type(None))
        with open(path, "w") as df_string:
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


def test_read_sql_via_sqlalchemy_connection():
    with ensure_clean() as path:
        db_uri = "sqlite:///" + path
        engine = sqlalchemy.create_engine(db_uri)

        with engine.connect() as conn:
            check(assert_type(DF.to_sql("test", con=conn), Union[int, None]), int)
            check(
                assert_type(read_sql("select * from test", con=conn), DataFrame),
                DataFrame,
            )
        engine.dispose()


def test_read_sql_via_sqlalchemy_engine():
    with ensure_clean() as path:
        db_uri = "sqlite:///" + path
        engine = sqlalchemy.create_engine(db_uri)

        check(assert_type(DF.to_sql("test", con=engine), Union[int, None]), int)
        check(
            assert_type(read_sql("select * from test", con=engine), DataFrame),
            DataFrame,
        )
        engine.dispose()


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


def test_csv_quoting():
    with ensure_clean() as path:
        check(assert_type(DF.to_csv(path, quoting=csv.QUOTE_ALL), None), type(None))
        check(assert_type(DF.to_csv(path, quoting=csv.QUOTE_NONE), None), type(None))
        check(
            assert_type(DF.to_csv(path, quoting=csv.QUOTE_NONNUMERIC), None), type(None)
        )
        check(assert_type(DF.to_csv(path, quoting=csv.QUOTE_MINIMAL), None), type(None))


def test_sqlalchemy_selectable() -> None:
    with ensure_clean() as path:
        db_uri = "sqlite:///" + path
        engine = sqlalchemy.create_engine(db_uri)

        if TYPE_CHECKING:
            # Just type checking since underlying dB does not exist
            class Base(metaclass=sqlalchemy.orm.decl_api.DeclarativeMeta):
                __abstract__ = True

            class Temp(Base):
                __tablename__ = "part"
                quantity = sqlalchemy.Column(sqlalchemy.Integer)

            Session = sqlalchemy.orm.sessionmaker(engine)
            with Session() as session:
                pd.read_sql(
                    session.query(Temp.quantity).statement, session.connection()
                )


def test_sqlalchemy_text() -> None:
    with ensure_clean() as path:
        db_uri = "sqlite:///" + path
        engine = sqlalchemy.create_engine(db_uri)
        sql_select = sqlalchemy.text("select * from test")
        with engine.connect() as conn:
            check(assert_type(DF.to_sql("test", con=conn), Union[int, None]), int)
            check(
                assert_type(read_sql(sql_select, con=conn), DataFrame),
                DataFrame,
            )
        engine.dispose()


def test_read_sql_dtype() -> None:
    with ensure_clean() as path:
        conn = sqlite3.connect(path)
        df = pd.DataFrame(
            data=[[0, "10/11/12"], [1, "12/11/10"]],
            columns=["int_column", "date_column"],
        )
        check(assert_type(df.to_sql("test_data", con=conn), Union[int, None]), int)
        check(
            assert_type(
                pd.read_sql(
                    "SELECT int_column, date_column FROM test_data",
                    con=conn,
                    dtype=None,
                ),
                pd.DataFrame,
            ),
            pd.DataFrame,
        )
        check(
            assert_type(
                pd.read_sql(
                    "SELECT int_column, date_column FROM test_data",
                    con=conn,
                    dtype={"int_column": int},
                ),
                pd.DataFrame,
            ),
            pd.DataFrame,
        )
        check(assert_type(DF.to_sql("test", con=conn), Union[int, None]), int)

        check(
            assert_type(
                read_sql("select * from test", con=conn, dtype=int),
                pd.DataFrame,
            ),
            pd.DataFrame,
        )
        conn.close()


def test_read_sql_dtype_backend() -> None:
    with ensure_clean() as path:
        conn2 = sqlite3.connect(path)
        check(assert_type(DF.to_sql("test", con=conn2), Union[int, None]), int)
        check(
            assert_type(
                read_sql("select * from test", con=conn2, dtype_backend="pyarrow"),
                pd.DataFrame,
            ),
            pd.DataFrame,
        )
        check(
            assert_type(
                read_sql(
                    "select * from test", con=conn2, dtype_backend="numpy_nullable"
                ),
                pd.DataFrame,
            ),
            pd.DataFrame,
        )
        conn2.close()


def test_all_read_without_lxml_dtype_backend() -> None:
    with ensure_clean() as path:
        check(assert_type(DF.to_csv(path), None), type(None))
        s1 = read_csv(path, iterator=True, dtype_backend="pyarrow")
        check(assert_type(s1, TextFileReader), TextFileReader)
        s1.close()

        DF.to_string(path, index=False)
        check(
            assert_type(read_fwf(path, dtype_backend="pyarrow"), DataFrame), DataFrame
        )

        check(assert_type(DF.to_json(path), None), type(None))
        check(
            assert_type(read_json(path, dtype_backend="pyarrow"), DataFrame), DataFrame
        )
        check(
            assert_type(read_json(path, dtype={"MatchID": str}), DataFrame), DataFrame
        )

    with ensure_clean() as path:
        con = sqlite3.connect(path)
        check(assert_type(DF.to_sql("test", con=con), Union[int, None]), int)
        check(
            assert_type(
                read_sql_query(
                    "select * from test",
                    con=con,
                    index_col="index",
                    dtype_backend="pyarrow",
                ),
                DataFrame,
            ),
            DataFrame,
        )
        con.close()

        if not WINDOWS:
            check(assert_type(DF.to_orc(path), None), type(None))
            check(
                assert_type(read_orc(path, dtype_backend="numpy_nullable"), DataFrame),
                DataFrame,
            )
        check(assert_type(DF.to_feather(path), None), type(None))
        check(
            assert_type(read_feather(path, dtype_backend="pyarrow"), DataFrame),
            DataFrame,
        )

        check(
            assert_type(
                pd.to_numeric(
                    [1.0, 2.0, "blerg"], errors="ignore", dtype_backend="numpy_nullable"
                ),
                npt.NDArray,
            ),
            np.ndarray,
        )

    with ensure_clean(".xlsx") as path:
        as_str: str = path
        DF.to_excel(path)
        check(
            assert_type(pd.read_excel(as_str, dtype_backend="pyarrow"), pd.DataFrame),
            pd.DataFrame,
        )

    try:
        DF.to_clipboard()
    except errors.PyperclipException:
        pytest.skip("clipboard not available for testing")
    check(
        assert_type(
            read_clipboard(iterator=True, dtype_backend="pyarrow"), TextFileReader
        ),
        TextFileReader,
    )

    if TYPE_CHECKING:
        # sqlite3 doesn't support read_table, which is required for this function
        # Could only run in pytest if SQLAlchemy was installed
        with ensure_clean() as path:
            co1 = sqlite3.connect(path)
            assert_type(DF.to_sql("test", con=co1), Union[int, None])
            assert_type(
                read_sql_table("test", con=co1, dtype_backend="numpy_nullable"),
                DataFrame,
            )
            co1.close()


def test_read_with_lxml_dtype_backend() -> None:
    with ensure_clean() as path:
        check(assert_type(DF.to_html(path), None), type(None))
        check(
            assert_type(
                read_html(path, dtype_backend="numpy_nullable"), List[DataFrame]
            ),
            list,
        )

        check(assert_type(DF.to_xml(path), None), type(None))
        check(
            assert_type(read_xml(path, dtype_backend="pyarrow"), DataFrame), DataFrame
        )


def test_read_sql_dict_str_value_dtype() -> None:
    # GH 676
    with ensure_clean() as path:
        con = sqlite3.connect(path)
        check(assert_type(DF.to_sql("test", con), Union[int, None]), int)
        check(
            assert_type(
                read_sql_query(
                    "select * from test",
                    con=con,
                    index_col="index",
                    dtype={"a": "int"},
                ),
                DataFrame,
            ),
            DataFrame,
        )
        con.close()


def test_added_date_format() -> None:
    with ensure_clean() as path:
        df_dates = pd.DataFrame(
            data={
                "col1": ["2023-03-15", "2023-04-20"],
            }
        )
        df_dates.to_csv(path)

        check(
            assert_type(
                pd.read_table(
                    path, sep=",", parse_dates=["col1"], date_format="%Y-%m-%d"
                ),
                pd.DataFrame,
            ),
            pd.DataFrame,
        )
        check(
            assert_type(
                pd.read_table(
                    path,
                    sep=",",
                    parse_dates=["col1"],
                    date_format={"col1": "%Y-%m-%d"},
                ),
                pd.DataFrame,
            ),
            pd.DataFrame,
        )
        check(
            assert_type(
                pd.read_table(
                    path, sep=",", parse_dates=["col1"], date_format={0: "%Y-%m-%d"}
                ),
                pd.DataFrame,
            ),
            pd.DataFrame,
        )

        check(
            assert_type(
                pd.read_fwf(path, date_format="%Y-%m-%d"),
                pd.DataFrame,
            ),
            pd.DataFrame,
        )
        check(
            assert_type(
                pd.read_fwf(path, date_format={"col1": "%Y-%m-%d"}),
                pd.DataFrame,
            ),
            pd.DataFrame,
        )
        check(
            assert_type(
                pd.read_fwf(path, date_format={0: "%Y-%m-%d"}),
                pd.DataFrame,
            ),
            pd.DataFrame,
        )
    with ensure_clean(".xlsx") as path:
        check(
            assert_type(
                pd.DataFrame(
                    data={
                        "col1": ["2023-03-15", "2023-04-20"],
                    }
                ).to_excel(path),
                None,
            ),
            type(None),
        )
        check(
            assert_type(
                pd.read_excel(path, parse_dates=["col1"], date_format={0: "%Y-%m-%d"}),
                pd.DataFrame,
            ),
            pd.DataFrame,
        )
        check(
            assert_type(
                pd.read_excel(
                    path, parse_dates=["col1"], date_format={"col1": "%Y-%m-%d"}
                ),
                pd.DataFrame,
            ),
            pd.DataFrame,
        )
        check(
            assert_type(
                pd.read_excel(path, parse_dates=["col1"], date_format="%Y-%m-%d"),
                pd.DataFrame,
            ),
            pd.DataFrame,
        )
