from collections import defaultdict
from collections.abc import Generator
import csv
from functools import partial
import io
import os
from pathlib import Path
import sqlite3
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)
import uuid

import numpy as np
from openpyxl.workbook.workbook import Workbook as OpenXlWorkbook
import pandas as pd
from pandas import (
    DataFrame,
    HDFStore,
    Series,
    errors,
    read_clipboard,
    read_csv,
    read_feather,
    read_fwf,
    read_hdf,
    read_html,
    read_json,
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
from pandas import read_excel  # pyright: ignore[reportUnknownVariableType]
from pandas import read_orc  # pyright: ignore[reportUnknownVariableType]
from pandas.api.typing import JsonReader
import pytest
import sqlalchemy
import sqlalchemy.orm
import sqlalchemy.orm.decl_api
from typing_extensions import assert_type

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    WINDOWS,
    check,
)

from pandas.io.parsers import TextFileReader
from pandas.io.pytables import (
    TableIterator,
    Term,
)
from pandas.io.sas.sas7bdat import SAS7BDATReader
from pandas.io.sas.sas_xport import XportReader
from pandas.io.stata import StataReader

from odf.opendocument import (  # pyright: ignore[reportMissingTypeStubs] # isort: skip
    OpenDocument,  # pyright: ignore[reportUnknownVariableType]
)
from xlsxwriter.workbook import (  # pyright: ignore[reportMissingTypeStubs] # isort: skip
    Workbook as XlsxWorkbook,  # pyright: ignore[reportUnknownVariableType]
)


DF = DataFrame({"a": [1, 2, 3], "b": [0.0, 0.0, 0.0]})
CWD = Path(__file__).parent.resolve()


def test_orc(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    check(assert_type(DF.to_orc(path_str), None), type(None))
    check(assert_type(read_orc(path_str), DataFrame), DataFrame)


def test_orc_path(tmp_path: Path) -> None:
    path = tmp_path / str(uuid.uuid4())
    check(assert_type(DF.to_orc(path), None), type(None))
    check(assert_type(read_orc(path), DataFrame), DataFrame)


def test_orc_buffer(tmp_path: Path) -> None:
    path = tmp_path / str(uuid.uuid4())
    with path.open("wb") as file_w:
        check(assert_type(DF.to_orc(file_w), None), type(None))

    with path.open("rb") as file_r:
        check(assert_type(read_orc(file_r), DataFrame), DataFrame)


def test_orc_columns(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    check(assert_type(DF.to_orc(path_str, index=False), None), type(None))
    check(assert_type(read_orc(path_str, columns=["a"]), DataFrame), DataFrame)


def test_orc_bytes() -> None:
    check(assert_type(DF.to_orc(index=False), bytes), bytes)


def test_xml(tmp_path: Path) -> None:
    path = tmp_path / str(uuid.uuid4())
    path_str = str(path)
    check(assert_type(DF.to_xml(path_str), None), type(None))
    check(assert_type(read_xml(path_str), DataFrame), DataFrame)
    with path.open("rb") as f:
        check(assert_type(read_xml(f), DataFrame), DataFrame)


def test_xml_str() -> None:
    out = check(assert_type(DF.to_xml(), str), str)
    check(assert_type(read_xml(io.StringIO(out)), DataFrame), DataFrame)


def test_pickle(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    check(assert_type(DF.to_pickle(path_str), None), type(None))
    check(assert_type(read_pickle(path_str), Any), DataFrame)


def test_pickle_file_handle(tmp_path: Path) -> None:
    path = tmp_path / str(uuid.uuid4())
    check(assert_type(DF.to_pickle(str(path)), None), type(None))
    with path.open("rb") as file:
        check(assert_type(read_pickle(file), Any), DataFrame)


def test_pickle_path(tmp_path: Path) -> None:
    path = tmp_path / str(uuid.uuid4())
    path_str = str(path)
    check(assert_type(DF.to_pickle(path_str), None), type(None))
    check(assert_type(read_pickle(path), Any), DataFrame)


def test_pickle_protocol(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    DF.to_pickle(path_str, protocol=3)
    check(assert_type(read_pickle(path_str), Any), DataFrame)


def test_pickle_compression(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    DF.to_pickle(path_str, compression="gzip")
    check(assert_type(read_pickle(path_str, compression="gzip"), Any), DataFrame)

    check(assert_type(read_pickle(path_str, compression="gzip"), Any), DataFrame)


def test_pickle_storage_options(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    DF.to_pickle(path_str, storage_options={})

    check(assert_type(read_pickle(path_str, storage_options={}), Any), DataFrame)


def test_to_pickle_series(tmp_path: Path) -> None:
    s: Series = DF["a"]
    path_str = str(tmp_path / str(uuid.uuid4()))
    check(assert_type(s.to_pickle(path_str), None), type(None))
    check(assert_type(read_pickle(path_str), Any), Series)


def test_read_stata_df(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    DF.to_stata(path_str)
    check(assert_type(read_stata(path_str), pd.DataFrame), pd.DataFrame)


def test_read_stata_iterator(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    DF.to_stata(path_str)
    check(assert_type(read_stata(path_str, iterator=True), StataReader), StataReader)
    reader = read_stata(path_str, chunksize=1)
    check(assert_type(reader, StataReader), StataReader)


def _true_if_b(s: str) -> bool:
    return s == "b"


def _true_if_greater_than_0(i: int) -> bool:
    return i > 0


def _true_if_first_param_is_head(t: tuple[str, int]) -> bool:
    return t[0] == "head"


def test_clipboard() -> None:
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
    # Passing kwargs for to_csv
    DF.to_clipboard(quoting=csv.QUOTE_ALL)
    DF.to_clipboard(sep=",", index=False)
    if TYPE_CHECKING_INVALID_USAGE:
        pd.read_clipboard(names="abcd")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType]
        pd.read_clipboard(usecols="abcd")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType]


def test_clipboard_iterator() -> None:
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
    path = Path(CWD, "data", "airline.sas7bdat")
    check(assert_type(read_sas(path), DataFrame), DataFrame)
    with check(
        assert_type(read_sas(path, iterator=True), SAS7BDATReader | XportReader),
        SAS7BDATReader,
    ):
        pass
    with check(
        assert_type(read_sas(path, iterator=True, format="sas7bdat"), SAS7BDATReader),
        SAS7BDATReader,
    ):
        pass
    with check(
        assert_type(read_sas(path, chunksize=1), SAS7BDATReader | XportReader),
        SAS7BDATReader,
    ):
        pass
    with check(
        assert_type(read_sas(path, chunksize=1, format="sas7bdat"), SAS7BDATReader),
        SAS7BDATReader,
    ):
        pass


def test_sas_xport() -> None:
    path = Path(CWD, "data", "SSHSV1_A.xpt")
    check(assert_type(read_sas(path), DataFrame), DataFrame)
    with check(
        assert_type(read_sas(path, iterator=True), SAS7BDATReader | XportReader),
        XportReader,
    ):
        pass
    with check(
        assert_type(read_sas(path, iterator=True, format="xport"), XportReader),
        XportReader,
    ):
        pass
    with check(
        assert_type(read_sas(path, chunksize=1), SAS7BDATReader | XportReader),
        XportReader,
    ):
        pass
    with check(
        assert_type(read_sas(path, chunksize=1, format="xport"), XportReader),
        XportReader,
    ):
        pass


MESSAGE_PYTABLE_314 = (
    "PyTables does not support Python 3.14+ yet, see PyTables/PyTables#1261"
)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="PyTables requires Python 3.11+")
@pytest.mark.skipif(sys.version_info >= (3, 14), reason=MESSAGE_PYTABLE_314)
def test_hdf(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    check(assert_type(DF.to_hdf(path_str, key="df"), None), type(None))
    check(assert_type(read_hdf(path_str), DataFrame | Series), DataFrame)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="PyTables requires Python 3.11+")
@pytest.mark.skipif(sys.version_info >= (3, 14), reason=MESSAGE_PYTABLE_314)
def test_hdfstore(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    store = HDFStore(path_str, model="w")
    check(assert_type(store, HDFStore), HDFStore)
    check(assert_type(store.put("df", DF, "table"), None), type(None))
    check(assert_type(store.append("df2", DF, "table"), None), type(None))
    check(assert_type(store.keys(), list[str]), list)
    check(assert_type(store.info(), str), str)
    check(
        assert_type(store.select("df", start=0, stop=1), DataFrame | Series),
        DataFrame,
    )
    check(
        assert_type(store.select("df", where="index>=1"), DataFrame | Series),
        DataFrame,
    )
    check(
        assert_type(store.select("df", where=Term("index>=1")), DataFrame | Series),
        DataFrame,
    )
    check(
        assert_type(store.select("df", where=[Term("index>=1")]), DataFrame | Series),
        DataFrame,
    )
    check(assert_type(store.get("df"), DataFrame | Series), DataFrame)
    for key in store:
        check(assert_type(key, str), str)
    check(assert_type(store.close(), None), type(None))

    store = HDFStore(path_str, model="r")
    check(assert_type(read_hdf(store, "df"), DataFrame | Series), DataFrame)
    store.close()


@pytest.mark.skipif(sys.version_info < (3, 11), reason="PyTables requires Python 3.11+")
@pytest.mark.skipif(sys.version_info >= (3, 14), reason=MESSAGE_PYTABLE_314)
def test_read_hdf_iterator(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    check(assert_type(DF.to_hdf(path_str, key="df", format="table"), None), type(None))
    ti = read_hdf(path_str, chunksize=1)
    check(assert_type(ti, TableIterator), TableIterator)
    ti.close()

    ti = read_hdf(path_str, "df", iterator=True)
    check(assert_type(ti, TableIterator), TableIterator)
    for _ in ti:
        pass
    ti.close()


@pytest.mark.skipif(sys.version_info < (3, 11), reason="PyTables requires Python 3.11+")
@pytest.mark.skipif(sys.version_info >= (3, 14), reason=MESSAGE_PYTABLE_314)
def test_hdf_context_manager(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    check(assert_type(DF.to_hdf(path_str, key="df", format="table"), None), type(None))
    with HDFStore(path_str, mode="r") as store:
        check(assert_type(store.is_open, bool), bool)
        check(assert_type(store.get("df"), DataFrame | Series), DataFrame)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="PyTables requires Python 3.11+")
@pytest.mark.skipif(sys.version_info >= (3, 14), reason=MESSAGE_PYTABLE_314)
def test_hdf_series(tmp_path: Path) -> None:
    s = DF["a"]
    path_str = str(tmp_path / str(uuid.uuid4()))
    check(assert_type(s.to_hdf(path_str, key="s"), None), type(None))
    check(assert_type(read_hdf(path_str, "s"), DataFrame | Series), Series)


def test_spss() -> None:
    path_str = Path(CWD, "data", "labelled-num.sav")
    check(
        assert_type(read_spss(path_str, convert_categoricals=True), DataFrame),
        DataFrame,
    )
    check(
        assert_type(read_spss(str(path_str), usecols=["VAR00002"]), DataFrame),
        DataFrame,
    )


def test_json(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    check(assert_type(DF.to_json(path_str), None), type(None))
    check(assert_type(read_json(path_str), DataFrame), DataFrame)
    json_str = DF.to_json()
    check(assert_type(json_str, str), str)
    bin_json = io.StringIO(json_str)
    check(assert_type(read_json(bin_json), DataFrame), DataFrame)


def test_json_dataframe_bytes() -> None:
    """Test DataFrame.to_json with bytesIO buffer."""
    buffer = io.BytesIO()
    df = pd.DataFrame()
    check(assert_type(df.to_json(buffer, compression="gzip"), None), type(None))
    check(assert_type(df.to_json(buffer), None), type(None))


def test_json_series_bytes() -> None:
    """Test Series.to_json with bytesIO buffer."""
    buffer = io.BytesIO()
    sr = pd.Series()
    check(assert_type(sr.to_json(buffer, compression="gzip"), None), type(None))
    check(assert_type(sr.to_json(buffer), None), type(None))


def test_json_series(tmp_path: Path) -> None:
    s = DF["a"]
    path_str = str(tmp_path / str(uuid.uuid4()))
    check(assert_type(s.to_json(path_str), None), type(None))
    check(assert_type(read_json(path_str, typ="series"), Series), Series)
    check(assert_type(DF.to_json(), str), str)
    check(
        assert_type(
            read_json(io.StringIO(s.to_json(orient=None)), typ="series", orient=None),
            Series,
        ),
        Series,
    )
    check(
        assert_type(
            read_json(
                io.StringIO(s.to_json(orient="split")), typ="series", orient="split"
            ),
            Series,
        ),
        Series,
    )
    check(
        assert_type(
            read_json(
                io.StringIO(s.to_json(orient="records")), typ="series", orient="records"
            ),
            Series,
        ),
        Series,
    )
    check(
        assert_type(
            read_json(
                io.StringIO(s.to_json(orient="index")), typ="series", orient="index"
            ),
            Series,
        ),
        Series,
    )
    check(
        assert_type(
            read_json(
                io.StringIO(s.to_json(orient="table")), typ="series", orient="table"
            ),
            Series,
        ),
        Series,
    )


def test_json_chunk(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    check(assert_type(DF.to_json(path_str), None), type(None))
    json_reader = read_json(path_str, chunksize=1, lines=True)
    check(assert_type(json_reader, "JsonReader[DataFrame]"), JsonReader)
    for sub_df in json_reader:
        check(assert_type(sub_df, DataFrame), DataFrame)
    check(assert_type(DF.to_json(), str), str)


def test_parquet(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    check(assert_type(DF.to_parquet(path_str), None), type(None))
    check(assert_type(DF.to_parquet(), bytes), bytes)
    check(assert_type(read_parquet(path_str), DataFrame), DataFrame)


def test_parquet_options(tmp_path: Path) -> None:
    path_str = str(tmp_path / f"{uuid.uuid4()}test.parquet")
    check(
        assert_type(DF.to_parquet(path_str, compression=None, index=True), None),
        type(None),
    )
    check(assert_type(read_parquet(path_str), DataFrame), DataFrame)
    check(assert_type(read_parquet(path_str), DataFrame), DataFrame)

    sel = [("a", ">", 2)]
    check(assert_type(read_parquet(path_str, filters=sel), DataFrame), DataFrame)


def test_feather(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    check(assert_type(DF.to_feather(path_str), None), type(None))
    check(assert_type(read_feather(path_str), DataFrame), DataFrame)
    check(assert_type(read_feather(path_str, columns=["a"]), DataFrame), DataFrame)
    with io.BytesIO() as bio:
        check(assert_type(DF.to_feather(bio), None), type(None))
        bio.seek(0)
        check(assert_type(read_feather(bio), DataFrame), DataFrame)


def test_read_csv(tmp_path: Path) -> None:
    path = tmp_path / str(uuid.uuid4())
    path_str = str(path)
    check(assert_type(DF.to_csv(path_str), None), type(None))
    check(assert_type(read_csv(path_str), DataFrame), DataFrame)
    with path.open() as csv_file:
        check(assert_type(read_csv(csv_file), DataFrame), DataFrame)
    with path.open() as csv_file:
        sio = io.StringIO(csv_file.read())
        check(assert_type(read_csv(sio), DataFrame), DataFrame)
    check(assert_type(read_csv(path_str, iterator=False), DataFrame), DataFrame)
    check(assert_type(read_csv(path_str, chunksize=None), DataFrame), DataFrame)
    check(
        assert_type(read_csv(path_str, dtype=defaultdict(lambda: "f8")), DataFrame),
        DataFrame,
    )

    def cols(x: str) -> bool:
        return x in ["a", "b"]

    pd.read_csv(path_str, usecols=cols)


def test_read_csv_iterator(tmp_path: Path) -> None:
    path = tmp_path / str(uuid.uuid4())
    path_str = str(path)
    check(assert_type(DF.to_csv(path_str), None), type(None))
    tfr = read_csv(path_str, iterator=True)
    check(assert_type(tfr, TextFileReader), TextFileReader)
    tfr.close()
    tfr2 = read_csv(path, chunksize=1)
    check(assert_type(tfr2, TextFileReader), TextFileReader)
    tfr2.close()


def _true_if_col1(s: str) -> bool:
    return s == "col1"


def test_types_read_csv_num(tmp_path: Path) -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    check(assert_type(df.to_csv(), str), str)

    path_str = str(tmp_path / str(uuid.uuid4()))
    df.to_csv(path_str)
    check(assert_type(pd.read_csv(path_str), pd.DataFrame), pd.DataFrame)
    check(assert_type(pd.read_csv(path_str, sep="a"), pd.DataFrame), pd.DataFrame)
    check(assert_type(pd.read_csv(path_str, header=None), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(
            pd.read_csv(
                path_str,
                engine="python",
                true_values=["no", "No", "NO"],
                na_filter=False,
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.read_csv(
                path_str,
                skiprows=lambda x: x in [0, 2],
                skip_blank_lines=True,
                dayfirst=False,
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(assert_type(pd.read_csv(path_str, nrows=2), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(pd.read_csv(path_str, dtype={"a": float, "b": int}), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(pd.read_csv(path_str, usecols=["col1"]), pd.DataFrame), pd.DataFrame
    )
    check(assert_type(pd.read_csv(path_str, usecols=[0]), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(pd.read_csv(path_str, usecols=np.array([0])), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(pd.read_csv(path_str, usecols=("col1",)), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.read_csv(path_str, usecols=pd.Series(data=["col1"])), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(pd.read_csv(path_str, converters=None), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(
            pd.read_csv(path_str, names=("first", "second"), header=0), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(pd.read_csv(path_str, names=range(2), header=0), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(pd.read_csv(path_str, names=(1, "two"), header=0), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.read_csv(path_str, names=(("first", 1), ("last", 2)), header=0),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(assert_type(pd.read_csv(path_str, usecols=None), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(pd.read_csv(path_str, usecols=["col1"]), pd.DataFrame), pd.DataFrame
    )
    check(assert_type(pd.read_csv(path_str, usecols=(0,)), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(pd.read_csv(path_str, usecols=range(1)), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(pd.read_csv(path_str, usecols=_true_if_col1), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.read_csv(
                path_str,
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
            pd.read_csv(
                path_str,
                names=(("head", 1), ("tail", 2)),
                usecols=_true_if_first_param_is_head,
                header=0,
                index_col=0,
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )

    if TYPE_CHECKING_INVALID_USAGE:
        pd.read_csv(path_str, names="abcd")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType]
        pd.read_csv(path_str, usecols="abcd")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType]

        def cols2(x: set[float]) -> bool:
            return sum(x) < 1.0

        pd.read_csv("file.csv", usecols=cols2)  # type: ignore[type-var] # pyright: ignore[reportArgumentType]

    tfr1 = pd.read_csv(path_str, nrows=2, iterator=True, chunksize=3)
    check(assert_type(tfr1, TextFileReader), TextFileReader)
    tfr1.close()

    tfr2 = pd.read_csv(path_str, nrows=2, chunksize=1)
    check(assert_type(tfr2, TextFileReader), TextFileReader)
    tfr2.close()

    tfr3 = pd.read_csv(path_str, nrows=2, iterator=False, chunksize=1)
    check(assert_type(tfr3, TextFileReader), TextFileReader)
    tfr3.close()

    tfr4 = pd.read_csv(path_str, nrows=2, iterator=True)
    check(assert_type(tfr4, TextFileReader), TextFileReader)
    tfr4.close()


def test_types_read_csv_date(tmp_path: Path) -> None:
    df_dates = pd.DataFrame(data={"col1": ["2023-03-15", "2023-04-20"]})

    path_str = str(tmp_path / str(uuid.uuid4()))
    df_dates.to_csv(path_str)

    check(
        assert_type(
            pd.read_csv(path_str, parse_dates=["col1"], date_format="%Y-%m-%d"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.read_csv(
                path_str, parse_dates=["col1"], date_format={"col1": "%Y-%m-%d"}
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.read_csv(path_str, parse_dates=["col1"], date_format={1: "%Y-%m-%d"}),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )


def test_read_table(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    check(assert_type(DF.to_csv(path_str, sep="\t"), None), type(None))
    check(assert_type(read_table(path_str), DataFrame), DataFrame)
    check(assert_type(read_table(path_str, iterator=False), DataFrame), DataFrame)
    check(assert_type(read_table(path_str, chunksize=None), DataFrame), DataFrame)
    check(
        assert_type(read_table(path_str, dtype=defaultdict(lambda: "f8")), DataFrame),
        DataFrame,
    )
    check(
        assert_type(
            read_table(path_str, names=("first", "second"), header=0), DataFrame
        ),
        DataFrame,
    )
    check(
        assert_type(read_table(path_str, names=range(2), header=0), DataFrame),
        DataFrame,
    )
    check(
        assert_type(read_table(path_str, names=(1, "two"), header=0), DataFrame),
        DataFrame,
    )
    check(
        assert_type(
            read_table(path_str, names=(("first", 1), ("last", 2)), header=0), DataFrame
        ),
        DataFrame,
    )
    check(
        assert_type(read_table(path_str, usecols=None), DataFrame),
        DataFrame,
    )
    check(assert_type(read_table(path_str, usecols=["a"]), DataFrame), DataFrame)
    check(assert_type(read_table(path_str, usecols=(0,)), DataFrame), DataFrame)
    check(assert_type(read_table(path_str, usecols=range(1)), DataFrame), DataFrame)
    check(assert_type(read_table(path_str, usecols=_true_if_b), DataFrame), DataFrame)
    check(
        assert_type(
            read_table(
                path_str,
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
                path_str,
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
        pd.read_table(path_str, names="abcd")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType]
        pd.read_table(path_str, usecols="abcd")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType]


def test_read_table_iterator(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    check(assert_type(DF.to_csv(path_str, sep="\t"), None), type(None))
    tfr = read_table(path_str, iterator=True)
    check(assert_type(tfr, TextFileReader), TextFileReader)
    tfr.close()
    tfr2 = read_table(path_str, chunksize=1)
    check(assert_type(tfr2, TextFileReader), TextFileReader)
    tfr2.close()


def test_types_read_table(tmp_path: Path) -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})

    path_str = str(tmp_path / str(uuid.uuid4()))
    df.to_csv(path_str)
    check(
        assert_type(pd.read_table(path_str, sep=",", converters=None), pd.DataFrame),
        pd.DataFrame,
    )


def test_btest_read_fwf(tmp_path: Path) -> None:
    path = tmp_path / str(uuid.uuid4())
    path_str = str(path)
    DF.to_string(path_str, index=False)
    check(assert_type(read_fwf(path_str), DataFrame), DataFrame)
    check(assert_type(read_fwf(path), DataFrame), DataFrame)

    with path.open() as fwf_file:
        check(
            assert_type(read_fwf(fwf_file), DataFrame),
            DataFrame,
        )
    with path.open() as fwf_file:
        sio = io.StringIO(fwf_file.read())
        check(assert_type(read_fwf(sio), DataFrame), DataFrame)
    with path.open("rb") as fwf_file:
        bio = io.BytesIO(fwf_file.read())
        check(assert_type(read_fwf(bio), DataFrame), DataFrame)
    with read_fwf(path_str, iterator=True) as fwf_iterator:
        check(assert_type(fwf_iterator, TextFileReader), TextFileReader)
    with read_fwf(path_str, chunksize=1) as fwf_iterator2:
        check(assert_type(fwf_iterator2, TextFileReader), TextFileReader)


def test_text_file_reader(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    DF.to_string(path_str, index=False)
    tfr = TextFileReader(path_str, engine="python")
    check(assert_type(tfr, TextFileReader), TextFileReader)
    check(assert_type(tfr.read(1), DataFrame), DataFrame)
    check(assert_type(tfr.close(), None), type(None))
    with TextFileReader(path_str, engine="python") as tfr:
        check(assert_type(tfr.read(1), DataFrame), DataFrame)
    with TextFileReader(path_str, engine="python") as tfr:
        check(assert_type(tfr.__next__(), DataFrame), DataFrame)
        df_iter: DataFrame
        for df_iter in tfr:
            check(df_iter, DataFrame)


def test_to_csv_series(tmp_path: Path) -> None:
    s = DF.iloc[:, 0]
    check(assert_type(s.to_csv(), str), str)
    path_str = str(tmp_path / str(uuid.uuid4()))
    check(assert_type(s.to_csv(path_str), None), type(None))


def test_read_excel(tmp_path: Path) -> None:
    path_str = str(tmp_path / f"{uuid.uuid4()}test.xlsx")
    # https://github.com/pandas-dev/pandas-stubs/pull/33
    check(
        assert_type(pd.DataFrame({"A": [1, 2, 3]}).to_excel(path_str), None), type(None)
    )
    check(assert_type(pd.read_excel(path_str), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(pd.read_excel(path_str, sheet_name="Sheet1"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.read_excel(path_str, sheet_name=["Sheet1"]), dict[str, pd.DataFrame]
        ),
        dict,
    )
    # GH 98
    check(
        assert_type(pd.read_excel(path_str, sheet_name=0), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(pd.read_excel(path_str, sheet_name=[0]), dict[int, pd.DataFrame]),
        dict,
    )
    check(
        assert_type(
            pd.read_excel(path_str, sheet_name=[0, "Sheet1"]),
            dict[int | str, pd.DataFrame],
        ),
        dict,
    )
    # GH 641
    check(
        assert_type(pd.read_excel(path_str, sheet_name=None), dict[str, pd.DataFrame]),
        dict,
    )
    check(
        assert_type(pd.read_excel(path_str, names=("test",), header=0), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(pd.read_excel(path_str, names=(1,), header=0), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.read_excel(path_str, names=(("higher", "lower"),), header=0),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(pd.read_excel(path_str, names=range(1), header=0), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(pd.read_excel(path_str, usecols=None), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(pd.read_excel(path_str, usecols=["A"]), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(pd.read_excel(path_str, usecols=(0,)), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(pd.read_excel(path_str, usecols=range(1)), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(pd.read_excel(path_str, usecols=_true_if_b), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.read_excel(
                path_str,
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
                path_str,
                names=(("head", 1), ("tail", 2)),
                usecols=_true_if_first_param_is_head,
                header=0,
                index_col=0,
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(assert_type(pd.read_excel(path_str, usecols="A"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(pd.read_excel(path_str, engine="calamine"), pd.DataFrame),
        pd.DataFrame,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        pd.read_excel(path_str, names="abcd")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType]


def test_read_excel_io_types(tmp_path: Path) -> None:
    # GH 195
    df = pd.DataFrame([[1, 2], [8, 9]], columns=["A", "B"])
    as_str = check(assert_type(str(tmp_path / f"{uuid.uuid4()}test.xlsx"), str), str)
    df.to_excel(as_str)

    check(assert_type(pd.read_excel(as_str), pd.DataFrame), pd.DataFrame)

    as_path = Path(as_str)
    check(assert_type(pd.read_excel(as_path), pd.DataFrame), pd.DataFrame)

    with as_path.open("rb") as as_file:
        check(assert_type(pd.read_excel(as_file), pd.DataFrame), pd.DataFrame)


def test_read_excel_basic(tmp_path: Path) -> None:
    path_str = str(tmp_path / f"{uuid.uuid4()}test.xlsx")
    check(assert_type(DF.to_excel(path_str), None), type(None))
    check(assert_type(read_excel(path_str), DataFrame), DataFrame)
    check(assert_type(read_excel(path_str, sheet_name="Sheet1"), DataFrame), DataFrame)
    check(assert_type(read_excel(path_str, sheet_name=0), DataFrame), DataFrame)


def test_read_excel_list(tmp_path: Path) -> None:
    path_str = str(tmp_path / f"{uuid.uuid4()}test.xlsx")
    check(assert_type(DF.to_excel(path_str), None), type(None))
    check(
        assert_type(read_excel(path_str, sheet_name=["Sheet1"]), dict[str, DataFrame]),
        dict,
    )
    check(assert_type(read_excel(path_str, sheet_name=[0]), dict[int, DataFrame]), dict)


def test_read_excel_dtypes(tmp_path: Path) -> None:
    # GH 440
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [10.0, 20.0, 30.3]})
    path_str = str(tmp_path / f"{uuid.uuid4()}test.xlsx")
    check(assert_type(df.to_excel(path_str), None), type(None))
    dtypes = {"a": np.int64, "b": str, "c": np.float64}
    check(assert_type(read_excel(path_str, dtype=dtypes), pd.DataFrame), pd.DataFrame)


def test_excel_reader(tmp_path: Path) -> None:
    path_str = str(tmp_path / f"{uuid.uuid4()}test.xlsx")
    check(assert_type(DF.to_excel(path_str), None), type(None))
    with pd.ExcelFile(path_str, engine="calamine") as ef:
        check(assert_type(ef, pd.ExcelFile), pd.ExcelFile)
        check(assert_type(pd.read_excel(ef), pd.DataFrame), pd.DataFrame)

    with pd.ExcelFile(
        path_or_buffer=path_str, engine="openpyxl", engine_kwargs={"data_only": True}
    ) as ef:
        check(assert_type(ef, pd.ExcelFile), pd.ExcelFile)
        check(assert_type(pd.read_excel(ef), pd.DataFrame), pd.DataFrame)


def test_excel_fspath(tmp_path: Path) -> None:
    """Test ExcelFile.__fspath__ type."""
    path_str = str(tmp_path / f"{uuid.uuid4()}test.xlsx")
    check(assert_type(DF.to_excel(path_str), None), type(None))
    with pd.ExcelFile(
        path_or_buffer=path_str, engine="openpyxl", engine_kwargs={"data_only": True}
    ) as ef:
        check(assert_type(os.fspath(ef), str), str)


def test_excel_writer(tmp_path: Path) -> None:
    path_str = str(tmp_path / f"{uuid.uuid4()}test.xlsx")
    with pd.ExcelWriter(path_str) as ew:  # pyright: ignore[reportUnknownVariableType]
        check(
            assert_type(
                ew, pd.ExcelWriter  # pyright: ignore[reportUnknownArgumentType]
            ),
            pd.ExcelWriter,
        )
        DF.to_excel(ew, sheet_name="A")
    check(assert_type(read_excel(path_str, sheet_name="A"), DataFrame), DataFrame)
    check(assert_type(read_excel(path_str), DataFrame), DataFrame)
    ef = pd.ExcelFile(path_str)
    check(assert_type(ef, pd.ExcelFile), pd.ExcelFile)
    check(assert_type(read_excel(ef, sheet_name="A"), DataFrame), DataFrame)
    check(assert_type(read_excel(ef), DataFrame), DataFrame)
    check(assert_type(ef.parse(sheet_name=0), DataFrame), DataFrame)
    check(
        assert_type(ef.parse(sheet_name=[0]), dict[str | int, DataFrame]),
        dict,
    )
    check(assert_type(ef.close(), None), type(None))


def test_excel_writer_io() -> None:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:  # pyright: ignore[reportUnknownVariableType]
        DF.to_excel(writer, sheet_name="A")

    ef = pd.ExcelFile(buffer)
    check(assert_type(ef, pd.ExcelFile), pd.ExcelFile)
    check(assert_type(read_excel(ef, sheet_name="A"), DataFrame), DataFrame)


def test_excel_writer_engine(tmp_path: Path) -> None:
    path_str = str(tmp_path / f"{uuid.uuid4()}test0.xlsx")
    with pd.ExcelWriter(
        path_str, engine="auto"
    ) as ew:  # pyright: ignore[reportUnknownVariableType]
        check(
            assert_type(
                ew,
                pd.ExcelWriter,  # pyright: ignore[reportUnknownArgumentType]
            ),
            pd.ExcelWriter,
        )
        DF.to_excel(ew, sheet_name="A")

    path_str = str(tmp_path / f"{uuid.uuid4()}test1.xlsx")
    with pd.ExcelWriter(path_str, engine="openpyxl") as ew:
        check(
            assert_type(ew, pd.ExcelWriter[OpenXlWorkbook]),
            pd.ExcelWriter[OpenXlWorkbook],
        )
        DF.to_excel(ew, sheet_name="A")
        check(
            assert_type(ew.engine, Literal["openpyxl", "odf", "xlsxwriter"]),
            str,
        )

    path_str = str(tmp_path / f"{uuid.uuid4()}test2.ods")
    with pd.ExcelWriter(
        path_str, engine="odf"
    ) as ew:  # pyright: ignore[reportUnknownVariableType]
        check(
            assert_type(
                ew,
                pd.ExcelWriter[
                    OpenDocument
                ],  # pyright: ignore[reportUnknownArgumentType]
            ),
            pd.ExcelWriter[OpenDocument],
        )
        DF.to_excel(ew, sheet_name="A")
        check(
            assert_type(ew.engine, Literal["openpyxl", "odf", "xlsxwriter"]),
            str,
        )

    path_str = str(tmp_path / f"{uuid.uuid4()}test3.xlsx")
    with pd.ExcelWriter(
        path_str, engine="xlsxwriter"
    ) as ew:  # pyright: ignore[reportUnknownVariableType]
        check(
            assert_type(
                ew, pd.ExcelWriter[XlsxWorkbook]
            ),  # pyright: ignore[reportUnknownArgumentType]
            pd.ExcelWriter[XlsxWorkbook],
        )
        DF.to_excel(ew, sheet_name="A")
        check(
            assert_type(ew.engine, Literal["openpyxl", "odf", "xlsxwriter"]),
            str,
        )


def test_excel_writer_append_mode(tmp_path: Path) -> None:
    path_str = str(tmp_path / f"{uuid.uuid4()}test.xlsx")
    with pd.ExcelWriter(
        path_str, mode="w"
    ) as ew:  # pyright: ignore[reportUnknownVariableType]
        DF.to_excel(ew, sheet_name="A")
    with pd.ExcelWriter(path_str, mode="a", engine="openpyxl") as ew:
        DF.to_excel(ew, sheet_name="B")


def test_to_string(tmp_path: Path) -> None:
    check(assert_type(DF.to_string(), str), str)
    path = tmp_path / str(uuid.uuid4())
    path_str = str(path)
    check(assert_type(DF.to_string(path_str), None), type(None))
    check(assert_type(DF.to_string(path), None), type(None))
    with path.open("w") as df_string:
        check(assert_type(DF.to_string(df_string), None), type(None))
    sio = io.StringIO()
    check(assert_type(DF.to_string(sio), None), type(None))


def test_read_sql(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    con = sqlite3.connect(path_str)
    check(assert_type(DF.to_sql("test", con=con), int | None), int)
    check(assert_type(read_sql("select * from test", con=con), DataFrame), DataFrame)
    con.close()


def test_read_sql_via_sqlalchemy_connection(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    db_uri = "sqlite:///" + path_str
    engine = sqlalchemy.create_engine(db_uri)

    with engine.connect() as conn:
        check(assert_type(DF.to_sql("test", con=conn), int | None), int)
        check(
            assert_type(read_sql("select * from test", con=conn), DataFrame),
            DataFrame,
        )
    engine.dispose()


def test_read_sql_via_sqlalchemy_engine(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    db_uri = "sqlite:///" + path_str
    engine = sqlalchemy.create_engine(db_uri)

    check(assert_type(DF.to_sql("test", con=engine), int | None), int)
    check(
        assert_type(read_sql("select * from test", con=engine), DataFrame),
        DataFrame,
    )
    engine.dispose()


def test_read_sql_via_sqlalchemy_engine_with_params(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    db_uri = "sqlite:///" + path_str
    engine = sqlalchemy.create_engine(db_uri)

    check(assert_type(DF.to_sql("test", con=engine), int | None), int)
    check(
        assert_type(
            read_sql(
                "select * from test where a = :a and b = :b",
                con=engine,
                params={"a": 2, "b": 0.0},
            ),
            DataFrame,
        ),
        DataFrame,
    )
    engine.dispose()


def test_read_sql_generator(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    con = sqlite3.connect(path_str)
    check(assert_type(DF.to_sql("test", con=con), int | None), int)

    check(
        assert_type(
            read_sql("select * from test", con=con, chunksize=1),
            Generator[DataFrame, None, None],
        ),
        Generator,
    )
    con.close()


def test_read_sql_table(tmp_path: Path) -> None:
    # sqlite3 doesn't support read_table, which is required for this function
    # Could only run in pytest if SQLAlchemy was installed
    path_str = str(tmp_path / str(uuid.uuid4()))
    con = sqlite3.connect(path_str)
    check(assert_type(DF.to_sql("test", con=con), int | None), int)
    if TYPE_CHECKING:
        assert_type(read_sql_table("test", con=con), DataFrame)
        assert_type(
            read_sql_table("test", con=con, chunksize=1),
            Generator[DataFrame, None, None],
        )
    con.close()


def test_read_sql_query(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    con = sqlite3.connect(path_str)
    check(assert_type(DF.to_sql("test", con=con), int | None), int)
    check(
        assert_type(
            read_sql_query("select * from test", con=con, index_col="index"),
            DataFrame,
        ),
        DataFrame,
    )
    con.close()


def test_read_sql_query_generator(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    con = sqlite3.connect(path_str)
    check(assert_type(DF.to_sql("test", con=con), int | None), int)

    check(
        assert_type(
            read_sql_query("select * from test", con=con, chunksize=1),
            Generator[DataFrame, None, None],
        ),
        Generator,
    )
    con.close()


def test_read_sql_query_via_sqlalchemy_engine_with_params(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    db_uri = "sqlite:///" + path_str
    engine = sqlalchemy.create_engine(db_uri)

    check(assert_type(DF.to_sql("test", con=engine), int | None), int)
    check(
        assert_type(
            read_sql_query(
                "select * from test where a = :a and b = :b",
                con=engine,
                params={"a": 2, "b": 0.0},
            ),
            DataFrame,
        ),
        DataFrame,
    )
    engine.dispose()


@pytest.mark.skip(
    reason="Only works in Postgres (and MySQL, but with different query syntax)"
)
def test_read_sql_query_via_sqlalchemy_engine_with_tuple_valued_params() -> None:
    db_uri = "postgresql+psycopg2://postgres@localhost:5432/postgres"
    engine = sqlalchemy.create_engine(db_uri)

    check(
        assert_type(
            read_sql_query(
                "select * from test where a in %(a)s", con=engine, params={"a": (1, 2)}
            ),
            DataFrame,
        ),
        DataFrame,
    )
    check(
        assert_type(
            read_sql_query(
                "select * from test where a in %s", con=engine, params=((1, 2),)
            ),
            DataFrame,
        ),
        DataFrame,
    )
    engine.dispose()


def test_read_html(tmp_path: Path) -> None:
    check(assert_type(DF.to_html(), str), str)
    path_str = str(tmp_path / str(uuid.uuid4()))
    check(assert_type(DF.to_html(path_str), None), type(None))
    check(assert_type(read_html(path_str), list[DataFrame]), list)
    check(assert_type(read_html(path_str, flavor=None), list[DataFrame]), list)
    check(assert_type(read_html(path_str, flavor="bs4"), list[DataFrame]), list)
    check(assert_type(read_html(path_str, flavor=["bs4"]), list[DataFrame]), list)
    check(
        assert_type(read_html(path_str, flavor=["bs4", "lxml"]), list[DataFrame]), list
    )


def test_csv_quoting(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    check(assert_type(DF.to_csv(path_str, quoting=csv.QUOTE_ALL), None), type(None))
    check(assert_type(DF.to_csv(path_str, quoting=csv.QUOTE_NONE), None), type(None))
    check(
        assert_type(DF.to_csv(path_str, quoting=csv.QUOTE_NONNUMERIC), None), type(None)
    )
    check(assert_type(DF.to_csv(path_str, quoting=csv.QUOTE_MINIMAL), None), type(None))


def test_sqlalchemy_selectable(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    db_uri = "sqlite:///" + path_str
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
            pd.read_sql(session.query(Temp.quantity).statement, session.connection())


def test_sqlalchemy_text(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    db_uri = "sqlite:///" + path_str
    engine = sqlalchemy.create_engine(db_uri)
    sql_select = sqlalchemy.text("select * from test")
    with engine.connect() as conn:
        check(assert_type(DF.to_sql("test", con=conn), int | None), int)
        check(assert_type(read_sql(sql_select, con=conn), DataFrame), DataFrame)
    engine.dispose()


def test_read_sql_dtype(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    conn = sqlite3.connect(path_str)
    df = pd.DataFrame(
        data=[[0, "10/11/12"], [1, "12/11/10"]],
        columns=["int_column", "date_column"],
    )
    check(assert_type(df.to_sql("test_data", con=conn), int | None), int)
    check(
        assert_type(
            pd.read_sql(
                "SELECT int_column, date_column FROM test_data", con=conn, dtype=None
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
    check(assert_type(DF.to_sql("test", con=conn), int | None), int)

    check(
        assert_type(read_sql("select * from test", con=conn, dtype=int), pd.DataFrame),
        pd.DataFrame,
    )
    conn.close()


def test_read_sql_dtype_backend(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    conn2 = sqlite3.connect(path_str)
    check(assert_type(DF.to_sql("test", con=conn2), int | None), int)
    check(
        assert_type(
            read_sql("select * from test", con=conn2, dtype_backend="pyarrow"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            read_sql("select * from test", con=conn2, dtype_backend="numpy_nullable"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    conn2.close()


def test_all_read_without_lxml_dtype_backend(tmp_path: Path) -> None:
    path_str = str(tmp_path / f"{uuid.uuid4()}test0.csv")
    check(assert_type(DF.to_csv(path_str), None), type(None))
    s1 = read_csv(path_str, iterator=True, dtype_backend="pyarrow")
    check(assert_type(s1, TextFileReader), TextFileReader)
    s1.close()

    DF.to_string(path_str, index=False)
    check(
        assert_type(read_fwf(path_str, dtype_backend="pyarrow"), DataFrame), DataFrame
    )

    check(assert_type(DF.to_json(path_str), None), type(None))
    check(
        assert_type(read_json(path_str, dtype_backend="pyarrow"), DataFrame), DataFrame
    )
    check(
        assert_type(read_json(path_str, dtype={"MatchID": str}), DataFrame), DataFrame
    )

    path_str = str(tmp_path / str(uuid.uuid4()))
    con = sqlite3.connect(path_str)
    check(assert_type(DF.to_sql("test", con=con), int | None), int)
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
        check(assert_type(DF.to_orc(path_str), None), type(None))
        check(
            assert_type(read_orc(path_str, dtype_backend="numpy_nullable"), DataFrame),
            DataFrame,
        )
    check(assert_type(DF.to_feather(path_str), None), type(None))
    check(
        assert_type(read_feather(path_str, dtype_backend="pyarrow"), DataFrame),
        DataFrame,
    )

    path_str = str(tmp_path / f"{uuid.uuid4()}test.xlsx")
    as_str: str = path_str
    DF.to_excel(path_str)
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
        path_str = str(tmp_path / str(uuid.uuid4()))
        co1 = sqlite3.connect(path_str)
        assert_type(DF.to_sql("test", con=co1), int | None)
        assert_type(
            read_sql_table("test", con=co1, dtype_backend="numpy_nullable"),
            DataFrame,
        )
        co1.close()


def test_read_with_lxml_dtype_backend(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    check(assert_type(DF.to_html(path_str), None), type(None))
    check(
        assert_type(
            read_html(path_str, dtype_backend="numpy_nullable"), list[DataFrame]
        ),
        list,
    )

    check(assert_type(DF.to_xml(path_str), None), type(None))
    check(
        assert_type(read_xml(path_str, dtype_backend="pyarrow"), DataFrame), DataFrame
    )


def test_read_sql_dict_str_value_dtype(tmp_path: Path) -> None:
    # GH 676
    path_str = str(tmp_path / str(uuid.uuid4()))
    con = sqlite3.connect(path_str)
    check(assert_type(DF.to_sql("test", con), int | None), int)
    check(
        assert_type(
            read_sql_query(
                "select * from test", con=con, index_col="index", dtype={"a": "int"}
            ),
            DataFrame,
        ),
        DataFrame,
    )
    con.close()


def test_added_date_format(tmp_path: Path) -> None:
    path_str = str(tmp_path / str(uuid.uuid4()))
    df_dates = pd.DataFrame(data={"col1": ["2023-03-15", "2023-04-20"]})
    df_dates.to_csv(path_str)

    check(
        assert_type(
            pd.read_table(
                path_str, sep=",", parse_dates=["col1"], date_format="%Y-%m-%d"
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.read_table(
                path_str,
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
                path_str, sep=",", parse_dates=["col1"], date_format={0: "%Y-%m-%d"}
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )

    check(
        assert_type(pd.read_fwf(path_str, date_format="%Y-%m-%d"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.read_fwf(path_str, date_format={"col1": "%Y-%m-%d"}), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(pd.read_fwf(path_str, date_format={0: "%Y-%m-%d"}), pd.DataFrame),
        pd.DataFrame,
    )
    path_str = str(tmp_path / f"{uuid.uuid4()}test.xlsx")
    check(
        assert_type(
            pd.DataFrame(data={"col1": ["2023-03-15", "2023-04-20"]}).to_excel(
                path_str
            ),
            None,
        ),
        type(None),
    )
    check(
        assert_type(
            pd.read_excel(path_str, parse_dates=["col1"], date_format={0: "%Y-%m-%d"}),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.read_excel(
                path_str, parse_dates=["col1"], date_format={"col1": "%Y-%m-%d"}
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.read_excel(path_str, parse_dates=["col1"], date_format="%Y-%m-%d"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )


def test_read_excel_index_col(tmp_path: Path) -> None:
    path_str = str(tmp_path / f"{uuid.uuid4()}test.xlsx")
    pd.DataFrame(data={"foo": [1, 3], "bar": [2, 4]}).to_excel(path_str)

    check(
        assert_type(pd.read_excel(path_str, index_col="bar"), pd.DataFrame),
        pd.DataFrame,
    )


def test_read_json_engine() -> None:
    """Test the engine argument for `pd.read_json` introduced with pandas 2.0."""
    data = """{"index": {"0": 0, "1": 1},
       "a": {"0": 1, "1": null},
       "b": {"0": 2.5, "1": 4.5},
       "c": {"0": true, "1": false},
       "d": {"0": "a", "1": "b"},
       "e": {"0": 1577.2, "1": 1577.1}}"""
    check(
        assert_type(pd.read_json(io.StringIO(data), engine="ujson"), pd.DataFrame),
        pd.DataFrame,
    )

    data_lines = b"""{"col 1":"a","col 2":"b"}
    {"col 1":"c","col 2":"d"}"""
    dd = io.BytesIO(data_lines)
    check(
        assert_type(
            pd.read_json(dd, lines=True, engine="pyarrow"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )

    if TYPE_CHECKING_INVALID_USAGE:
        pd.read_json(dd, lines=False, engine="pyarrow")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType, reportCallIssue]
        pd.read_json(io.StringIO(data), engine="pyarrow")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType]
        pd.read_json(io.StringIO(data), lines=True, engine="pyarrow")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType, reportCallIssue]


def test_converters_partial(tmp_path: Path) -> None:
    df = pd.DataFrame({"field_1": ["2020-01-01", "not a date"]})
    partial_func = partial(pd.to_datetime, errors="coerce")

    path_str = str(tmp_path / f"{uuid.uuid4()}test.xlsx")
    check(assert_type(df.to_excel(path_str, index=False), None), type(None))

    result = pd.read_excel(path_str, converters={"field_1": partial_func})
    check(assert_type(result, pd.DataFrame), pd.DataFrame)
