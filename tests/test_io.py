import io
from typing import (
    List,
    Union,
)

import pandas as pd
from pandas import (
    DataFrame,
    HDFStore,
    Index,
    Series,
    read_clipboard,
    read_hdf,
    read_stata,
    read_xml,
)
from pandas._testing import ensure_clean
import pytest
from typing_extensions import assert_type

from tests import check

from pandas.io.clipboard import PyperclipException
from pandas.io.parsers import TextFileReader
from pandas.io.pytables import (
    TableIterator,
    Term,
)
from pandas.io.stata import StataReader

DF = DataFrame({"a": [1, 2, 3], "b": [0.0, 0.0, 0.0]})


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


def test_hdf():
    with ensure_clean() as path:
        check(assert_type(DF.to_hdf(path, "df"), None), type(None))
        check(assert_type(read_hdf(path), Union[DataFrame, Series, Index]), DataFrame)


def test_hdfstore():
    with ensure_clean() as path:
        store = HDFStore(path, model="w")
        check(assert_type(store, HDFStore), HDFStore)
        check(assert_type(store.put("df", DF, "table"), None), type(None))
        check(assert_type(store.append("df2", DF, "table"), None), type(None))
        check(assert_type(store.keys(), List[str]), list)
        check(assert_type(store.info(), str), str)
        check(
            assert_type(
                store.select("df", start=0, stop=1), Union[DataFrame, Series, Index]
            ),
            DataFrame,
        )
        check(
            assert_type(
                store.select("df", where="index>=1"), Union[DataFrame, Series, Index]
            ),
            DataFrame,
        )
        check(
            assert_type(
                store.select("df", where=Term("index>=1")),
                Union[DataFrame, Series, Index],
            ),
            DataFrame,
        )
        check(
            assert_type(
                store.select("df", where=[Term("index>=1")]),
                Union[DataFrame, Series, Index],
            ),
            DataFrame,
        )
        check(assert_type(store.get("df"), Union[DataFrame, Series, Index]), DataFrame)
        check(assert_type(store.close(), None), type(None))

        store = HDFStore(path, model="r")
        check(
            assert_type(read_hdf(store, "df"), Union[DataFrame, Series, Index]),
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


def test_hdf_context_manaeger():
    with ensure_clean() as path:
        check(assert_type(DF.to_hdf(path, "df", format="table"), None), type(None))
        with HDFStore(path, mode="r") as store:
            check(assert_type(store.is_open, bool), bool)
            check(
                assert_type(store.get("df"), Union[DataFrame, Series, Index]), DataFrame
            )
