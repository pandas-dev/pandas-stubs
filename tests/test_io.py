from pathlib import Path
from typing import Any

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    read_clipboard,
    read_pickle,
    read_stata,
)
from pandas._testing import ensure_clean
import pytest
from typing_extensions import assert_type

from tests import check

from pandas.io.api import to_pickle
from pandas.io.clipboard import PyperclipException
from pandas.io.parsers import TextFileReader
from pandas.io.stata import StataReader

DF = DataFrame({"a": [1, 2, 3], "b": [0.0, 0.0, 0.0]})


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
