from pathlib import Path

from packaging.version import parse
from pandas import (
    DataFrame,
    __version__,
    read_clipboard,
    read_orc,
)
from pandas._testing import ensure_clean
import pytest
from typing_extensions import assert_type

from tests import check

from pandas.io.clipboard import PyperclipException
from pandas.io.parsers import TextFileReader

DF = DataFrame({"a": [1, 2, 3], "b": [0.0, 0.0, 0.0]})

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
