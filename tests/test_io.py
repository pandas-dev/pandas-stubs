import io
import os.path
import pathlib
from pathlib import Path
from typing import Union

from packaging.version import parse
import pandas as pd
from pandas import (
    DataFrame,
    __version__,
    read_clipboard,
    read_orc,
    read_sas,
    read_stata,
    read_xml,
)
from pandas._testing import ensure_clean
import pytest
from typing_extensions import assert_type

from tests import check

from pandas.io.clipboard import PyperclipException
from pandas.io.parsers import TextFileReader
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


@pytest.mark.parametrize("file_name", ["airline.sas7bdat", "SSHSV1_A.xpt"])
def test_sas(file_name: str) -> None:
    path = pathlib.Path(CWD, "data", file_name)
    actual_type = SAS7BDATReader if file_name.endswith("bdat") else XportReader
    check(assert_type(read_sas(path), DataFrame), DataFrame)
    check(
        assert_type(read_sas(path, iterator=True), Union[SAS7BDATReader, XportReader]),
        actual_type,
    )
    check(
        assert_type(read_sas(path, chunksize=1), Union[SAS7BDATReader, XportReader]),
        actual_type,
    )
