from pathlib import Path

from packaging.version import parse
from pandas import (
    DataFrame,
    __version__,
    read_orc,
)
from pandas._testing import ensure_clean
import pytest
from typing_extensions import assert_type

from tests import check

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
