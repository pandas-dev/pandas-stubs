import io

from pandas import (
    DataFrame,
    read_xml,
)
from pandas._testing import ensure_clean
from typing_extensions import assert_type

from tests import check

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
