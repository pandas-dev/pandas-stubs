from pathlib import Path
from typing import Any

from pandas import (
    DataFrame,
    read_pickle,
)
from pandas._testing import ensure_clean
from typing_extensions import assert_type

from tests import check

from pandas.io.api import to_pickle

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
