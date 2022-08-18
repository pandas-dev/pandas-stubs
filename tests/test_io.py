import os
import tempfile
from typing import Any

from pandas import DataFrame
from typing_extensions import assert_type

from tests import check

from pandas.io.pickle import (
    read_pickle,
    to_pickle,
)

DF = DataFrame({"a": [1, 2, 3], "b": [0.0, 0.0, 0.0]})


def test_pickle():
    with tempfile.NamedTemporaryFile(delete=False) as file:
        check(assert_type(DF.to_pickle(file), None), type(None))
        file.seek(0)
        check(assert_type(read_pickle(file.name), Any), DataFrame)
        file.close()
        check(assert_type(read_pickle(file.name), Any), DataFrame)
        os.unlink(file.name)

    with tempfile.NamedTemporaryFile(delete=False) as file:
        check(assert_type(to_pickle(DF, file), None), type(None))
        file.seek(0)
        check(assert_type(read_pickle(file.name), Any), DataFrame)
        file.close()
        check(assert_type(read_pickle(file.name), Any), DataFrame)
        os.unlink(file.name)


def test_pickle_protocol():
    with tempfile.NamedTemporaryFile(delete=False) as file:
        DF.to_pickle(file, protocol=3)
        file.seek(0)
        check(assert_type(read_pickle(file.name), Any), DataFrame)
        file.close()
        check(assert_type(read_pickle(file.name), Any), DataFrame)
        os.unlink(file.name)


def test_pickle_compression():
    with tempfile.NamedTemporaryFile(delete=False) as file:
        DF.to_pickle(file, compression="gzip")
        file.seek(0)
        check(
            assert_type(read_pickle(file.name, compression="gzip"), Any),
            DataFrame,
        )
        file.close()
        check(
            assert_type(read_pickle(file.name, compression="gzip"), Any),
            DataFrame,
        )
        os.unlink(file.name)


def test_pickle_storage_options():
    with tempfile.NamedTemporaryFile(delete=False) as file:
        DF.to_pickle(file, storage_options={})
        file.seek(0)
        check(assert_type(read_pickle(file, storage_options={}), Any), DataFrame)
        file.close()
        check(
            assert_type(read_pickle(file.name, storage_options={}), Any),
            DataFrame,
        )
        os.unlink(file.name)
