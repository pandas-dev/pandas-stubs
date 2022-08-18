from pandas import DataFrame
from typing_extensions import assert_type

from tests import check

from pandas.io.clipboards import read_clipboard

DF = DataFrame({"a": [1, 2, 3], "b": [0.0, 0.0, 0.0]})


def test_clipboard():
    DF.to_clipboard()
    check(assert_type(read_clipboard(), DataFrame), DataFrame)
