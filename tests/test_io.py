from pandas import (
    DataFrame,
    read_clipboard,
)
import pytest
from typing_extensions import assert_type

from tests import check

from pandas.io.clipboard import PyperclipException

DF = DataFrame({"a": [1, 2, 3], "b": [0.0, 0.0, 0.0]})


def test_clipboard():
    try:
        DF.to_clipboard()
    except PyperclipException:
        pytest.skip("clipboard not available for testing")
    check(assert_type(read_clipboard(), DataFrame), DataFrame)
