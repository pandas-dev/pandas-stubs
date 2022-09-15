from typing import (
    Any,
    Union,
)

import pandas as pd
from typing_extensions import assert_type

from pandas._config.config import DictWrapper

from tests import check


def test_option_tools():
    check(assert_type(pd.reset_option("display.width"), None), type(None))
    check(assert_type(pd.set_option("display.width", 80), None), type(None))
    check(assert_type(pd.describe_option("display.width", False), str), str)
    check(assert_type(pd.describe_option("display.width", True), None), type(None))
    check(assert_type(pd.options, DictWrapper), DictWrapper)
    check(
        assert_type(pd.options.display, Union[str, bool, int, None, DictWrapper]),
        DictWrapper,
    )
    check(assert_type(pd.get_option("display.width"), Any), int)
    with pd.option_context("display.width", 120):
        assert pd.get_option("display.width") == 120
