from typing import (
    TYPE_CHECKING,
    Any,
)

import pandas as pd
from typing_extensions import assert_type

from pandas._config.config import DictWrapper

from tests import check

if TYPE_CHECKING:
    from pandas._config.config import (
        Display,
        Options,
    )
else:
    Display = Options = Any


def test_option_tools():
    check(assert_type(pd.reset_option("display.width"), None), type(None))
    check(assert_type(pd.set_option("display.width", 80), None), type(None))
    check(assert_type(pd.describe_option("display.width", False), str), str)
    check(assert_type(pd.describe_option("display.width", True), None), type(None))
    check(assert_type(pd.options, Options), DictWrapper)
    check(assert_type(pd.options.display, Display), DictWrapper)
    check(assert_type(pd.get_option("display.width"), Any), int)
    with pd.option_context("display.width", 120):
        assert pd.get_option("display.width") == 120


def test_specific_option():
    # GH 294
    check(assert_type(pd.options.plotting.backend, str), str)
    # Just check assignment
    pd.options.plotting.backend = "matplotlib"
