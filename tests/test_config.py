from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
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


def test_display_float_format():
    check(
        assert_type(pd.options.display.float_format, Optional[Callable[[float], str]]),
        type(None),
    )
    formatter = "{,.2f}".format
    with pd.option_context("display.float_format", formatter):
        assert pd.get_option("display.float_format") == formatter


def test_display_types_none_allowed():
    # GH 1230
    # Initial values
    assert_type(pd.options.display.max_columns, Optional[int])
    assert_type(pd.options.display.max_colwidth, Optional[int])
    assert_type(pd.options.display.max_dir_items, Optional[int])
    assert_type(pd.options.display.max_rows, Optional[int])
    assert_type(pd.options.display.max_seq_items, Optional[int])
    assert_type(pd.options.display.min_rows, Optional[int])
    # Test with None
    pd.options.display.max_columns = None
    check(assert_type(pd.options.display.max_columns, None), type(None))
    pd.options.display.max_colwidth = None
    check(assert_type(pd.options.display.max_colwidth, None), type(None))
    pd.options.display.max_dir_items = None
    check(assert_type(pd.options.display.max_dir_items, None), type(None))
    pd.options.display.max_rows = None
    check(assert_type(pd.options.display.max_rows, None), type(None))
    pd.options.display.max_seq_items = None
    check(assert_type(pd.options.display.max_seq_items, None), type(None))
    pd.options.display.min_rows = None
    check(assert_type(pd.options.display.min_rows, None), type(None))
    # Test with integer values
    pd.options.display.max_columns = 100
    check(assert_type(pd.options.display.max_columns, int), int)
    pd.options.display.max_colwidth = 100
    check(assert_type(pd.options.display.max_colwidth, int), int)
    pd.options.display.max_dir_items = 100
    check(assert_type(pd.options.display.max_dir_items, int), int)
    pd.options.display.max_rows = 100
    check(assert_type(pd.options.display.max_rows, int), int)
    pd.options.display.max_seq_items = 100
    check(assert_type(pd.options.display.max_seq_items, int), int)
    pd.options.display.min_rows = 100
    check(assert_type(pd.options.display.min_rows, int), int)


def test_display_types_literal_constraints():
    # GH 1230
    # Various display options have specific allowed values
    # Test colheader_justify with allowed values
    assert_type(pd.options.display.colheader_justify, Literal["left", "right"])
    pd.options.display.colheader_justify = "left"
    check(assert_type(pd.options.display.colheader_justify, Literal["left"]), str)
    pd.options.display.colheader_justify = "right"
    check(assert_type(pd.options.display.colheader_justify, Literal["right"]), str)

    # Test large_repr with allowed values
    assert_type(pd.options.display.large_repr, Literal["truncate", "info"])
    pd.options.display.large_repr = "truncate"
    check(assert_type(pd.options.display.large_repr, Literal["truncate"]), str)
    pd.options.display.large_repr = "info"
    check(assert_type(pd.options.display.large_repr, Literal["info"]), str)

    # Test memory_usage with allowed values
    assert_type(pd.options.display.memory_usage, Optional[Literal[True, False, "deep"]])
    pd.options.display.memory_usage = True
    check(assert_type(pd.options.display.memory_usage, Literal[True]), bool)
    pd.options.display.memory_usage = False
    check(assert_type(pd.options.display.memory_usage, Literal[False]), bool)
    pd.options.display.memory_usage = "deep"
    check(assert_type(pd.options.display.memory_usage, Literal["deep"]), str)
    pd.options.display.memory_usage = None
    check(assert_type(pd.options.display.memory_usage, None), type(None))

    # Test show_dimensions with allowed values
    assert_type(pd.options.display.show_dimensions, Literal[True, False, "truncate"])
    pd.options.display.show_dimensions = True
    check(assert_type(pd.options.display.show_dimensions, Literal[True]), bool)
    pd.options.display.show_dimensions = False
    check(assert_type(pd.options.display.show_dimensions, Literal[False]), bool)
    pd.options.display.show_dimensions = "truncate"
    check(assert_type(pd.options.display.show_dimensions, Literal["truncate"]), str)
