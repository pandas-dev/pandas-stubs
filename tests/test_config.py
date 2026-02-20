from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    assert_type,
)

import pandas as pd

from pandas._config.config import DictWrapper

from tests import check

if TYPE_CHECKING:
    from pandas._config.config import (
        Display,
        Options,
    )
else:
    Display = Options = Any


def test_option_tools() -> None:
    check(assert_type(pd.reset_option("display.width"), None), type(None))
    check(assert_type(pd.set_option("display.width", 80), None), type(None))
    check(assert_type(pd.set_option({"display.width": 80}), None), type(None))
    check(
        assert_type(
            pd.set_option("display.max_columns", 4, "display.precision", 1), None
        ),
        type(None),
    )
    check(assert_type(pd.describe_option("display.width", False), str), str)
    check(assert_type(pd.describe_option("display.width", True), None), type(None))
    check(assert_type(pd.options, Options), DictWrapper)
    check(assert_type(pd.options.display, Display), DictWrapper)
    check(assert_type(pd.get_option("display.width"), Any), int)
    with pd.option_context("display.width", 120):
        assert pd.get_option("display.width") == 120


def test_set_option_overload() -> None:
    """Test the different overloads of `pd.set_option`."""
    # dict
    check(assert_type(pd.set_option({"display.width": 80}), None), type(None))
    check(
        assert_type(pd.set_option({"display.width": 80, "display.precision": 6}), None),
        type(None),
    )

    # one pair of arguments
    check(assert_type(pd.set_option("display.width", 80), None), type(None))
    check(assert_type(pd.set_option("display.precision", 6), None), type(None))
    check(assert_type(pd.set_option("display.max_rows", 60), None), type(None))
    check(assert_type(pd.set_option("display.max_rows", None), None), type(None))
    check(
        assert_type(pd.set_option("display.expand_frame_repr", True), None), type(None)
    )

    # two pairs of arguments
    check(
        assert_type(pd.set_option("display.width", 80, "display.precision", 6), None),
        type(None),
    )
    check(
        assert_type(
            pd.set_option("display.max_rows", 60, "display.max_columns", 20), None
        ),
        type(None),
    )

    # three pairs of arguments
    check(
        assert_type(
            pd.set_option(
                "display.width",
                80,
                "display.precision",
                6,
                "display.max_rows",
                60,
            ),
            None,
        ),
        type(None),
    )

    # four pairs of arguments
    check(
        assert_type(
            pd.set_option(
                "display.width",
                80,
                "display.precision",
                6,
                "display.max_rows",
                60,
                "display.max_columns",
                20,
            ),
            None,
        ),
        type(None),
    )

    # five pairs of arguments
    check(
        assert_type(
            pd.set_option(
                "display.width",
                80,
                "display.precision",
                6,
                "display.max_rows",
                60,
                "display.max_columns",
                20,
                "display.max_colwidth",
                50,
            ),
            None,
        ),
        type(None),
    )


def test_set_option_six_pairs() -> None:
    # GH 1680: set_option(pat0..pat5, val0..val5)
    check(
        assert_type(
            pd.set_option(
                "display.width",
                80,
                "display.precision",
                6,
                "display.max_rows",
                60,
                "display.max_columns",
                20,
                "display.max_colwidth",
                50,
                "display.min_rows",
                10,
            ),
            None,
        ),
        type(None),
    )


def test_specific_option() -> None:
    # GH 294
    check(assert_type(pd.options.plotting.backend, str), str)
    # Just check assignment
    pd.options.plotting.backend = "matplotlib"


def test_display_float_format() -> None:
    check(
        assert_type(pd.options.display.float_format, Callable[[float], str] | None),
        type(None),
    )
    formatter = "{,.2f}".format
    with pd.option_context("display.float_format", formatter):
        assert pd.get_option("display.float_format") == formatter


def test_display_types_none_allowed_get_options() -> None:
    # GH 1230
    # Initial values
    check(assert_type(pd.options.display.chop_threshold, float | None), type(None))
    check(assert_type(pd.options.display.max_columns, int | None), int)
    check(assert_type(pd.options.display.max_colwidth, int | None), int)
    check(assert_type(pd.options.display.max_dir_items, int | None), int)
    check(assert_type(pd.options.display.max_rows, int | None), int)
    check(assert_type(pd.options.display.max_seq_items, int | None), int)
    check(assert_type(pd.options.display.min_rows, int | None), int)


def test_display_types_none_allowed_set_options() -> None:
    # GH 1230
    # Test setting each option as None and then to a specific value
    pd.options.display.chop_threshold = None
    pd.options.display.chop_threshold = 0.9
    pd.options.display.max_columns = None
    pd.options.display.max_columns = 100
    pd.options.display.max_colwidth = None
    pd.options.display.max_colwidth = 100
    pd.options.display.max_dir_items = None
    pd.options.display.max_dir_items = 100
    pd.options.display.max_rows = None
    pd.options.display.max_rows = 100
    pd.options.display.max_seq_items = None
    pd.options.display.max_seq_items = 100
    pd.options.display.min_rows = None
    pd.options.display.min_rows = 100


def test_display_types_literal_constraints() -> None:
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
    assert_type(pd.options.display.memory_usage, Literal[True, False, "deep"] | None)
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
