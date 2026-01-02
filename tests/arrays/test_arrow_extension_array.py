from collections.abc import Sequence
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

from pandas.core.arrays.arrow.array import ArrowExtensionArray
import pyarrow as pa
import pytest
from typing_extensions import assert_type

from tests import check


@pytest.mark.parametrize(
    "data",
    [
        [True],
        [1],
        [1.0],
        ["1"],
        [datetime(2026, 1, 1)],
        [datetime(2026, 1, 1, tzinfo=timezone.utc)],
        [timedelta(seconds=1)],
    ],
)
def test_constructor(data: Sequence[Any]) -> None:
    check(ArrowExtensionArray(pa.array(data)), ArrowExtensionArray)
    check(ArrowExtensionArray(pa.chunked_array([data])), ArrowExtensionArray)

    if TYPE_CHECKING:
        assert_type(ArrowExtensionArray(pa.array([True])), ArrowExtensionArray)
        assert_type(ArrowExtensionArray(pa.array([1])), ArrowExtensionArray)
        assert_type(ArrowExtensionArray(pa.array([1.0])), ArrowExtensionArray)
        assert_type(ArrowExtensionArray(pa.array(["1"])), ArrowExtensionArray)
        assert_type(
            ArrowExtensionArray(pa.array([datetime(2026, 1, 1)])), ArrowExtensionArray
        )
        assert_type(
            ArrowExtensionArray(pa.array([datetime(2026, 1, 1, tzinfo=timezone.utc)])),
            ArrowExtensionArray,
        )
        assert_type(
            ArrowExtensionArray(pa.array([timedelta(seconds=1)])), ArrowExtensionArray
        )

        assert_type(
            ArrowExtensionArray(pa.chunked_array([[True]])), ArrowExtensionArray
        )
        assert_type(ArrowExtensionArray(pa.chunked_array([[1]])), ArrowExtensionArray)
        assert_type(ArrowExtensionArray(pa.chunked_array([[1.0]])), ArrowExtensionArray)
        assert_type(ArrowExtensionArray(pa.chunked_array([["1"]])), ArrowExtensionArray)
        assert_type(
            ArrowExtensionArray(pa.chunked_array([[datetime(2026, 1, 1)]])),
            ArrowExtensionArray,
        )
        assert_type(
            ArrowExtensionArray(
                pa.chunked_array([[datetime(2026, 1, 1, tzinfo=timezone.utc)]])
            ),
            ArrowExtensionArray,
        )
        assert_type(
            ArrowExtensionArray(pa.chunked_array([[timedelta(seconds=1)]])),
            ArrowExtensionArray,
        )
