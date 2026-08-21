from collections.abc import Sequence
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from typing import (
    TYPE_CHECKING,
    Any,
    assert_type,
)

import pandas as pd
from pandas.arrays import ArrowExtensionArray
from pandas.arrays import ArrowStringArray
import pyarrow as pa
import pytest

from tests import check


@pytest.mark.parametrize(
    "data",
    [
        [True],
        [1],
        [1.0],
        ["1"],
        [datetime(2026, 1, 1)],
        [datetime(2026, 1, 1, tzinfo=UTC)],
        [timedelta(seconds=1)],
    ],
)
def test_constructor(data: Sequence[Any]) -> None:
    check(
        assert_type(ArrowExtensionArray(pa.array(data)), ArrowExtensionArray),
        ArrowExtensionArray,
    )
    check(
        assert_type(ArrowExtensionArray(pa.chunked_array([data])), ArrowExtensionArray),
        ArrowExtensionArray,
    )

    if TYPE_CHECKING:
        assert_type(ArrowExtensionArray(pa.array([True])), ArrowExtensionArray)
        assert_type(ArrowExtensionArray(pa.array([1])), ArrowExtensionArray)
        assert_type(ArrowExtensionArray(pa.array([1.0])), ArrowExtensionArray)
        assert_type(ArrowExtensionArray(pa.array(["1"])), ArrowExtensionArray)
        assert_type(
            ArrowExtensionArray(pa.array([datetime(2026, 1, 1)])), ArrowExtensionArray
        )
        assert_type(
            ArrowExtensionArray(pa.array([datetime(2026, 1, 1, tzinfo=UTC)])),
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
            ArrowExtensionArray(pa.chunked_array([[datetime(2026, 1, 1, tzinfo=UTC)]])),
            ArrowExtensionArray,
        )
        assert_type(
            ArrowExtensionArray(pa.chunked_array([[timedelta(seconds=1)]])),
            ArrowExtensionArray,
        )


def test_pd_array_bool_bool_pyarrow() -> None:
    check(
        assert_type(pd.array([True, False], "bool[pyarrow]"), ArrowExtensionArray),
        ArrowExtensionArray,
        bool,
    )


def test_pd_array_bool_boolean_pyarrow() -> None:
    check(
        assert_type(pd.array([True, False], "boolean[pyarrow]"), ArrowExtensionArray),
        ArrowExtensionArray,
        bool,
    )


def test_pd_array_int_int8_pyarrow() -> None:
    check(
        assert_type(pd.array([1, 2, 3], "int8[pyarrow]"), ArrowExtensionArray),
        ArrowExtensionArray,
        int,
    )


def test_pd_array_int_int16_pyarrow() -> None:
    check(
        assert_type(pd.array([1, 2, 3], "int16[pyarrow]"), ArrowExtensionArray),
        ArrowExtensionArray,
        int,
    )


def test_pd_array_int_int32_pyarrow() -> None:
    check(
        assert_type(pd.array([1, 2, 3], "int32[pyarrow]"), ArrowExtensionArray),
        ArrowExtensionArray,
        int,
    )


def test_pd_array_int_int64_pyarrow() -> None:
    check(
        assert_type(pd.array([1, 2, 3], "int64[pyarrow]"), ArrowExtensionArray),
        ArrowExtensionArray,
        int,
    )


def test_pd_array_uint_uint8_pyarrow() -> None:
    check(
        assert_type(pd.array([1, 2, 3], "uint8[pyarrow]"), ArrowExtensionArray),
        ArrowExtensionArray,
        int,
    )


def test_pd_array_uint_uint16_pyarrow() -> None:
    check(
        assert_type(pd.array([1, 2, 3], "uint16[pyarrow]"), ArrowExtensionArray),
        ArrowExtensionArray,
        int,
    )


def test_pd_array_uint_uint32_pyarrow() -> None:
    check(
        assert_type(pd.array([1, 2, 3], "uint32[pyarrow]"), ArrowExtensionArray),
        ArrowExtensionArray,
        int,
    )


def test_pd_array_uint_uint64_pyarrow() -> None:
    check(
        assert_type(pd.array([1, 2, 3], "uint64[pyarrow]"), ArrowExtensionArray),
        ArrowExtensionArray,
        int,
    )


def test_pd_array_float_float16_pyarrow() -> None:
    check(
        assert_type(pd.array([1.0, 2.0, 3.0], "float16[pyarrow]"), ArrowExtensionArray),
        ArrowExtensionArray,
        float,
    )


def test_pd_array_float_float32_pyarrow() -> None:
    check(
        assert_type(pd.array([1.0, 2.0, 3.0], "float32[pyarrow]"), ArrowExtensionArray),
        ArrowExtensionArray,
        float,
    )


def test_pd_array_float_float_pyarrow() -> None:
    check(
        assert_type(pd.array([1.0, 2.0, 3.0], "float[pyarrow]"), ArrowExtensionArray),
        ArrowExtensionArray,
        float,
    )


def test_pd_array_float_float64_pyarrow() -> None:
    check(
        assert_type(pd.array([1.0, 2.0, 3.0], "float64[pyarrow]"), ArrowExtensionArray),
        ArrowExtensionArray,
        float,
    )


def test_pd_array_float_double_pyarrow() -> None:
    check(
        assert_type(pd.array([1.0, 2.0, 3.0], "double[pyarrow]"), ArrowExtensionArray),
        ArrowExtensionArray,
        float,
    )


def test_pd_array_mixed_int_bool_data_float_pyarrow() -> None:
    # data is a mix of int and bool elements, coerced by pyarrow to float64
    check(
        assert_type(pd.array([1, True], "float64[pyarrow]"), ArrowExtensionArray),
        ArrowExtensionArray,
        float,
    )


def test_pd_array_str_data_float_pyarrow() -> None:
    # pyarrow parses the numeric strings into a float64 array
    check(
        assert_type(pd.array(["1", "1"], "float64[pyarrow]"), ArrowExtensionArray),
        ArrowExtensionArray,
        float,
    )


def test_pd_array_float_data_string_pyarrow() -> None:
    # dtype-specific string overload takes precedence, giving ArrowStringArray
    check(
        assert_type(pd.array([0.1, 0.2], "string[pyarrow]"), ArrowStringArray),
        ArrowStringArray,
        str,
    )


def test_pd_array_float_data_duration_pyarrow() -> None:
    check(
        assert_type(pd.array([0.1, 0.2], "duration[s][pyarrow]"), ArrowExtensionArray),
        ArrowExtensionArray,
        pd.Timedelta,
    )


def test_pd_array_mixed_int_str_data_float_pyarrow_runtime_failure() -> None:
    # type checkers cannot statically detect this is invalid since pyarrow's
    # casting behavior depends on runtime values, not just element types;
    # it still raises at runtime because pyarrow cannot cast "1" once it has
    # inferred an int64 array from the first element.
    with pytest.raises(pa.ArrowInvalid):
        pd.array([1, "1"], "float64[pyarrow]")
