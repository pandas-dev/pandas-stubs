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
from pandas.core.arrays.arrow.array import ArrowExtensionArray
import pyarrow as pa
import pytest

from tests import check
from tests._typing import (
    PyArrowBooleanDtypeArg,
    PyArrowFloatDtypeArg,
    PyArrowIntDtypeArg,
    PyArrowUIntDtypeArg,
)
from tests.dtypes import (
    PYARROW_BOOL_ARGS,
    PYARROW_FLOAT_ARGS,
    PYARROW_INT_ARGS,
    PYARROW_UINT_ARGS,
)


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


@pytest.mark.parametrize(("dtype", "target_dtype"), PYARROW_BOOL_ARGS.items(), ids=repr)
def test_pd_array_boolean(dtype: PyArrowBooleanDtypeArg, target_dtype: type) -> None:
    check(
        assert_type(pd.array([True, False], dtype), ArrowExtensionArray),
        ArrowExtensionArray,
        target_dtype,
    )


@pytest.mark.parametrize(("dtype", "target_dtype"), PYARROW_INT_ARGS.items(), ids=repr)
def test_pd_array_int(dtype: PyArrowIntDtypeArg, target_dtype: type) -> None:
    check(
        assert_type(pd.array([1, 2, 3], dtype), ArrowExtensionArray),
        ArrowExtensionArray,
        target_dtype,
    )


@pytest.mark.parametrize(("dtype", "target_dtype"), PYARROW_UINT_ARGS.items(), ids=repr)
def test_pd_array_uint(dtype: PyArrowUIntDtypeArg, target_dtype: type) -> None:
    check(
        assert_type(pd.array([1, 2, 3], dtype), ArrowExtensionArray),
        ArrowExtensionArray,
        target_dtype,
    )


@pytest.mark.parametrize(
    ("dtype", "target_dtype"), PYARROW_FLOAT_ARGS.items(), ids=repr
)
def test_pd_array_float(dtype: PyArrowFloatDtypeArg, target_dtype: type) -> None:
    check(
        assert_type(pd.array([1.0, 2.0, 3.0], dtype), ArrowExtensionArray),
        ArrowExtensionArray,
        target_dtype,
    )
