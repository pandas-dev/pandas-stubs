from __future__ import annotations

from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)
from tests._typing import (
    np_ndarray_dt,
    np_ndarray_str,
    np_ndarray_td,
)


@pytest.fixture
def left() -> pd.Series[float]:
    """Left operand"""
    lo = pd.Series([1.0, 2.0, 3.0])
    return check(assert_type(lo, "pd.Series[float]"), pd.Series, np.floating)


def test_relops_py_scalar(left: pd.Series[float]) -> None:
    """Test pd.Series[float] <comparison> py scalar"""
    # GH 175
    f = 3.14

    check(assert_type(left < f, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= f, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > f, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= f, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(f), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(f), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(f), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(f), "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(f < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(f <= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(f > left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(f >= left, "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_mixed_numeric(left: pd.Series[float]) -> None:
    """Test pd.Series[float] <comparison> int scalar"""
    # GH 1903
    i = 10

    check(assert_type(left < i, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= i, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > i, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= i, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(i < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(i <= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(i > left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(i >= left, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(i), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(i), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(i), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(i), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_invalid_py_scalar(left: pd.Series[float]) -> None:
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_c = left < 1j  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_c = left <= 1j  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_c = left > 1j  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_c = left >= 1j  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]

        _c_lt = 1j < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _c_le = 1j <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _c_gt = 1j > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _c_ge = 1j >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]

        left.lt(1j)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(1j)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(1j)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(1j)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_py_sequence(left: pd.Series[float]) -> None:
    complex_values = [1j, 2j]

    if TYPE_CHECKING_INVALID_USAGE:
        _lt_cseq = left < complex_values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_cseq = left <= complex_values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_cseq = left > complex_values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_cseq = left >= complex_values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]

        _cseq_lt = complex_values < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _cseq_le = complex_values <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _cseq_gt = complex_values > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _cseq_ge = complex_values >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]

        left.lt(complex_values)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(complex_values)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(complex_values)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(complex_values)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_numpy_array(left: pd.Series[float]) -> None:
    complex_array = np.array([1j, 2j], np.complex128)

    if TYPE_CHECKING_INVALID_USAGE:
        left.__lt__(complex_array)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__le__(complex_array)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__gt__(complex_array)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__ge__(complex_array)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]

        left.lt(complex_array)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(complex_array)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(complex_array)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(complex_array)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_pd_index(left: pd.Series[float]) -> None:
    index = pd.Index(left)

    check(assert_type(left < index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= index, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(index < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(index <= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(index > left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(index >= left, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(index), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(index), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(index), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(index), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_invalid_pd_index(left: pd.Series[float]) -> None:
    complex_index = pd.Index([1j, 2j])

    if TYPE_CHECKING_INVALID_USAGE:
        _lt_cidx = left < complex_index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_cidx = left <= complex_index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_cidx = left > complex_index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_cidx = left >= complex_index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]

        _cidx_lt = complex_index < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _cidx_le = complex_index <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _cidx_gt = complex_index > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _cidx_ge = complex_index >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]

        left.lt(complex_index)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(complex_index)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(complex_index)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(complex_index)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_pd_series(left: pd.Series[float]) -> None:
    other = left.copy()

    check(assert_type(left < other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= other, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(other), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_py_scalar_bool(left: pd.Series[float]) -> None:
    other = True
    check(assert_type(left < other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(other < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.lt(other), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_numpy_scalar_bool(left: pd.Series[float]) -> None:
    other = np.bool_(True)
    check(assert_type(left < other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(
        assert_type(other < left, "pd.Series[bool]"),  # type: ignore[assert-type] # pyright: ignore[reportAssertTypeFailure] # pyrefly: ignore[assert-type] # ty: ignore[type-assertion-failure]
        pd.Series,
        np.bool_,
    )
    check(assert_type(left.lt(other), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_numpy_scalar_int64(left: pd.Series[float]) -> None:
    other = np.int64(2)
    check(assert_type(left < other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(
        assert_type(other < left, "pd.Series[bool]"),  # type: ignore[assert-type] # pyright: ignore[reportAssertTypeFailure] # pyrefly: ignore[assert-type] # ty: ignore[type-assertion-failure]
        pd.Series,
        np.bool_,
    )
    check(assert_type(left.lt(other), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_numpy_scalar_float32(left: pd.Series[float]) -> None:
    other = np.float32(2)
    check(assert_type(left < other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(
        assert_type(other < left, "pd.Series[bool]"),  # type: ignore[assert-type] # pyright: ignore[reportAssertTypeFailure] # pyrefly: ignore[assert-type] # ty: ignore[type-assertion-failure]
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(other <= left, "pd.Series[bool]"),  # type: ignore[assert-type] # pyright: ignore[reportAssertTypeFailure] # pyrefly: ignore[assert-type] # ty: ignore[type-assertion-failure]
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(other > left, "pd.Series[bool]"),  # type: ignore[assert-type] # pyright: ignore[reportAssertTypeFailure] # pyrefly: ignore[assert-type] # ty: ignore[type-assertion-failure]
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(other >= left, "pd.Series[bool]"),  # type: ignore[assert-type] # pyright: ignore[reportAssertTypeFailure] # pyrefly: ignore[assert-type] # ty: ignore[type-assertion-failure]
        pd.Series,
        np.bool_,
    )
    check(assert_type(left.lt(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(other), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_py_sequence_float(left: pd.Series[float]) -> None:
    values = [2.0, 2.0, 2.0]
    check(assert_type(left < values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(values < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(values <= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(values > left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(values >= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.lt(values), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(values), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(values), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(values), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_py_sequence_int(left: pd.Series[float]) -> None:
    values = [2, 2, 2]
    check(assert_type(left < values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(values < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.lt(values), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_py_sequence_bool(left: pd.Series[float]) -> None:
    values = [True, False, True]
    check(assert_type(left < values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(values < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.lt(values), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_numpy_array_float32(left: pd.Series[float]) -> None:
    array = np.array([2.0, 2.0, 2.0], np.float32)
    check(assert_type(left < array, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= array, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > array, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= array, "pd.Series[bool]"), pd.Series, np.bool_)
    check(
        assert_type(array < left, "pd.Series[bool]"),  # type: ignore[assert-type] # pyright: ignore[reportAssertTypeFailure] # pyrefly: ignore[assert-type] # ty: ignore[type-assertion-failure]
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(array <= left, "pd.Series[bool]"),  # type: ignore[assert-type] # pyright: ignore[reportAssertTypeFailure] # pyrefly: ignore[assert-type] # ty: ignore[type-assertion-failure]
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(array > left, "pd.Series[bool]"),  # type: ignore[assert-type] # pyright: ignore[reportAssertTypeFailure] # pyrefly: ignore[assert-type] # ty: ignore[type-assertion-failure]
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(array >= left, "pd.Series[bool]"),  # type: ignore[assert-type] # pyright: ignore[reportAssertTypeFailure] # pyrefly: ignore[assert-type] # ty: ignore[type-assertion-failure]
        pd.Series,
        np.bool_,
    )
    check(assert_type(left.lt(array), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(array), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(array), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(array), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_numpy_array_int64(left: pd.Series[float]) -> None:
    array = np.array([2, 2, 2], np.int64)
    check(assert_type(left < array, "pd.Series[bool]"), pd.Series, np.bool_)
    check(
        assert_type(array < left, "pd.Series[bool]"),  # type: ignore[assert-type] # pyright: ignore[reportAssertTypeFailure] # pyrefly: ignore[assert-type] # ty: ignore[type-assertion-failure]
        pd.Series,
        np.bool_,
    )
    check(assert_type(left.lt(array), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_numpy_array_bool(left: pd.Series[float]) -> None:
    array = np.array([True, False, True], np.bool_)
    check(assert_type(left < array, "pd.Series[bool]"), pd.Series, np.bool_)
    check(
        assert_type(array < left, "pd.Series[bool]"),  # type: ignore[assert-type] # pyright: ignore[reportAssertTypeFailure] # pyrefly: ignore[assert-type] # ty: ignore[type-assertion-failure]
        pd.Series,
        np.bool_,
    )
    check(assert_type(left.lt(array), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_pd_index_int(left: pd.Series[float]) -> None:
    index = pd.Index([2, 2, 2])
    check(assert_type(left < index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(index < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.lt(index), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_pd_index_bool(left: pd.Series[float]) -> None:
    index = pd.Index([True, False, True])
    check(assert_type(left < index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(index < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.lt(index), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_range_index(left: pd.Series[float]) -> None:
    index = pd.RangeIndex(3)
    check(assert_type(left < index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(index < left, "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_pd_series_bool(left: pd.Series[float]) -> None:
    other = pd.Series([True, False, True])
    check(assert_type(left < other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(other < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(other <= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(other > left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(other >= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.lt(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(other.lt(left), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(other.le(left), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(other.gt(left), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(other.ge(left), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_invalid_py_scalar_str(left: pd.Series[float]) -> None:
    other = "x"
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = other < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = other <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = other > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = other >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        left.lt(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_timestamp(left: pd.Series[float]) -> None:
    other = pd.Timestamp("2020-01-01")
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = other < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = other <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = other > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = other >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        left.lt(other)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(other)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(other)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(other)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_timedelta(left: pd.Series[float]) -> None:
    other = pd.Timedelta("1 day")
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = other < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = other <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = other > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = other >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        left.lt(other)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(other)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(other)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(other)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_py_sequence_str(left: pd.Series[float]) -> None:
    values: list[str] = ["x", "y", "z"]
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = values < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = values <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = values > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = values >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        left.lt(values)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(values)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(values)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(values)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_pd_index_str(left: pd.Series[float]) -> None:
    index: pd.Index[str] = pd.Index(["x", "y", "z"])
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = index < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = index <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = index > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = index >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        left.lt(index)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(index)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(index)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(index)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_datetime_index(left: pd.Series[float]) -> None:
    index = pd.date_range("2020-01-01", periods=3)
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = index < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = index <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = index > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = index >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        left.lt(index)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(index)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(index)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(index)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_timedelta_index(left: pd.Series[float]) -> None:
    index = pd.timedelta_range("1 day", periods=3)
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = index < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = index <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = index > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = index >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        left.lt(index)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(index)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(index)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(index)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_pd_series_str(left: pd.Series[float]) -> None:
    other = pd.Series(["x", "y", "z"])
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = other < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = other <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = other > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = other >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        left.lt(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        other.lt(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        other.le(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        other.gt(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        other.ge(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_pd_series_timestamp(left: pd.Series[float]) -> None:
    other = pd.Series(pd.date_range("2020-01-01", periods=3))
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = other < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = other <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = other > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = other >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        left.lt(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        other.lt(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        other.le(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        other.gt(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        other.ge(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_pd_series_timedelta(left: pd.Series[float]) -> None:
    other = pd.Series(pd.timedelta_range("1 day", periods=3))
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = other < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = other <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = other > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = other >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        left.lt(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        other.lt(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        other.le(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        other.gt(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        other.ge(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_numpy_array_str(left: pd.Series[float]) -> None:
    other: np_ndarray_str = np.array(["x", "y", "z"], np.str_)
    if TYPE_CHECKING_INVALID_USAGE:
        left.__lt__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__le__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__gt__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__ge__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.lt(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_numpy_array_datetime64(
    left: pd.Series[float],
) -> None:
    other: np_ndarray_dt = np.array(
        [np.datetime64("2020-01-01", "ns")] * 3, np.datetime64
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.__lt__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__le__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__gt__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__ge__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.lt(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_numpy_array_timedelta64(
    left: pd.Series[float],
) -> None:
    other: np_ndarray_td = np.array([np.timedelta64(1, "ns")] * 3, np.timedelta64)
    if TYPE_CHECKING_INVALID_USAGE:
        left.__lt__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__le__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__gt__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__ge__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.lt(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
