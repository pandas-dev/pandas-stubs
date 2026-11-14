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
    np_ndarray_bool,
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


def test_relops_bool_scalar(left: pd.Series[float]) -> None:
    check(assert_type(left < True, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= True, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > True, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= True, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(True < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(True <= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(True > left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(True >= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.lt(True), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(True), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(True), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(True), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_numpy_scalar_bool(left: pd.Series[float]) -> None:
    check(assert_type(left < np.bool_(True), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= np.bool_(True), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > np.bool_(True), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= np.bool_(True), "pd.Series[bool]"), pd.Series, np.bool_)
    # NumPy infers ndarray, while pandas dispatch returns Series.
    check(assert_type(np.bool_(True) < left, np_ndarray_bool), pd.Series, np.bool_)
    check(assert_type(np.bool_(True) <= left, np_ndarray_bool), pd.Series, np.bool_)
    check(assert_type(np.bool_(True) > left, np_ndarray_bool), pd.Series, np.bool_)
    check(assert_type(np.bool_(True) >= left, np_ndarray_bool), pd.Series, np.bool_)
    check(assert_type(left.lt(np.bool_(True)), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(np.bool_(True)), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(np.bool_(True)), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(np.bool_(True)), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_numpy_scalar_int64(left: pd.Series[float]) -> None:
    check(assert_type(left < np.int64(2), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= np.int64(2), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > np.int64(2), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= np.int64(2), "pd.Series[bool]"), pd.Series, np.bool_)
    # NumPy infers ndarray, while pandas dispatch returns Series.
    check(assert_type(np.int64(2) < left, np_ndarray_bool), pd.Series, np.bool_)
    check(assert_type(np.int64(2) <= left, np_ndarray_bool), pd.Series, np.bool_)
    check(assert_type(np.int64(2) > left, np_ndarray_bool), pd.Series, np.bool_)
    check(assert_type(np.int64(2) >= left, np_ndarray_bool), pd.Series, np.bool_)
    check(assert_type(left.lt(np.int64(2)), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(np.int64(2)), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(np.int64(2)), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(np.int64(2)), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_numpy_scalar_float32(left: pd.Series[float]) -> None:
    check(assert_type(left < np.float32(2), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= np.float32(2), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > np.float32(2), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= np.float32(2), "pd.Series[bool]"), pd.Series, np.bool_)
    # NumPy infers ndarray, while pandas dispatch returns Series.
    check(assert_type(np.float32(2) < left, np_ndarray_bool), pd.Series, np.bool_)
    check(assert_type(np.float32(2) <= left, np_ndarray_bool), pd.Series, np.bool_)
    check(assert_type(np.float32(2) > left, np_ndarray_bool), pd.Series, np.bool_)
    check(assert_type(np.float32(2) >= left, np_ndarray_bool), pd.Series, np.bool_)
    check(assert_type(left.lt(np.float32(2)), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(np.float32(2)), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(np.float32(2)), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(np.float32(2)), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_numpy_scalar_float64(left: pd.Series[float]) -> None:
    check(assert_type(left < np.float64(2), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= np.float64(2), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > np.float64(2), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= np.float64(2), "pd.Series[bool]"), pd.Series, np.bool_)
    # NumPy infers ndarray, while pandas dispatch returns Series.
    check(assert_type(np.float64(2) < left, np_ndarray_bool), pd.Series, np.bool_)
    check(assert_type(np.float64(2) <= left, np_ndarray_bool), pd.Series, np.bool_)
    check(assert_type(np.float64(2) > left, np_ndarray_bool), pd.Series, np.bool_)
    check(assert_type(np.float64(2) >= left, np_ndarray_bool), pd.Series, np.bool_)
    check(assert_type(left.lt(np.float64(2)), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(np.float64(2)), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(np.float64(2)), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(np.float64(2)), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_py_sequence_float(left: pd.Series[float]) -> None:
    check(assert_type(left < [2.0, 2.0, 2.0], "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= [2.0, 2.0, 2.0], "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > [2.0, 2.0, 2.0], "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= [2.0, 2.0, 2.0], "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type([2.0, 2.0, 2.0] < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type([2.0, 2.0, 2.0] <= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type([2.0, 2.0, 2.0] > left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type([2.0, 2.0, 2.0] >= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.lt([2.0, 2.0, 2.0]), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le([2.0, 2.0, 2.0]), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt([2.0, 2.0, 2.0]), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge([2.0, 2.0, 2.0]), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_py_sequence_int(left: pd.Series[float]) -> None:
    check(assert_type(left < [2, 2, 2], "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= [2, 2, 2], "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > [2, 2, 2], "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= [2, 2, 2], "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type([2, 2, 2] < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type([2, 2, 2] <= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type([2, 2, 2] > left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type([2, 2, 2] >= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.lt([2, 2, 2]), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le([2, 2, 2]), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt([2, 2, 2]), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge([2, 2, 2]), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_py_sequence_bool(left: pd.Series[float]) -> None:
    check(
        assert_type(left < [True, False, True], "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type(left <= [True, False, True], "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type(left > [True, False, True], "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type(left >= [True, False, True], "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type([True, False, True] < left, "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type([True, False, True] <= left, "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type([True, False, True] > left, "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type([True, False, True] >= left, "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type(left.lt([True, False, True]), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.le([True, False, True]), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.gt([True, False, True]), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.ge([True, False, True]), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )


def test_relops_numpy_array_float32(left: pd.Series[float]) -> None:
    check(
        assert_type(left < np.array([2.0, 2.0, 2.0], np.float32), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left <= np.array([2.0, 2.0, 2.0], np.float32), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left > np.array([2.0, 2.0, 2.0], np.float32), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left >= np.array([2.0, 2.0, 2.0], np.float32), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    # NumPy infers ndarray, while pandas dispatch returns Series.
    check(
        assert_type(np.array([2.0, 2.0, 2.0], np.float32) < left, np_ndarray_bool),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(np.array([2.0, 2.0, 2.0], np.float32) <= left, np_ndarray_bool),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(np.array([2.0, 2.0, 2.0], np.float32) > left, np_ndarray_bool),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(np.array([2.0, 2.0, 2.0], np.float32) >= left, np_ndarray_bool),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.lt(np.array([2.0, 2.0, 2.0], np.float32)), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.le(np.array([2.0, 2.0, 2.0], np.float32)), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.gt(np.array([2.0, 2.0, 2.0], np.float32)), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.ge(np.array([2.0, 2.0, 2.0], np.float32)), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )


def test_relops_numpy_array_float64(left: pd.Series[float]) -> None:
    check(
        assert_type(left < np.array([2.0, 2.0, 2.0], np.float64), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left <= np.array([2.0, 2.0, 2.0], np.float64), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left > np.array([2.0, 2.0, 2.0], np.float64), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left >= np.array([2.0, 2.0, 2.0], np.float64), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    # NumPy infers ndarray, while pandas dispatch returns Series.
    check(
        assert_type(np.array([2.0, 2.0, 2.0], np.float64) < left, np_ndarray_bool),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(np.array([2.0, 2.0, 2.0], np.float64) <= left, np_ndarray_bool),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(np.array([2.0, 2.0, 2.0], np.float64) > left, np_ndarray_bool),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(np.array([2.0, 2.0, 2.0], np.float64) >= left, np_ndarray_bool),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.lt(np.array([2.0, 2.0, 2.0], np.float64)), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.le(np.array([2.0, 2.0, 2.0], np.float64)), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.gt(np.array([2.0, 2.0, 2.0], np.float64)), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.ge(np.array([2.0, 2.0, 2.0], np.float64)), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )


def test_relops_numpy_array_int64(left: pd.Series[float]) -> None:
    check(
        assert_type(left < np.array([2, 2, 2], np.int64), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left <= np.array([2, 2, 2], np.int64), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left > np.array([2, 2, 2], np.int64), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left >= np.array([2, 2, 2], np.int64), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    # NumPy infers ndarray, while pandas dispatch returns Series.
    check(
        assert_type(np.array([2, 2, 2], np.int64) < left, np_ndarray_bool),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(np.array([2, 2, 2], np.int64) <= left, np_ndarray_bool),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(np.array([2, 2, 2], np.int64) > left, np_ndarray_bool),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(np.array([2, 2, 2], np.int64) >= left, np_ndarray_bool),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.lt(np.array([2, 2, 2], np.int64)), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.le(np.array([2, 2, 2], np.int64)), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.gt(np.array([2, 2, 2], np.int64)), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.ge(np.array([2, 2, 2], np.int64)), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )


def test_relops_numpy_array_bool(left: pd.Series[float]) -> None:
    check(
        assert_type(left < np.array([True, False, True], np.bool_), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left <= np.array([True, False, True], np.bool_), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left > np.array([True, False, True], np.bool_), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left >= np.array([True, False, True], np.bool_), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    # NumPy infers ndarray, while pandas dispatch returns Series.
    check(
        assert_type(np.array([True, False, True], np.bool_) < left, np_ndarray_bool),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(np.array([True, False, True], np.bool_) <= left, np_ndarray_bool),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(np.array([True, False, True], np.bool_) > left, np_ndarray_bool),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(np.array([True, False, True], np.bool_) >= left, np_ndarray_bool),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(
            left.lt(np.array([True, False, True], np.bool_)), "pd.Series[bool]"
        ),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(
            left.le(np.array([True, False, True], np.bool_)), "pd.Series[bool]"
        ),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(
            left.gt(np.array([True, False, True], np.bool_)), "pd.Series[bool]"
        ),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(
            left.ge(np.array([True, False, True], np.bool_)), "pd.Series[bool]"
        ),
        pd.Series,
        np.bool_,
    )


def test_relops_mixed_numeric_index_int(left: pd.Series[float]) -> None:
    check(
        assert_type(left < pd.Index([2, 2, 2]), "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type(left <= pd.Index([2, 2, 2]), "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type(left > pd.Index([2, 2, 2]), "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type(left >= pd.Index([2, 2, 2]), "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type(pd.Index([2, 2, 2]) < left, "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type(pd.Index([2, 2, 2]) <= left, "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type(pd.Index([2, 2, 2]) > left, "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type(pd.Index([2, 2, 2]) >= left, "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type(left.lt(pd.Index([2, 2, 2])), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.le(pd.Index([2, 2, 2])), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.gt(pd.Index([2, 2, 2])), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.ge(pd.Index([2, 2, 2])), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )


def test_relops_mixed_numeric_index_bool(left: pd.Series[float]) -> None:
    check(
        assert_type(left < pd.Index([True, False, True]), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left <= pd.Index([True, False, True]), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left > pd.Index([True, False, True]), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left >= pd.Index([True, False, True]), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(pd.Index([True, False, True]) < left, "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(pd.Index([True, False, True]) <= left, "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(pd.Index([True, False, True]) > left, "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(pd.Index([True, False, True]) >= left, "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.lt(pd.Index([True, False, True])), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.le(pd.Index([True, False, True])), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.gt(pd.Index([True, False, True])), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.ge(pd.Index([True, False, True])), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )


def test_relops_mixed_numeric_index_range(left: pd.Series[float]) -> None:
    check(assert_type(left < pd.RangeIndex(3), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= pd.RangeIndex(3), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > pd.RangeIndex(3), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= pd.RangeIndex(3), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.RangeIndex(3) < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.RangeIndex(3) <= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.RangeIndex(3) > left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.RangeIndex(3) >= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(
        assert_type(left.lt(pd.RangeIndex(3)), "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type(left.le(pd.RangeIndex(3)), "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type(left.gt(pd.RangeIndex(3)), "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type(left.ge(pd.RangeIndex(3)), "pd.Series[bool]"), pd.Series, np.bool_
    )


def test_relops_bool_series(left: pd.Series[float]) -> None:
    check(
        assert_type(left < pd.Series([True, False, True]), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left <= pd.Series([True, False, True]), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left > pd.Series([True, False, True]), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left >= pd.Series([True, False, True]), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(pd.Series([True, False, True]) < left, "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(pd.Series([True, False, True]) <= left, "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(pd.Series([True, False, True]) > left, "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(pd.Series([True, False, True]) >= left, "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.lt(pd.Series([True, False, True])), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.le(pd.Series([True, False, True])), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.gt(pd.Series([True, False, True])), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(left.ge(pd.Series([True, False, True])), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )

    check(
        assert_type(pd.Series([True, False, True]).lt(left), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )

    check(
        assert_type(pd.Series([True, False, True]).le(left), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )

    check(
        assert_type(pd.Series([True, False, True]).gt(left), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )

    check(
        assert_type(pd.Series([True, False, True]).ge(left), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )


def test_relops_mixed_numeric_series(left: pd.Series[float]) -> None:
    other = pd.Series([2, 2, 2])
    check(assert_type(left.lt(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(other), "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left < other, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left <= other, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left > other, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left >= other, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(other < left, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(other <= left, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(other > left, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(other >= left, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(other.lt(left), "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(other.le(left), "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(other.gt(left), "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(other.ge(left), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_invalid_str_scalar(left: pd.Series[float]) -> None:
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < "x"  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= "x"  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > "x"  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= "x"  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = "x" < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = "x" <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = "x" > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = "x" >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        left.lt("x")  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le("x")  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt("x")  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge("x")  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_temporal_scalar_timestamp(left: pd.Series[float]) -> None:
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < pd.Timestamp("2020-01-01")  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= pd.Timestamp("2020-01-01")  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > pd.Timestamp("2020-01-01")  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= pd.Timestamp("2020-01-01")  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = pd.Timestamp("2020-01-01") < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = pd.Timestamp("2020-01-01") <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = pd.Timestamp("2020-01-01") > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = pd.Timestamp("2020-01-01") >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        left.lt(pd.Timestamp("2020-01-01"))  # type: ignore[call-overload] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(pd.Timestamp("2020-01-01"))  # type: ignore[call-overload] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(pd.Timestamp("2020-01-01"))  # type: ignore[call-overload] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(pd.Timestamp("2020-01-01"))  # type: ignore[call-overload] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_temporal_scalar_timedelta(left: pd.Series[float]) -> None:
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < pd.Timedelta("1 day")  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= pd.Timedelta("1 day")  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > pd.Timedelta("1 day")  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= pd.Timedelta("1 day")  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = pd.Timedelta("1 day") < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = pd.Timedelta("1 day") <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = pd.Timedelta("1 day") > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = pd.Timedelta("1 day") >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        left.lt(pd.Timedelta("1 day"))  # type: ignore[call-overload] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(pd.Timedelta("1 day"))  # type: ignore[call-overload] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(pd.Timedelta("1 day"))  # type: ignore[call-overload] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(pd.Timedelta("1 day"))  # type: ignore[call-overload] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_str_sequence(left: pd.Series[float]) -> None:
    other: list[str] = ["x", "y", "z"]
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


def test_relops_invalid_incompatible_index_str(left: pd.Series[float]) -> None:
    other: pd.Index[str] = pd.Index(["x", "y", "z"])
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


def test_relops_invalid_incompatible_index_timestamp(left: pd.Series[float]) -> None:
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < pd.date_range("2020-01-01", periods=3)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= pd.date_range("2020-01-01", periods=3)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > pd.date_range("2020-01-01", periods=3)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= pd.date_range("2020-01-01", periods=3)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = pd.date_range("2020-01-01", periods=3) < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = pd.date_range("2020-01-01", periods=3) <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = pd.date_range("2020-01-01", periods=3) > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = pd.date_range("2020-01-01", periods=3) >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        left.lt(pd.date_range("2020-01-01", periods=3))  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(pd.date_range("2020-01-01", periods=3))  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(pd.date_range("2020-01-01", periods=3))  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(pd.date_range("2020-01-01", periods=3))  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_incompatible_index_timedelta(left: pd.Series[float]) -> None:
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < pd.timedelta_range("1 day", periods=3)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= pd.timedelta_range("1 day", periods=3)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > pd.timedelta_range("1 day", periods=3)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= pd.timedelta_range("1 day", periods=3)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = pd.timedelta_range("1 day", periods=3) < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = pd.timedelta_range("1 day", periods=3) <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = pd.timedelta_range("1 day", periods=3) > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = pd.timedelta_range("1 day", periods=3) >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        left.lt(pd.timedelta_range("1 day", periods=3))  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(pd.timedelta_range("1 day", periods=3))  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(pd.timedelta_range("1 day", periods=3))  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(pd.timedelta_range("1 day", periods=3))  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_incompatible_series_str(left: pd.Series[float]) -> None:
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < pd.Series(["x", "y", "z"])  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= pd.Series(["x", "y", "z"])  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > pd.Series(["x", "y", "z"])  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= pd.Series(["x", "y", "z"])  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = pd.Series(["x", "y", "z"]) < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = pd.Series(["x", "y", "z"]) <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = pd.Series(["x", "y", "z"]) > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = pd.Series(["x", "y", "z"]) >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        left.lt(pd.Series(["x", "y", "z"]))  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(pd.Series(["x", "y", "z"]))  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(pd.Series(["x", "y", "z"]))  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(pd.Series(["x", "y", "z"]))  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        pd.Series(["x", "y", "z"]).lt(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        pd.Series(["x", "y", "z"]).le(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        pd.Series(["x", "y", "z"]).gt(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        pd.Series(["x", "y", "z"]).ge(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_incompatible_series_timestamp(left: pd.Series[float]) -> None:
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < pd.Series(pd.date_range("2020-01-01", periods=3))  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= pd.Series(pd.date_range("2020-01-01", periods=3))  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > pd.Series(pd.date_range("2020-01-01", periods=3))  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= pd.Series(pd.date_range("2020-01-01", periods=3))  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = pd.Series(pd.date_range("2020-01-01", periods=3)) < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = pd.Series(pd.date_range("2020-01-01", periods=3)) <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = pd.Series(pd.date_range("2020-01-01", periods=3)) > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = pd.Series(pd.date_range("2020-01-01", periods=3)) >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        left.lt(pd.Series(pd.date_range("2020-01-01", periods=3)))  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(pd.Series(pd.date_range("2020-01-01", periods=3)))  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(pd.Series(pd.date_range("2020-01-01", periods=3)))  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(pd.Series(pd.date_range("2020-01-01", periods=3)))  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        pd.Series(pd.date_range("2020-01-01", periods=3)).lt(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        pd.Series(pd.date_range("2020-01-01", periods=3)).le(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        pd.Series(pd.date_range("2020-01-01", periods=3)).gt(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        pd.Series(pd.date_range("2020-01-01", periods=3)).ge(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_incompatible_series_timedelta(left: pd.Series[float]) -> None:
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < pd.Series(pd.timedelta_range("1 day", periods=3))  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= pd.Series(pd.timedelta_range("1 day", periods=3))  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > pd.Series(pd.timedelta_range("1 day", periods=3))  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= pd.Series(pd.timedelta_range("1 day", periods=3))  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = pd.Series(pd.timedelta_range("1 day", periods=3)) < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = pd.Series(pd.timedelta_range("1 day", periods=3)) <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = pd.Series(pd.timedelta_range("1 day", periods=3)) > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = pd.Series(pd.timedelta_range("1 day", periods=3)) >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        left.lt(pd.Series(pd.timedelta_range("1 day", periods=3)))  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(pd.Series(pd.timedelta_range("1 day", periods=3)))  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(pd.Series(pd.timedelta_range("1 day", periods=3)))  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(pd.Series(pd.timedelta_range("1 day", periods=3)))  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        pd.Series(pd.timedelta_range("1 day", periods=3)).lt(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        pd.Series(pd.timedelta_range("1 day", periods=3)).le(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        pd.Series(pd.timedelta_range("1 day", periods=3)).gt(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        pd.Series(pd.timedelta_range("1 day", periods=3)).ge(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_incompatible_numpy_array_str(left: pd.Series[float]) -> None:
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


def test_relops_invalid_incompatible_numpy_array_datetime64(
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


def test_relops_invalid_incompatible_numpy_array_timedelta64(
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


def test_relops_float_index_int_series() -> None:
    left = pd.Index([1.0, 2.0, 3.0])
    other = pd.Series([2, 2, 2])
    check(assert_type(left < other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= other, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(other < left, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(other <= left, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(other > left, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(other >= left, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(other.lt(left), "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(other.le(left), "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(other.gt(left), "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(other.ge(left), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_float_index_bool_series() -> None:
    left = pd.Index([1.0, 2.0, 3.0])
    other = pd.Series([True, False, True])
    check(assert_type(left < other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(other < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(other <= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(other > left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(other >= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(other.lt(left), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(other.le(left), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(other.gt(left), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(other.ge(left), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_invalid_float_index_temporal_series_timestamp() -> None:
    left = pd.Index([1.0, 2.0, 3.0])
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < pd.Series(pd.date_range("2020-01-01", periods=3))  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= pd.Series(pd.date_range("2020-01-01", periods=3))  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > pd.Series(pd.date_range("2020-01-01", periods=3))  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= pd.Series(pd.date_range("2020-01-01", periods=3))  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = pd.Series(pd.date_range("2020-01-01", periods=3)) < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = pd.Series(pd.date_range("2020-01-01", periods=3)) <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = pd.Series(pd.date_range("2020-01-01", periods=3)) > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = pd.Series(pd.date_range("2020-01-01", periods=3)) >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        pd.Series(pd.date_range("2020-01-01", periods=3)).lt(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        pd.Series(pd.date_range("2020-01-01", periods=3)).le(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        pd.Series(pd.date_range("2020-01-01", periods=3)).gt(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        pd.Series(pd.date_range("2020-01-01", periods=3)).ge(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_float_index_temporal_series_timedelta() -> None:
    left = pd.Index([1.0, 2.0, 3.0])
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < pd.Series(pd.timedelta_range("1 day", periods=3))  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= pd.Series(pd.timedelta_range("1 day", periods=3))  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > pd.Series(pd.timedelta_range("1 day", periods=3))  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= pd.Series(pd.timedelta_range("1 day", periods=3))  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = pd.Series(pd.timedelta_range("1 day", periods=3)) < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = pd.Series(pd.timedelta_range("1 day", periods=3)) <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = pd.Series(pd.timedelta_range("1 day", periods=3)) > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = pd.Series(pd.timedelta_range("1 day", periods=3)) >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        pd.Series(pd.timedelta_range("1 day", periods=3)).lt(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        pd.Series(pd.timedelta_range("1 day", periods=3)).le(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        pd.Series(pd.timedelta_range("1 day", periods=3)).gt(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        pd.Series(pd.timedelta_range("1 day", periods=3)).ge(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
