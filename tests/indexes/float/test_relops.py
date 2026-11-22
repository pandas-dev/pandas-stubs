from __future__ import annotations

from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
    pytest_warns_listlike_pandas4,
)
from tests._typing import (
    np_1darray_bool,
    np_ndarray_bool,
    np_ndarray_complex,
    np_ndarray_dt,
    np_ndarray_str,
    np_ndarray_td,
)


@pytest.fixture
def left() -> pd.Index[float]:
    """Left operand"""
    lo = pd.Index([1.0, 2.0, 3.0])
    return check(assert_type(lo, "pd.Index[float]"), pd.Index, np.floating)


def test_relops_invalid_py_scalar(left: pd.Index[float]) -> None:
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_c = left < 1j  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_c = left <= 1j  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_c = left > 1j  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_c = left >= 1j  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]

        _c_lt = 1j < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _c_le = 1j <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _c_gt = 1j > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _c_ge = 1j >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_relops_py_sequence_float(left: pd.Index[float]) -> None:
    values = [2.0, 2.0, 2.0]
    check(assert_type(left < values, np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= values, np_1darray_bool), np_1darray_bool)
    check(assert_type(left > values, np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= values, np_1darray_bool), np_1darray_bool)
    check(assert_type(values < left, np_1darray_bool), np_1darray_bool)
    check(assert_type(values <= left, np_1darray_bool), np_1darray_bool)
    check(assert_type(values > left, np_1darray_bool), np_1darray_bool)
    check(assert_type(values >= left, np_1darray_bool), np_1darray_bool)


def test_relops_py_sequence_int(left: pd.Index[float]) -> None:
    values = [2, 2, 2]
    check(assert_type(left < values, np_1darray_bool), np_1darray_bool)
    check(assert_type(values < left, np_1darray_bool), np_1darray_bool)


def test_relops_py_sequence_bool(left: pd.Index[float]) -> None:
    values = [True, False, True]
    check(assert_type(left < values, np_1darray_bool), np_1darray_bool)
    check(assert_type(values < left, np_1darray_bool), np_1darray_bool)


def test_relops_py_sequence_tuple(left: pd.Index[float]) -> None:
    values = (2.0, 2.0, 2.0)
    with pytest_warns_listlike_pandas4("tuple"):
        check(assert_type(left < values, np_1darray_bool), np_1darray_bool)
        check(assert_type(values < left, np_1darray_bool), np_1darray_bool)


def test_relops_pd_index_int(left: pd.Index[float]) -> None:
    index = pd.Index([2, 2, 2])
    check(assert_type(left < index, np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= index, np_1darray_bool), np_1darray_bool)
    check(assert_type(left > index, np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= index, np_1darray_bool), np_1darray_bool)
    check(assert_type(index < left, np_1darray_bool), np_1darray_bool)
    check(assert_type(index <= left, np_1darray_bool), np_1darray_bool)
    check(assert_type(index > left, np_1darray_bool), np_1darray_bool)
    check(assert_type(index >= left, np_1darray_bool), np_1darray_bool)


def test_relops_pd_index_bool(left: pd.Index[float]) -> None:
    index = pd.Index([True, False, True])
    check(assert_type(left < index, np_1darray_bool), np_1darray_bool)
    check(assert_type(index < left, np_1darray_bool), np_1darray_bool)


def test_relops_range_index(left: pd.Index[float]) -> None:
    index = pd.RangeIndex(3)
    check(assert_type(left < index, np_1darray_bool), np_1darray_bool)
    check(assert_type(index < left, np_1darray_bool), np_1darray_bool)


def test_relops_py_scalar_float(left: pd.Index[float]) -> None:
    other = 2.0
    check(assert_type(left < other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left > other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= other, np_1darray_bool), np_1darray_bool)
    check(assert_type(other < left, np_1darray_bool), np_1darray_bool)
    check(assert_type(other <= left, np_1darray_bool), np_1darray_bool)
    check(assert_type(other > left, np_1darray_bool), np_1darray_bool)
    check(assert_type(other >= left, np_1darray_bool), np_1darray_bool)


def test_relops_py_scalar_int(left: pd.Index[float]) -> None:
    other = 2
    check(assert_type(left < other, np_1darray_bool), np_1darray_bool)
    check(assert_type(other < left, np_1darray_bool), np_1darray_bool)


def test_relops_py_scalar_bool(left: pd.Index[float]) -> None:
    other = True
    check(assert_type(left < other, np_1darray_bool), np_1darray_bool)
    check(assert_type(other < left, np_1darray_bool), np_1darray_bool)


def test_relops_numpy_scalar_bool(left: pd.Index[float]) -> None:
    other = np.bool_(True)
    check(assert_type(left < other, np_1darray_bool), np_1darray_bool)
    check(assert_type(other < left, np_ndarray_bool), np_ndarray_bool)


def test_relops_numpy_scalar_int64(left: pd.Index[float]) -> None:
    other = np.int64(2)
    check(assert_type(left < other, np_1darray_bool), np_1darray_bool)
    check(assert_type(other < left, np_ndarray_bool), np_ndarray_bool)


def test_relops_numpy_scalar_float32(left: pd.Index[float]) -> None:
    other = np.float32(2)
    check(assert_type(left < other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left > other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= other, np_1darray_bool), np_1darray_bool)
    check(assert_type(other < left, np_ndarray_bool), np_ndarray_bool)
    check(assert_type(other <= left, np_ndarray_bool), np_ndarray_bool)
    check(assert_type(other > left, np_ndarray_bool), np_ndarray_bool)
    check(assert_type(other >= left, np_ndarray_bool), np_ndarray_bool)


def test_relops_numpy_array_float32(left: pd.Index[float]) -> None:
    array = np.array([2.0, 2.0, 2.0], np.float32)
    check(assert_type(left < array, np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= array, np_1darray_bool), np_1darray_bool)
    check(assert_type(left > array, np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= array, np_1darray_bool), np_1darray_bool)
    check(assert_type(array < left, np_ndarray_bool), np_ndarray_bool)
    check(assert_type(array <= left, np_ndarray_bool), np_ndarray_bool)
    check(assert_type(array > left, np_ndarray_bool), np_ndarray_bool)
    check(assert_type(array >= left, np_ndarray_bool), np_ndarray_bool)


def test_relops_numpy_array_int64(left: pd.Index[float]) -> None:
    array = np.array([2, 2, 2], np.int64)
    check(assert_type(left < array, np_1darray_bool), np_1darray_bool)
    check(assert_type(array < left, np_ndarray_bool), np_ndarray_bool)


def test_relops_numpy_array_bool(left: pd.Index[float]) -> None:
    array = np.array([True, False, True], np.bool_)
    check(assert_type(left < array, np_1darray_bool), np_1darray_bool)
    check(assert_type(array < left, np_ndarray_bool), np_ndarray_bool)


def test_relops_pd_index_float(left: pd.Index[float]) -> None:
    index = pd.Index([2.0, 2.0, 2.0])
    check(assert_type(left < index, np_1darray_bool), np_1darray_bool)
    check(assert_type(index < left, np_1darray_bool), np_1darray_bool)


def test_relops_invalid_py_scalar_str(left: pd.Index[float]) -> None:
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


def test_relops_invalid_timestamp(left: pd.Index[float]) -> None:
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


def test_relops_invalid_timedelta(left: pd.Index[float]) -> None:
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


def test_relops_invalid_py_sequence_str(left: pd.Index[float]) -> None:
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


def test_relops_invalid_py_sequence_complex(left: pd.Index[float]) -> None:
    values: list[complex] = [1j, 2j, 3j]
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = values < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = values <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = values > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = values >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_relops_invalid_pd_index_str(left: pd.Index[float]) -> None:
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


def test_relops_invalid_datetime_index(left: pd.Index[float]) -> None:
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


def test_relops_invalid_timedelta_index(left: pd.Index[float]) -> None:
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


def test_relops_invalid_numpy_array_str(left: pd.Index[float]) -> None:
    other: np_ndarray_str = np.array(["x", "y", "z"], np.str_)
    if TYPE_CHECKING_INVALID_USAGE:
        left.__lt__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__le__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__gt__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__ge__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_numpy_array_complex(left: pd.Index[float]) -> None:
    other: np_ndarray_complex = np.array([1j, 2j, 3j], np.complex128)
    if TYPE_CHECKING_INVALID_USAGE:
        left.__lt__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__le__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__gt__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__ge__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_numpy_array_datetime64(left: pd.Index[float]) -> None:
    other: np_ndarray_dt = np.array(
        [np.datetime64("2020-01-01", "ns")] * 3, np.datetime64
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.__lt__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__le__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__gt__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__ge__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_numpy_array_timedelta64(left: pd.Index[float]) -> None:
    other: np_ndarray_td = np.array([np.timedelta64(1, "ns")] * 3, np.timedelta64)
    if TYPE_CHECKING_INVALID_USAGE:
        left.__lt__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__le__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__gt__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__ge__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
