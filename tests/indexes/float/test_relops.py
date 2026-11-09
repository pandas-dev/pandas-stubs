from __future__ import annotations

from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from pandas.errors import Pandas4Warning

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
    pytest_warns_conditioned,
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


def test_relops_numeric_sequence_index_float_list(left: pd.Index[float]) -> None:
    check(assert_type(left < [2.0, 2.0, 2.0], np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= [2.0, 2.0, 2.0], np_1darray_bool), np_1darray_bool)
    check(assert_type(left > [2.0, 2.0, 2.0], np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= [2.0, 2.0, 2.0], np_1darray_bool), np_1darray_bool)
    check(assert_type([2.0, 2.0, 2.0] < left, np_1darray_bool), np_1darray_bool)
    check(assert_type([2.0, 2.0, 2.0] <= left, np_1darray_bool), np_1darray_bool)
    check(assert_type([2.0, 2.0, 2.0] > left, np_1darray_bool), np_1darray_bool)
    check(assert_type([2.0, 2.0, 2.0] >= left, np_1darray_bool), np_1darray_bool)


def test_relops_numeric_sequence_index_int_list(left: pd.Index[float]) -> None:
    check(assert_type(left < [2, 2, 2], np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= [2, 2, 2], np_1darray_bool), np_1darray_bool)
    check(assert_type(left > [2, 2, 2], np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= [2, 2, 2], np_1darray_bool), np_1darray_bool)
    check(assert_type([2, 2, 2] < left, np_1darray_bool), np_1darray_bool)
    check(assert_type([2, 2, 2] <= left, np_1darray_bool), np_1darray_bool)
    check(assert_type([2, 2, 2] > left, np_1darray_bool), np_1darray_bool)
    check(assert_type([2, 2, 2] >= left, np_1darray_bool), np_1darray_bool)


def test_relops_numeric_sequence_index_bool_list(left: pd.Index[float]) -> None:
    check(assert_type(left < [True, False, True], np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= [True, False, True], np_1darray_bool), np_1darray_bool)
    check(assert_type(left > [True, False, True], np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= [True, False, True], np_1darray_bool), np_1darray_bool)
    check(assert_type([True, False, True] < left, np_1darray_bool), np_1darray_bool)
    check(assert_type([True, False, True] <= left, np_1darray_bool), np_1darray_bool)
    check(assert_type([True, False, True] > left, np_1darray_bool), np_1darray_bool)
    check(assert_type([True, False, True] >= left, np_1darray_bool), np_1darray_bool)


def test_relops_numeric_sequence_index_tuple(left: pd.Index[float]) -> None:
    with pytest_warns_conditioned(
        Pandas4Warning, "Operation with tuple is deprecated", True
    ):
        check(assert_type(left < (2.0, 2.0, 2.0), np_1darray_bool), np_1darray_bool)
        check(assert_type(left <= (2.0, 2.0, 2.0), np_1darray_bool), np_1darray_bool)
        check(assert_type(left > (2.0, 2.0, 2.0), np_1darray_bool), np_1darray_bool)
        check(assert_type(left >= (2.0, 2.0, 2.0), np_1darray_bool), np_1darray_bool)
        check(assert_type((2.0, 2.0, 2.0) < left, np_1darray_bool), np_1darray_bool)
        check(assert_type((2.0, 2.0, 2.0) <= left, np_1darray_bool), np_1darray_bool)
        check(assert_type((2.0, 2.0, 2.0) > left, np_1darray_bool), np_1darray_bool)
        check(assert_type((2.0, 2.0, 2.0) >= left, np_1darray_bool), np_1darray_bool)


def test_relops_numeric_sequence_index_int_index(left: pd.Index[float]) -> None:
    check(assert_type(left < pd.Index([2, 2, 2]), np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= pd.Index([2, 2, 2]), np_1darray_bool), np_1darray_bool)
    check(assert_type(left > pd.Index([2, 2, 2]), np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= pd.Index([2, 2, 2]), np_1darray_bool), np_1darray_bool)
    check(assert_type(pd.Index([2, 2, 2]) < left, np_1darray_bool), np_1darray_bool)
    check(assert_type(pd.Index([2, 2, 2]) <= left, np_1darray_bool), np_1darray_bool)
    check(assert_type(pd.Index([2, 2, 2]) > left, np_1darray_bool), np_1darray_bool)
    check(assert_type(pd.Index([2, 2, 2]) >= left, np_1darray_bool), np_1darray_bool)


def test_relops_numeric_sequence_index_bool_index(left: pd.Index[float]) -> None:
    check(
        assert_type(left < pd.Index([True, False, True]), np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(left <= pd.Index([True, False, True]), np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(left > pd.Index([True, False, True]), np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(left >= pd.Index([True, False, True]), np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(pd.Index([True, False, True]) < left, np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(pd.Index([True, False, True]) <= left, np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(pd.Index([True, False, True]) > left, np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(pd.Index([True, False, True]) >= left, np_1darray_bool),
        np_1darray_bool,
    )


def test_relops_numeric_sequence_index_range(left: pd.Index[float]) -> None:
    check(assert_type(left < pd.RangeIndex(3), np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= pd.RangeIndex(3), np_1darray_bool), np_1darray_bool)
    check(assert_type(left > pd.RangeIndex(3), np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= pd.RangeIndex(3), np_1darray_bool), np_1darray_bool)
    check(assert_type(pd.RangeIndex(3) < left, np_1darray_bool), np_1darray_bool)
    check(assert_type(pd.RangeIndex(3) <= left, np_1darray_bool), np_1darray_bool)
    check(assert_type(pd.RangeIndex(3) > left, np_1darray_bool), np_1darray_bool)
    check(assert_type(pd.RangeIndex(3) >= left, np_1darray_bool), np_1darray_bool)


def test_relops_py_scalar_float(left: pd.Index[float]) -> None:
    check(assert_type(left < 2.0, np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= 2.0, np_1darray_bool), np_1darray_bool)
    check(assert_type(left > 2.0, np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= 2.0, np_1darray_bool), np_1darray_bool)
    check(assert_type(2.0 < left, np_1darray_bool), np_1darray_bool)
    check(assert_type(2.0 <= left, np_1darray_bool), np_1darray_bool)
    check(assert_type(2.0 > left, np_1darray_bool), np_1darray_bool)
    check(assert_type(2.0 >= left, np_1darray_bool), np_1darray_bool)


def test_relops_py_scalar_int(left: pd.Index[float]) -> None:
    check(assert_type(left < 2, np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= 2, np_1darray_bool), np_1darray_bool)
    check(assert_type(left > 2, np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= 2, np_1darray_bool), np_1darray_bool)
    check(assert_type(2 < left, np_1darray_bool), np_1darray_bool)
    check(assert_type(2 <= left, np_1darray_bool), np_1darray_bool)
    check(assert_type(2 > left, np_1darray_bool), np_1darray_bool)
    check(assert_type(2 >= left, np_1darray_bool), np_1darray_bool)


def test_relops_py_scalar_bool(left: pd.Index[float]) -> None:
    check(assert_type(left < True, np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= True, np_1darray_bool), np_1darray_bool)
    check(assert_type(left > True, np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= True, np_1darray_bool), np_1darray_bool)
    check(assert_type(True < left, np_1darray_bool), np_1darray_bool)
    check(assert_type(True <= left, np_1darray_bool), np_1darray_bool)
    check(assert_type(True > left, np_1darray_bool), np_1darray_bool)
    check(assert_type(True >= left, np_1darray_bool), np_1darray_bool)


def test_relops_numpy_scalar_bool(left: pd.Index[float]) -> None:
    check(assert_type(left < np.bool_(True), np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= np.bool_(True), np_1darray_bool), np_1darray_bool)
    check(assert_type(left > np.bool_(True), np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= np.bool_(True), np_1darray_bool), np_1darray_bool)
    check(assert_type(np.bool_(True) < left, np_ndarray_bool), np_ndarray_bool)
    check(assert_type(np.bool_(True) <= left, np_ndarray_bool), np_ndarray_bool)
    check(assert_type(np.bool_(True) > left, np_ndarray_bool), np_ndarray_bool)
    check(assert_type(np.bool_(True) >= left, np_ndarray_bool), np_ndarray_bool)


def test_relops_numpy_scalar_int64(left: pd.Index[float]) -> None:
    check(assert_type(left < np.int64(2), np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= np.int64(2), np_1darray_bool), np_1darray_bool)
    check(assert_type(left > np.int64(2), np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= np.int64(2), np_1darray_bool), np_1darray_bool)
    check(assert_type(np.int64(2) < left, np_ndarray_bool), np_ndarray_bool)
    check(assert_type(np.int64(2) <= left, np_ndarray_bool), np_ndarray_bool)
    check(assert_type(np.int64(2) > left, np_ndarray_bool), np_ndarray_bool)
    check(assert_type(np.int64(2) >= left, np_ndarray_bool), np_ndarray_bool)


def test_relops_numpy_scalar_float32(left: pd.Index[float]) -> None:
    check(assert_type(left < np.float32(2), np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= np.float32(2), np_1darray_bool), np_1darray_bool)
    check(assert_type(left > np.float32(2), np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= np.float32(2), np_1darray_bool), np_1darray_bool)
    check(assert_type(np.float32(2) < left, np_ndarray_bool), np_ndarray_bool)
    check(assert_type(np.float32(2) <= left, np_ndarray_bool), np_ndarray_bool)
    check(assert_type(np.float32(2) > left, np_ndarray_bool), np_ndarray_bool)
    check(assert_type(np.float32(2) >= left, np_ndarray_bool), np_ndarray_bool)


def test_relops_numpy_scalar_float64(left: pd.Index[float]) -> None:
    check(assert_type(left < np.float64(2), np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= np.float64(2), np_1darray_bool), np_1darray_bool)
    check(assert_type(left > np.float64(2), np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= np.float64(2), np_1darray_bool), np_1darray_bool)
    check(assert_type(np.float64(2) < left, np_ndarray_bool), np_ndarray_bool)
    check(assert_type(np.float64(2) <= left, np_ndarray_bool), np_ndarray_bool)
    check(assert_type(np.float64(2) > left, np_ndarray_bool), np_ndarray_bool)
    check(assert_type(np.float64(2) >= left, np_ndarray_bool), np_ndarray_bool)


def test_relops_numpy_array_float32(left: pd.Index[float]) -> None:
    check(
        assert_type(left < np.array([2.0, 2.0, 2.0], np.float32), np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(left <= np.array([2.0, 2.0, 2.0], np.float32), np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(left > np.array([2.0, 2.0, 2.0], np.float32), np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(left >= np.array([2.0, 2.0, 2.0], np.float32), np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(np.array([2.0, 2.0, 2.0], np.float32) < left, np_ndarray_bool),
        np_ndarray_bool,
    )
    check(
        assert_type(np.array([2.0, 2.0, 2.0], np.float32) <= left, np_ndarray_bool),
        np_ndarray_bool,
    )
    check(
        assert_type(np.array([2.0, 2.0, 2.0], np.float32) > left, np_ndarray_bool),
        np_ndarray_bool,
    )
    check(
        assert_type(np.array([2.0, 2.0, 2.0], np.float32) >= left, np_ndarray_bool),
        np_ndarray_bool,
    )


def test_relops_numpy_array_float64(left: pd.Index[float]) -> None:
    check(
        assert_type(left < np.array([2.0, 2.0, 2.0], np.float64), np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(left <= np.array([2.0, 2.0, 2.0], np.float64), np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(left > np.array([2.0, 2.0, 2.0], np.float64), np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(left >= np.array([2.0, 2.0, 2.0], np.float64), np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(np.array([2.0, 2.0, 2.0], np.float64) < left, np_ndarray_bool),
        np_ndarray_bool,
    )
    check(
        assert_type(np.array([2.0, 2.0, 2.0], np.float64) <= left, np_ndarray_bool),
        np_ndarray_bool,
    )
    check(
        assert_type(np.array([2.0, 2.0, 2.0], np.float64) > left, np_ndarray_bool),
        np_ndarray_bool,
    )
    check(
        assert_type(np.array([2.0, 2.0, 2.0], np.float64) >= left, np_ndarray_bool),
        np_ndarray_bool,
    )


def test_relops_numpy_array_int64(left: pd.Index[float]) -> None:
    check(
        assert_type(left < np.array([2, 2, 2], np.int64), np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(left <= np.array([2, 2, 2], np.int64), np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(left > np.array([2, 2, 2], np.int64), np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(left >= np.array([2, 2, 2], np.int64), np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(np.array([2, 2, 2], np.int64) < left, np_ndarray_bool),
        np_ndarray_bool,
    )
    check(
        assert_type(np.array([2, 2, 2], np.int64) <= left, np_ndarray_bool),
        np_ndarray_bool,
    )
    check(
        assert_type(np.array([2, 2, 2], np.int64) > left, np_ndarray_bool),
        np_ndarray_bool,
    )
    check(
        assert_type(np.array([2, 2, 2], np.int64) >= left, np_ndarray_bool),
        np_ndarray_bool,
    )


def test_relops_numpy_array_bool(left: pd.Index[float]) -> None:
    check(
        assert_type(left < np.array([True, False, True], np.bool_), np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(left <= np.array([True, False, True], np.bool_), np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(left > np.array([True, False, True], np.bool_), np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(left >= np.array([True, False, True], np.bool_), np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(np.array([True, False, True], np.bool_) < left, np_ndarray_bool),
        np_ndarray_bool,
    )
    check(
        assert_type(np.array([True, False, True], np.bool_) <= left, np_ndarray_bool),
        np_ndarray_bool,
    )
    check(
        assert_type(np.array([True, False, True], np.bool_) > left, np_ndarray_bool),
        np_ndarray_bool,
    )
    check(
        assert_type(np.array([True, False, True], np.bool_) >= left, np_ndarray_bool),
        np_ndarray_bool,
    )


def test_relops_pd_index(left: pd.Index[float]) -> None:
    check(
        assert_type(left < pd.Index([2.0, 2.0, 2.0]), np_1darray_bool), np_1darray_bool
    )
    check(
        assert_type(left <= pd.Index([2.0, 2.0, 2.0]), np_1darray_bool), np_1darray_bool
    )
    check(
        assert_type(left > pd.Index([2.0, 2.0, 2.0]), np_1darray_bool), np_1darray_bool
    )
    check(
        assert_type(left >= pd.Index([2.0, 2.0, 2.0]), np_1darray_bool), np_1darray_bool
    )


def test_relops_invalid_incompatible_scalar_str(left: pd.Index[float]) -> None:
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < "x"  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= "x"  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > "x"  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= "x"  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = "x" < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = "x" <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = "x" > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = "x" >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_relops_invalid_incompatible_scalar_timestamp(left: pd.Index[float]) -> None:
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < pd.Timestamp("2020-01-01")  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= pd.Timestamp("2020-01-01")  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > pd.Timestamp("2020-01-01")  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= pd.Timestamp("2020-01-01")  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = pd.Timestamp("2020-01-01") < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = pd.Timestamp("2020-01-01") <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = pd.Timestamp("2020-01-01") > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = pd.Timestamp("2020-01-01") >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_relops_invalid_incompatible_scalar_timedelta(left: pd.Index[float]) -> None:
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < pd.Timedelta("1 day")  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= pd.Timedelta("1 day")  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > pd.Timedelta("1 day")  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= pd.Timedelta("1 day")  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = pd.Timedelta("1 day") < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = pd.Timedelta("1 day") <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = pd.Timedelta("1 day") > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = pd.Timedelta("1 day") >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_relops_invalid_py_sequence_str(left: pd.Index[float]) -> None:
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


def test_relops_invalid_py_sequence_complex(left: pd.Index[float]) -> None:
    other: list[complex] = [1j, 2j, 3j]
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= other  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = other < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = other <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = other > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = other >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_relops_invalid_pd_index_str(left: pd.Index[float]) -> None:
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


def test_relops_invalid_pd_index_complex(left: pd.Index[float]) -> None:
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < pd.Index([1j, 2j, 3j])  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= pd.Index([1j, 2j, 3j])  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > pd.Index([1j, 2j, 3j])  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= pd.Index([1j, 2j, 3j])  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = pd.Index([1j, 2j, 3j]) < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = pd.Index([1j, 2j, 3j]) <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = pd.Index([1j, 2j, 3j]) > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = pd.Index([1j, 2j, 3j]) >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_relops_invalid_pd_index_timestamp(left: pd.Index[float]) -> None:
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < pd.date_range("2020-01-01", periods=3)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= pd.date_range("2020-01-01", periods=3)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > pd.date_range("2020-01-01", periods=3)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= pd.date_range("2020-01-01", periods=3)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = pd.date_range("2020-01-01", periods=3) < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = pd.date_range("2020-01-01", periods=3) <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = pd.date_range("2020-01-01", periods=3) > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = pd.date_range("2020-01-01", periods=3) >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_relops_invalid_pd_index_timedelta(left: pd.Index[float]) -> None:
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_left = left < pd.timedelta_range("1 day", periods=3)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_left = left <= pd.timedelta_range("1 day", periods=3)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_left = left > pd.timedelta_range("1 day", periods=3)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_left = left >= pd.timedelta_range("1 day", periods=3)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_other = pd.timedelta_range("1 day", periods=3) < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_other = pd.timedelta_range("1 day", periods=3) <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_other = pd.timedelta_range("1 day", periods=3) > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_other = pd.timedelta_range("1 day", periods=3) >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_relops_invalid_numpy_array_complex(left: pd.Index[float]) -> None:
    other: np_ndarray_str = np.array(["x", "y", "z"], np.str_)
    if TYPE_CHECKING_INVALID_USAGE:
        left.__lt__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__le__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__gt__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__ge__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_numpy_array_str(left: pd.Index[float]) -> None:
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
