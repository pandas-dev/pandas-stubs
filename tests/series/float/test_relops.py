from __future__ import annotations

from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
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


def test_relops_invalid_pd_series(left: pd.Series[float]) -> None:
    complex_series = pd.Series([1j, 2j])

    if TYPE_CHECKING_INVALID_USAGE:
        left.lt(complex_series)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(complex_series)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(complex_series)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(complex_series)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
