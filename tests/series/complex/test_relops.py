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
def left() -> pd.Series[complex]:
    """Left operand"""
    lo = pd.Series([1j, 2j, 4j])
    return check(assert_type(lo, "pd.Series[complex]"), pd.Series, np.complexfloating)


def test_relops_equality(left: pd.Series[complex]) -> None:
    """Test valid equality comparisons for pd.Series[complex]"""
    check(assert_type(left == 1j, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left != 1j, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.eq(left), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ne(1.0), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_invalid_py_scalar(left: pd.Series[complex]) -> None:
    """Test invalid pd.Series[complex] comparisons with py scalars"""
    if TYPE_CHECKING_INVALID_USAGE:
        _lt_f = left < 1.0  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_f = left <= 1.0  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_f = left > 1.0  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_f = left >= 1.0  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]

        _lt_c = left < 1j  # type: ignore[misc] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_c = left <= 1j  # type: ignore[misc] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_c = left > 1j  # type: ignore[misc] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_c = left >= 1j  # type: ignore[misc] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]

        _f_lt = 1.0 < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _f_le = 1.0 <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _f_gt = 1.0 > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _f_ge = 1.0 >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]

        left.lt(1.0)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(1.0)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(1.0)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(1.0)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]

        left.lt(1j)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(1j)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(1j)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(1j)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_py_sequence(left: pd.Series[complex]) -> None:
    """Test invalid pd.Series[complex] comparisons with py sequences"""
    real_values = [1.0, 2.0]
    complex_values = [1j, 2j]

    if TYPE_CHECKING_INVALID_USAGE:
        _lt_rseq = left < real_values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_rseq = left <= real_values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_rseq = left > real_values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_rseq = left >= real_values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]

        _lt_cseq = left < complex_values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_cseq = left <= complex_values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_cseq = left > complex_values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_cseq = left >= complex_values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]

        left.lt(real_values)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(real_values)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(real_values)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(real_values)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]

        left.lt(complex_values)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(complex_values)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(complex_values)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(complex_values)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_numpy_array(left: pd.Series[complex]) -> None:
    """Test invalid pd.Series[complex] comparisons with numpy arrays"""
    real_array = np.array([1.0, 2.0], np.float64)
    complex_array = np.array([1j, 2j], np.complex128)

    if TYPE_CHECKING_INVALID_USAGE:
        left.__lt__(real_array)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__le__(real_array)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__gt__(real_array)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__ge__(real_array)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]

        left.lt(real_array)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(real_array)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(real_array)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(real_array)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]

        left.__lt__(complex_array)  # type: ignore[misc,operator] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__le__(complex_array)  # type: ignore[misc,operator] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__gt__(complex_array)  # type: ignore[misc,operator] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__ge__(complex_array)  # type: ignore[misc,operator] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]

        left.lt(complex_array)  # type: ignore[arg-type,misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(complex_array)  # type: ignore[arg-type,misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(complex_array)  # type: ignore[arg-type,misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(complex_array)  # type: ignore[arg-type,misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_pd_index(left: pd.Series[complex]) -> None:
    """Test invalid pd.Series[complex] comparisons with pandas Indexes"""
    real_index = pd.Index([1.0, 2.0])
    complex_index = pd.Index([1j, 2j])

    if TYPE_CHECKING_INVALID_USAGE:
        _lt_ridx = left < real_index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_ridx = left <= real_index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_ridx = left > real_index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_ridx = left >= real_index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]

        _lt_cidx = left < complex_index  # type: ignore[misc] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_cidx = left <= complex_index  # type: ignore[misc] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_cidx = left > complex_index  # type: ignore[misc] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_cidx = left >= complex_index  # type: ignore[misc] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]

        _ridx_lt = real_index < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ridx_le = real_index <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ridx_gt = real_index > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ridx_ge = real_index >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]

        left.lt(real_index)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(real_index)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(real_index)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(real_index)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]

        left.lt(complex_index)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(complex_index)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(complex_index)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(complex_index)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_invalid_pd_series(left: pd.Series[complex]) -> None:
    """Test invalid pd.Series[complex] comparisons with pandas Series"""
    real_series = pd.Series([1.0, 2.0])
    complex_series = pd.Series([1j, 2j])

    if TYPE_CHECKING_INVALID_USAGE:
        _lt_rser = left < real_series  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_rser = left <= real_series  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_rser = left > real_series  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_rser = left >= real_series  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]

        _lt_cser = left < complex_series  # type: ignore[misc] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_cser = left <= complex_series  # type: ignore[misc] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_cser = left > complex_series  # type: ignore[misc] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_cser = left >= complex_series  # type: ignore[misc] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]

        _rser_lt = real_series < left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _rser_le = real_series <= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _rser_gt = real_series > left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _rser_ge = real_series >= left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]

        left.lt(real_series)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(real_series)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(real_series)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(real_series)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]

        left.lt(complex_series)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(complex_series)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(complex_series)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(complex_series)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]

        real_series.lt(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        real_series.le(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        real_series.gt(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        real_series.ge(left)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
