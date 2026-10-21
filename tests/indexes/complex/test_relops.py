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
    np_1darray_bool,
    np_ndarray_complex,
    np_ndarray_float,
)


@pytest.fixture
def left() -> pd.Index[complex]:
    """Left operand"""
    lo = pd.Index([1j, 2j, 4j])
    return check(assert_type(lo, "pd.Index[complex]"), pd.Index, np.complexfloating)


def test_relops_equality(left: pd.Index[complex]) -> None:
    """Test valid equality comparisons for pd.Index[complex]"""
    check(assert_type(left == 1j, np_1darray_bool), np_1darray_bool)
    check(assert_type(left != 1j, np_1darray_bool), np_1darray_bool)
    check(assert_type(left == left, np_1darray_bool), np_1darray_bool)


def test_relops_invalid_py_scalar(left: pd.Index[complex]) -> None:
    """Test invalid pd.Index[complex] comparisons with py scalars"""
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
        _lt_self = left < left  # type: ignore[misc] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_self = left <= left  # type: ignore[misc] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_self = left > left  # type: ignore[misc] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_self = left >= left  # type: ignore[misc] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_relops_invalid_py_sequence(left: pd.Index[complex]) -> None:
    """Test invalid pd.Index[complex] comparisons with py sequences"""
    real_values: list[float] = [1.0, 2.0]
    complex_values: list[complex] = [1j, 2j]

    if TYPE_CHECKING_INVALID_USAGE:
        _lt_rseq = left < real_values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_rseq = left <= real_values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_rseq = left > real_values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_rseq = left >= real_values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _lt_cseq = left < complex_values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_cseq = left <= complex_values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_cseq = left > complex_values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_cseq = left >= complex_values  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_relops_invalid_numpy_array(left: pd.Index[complex]) -> None:
    """Test invalid pd.Index[complex] comparisons with numpy arrays"""
    real_array: np_ndarray_float = np.array([1.0, 2.0], np.float64)
    complex_array: np_ndarray_complex = np.array([1j, 2j], np.complex128)

    if TYPE_CHECKING_INVALID_USAGE:
        left.__lt__(real_array)  # type: ignore[misc,operator] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type] # ty: ignore[invalid-argument-type]
        left.__le__(real_array)  # type: ignore[misc,operator] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type] # ty: ignore[invalid-argument-type]
        left.__gt__(real_array)  # type: ignore[misc,operator] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type] # ty: ignore[invalid-argument-type]
        left.__ge__(real_array)  # type: ignore[misc,operator] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type] # ty: ignore[invalid-argument-type]
        left.__lt__(complex_array)  # type: ignore[misc,operator] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type] # ty: ignore[invalid-argument-type]
        left.__le__(complex_array)  # type: ignore[misc,operator] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type] # ty: ignore[invalid-argument-type]
        left.__gt__(complex_array)  # type: ignore[misc,operator] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type] # ty: ignore[invalid-argument-type]
        left.__ge__(complex_array)  # type: ignore[misc,operator] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type] # ty: ignore[invalid-argument-type]


def test_relops_invalid_pd_index(left: pd.Index[complex]) -> None:
    """Test invalid pd.Index[complex] comparisons with pandas Indexes"""
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
        _cidx_lt = complex_index < left  # type: ignore[misc] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _cidx_le = complex_index <= left  # type: ignore[misc] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _cidx_gt = complex_index > left  # type: ignore[misc] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _cidx_ge = complex_index >= left  # type: ignore[misc] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
