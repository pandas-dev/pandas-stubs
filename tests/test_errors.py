import os
import platform
from typing import TYPE_CHECKING
import warnings

from packaging.version import parse
import pandas as pd
from pandas import errors
import pytest

PD_LT_15 = parse(pd.__version__) < parse("1.5.0")

if TYPE_CHECKING or PD_LT_15:
    # TODO: Remove all imports below after switch to 1.5.x, these moved to pandas.errors
    from pandas.core.base import (
        DataError,
        SpecificationError,
    )
    from pandas.core.common import (
        SettingWithCopyError,
        SettingWithCopyWarning,
    )
    from pandas.core.computation.engines import NumExprClobberingError
    from pandas.core.computation.ops import UndefinedVariableError
    from pandas.core.indexing import IndexingError

    from pandas.io.clipboard import (
        PyperclipException,
        PyperclipWindowsException,
    )
    from pandas.io.formats.css import CSSWarning
    from pandas.io.pytables import (
        AttributeConflictWarning,
        ClosedFileError,
        IncompatibilityWarning,
        PossibleDataLossError,
    )
    from pandas.io.sql import DatabaseError
    from pandas.io.stata import (
        CategoricalConversionWarning,
        InvalidColumnName,
        PossiblePrecisionLoss,
        ValueLabelTypeMismatch,
    )
else:
    from pandas.errors import (
        AttributeConflictWarning,
        CategoricalConversionWarning,
        ClosedFileError,
        CSSWarning,
        DatabaseError,
        DataError,
        IncompatibilityWarning,
        IndexingError,
        InvalidColumnName,
        NumExprClobberingError,
        PossibleDataLossError,
        PossiblePrecisionLoss,
        PyperclipException,
        PyperclipWindowsException,
        SettingWithCopyError,
        SettingWithCopyWarning,
        SpecificationError,
        UndefinedVariableError,
        ValueLabelTypeMismatch,
    )

WINDOWS = os.name == "nt" or "cygwin" in platform.system().lower()


def test_abstract_method_error() -> None:
    class Foo:
        pass

    with pytest.raises(errors.AbstractMethodError):
        raise errors.AbstractMethodError(Foo)


def test_accessor_registration_warning() -> None:
    with pytest.warns(errors.AccessorRegistrationWarning):
        warnings.warn("", errors.AccessorRegistrationWarning)


def test_dtype_warning() -> None:
    with pytest.warns(errors.DtypeWarning):
        warnings.warn("", errors.DtypeWarning)


def test_duplicate_label_error() -> None:
    with pytest.raises(errors.DuplicateLabelError):
        raise errors.DuplicateLabelError


def test_empry_data_error() -> None:
    with pytest.raises(errors.EmptyDataError):
        raise errors.EmptyDataError()


def test_in_casting_nan_error() -> None:
    with pytest.raises(errors.IntCastingNaNError):
        raise errors.IntCastingNaNError


def test_invalid_index_error() -> None:
    with pytest.raises(errors.InvalidIndexError):
        raise errors.InvalidIndexError


def test_merge_error() -> None:
    with pytest.raises(errors.MergeError):
        raise errors.MergeError


def test_null_frequency_error() -> None:
    with pytest.raises(errors.NullFrequencyError):
        raise errors.NullFrequencyError


def test_numba_util_error() -> None:
    with pytest.raises(errors.NumbaUtilError):
        raise errors.NumbaUtilError


def test_option_error() -> None:
    with pytest.raises(errors.OptionError):
        raise errors.OptionError()


def test_out_of_bounds_datetime() -> None:
    with pytest.raises(errors.OutOfBoundsDatetime):
        raise errors.OutOfBoundsDatetime()


def test_out_of_bounds_timedelta() -> None:
    with pytest.raises(errors.OutOfBoundsTimedelta):
        raise errors.OutOfBoundsTimedelta()


def test_parser_error() -> None:
    with pytest.raises(errors.ParserError):
        raise errors.ParserError()


def test_parser_warning() -> None:
    with pytest.warns(errors.ParserWarning):
        warnings.warn("", errors.ParserWarning)


def test_performance_warning() -> None:
    with pytest.warns(errors.PerformanceWarning):
        warnings.warn("", errors.PerformanceWarning)


def test_unsorted_index_error() -> None:
    with pytest.raises(errors.UnsortedIndexError):
        raise errors.UnsortedIndexError()


def test_unsupported_function_call() -> None:
    with pytest.raises(errors.UnsupportedFunctionCall):
        raise errors.UnsupportedFunctionCall()


def test_data_error() -> None:
    with pytest.raises(DataError):
        raise DataError()


def test_specification_error() -> None:
    with pytest.raises(SpecificationError):
        raise SpecificationError()


def test_setting_with_copy_error() -> None:
    with pytest.raises(SettingWithCopyError):
        raise SettingWithCopyError()


def test_setting_with_copy_warning() -> None:
    with pytest.warns(SettingWithCopyWarning):
        warnings.warn("", SettingWithCopyWarning)


def test_numexpr_clobbering_error() -> None:
    with pytest.raises(NumExprClobberingError):
        raise NumExprClobberingError()


def test_undefined_variable_error() -> None:
    with pytest.raises(UndefinedVariableError):
        raise UndefinedVariableError("x")


def test_indexing_error() -> None:
    with pytest.raises(IndexingError):
        raise IndexingError()


def test_pyperclip_exception() -> None:
    with pytest.raises(PyperclipException):
        raise PyperclipException()


@pytest.mark.skipif(not PD_LT_15 or not WINDOWS, reason="Feature moved in 1.5.0")
def test_pyperclip_windows_exception() -> None:
    with pytest.raises(PyperclipWindowsException):
        raise PyperclipWindowsException("message")


def test_css_warning() -> None:
    with pytest.warns(CSSWarning):
        warnings.warn("", CSSWarning)


def test_possible_data_loss_error() -> None:
    with pytest.raises(PossibleDataLossError):
        raise PossibleDataLossError()


def test_closed_file_error() -> None:
    with pytest.raises(ClosedFileError):
        raise ClosedFileError()


def test_incompatibility_warning() -> None:
    with pytest.warns(IncompatibilityWarning):
        warnings.warn("", IncompatibilityWarning)


def test_attribute_conflict_warning() -> None:
    with pytest.warns(AttributeConflictWarning):
        warnings.warn("", AttributeConflictWarning)


def test_database_error() -> None:
    with pytest.raises(DatabaseError):
        raise DatabaseError()


def test_possible_precision_loss() -> None:
    with pytest.warns(PossiblePrecisionLoss):
        warnings.warn("", PossiblePrecisionLoss)


def test_value_label_type_mismatch() -> None:
    with pytest.warns(ValueLabelTypeMismatch):
        warnings.warn("", ValueLabelTypeMismatch)


def test_invalid_column_name() -> None:
    with pytest.warns(InvalidColumnName):
        warnings.warn("", InvalidColumnName)


def test_categorical_conversion_warning() -> None:
    with pytest.warns(CategoricalConversionWarning):
        warnings.warn("", CategoricalConversionWarning)
