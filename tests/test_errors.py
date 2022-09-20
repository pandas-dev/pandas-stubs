import warnings

from pandas import errors
import pytest

from tests import WINDOWS


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
    with pytest.raises(errors.DataError):
        raise errors.DataError()


def test_specification_error() -> None:
    with pytest.raises(errors.SpecificationError):
        raise errors.SpecificationError()


def test_setting_with_copy_error() -> None:
    with pytest.raises(errors.SettingWithCopyError):
        raise errors.SettingWithCopyError()


def test_setting_with_copy_warning() -> None:
    with pytest.warns(errors.SettingWithCopyWarning):
        warnings.warn("", errors.SettingWithCopyWarning)


def test_numexpr_clobbering_error() -> None:
    with pytest.raises(errors.NumExprClobberingError):
        raise errors.NumExprClobberingError()


def test_undefined_variable_error() -> None:
    with pytest.raises(errors.UndefinedVariableError):
        raise errors.UndefinedVariableError("x")


def test_indexing_error() -> None:
    with pytest.raises(errors.IndexingError):
        raise errors.IndexingError()


def test_pyperclip_exception() -> None:
    with pytest.raises(errors.PyperclipException):
        raise errors.PyperclipException()


@pytest.mark.skipif(not WINDOWS, reason="Windows only")
def test_pyperclip_windows_exception() -> None:
    with pytest.raises(errors.PyperclipWindowsException):
        raise errors.PyperclipWindowsException("message")


def test_css_warning() -> None:
    with pytest.warns(errors.CSSWarning):
        warnings.warn("", errors.CSSWarning)


def test_possible_data_loss_error() -> None:
    with pytest.raises(errors.PossibleDataLossError):
        raise errors.PossibleDataLossError()


def test_closed_file_error() -> None:
    with pytest.raises(errors.ClosedFileError):
        raise errors.ClosedFileError()


def test_incompatibility_warning() -> None:
    with pytest.warns(errors.IncompatibilityWarning):
        warnings.warn("", errors.IncompatibilityWarning)


def test_attribute_conflict_warning() -> None:
    with pytest.warns(errors.AttributeConflictWarning):
        warnings.warn("", errors.AttributeConflictWarning)


def test_database_error() -> None:
    with pytest.raises(errors.DatabaseError):
        raise errors.DatabaseError()


def test_possible_precision_loss() -> None:
    with pytest.warns(errors.PossiblePrecisionLoss):
        warnings.warn("", errors.PossiblePrecisionLoss)


def test_value_label_type_mismatch() -> None:
    with pytest.warns(errors.ValueLabelTypeMismatch):
        warnings.warn("", errors.ValueLabelTypeMismatch)


def test_invalid_column_name() -> None:
    with pytest.warns(errors.InvalidColumnName):
        warnings.warn("", errors.InvalidColumnName)


def test_categorical_conversion_warning() -> None:
    with pytest.warns(errors.CategoricalConversionWarning):
        warnings.warn("", errors.CategoricalConversionWarning)
