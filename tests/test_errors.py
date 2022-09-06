import warnings

from pandas import errors
import pytest


def test_abstract_method_error():
    class Foo:
        pass

    with pytest.raises(errors.AbstractMethodError):
        raise errors.AbstractMethodError(Foo)


def test_accessor_registration_warning():
    with pytest.warns(errors.AccessorRegistrationWarning):
        warnings.warn("", errors.AccessorRegistrationWarning)


def test_dtype_warning():
    with pytest.warns(errors.DtypeWarning):
        warnings.warn("", errors.DtypeWarning)


def test_duplicate_label_error():
    with pytest.raises(errors.DuplicateLabelError):
        raise errors.DuplicateLabelError


def test_empry_data_error():
    with pytest.raises(errors.EmptyDataError):
        raise errors.EmptyDataError()


def test_in_casting_nan_error():
    with pytest.raises(errors.IntCastingNaNError):
        raise errors.IntCastingNaNError


def test_invalid_index_error():
    with pytest.raises(errors.InvalidIndexError):
        raise errors.InvalidIndexError


def test_merge_error():
    with pytest.raises(errors.MergeError):
        raise errors.MergeError


def test_null_frequency_error():
    with pytest.raises(errors.NullFrequencyError):
        raise errors.NullFrequencyError


def test_numba_util_error():
    with pytest.raises(errors.NumbaUtilError):
        raise errors.NumbaUtilError


def test_option_error():
    with pytest.raises(errors.OptionError):
        raise errors.OptionError()


def test_out_of_bounds_datetime():
    with pytest.raises(errors.OutOfBoundsDatetime):
        raise errors.OutOfBoundsDatetime()


def test_out_of_bounds_timedelta():
    with pytest.raises(errors.OutOfBoundsTimedelta):
        raise errors.OutOfBoundsTimedelta()


def test_parser_error():
    with pytest.raises(errors.ParserError):
        raise errors.ParserError()


def test_parser_warning():
    with pytest.warns(errors.ParserWarning):
        warnings.warn("", errors.ParserWarning)


def test_performance_warning():
    with pytest.warns(errors.PerformanceWarning):
        warnings.warn("", errors.PerformanceWarning)


def test_unsorted_index_error():
    with pytest.raises(errors.UnsortedIndexError):
        raise errors.UnsortedIndexError()


def test_unsupported_function_call():
    with pytest.raises(errors.UnsupportedFunctionCall):
        raise errors.UnsupportedFunctionCall()
