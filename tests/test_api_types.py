import numpy as np
import pandas as pd
from typing_extensions import assert_type

from pandas._typing import DtypeObj

import pandas.core.dtypes.api as api

from tests import check

nparr = np.array([1, 2, 3])
arr = pd.Series([1, 2, 3])
obj = "True"
mapping = {"a": "a"}


def test_is_array_like() -> None:
    check(assert_type(api.is_array_like(arr), bool), bool)


def test_is_bool() -> None:
    check(assert_type(api.is_bool(obj), bool), bool)


def test_is_bool_dtype() -> None:
    check(assert_type(api.is_bool_dtype(arr), bool), bool)


def test_is_categorical_dtype() -> None:
    check(assert_type(api.is_categorical_dtype(arr), bool), bool)


def test_is_complex() -> None:
    check(assert_type(api.is_complex(obj), bool), bool)


def test_is_complex_dtype() -> None:
    check(assert_type(api.is_complex_dtype(arr), bool), bool)


def test_is_datetime64_any_dtype() -> None:
    check(assert_type(api.is_datetime64_any_dtype(arr), bool), bool)


def test_is_datetime64_dtype() -> None:
    check(assert_type(api.is_datetime64_dtype(arr), bool), bool)


def test_is_datetime64_ns_dtype() -> None:
    check(assert_type(api.is_datetime64_ns_dtype(arr), bool), bool)


def test_is_datetime64tz_dtype() -> None:
    check(assert_type(api.is_datetime64tz_dtype(arr), bool), bool)


def test_is_dict_like() -> None:
    check(assert_type(api.is_dict_like(mapping), bool), bool)


def test_is_dtype_equal() -> None:
    check(assert_type(api.is_dtype_equal("i4", np.int8), bool), bool)


def test_is_extension_array_dtype() -> None:
    check(assert_type(api.is_extension_array_dtype(arr), bool), bool)


def test_is_file_like() -> None:
    check(assert_type(api.is_file_like(obj), bool), bool)


def test_is_float() -> None:
    check(assert_type(api.is_float(obj), bool), bool)


def test_is_float_dtype() -> None:
    check(assert_type(api.is_float_dtype(arr), bool), bool)


def test_is_hashable() -> None:
    check(assert_type(api.is_hashable(obj), bool), bool)


def test_is_int64_dtype() -> None:
    check(assert_type(api.is_int64_dtype(arr), bool), bool)


def test_is_integer() -> None:
    check(assert_type(api.is_integer(obj), bool), bool)


def test_is_integer_dtype() -> None:
    check(assert_type(api.is_integer_dtype(arr), bool), bool)


def test_is_interval() -> None:
    check(assert_type(api.is_interval(obj), bool), bool)


def test_is_interval_dtype() -> None:
    check(assert_type(api.is_interval_dtype(obj), bool), bool)


def test_is_iterator() -> None:
    check(assert_type(api.is_iterator(obj), bool), bool)


def test_is_list_like() -> None:
    check(assert_type(api.is_list_like(obj), bool), bool)


def test_is_named_tuple() -> None:
    check(assert_type(api.is_named_tuple(obj), bool), bool)


def test_is_number() -> None:
    check(assert_type(api.is_number(obj), bool), bool)


def test_is_numeric_dtype() -> None:
    check(assert_type(api.is_numeric_dtype(arr), bool), bool)


def test_is_object_dtype() -> None:
    check(assert_type(api.is_object_dtype(arr), bool), bool)


def test_is_period_dtype() -> None:
    check(assert_type(api.is_period_dtype(arr), bool), bool)


def test_is_re() -> None:
    check(assert_type(api.is_re(obj), bool), bool)


def test_is_re_compilable() -> None:
    check(assert_type(api.is_re_compilable(obj), bool), bool)


def test_is_scalar() -> None:
    check(assert_type(api.is_scalar(obj), bool), bool)


def test_is_signed_integer_dtype() -> None:
    check(assert_type(api.is_signed_integer_dtype(arr), bool), bool)


def test_is_sparse() -> None:
    check(assert_type(api.is_sparse(arr), bool), bool)


def test_is_string_dtype() -> None:
    check(assert_type(api.is_string_dtype(arr), bool), bool)


def test_is_timedelta64_dtype() -> None:
    check(assert_type(api.is_timedelta64_dtype(arr), bool), bool)


def test_is_timedelta64_ns_dtype() -> None:
    check(assert_type(api.is_timedelta64_ns_dtype(arr), bool), bool)


def test_is_unsigned_integer_dtype() -> None:
    check(assert_type(api.is_unsigned_integer_dtype(arr), bool), bool)


def test_pandas_dtype() -> None:
    # Can't use check because these are both types
    assert assert_type(api.pandas_dtype(arr), DtypeObj) == np.dtype("i8")
