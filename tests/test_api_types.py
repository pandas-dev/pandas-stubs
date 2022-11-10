from typing import Type

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionDtype
import pandas.api.types as api
from typing_extensions import assert_type

from pandas._typing import DtypeObj

from tests import check

nparr = np.array([1, 2, 3])
arr = pd.Series([1, 2, 3])
obj = "True"
mapping = {"a": "a"}


def test_is_array_like() -> None:
    check(assert_type(api.is_array_like(arr), bool), bool)
    check(assert_type(api.is_array_like(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_array_like(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_array_like(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_array_like(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_array_like(pd.Index([1, 2.0])), bool), bool)


def test_is_bool() -> None:
    check(assert_type(api.is_bool(obj), bool), bool)
    check(assert_type(api.is_bool(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_bool(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_bool(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_bool(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_bool(pd.Index([1, 2.0])), bool), bool)


def test_is_bool_dtype() -> None:
    check(assert_type(api.is_bool_dtype(arr), bool), bool)
    check(assert_type(api.is_bool_dtype(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_bool_dtype(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_bool_dtype(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_bool_dtype(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_bool_dtype(pd.Index([1, 2.0])), bool), bool)


def test_is_categorical_dtype() -> None:
    check(assert_type(api.is_categorical_dtype(arr), bool), bool)
    check(assert_type(api.is_categorical_dtype(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_categorical_dtype(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_categorical_dtype(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_categorical_dtype(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_categorical_dtype(pd.Index([1, 2.0])), bool), bool)


def test_is_complex() -> None:
    check(assert_type(api.is_complex(obj), bool), bool)
    check(assert_type(api.is_complex(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_complex(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_complex(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_complex(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_complex(pd.Index([1, 2.0])), bool), bool)


def test_is_complex_dtype() -> None:
    check(assert_type(api.is_complex_dtype(arr), bool), bool)
    check(assert_type(api.is_complex_dtype(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_complex_dtype(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_complex_dtype(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_complex_dtype(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_complex_dtype(pd.Index([1, 2.0])), bool), bool)


def test_is_datetime64_any_dtype() -> None:
    check(assert_type(api.is_datetime64_any_dtype(arr), bool), bool)
    check(assert_type(api.is_datetime64_any_dtype(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_datetime64_any_dtype(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_datetime64_any_dtype(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_datetime64_any_dtype(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_datetime64_any_dtype(pd.Index([1, 2.0])), bool), bool)


def test_is_datetime64_dtype() -> None:
    check(assert_type(api.is_datetime64_dtype(arr), bool), bool)
    check(assert_type(api.is_datetime64_dtype(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_datetime64_dtype(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_datetime64_dtype(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_datetime64_dtype(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_datetime64_dtype(pd.Index([1, 2.0])), bool), bool)


def test_is_datetime64_ns_dtype() -> None:
    check(assert_type(api.is_datetime64_ns_dtype(arr), bool), bool)
    check(assert_type(api.is_datetime64_ns_dtype(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_datetime64_ns_dtype(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_datetime64_ns_dtype(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_datetime64_ns_dtype(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_datetime64_ns_dtype(pd.Index([1, 2.0])), bool), bool)


def test_is_datetime64tz_dtype() -> None:
    check(assert_type(api.is_datetime64tz_dtype(arr), bool), bool)
    check(assert_type(api.is_datetime64tz_dtype(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_datetime64tz_dtype(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_datetime64tz_dtype(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_datetime64tz_dtype(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_datetime64tz_dtype(pd.Index([1, 2.0])), bool), bool)


def test_is_dict_like() -> None:
    check(assert_type(api.is_dict_like(mapping), bool), bool)
    check(assert_type(api.is_dict_like(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_dict_like(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_dict_like(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_dict_like(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_dict_like(pd.Index([1, 2.0])), bool), bool)


def test_is_dtype_equal() -> None:
    check(assert_type(api.is_dtype_equal("i4", np.int8), bool), bool)


def test_is_extension_array_dtype() -> None:
    check(assert_type(api.is_extension_array_dtype(arr), bool), bool)
    check(assert_type(api.is_extension_array_dtype(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_extension_array_dtype(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_extension_array_dtype(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_extension_array_dtype(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_extension_array_dtype(pd.Index([1, 2.0])), bool), bool)


def test_is_file_like() -> None:
    check(assert_type(api.is_file_like(obj), bool), bool)
    check(assert_type(api.is_file_like(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_file_like(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_file_like(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_file_like(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_file_like(pd.Index([1, 2.0])), bool), bool)


def test_is_float() -> None:
    check(assert_type(api.is_float(obj), bool), bool)
    check(assert_type(api.is_float(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_float(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_float(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_float(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_float(pd.Index([1, 2.0])), bool), bool)


def test_is_float_dtype() -> None:
    check(assert_type(api.is_float_dtype(arr), bool), bool)
    check(assert_type(api.is_float_dtype(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_float_dtype(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_float_dtype(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_float_dtype(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_float_dtype(pd.Index([1, 2.0])), bool), bool)


def test_is_hashable() -> None:
    check(assert_type(api.is_hashable(obj), bool), bool)
    check(assert_type(api.is_hashable(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_hashable(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_hashable(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_hashable(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_hashable(pd.Index([1, 2.0])), bool), bool)


def test_is_int64_dtype() -> None:
    check(assert_type(api.is_int64_dtype(arr), bool), bool)
    check(assert_type(api.is_int64_dtype(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_int64_dtype(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_int64_dtype(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_int64_dtype(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_int64_dtype(pd.Index([1, 2.0])), bool), bool)


def test_is_integer() -> None:
    check(assert_type(api.is_integer(obj), bool), bool)
    check(assert_type(api.is_integer(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_integer(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_integer(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_integer(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_integer(pd.Index([1, 2.0])), bool), bool)


def test_is_integer_dtype() -> None:
    check(assert_type(api.is_integer_dtype(arr), bool), bool)
    check(assert_type(api.is_integer_dtype(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_integer_dtype(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_integer_dtype(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_integer_dtype(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_integer_dtype(pd.Index([1, 2.0])), bool), bool)


def test_is_interval() -> None:
    check(assert_type(api.is_interval(obj), bool), bool)
    check(assert_type(api.is_interval(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_interval(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_interval(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_interval(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_interval(pd.Index([1, 2.0])), bool), bool)


def test_is_interval_dtype() -> None:
    check(assert_type(api.is_interval_dtype(obj), bool), bool)
    check(assert_type(api.is_interval(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_interval(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_interval(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_interval(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_interval(pd.Index([1, 2.0])), bool), bool)


def test_is_iterator() -> None:
    check(assert_type(api.is_iterator(obj), bool), bool)
    check(assert_type(api.is_iterator(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_iterator(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_iterator(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_iterator(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_iterator(pd.Index([1, 2.0])), bool), bool)


def test_is_list_like() -> None:
    check(assert_type(api.is_list_like(obj), bool), bool)
    check(assert_type(api.is_list_like(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_list_like(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_list_like(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_list_like(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_list_like(pd.Index([1, 2.0])), bool), bool)


def test_is_named_tuple() -> None:
    check(assert_type(api.is_named_tuple(obj), bool), bool)
    check(assert_type(api.is_named_tuple(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_named_tuple(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_named_tuple(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_named_tuple(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_named_tuple(pd.Index([1, 2.0])), bool), bool)


def test_is_number() -> None:
    check(assert_type(api.is_number(obj), bool), bool)
    check(assert_type(api.is_number(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_number(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_number(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_number(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_number(pd.Index([1, 2.0])), bool), bool)


def test_is_numeric_dtype() -> None:
    check(assert_type(api.is_numeric_dtype(arr), bool), bool)
    check(assert_type(api.is_numeric_dtype(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_numeric_dtype(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_numeric_dtype(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_numeric_dtype(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_numeric_dtype(pd.Index([1, 2.0])), bool), bool)


def test_is_object_dtype() -> None:
    check(assert_type(api.is_object_dtype(arr), bool), bool)
    check(assert_type(api.is_object_dtype(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_object_dtype(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_object_dtype(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_object_dtype(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_object_dtype(pd.Index([1, 2.0])), bool), bool)


def test_is_period_dtype() -> None:
    check(assert_type(api.is_period_dtype(arr), bool), bool)
    check(assert_type(api.is_period_dtype(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_period_dtype(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_period_dtype(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_period_dtype(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_period_dtype(pd.Index([1, 2.0])), bool), bool)


def test_is_re() -> None:
    check(assert_type(api.is_re(obj), bool), bool)
    check(assert_type(api.is_re(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_re(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_re(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_re(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_re(pd.Index([1, 2.0])), bool), bool)


def test_is_re_compilable() -> None:
    check(assert_type(api.is_re_compilable(obj), bool), bool)
    check(assert_type(api.is_re_compilable(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_re_compilable(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_re_compilable(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_re_compilable(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_re_compilable(pd.Index([1, 2.0])), bool), bool)


def test_is_scalar() -> None:
    check(assert_type(api.is_scalar(obj), bool), bool)
    check(assert_type(api.is_scalar(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_scalar(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_scalar(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_scalar(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_scalar(pd.Index([1, 2.0])), bool), bool)


def test_is_signed_integer_dtype() -> None:
    check(assert_type(api.is_signed_integer_dtype(arr), bool), bool)
    check(assert_type(api.is_signed_integer_dtype(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_signed_integer_dtype(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_signed_integer_dtype(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_signed_integer_dtype(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_signed_integer_dtype(pd.Index([1, 2.0])), bool), bool)


def test_is_sparse() -> None:
    check(assert_type(api.is_sparse(arr), bool), bool)
    check(assert_type(api.is_sparse(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_sparse(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_sparse(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)


def test_is_string_dtype() -> None:
    check(assert_type(api.is_string_dtype(arr), bool), bool)
    check(assert_type(api.is_string_dtype(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_string_dtype(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_string_dtype(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_string_dtype(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_string_dtype(pd.Index([1, 2.0])), bool), bool)


def test_is_timedelta64_dtype() -> None:
    check(assert_type(api.is_timedelta64_dtype(arr), bool), bool)
    check(assert_type(api.is_timedelta64_dtype(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_timedelta64_dtype(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_timedelta64_dtype(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_timedelta64_dtype(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_timedelta64_dtype(pd.Index([1, 2.0])), bool), bool)


def test_is_timedelta64_ns_dtype() -> None:
    check(assert_type(api.is_timedelta64_ns_dtype(arr), bool), bool)
    check(assert_type(api.is_timedelta64_ns_dtype(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_timedelta64_ns_dtype(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_timedelta64_ns_dtype(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_timedelta64_ns_dtype(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_timedelta64_ns_dtype(pd.Index([1, 2.0])), bool), bool)


def test_is_unsigned_integer_dtype() -> None:
    check(assert_type(api.is_unsigned_integer_dtype(arr), bool), bool)
    check(assert_type(api.is_unsigned_integer_dtype(np.array([1,2,3])), bool), bool)
    check(assert_type(api.is_unsigned_integer_dtype(np.dtype(np.int32)), bool), bool)
    check(assert_type(api.is_unsigned_integer_dtype(pd.Series([1,2,3])), bool), bool)
    check(assert_type(api.is_unsigned_integer_dtype(pd.DataFrame({"a":[1,2], "b":[3,4]})), bool), bool)
    check(assert_type(api.is_unsigned_integer_dtype(pd.Index([1, 2.0])), bool), bool)


def test_pandas_dtype() -> None:
    check(assert_type(api.pandas_dtype(arr), DtypeObj), type(np.dtype("i8")))


def test_infer_dtype() -> None:
    check(assert_type(api.infer_dtype([1, 2, 3]), str), str)


def test_union_categoricals() -> None:
    to_union = [pd.Categorical([1, 2, 3]), pd.Categorical([3, 4, 5])]
    check(assert_type(api.union_categoricals(to_union), pd.Categorical), pd.Categorical)


def check_extension_dtypes() -> None:
    # GH 315
    def check_ext_dtype(etype: Type[ExtensionDtype]):
        assert issubclass(etype, ExtensionDtype)

    check_ext_dtype(pd.Int64Dtype)
    check_ext_dtype(pd.Int8Dtype)
    check_ext_dtype(pd.Int16Dtype)
    check_ext_dtype(pd.Int32Dtype)
    check_ext_dtype(pd.Int64Dtype)
    check_ext_dtype(pd.UInt8Dtype)
    check_ext_dtype(pd.UInt16Dtype)
    check_ext_dtype(pd.UInt32Dtype)
    check_ext_dtype(pd.UInt64Dtype)
    check_ext_dtype(pd.BooleanDtype)
    check_ext_dtype(pd.StringDtype)
    check_ext_dtype(pd.CategoricalDtype)
    check_ext_dtype(pd.DatetimeTZDtype)
    check_ext_dtype(pd.IntervalDtype)
    check_ext_dtype(pd.PeriodDtype)
    check_ext_dtype(pd.SparseDtype)
    check_ext_dtype(pd.Float32Dtype)
    check_ext_dtype(pd.Float64Dtype)
