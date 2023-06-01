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
dframe = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
dtylike = np.dtype(np.int32)
ind = pd.Index([1, 2.0])


def test_is_array_like() -> None:
    check(assert_type(api.is_array_like(arr), bool), bool)
    check(assert_type(api.is_array_like(nparr), bool), bool)
    check(assert_type(api.is_array_like(dtylike), bool), bool)
    check(
        assert_type(api.is_array_like(dframe), bool),
        bool,
    )
    check(assert_type(api.is_array_like(ind), bool), bool)


def test_is_bool() -> None:
    check(assert_type(api.is_bool(obj), bool), bool)
    check(assert_type(api.is_bool(nparr), bool), bool)
    check(assert_type(api.is_bool(dtylike), bool), bool)
    check(assert_type(api.is_bool(arr), bool), bool)
    check(assert_type(api.is_bool(dframe), bool), bool)
    check(assert_type(api.is_bool(ind), bool), bool)


def test_is_bool_dtype() -> None:
    check(assert_type(api.is_bool_dtype(arr), bool), bool)
    check(assert_type(api.is_bool_dtype(nparr), bool), bool)
    check(assert_type(api.is_bool_dtype(dtylike), bool), bool)
    check(
        assert_type(api.is_bool_dtype(dframe), bool),
        bool,
    )
    check(assert_type(api.is_bool_dtype(ind), bool), bool)
    check(assert_type(api.is_bool_dtype(ExtensionDtype), bool), bool)


def test_is_categorical_dtype() -> None:
    check(assert_type(api.is_categorical_dtype(arr), bool), bool)
    check(assert_type(api.is_categorical_dtype(nparr), bool), bool)
    check(assert_type(api.is_categorical_dtype(dtylike), bool), bool)
    check(assert_type(api.is_categorical_dtype(dframe), bool), bool)
    check(assert_type(api.is_categorical_dtype(ind), bool), bool)
    check(assert_type(api.is_categorical_dtype(ExtensionDtype), bool), bool)


def test_is_complex() -> None:
    check(assert_type(api.is_complex(obj), bool), bool)
    check(assert_type(api.is_complex(nparr), bool), bool)
    check(assert_type(api.is_complex(dtylike), bool), bool)
    check(assert_type(api.is_complex(arr), bool), bool)
    check(
        assert_type(api.is_complex(dframe), bool),
        bool,
    )
    check(assert_type(api.is_complex(ind), bool), bool)


def test_is_complex_dtype() -> None:
    check(assert_type(api.is_complex_dtype(arr), bool), bool)
    check(assert_type(api.is_complex_dtype(nparr), bool), bool)
    check(assert_type(api.is_complex_dtype(dtylike), bool), bool)
    check(
        assert_type(api.is_complex_dtype(dframe), bool),
        bool,
    )
    check(assert_type(api.is_complex_dtype(ind), bool), bool)
    # check(assert_type(api.is_complex_dtype(ExtensionDtype), bool), bool) pandas GH 50923


def test_is_datetime64_any_dtype() -> None:
    check(assert_type(api.is_datetime64_any_dtype(arr), bool), bool)
    check(assert_type(api.is_datetime64_any_dtype(nparr), bool), bool)
    check(assert_type(api.is_datetime64_any_dtype(dtylike), bool), bool)
    check(
        assert_type(api.is_datetime64_any_dtype(dframe), bool),
        bool,
    )
    check(assert_type(api.is_datetime64_any_dtype(ind), bool), bool)
    # check(assert_type(api.is_datetime64_any_dtype(ExtensionDtype), bool), bool) pandas GH 50923


def test_is_datetime64_dtype() -> None:
    check(assert_type(api.is_datetime64_dtype(arr), bool), bool)
    check(assert_type(api.is_datetime64_dtype(nparr), bool), bool)
    check(assert_type(api.is_datetime64_dtype(dtylike), bool), bool)
    check(
        assert_type(api.is_datetime64_dtype(dframe), bool),
        bool,
    )
    check(assert_type(api.is_datetime64_dtype(ind), bool), bool)
    # check(assert_type(api.is_datetime64_dtype(ExtensionDtype), bool), bool) pandas GH 50923


def test_is_datetime64_ns_dtype() -> None:
    check(assert_type(api.is_datetime64_ns_dtype(arr), bool), bool)
    check(assert_type(api.is_datetime64_ns_dtype(nparr), bool), bool)
    check(assert_type(api.is_datetime64_ns_dtype(dtylike), bool), bool)
    check(
        assert_type(api.is_datetime64_ns_dtype(dframe), bool),
        bool,
    )
    check(assert_type(api.is_datetime64_ns_dtype(ind), bool), bool)
    check(assert_type(api.is_datetime64_ns_dtype(ExtensionDtype), bool), bool)


def test_is_datetime64tz_dtype() -> None:
    check(assert_type(api.is_datetime64tz_dtype(arr), bool), bool)
    check(assert_type(api.is_datetime64tz_dtype(nparr), bool), bool)
    check(assert_type(api.is_datetime64tz_dtype(dtylike), bool), bool)
    check(assert_type(api.is_datetime64tz_dtype(dframe), bool), bool)
    check(assert_type(api.is_datetime64tz_dtype(ind), bool), bool)
    check(assert_type(api.is_datetime64tz_dtype(ExtensionDtype), bool), bool)


def test_is_dict_like() -> None:
    check(assert_type(api.is_dict_like(mapping), bool), bool)
    check(assert_type(api.is_dict_like(nparr), bool), bool)
    check(assert_type(api.is_dict_like(dtylike), bool), bool)
    check(assert_type(api.is_dict_like(arr), bool), bool)
    check(
        assert_type(api.is_dict_like(dframe), bool),
        bool,
    )
    check(assert_type(api.is_dict_like(ind), bool), bool)


def test_is_dtype_equal() -> None:
    check(assert_type(api.is_dtype_equal("i4", np.int8), bool), bool)


def test_is_extension_array_dtype() -> None:
    check(assert_type(api.is_extension_array_dtype(arr), bool), bool)
    check(assert_type(api.is_extension_array_dtype(nparr), bool), bool)
    check(assert_type(api.is_extension_array_dtype(dtylike), bool), bool)
    check(
        assert_type(api.is_extension_array_dtype(dframe), bool),
        bool,
    )
    check(assert_type(api.is_extension_array_dtype(ind), bool), bool)
    check(assert_type(api.is_extension_array_dtype(ExtensionDtype), bool), bool)


def test_is_file_like() -> None:
    check(assert_type(api.is_file_like(obj), bool), bool)
    check(assert_type(api.is_file_like(nparr), bool), bool)
    check(assert_type(api.is_file_like(dtylike), bool), bool)
    check(assert_type(api.is_file_like(arr), bool), bool)
    check(
        assert_type(api.is_file_like(dframe), bool),
        bool,
    )
    check(assert_type(api.is_file_like(ind), bool), bool)


def test_is_float() -> None:
    check(assert_type(api.is_float(obj), bool), bool)
    check(assert_type(api.is_float(nparr), bool), bool)
    check(assert_type(api.is_float(dtylike), bool), bool)
    check(assert_type(api.is_float(arr), bool), bool)
    check(assert_type(api.is_float(dframe), bool), bool)
    check(assert_type(api.is_float(ind), bool), bool)


def test_is_float_dtype() -> None:
    check(assert_type(api.is_float_dtype(arr), bool), bool)
    check(assert_type(api.is_float_dtype(nparr), bool), bool)
    check(assert_type(api.is_float_dtype(dtylike), bool), bool)
    check(
        assert_type(api.is_float_dtype(dframe), bool),
        bool,
    )
    check(assert_type(api.is_float_dtype(ind), bool), bool)
    # check(assert_type(api.is_float_dtype(ExtensionDtype), bool), bool) pandas GH 50923


def test_is_hashable() -> None:
    check(assert_type(api.is_hashable(obj), bool), bool)
    check(assert_type(api.is_hashable(nparr), bool), bool)
    check(assert_type(api.is_hashable(dtylike), bool), bool)
    check(assert_type(api.is_hashable(arr), bool), bool)
    check(
        assert_type(api.is_hashable(dframe), bool),
        bool,
    )
    check(assert_type(api.is_hashable(ind), bool), bool)


def test_is_int64_dtype() -> None:
    check(assert_type(api.is_int64_dtype(arr), bool), bool)
    check(assert_type(api.is_int64_dtype(nparr), bool), bool)
    check(assert_type(api.is_int64_dtype(dtylike), bool), bool)
    check(assert_type(api.is_int64_dtype(dframe), bool), bool)
    check(assert_type(api.is_int64_dtype(ind), bool), bool)
    # check(assert_type(api.is_int64_dtype(ExtensionDtype), bool), bool) pandas GH 50923


def test_is_integer() -> None:
    check(assert_type(api.is_integer(obj), bool), bool)
    check(assert_type(api.is_integer(nparr), bool), bool)
    check(assert_type(api.is_integer(dtylike), bool), bool)
    check(assert_type(api.is_integer(arr), bool), bool)
    check(
        assert_type(api.is_integer(dframe), bool),
        bool,
    )
    check(assert_type(api.is_integer(ind), bool), bool)


def test_is_integer_dtype() -> None:
    check(assert_type(api.is_integer_dtype(arr), bool), bool)
    check(assert_type(api.is_integer_dtype(nparr), bool), bool)
    check(assert_type(api.is_integer_dtype(dtylike), bool), bool)
    check(
        assert_type(api.is_integer_dtype(dframe), bool),
        bool,
    )
    check(assert_type(api.is_integer_dtype(ind), bool), bool)
    # check(assert_type(api.is_integer_dtype(ExtensionDtype), bool), bool) pandas GH 50923


def test_is_interval() -> None:
    check(assert_type(api.is_interval(obj), bool), bool)
    check(assert_type(api.is_interval(nparr), bool), bool)
    check(assert_type(api.is_interval(dtylike), bool), bool)
    check(assert_type(api.is_interval(arr), bool), bool)
    check(
        assert_type(api.is_interval(dframe), bool),
        bool,
    )
    check(assert_type(api.is_interval(ind), bool), bool)


def test_is_interval_dtype() -> None:
    check(assert_type(api.is_interval_dtype(obj), bool), bool)
    check(assert_type(api.is_interval_dtype(nparr), bool), bool)
    check(assert_type(api.is_interval_dtype(dtylike), bool), bool)
    check(assert_type(api.is_interval_dtype(arr), bool), bool)
    check(assert_type(api.is_interval_dtype(dframe), bool), bool)
    check(assert_type(api.is_interval_dtype(ind), bool), bool)
    check(assert_type(api.is_interval_dtype(ExtensionDtype), bool), bool)


def test_is_iterator() -> None:
    check(assert_type(api.is_iterator(obj), bool), bool)
    check(assert_type(api.is_iterator(nparr), bool), bool)
    check(assert_type(api.is_iterator(dtylike), bool), bool)
    check(assert_type(api.is_iterator(arr), bool), bool)
    check(
        assert_type(api.is_iterator(dframe), bool),
        bool,
    )
    check(assert_type(api.is_iterator(ind), bool), bool)


def test_is_list_like() -> None:
    check(assert_type(api.is_list_like(obj), bool), bool)
    check(assert_type(api.is_list_like(nparr), bool), bool)
    check(assert_type(api.is_list_like(dtylike), bool), bool)
    check(assert_type(api.is_list_like(arr), bool), bool)
    check(
        assert_type(api.is_list_like(dframe), bool),
        bool,
    )
    check(assert_type(api.is_list_like(ind), bool), bool)


def test_is_named_tuple() -> None:
    check(assert_type(api.is_named_tuple(obj), bool), bool)
    check(assert_type(api.is_named_tuple(nparr), bool), bool)
    check(assert_type(api.is_named_tuple(dtylike), bool), bool)
    check(assert_type(api.is_named_tuple(arr), bool), bool)
    check(
        assert_type(api.is_named_tuple(dframe), bool),
        bool,
    )
    check(assert_type(api.is_named_tuple(ind), bool), bool)


def test_is_number() -> None:
    check(assert_type(api.is_number(obj), bool), bool)
    check(assert_type(api.is_number(nparr), bool), bool)
    check(assert_type(api.is_number(dtylike), bool), bool)
    check(assert_type(api.is_number(arr), bool), bool)
    check(assert_type(api.is_number(dframe), bool), bool)
    check(assert_type(api.is_number(ind), bool), bool)


def test_is_numeric_dtype() -> None:
    check(assert_type(api.is_numeric_dtype(arr), bool), bool)
    check(assert_type(api.is_numeric_dtype(nparr), bool), bool)
    check(assert_type(api.is_numeric_dtype(dtylike), bool), bool)
    check(
        assert_type(api.is_numeric_dtype(dframe), bool),
        bool,
    )
    check(assert_type(api.is_numeric_dtype(ind), bool), bool)
    # check(assert_type(api.is_numeric_dtype(ExtensionDtype), bool), bool) pandas GH 50923


def test_is_object_dtype() -> None:
    check(assert_type(api.is_object_dtype(arr), bool), bool)
    check(assert_type(api.is_object_dtype(nparr), bool), bool)
    check(assert_type(api.is_object_dtype(dtylike), bool), bool)
    check(
        assert_type(api.is_object_dtype(dframe), bool),
        bool,
    )
    check(assert_type(api.is_object_dtype(ind), bool), bool)
    # check(assert_type(api.is_object_dtype(ExtensionDtype), bool), bool) pandas GH 50923


def test_is_period_dtype() -> None:
    check(assert_type(api.is_period_dtype(arr), bool), bool)
    check(assert_type(api.is_period_dtype(nparr), bool), bool)
    check(assert_type(api.is_period_dtype(dtylike), bool), bool)
    check(assert_type(api.is_period_dtype(dframe), bool), bool)
    check(assert_type(api.is_period_dtype(ind), bool), bool)
    check(assert_type(api.is_period_dtype(ExtensionDtype), bool), bool)


def test_is_re() -> None:
    check(assert_type(api.is_re(obj), bool), bool)
    check(assert_type(api.is_re(nparr), bool), bool)
    check(assert_type(api.is_re(dtylike), bool), bool)
    check(assert_type(api.is_re(arr), bool), bool)
    check(assert_type(api.is_re(dframe), bool), bool)
    check(assert_type(api.is_re(ind), bool), bool)


def test_is_re_compilable() -> None:
    check(assert_type(api.is_re_compilable(obj), bool), bool)
    check(assert_type(api.is_re_compilable(nparr), bool), bool)
    check(assert_type(api.is_re_compilable(dtylike), bool), bool)
    check(assert_type(api.is_re_compilable(arr), bool), bool)
    check(
        assert_type(api.is_re_compilable(dframe), bool),
        bool,
    )
    check(assert_type(api.is_re_compilable(ind), bool), bool)


def test_is_scalar() -> None:
    check(assert_type(api.is_scalar(obj), bool), bool)
    check(assert_type(api.is_scalar(nparr), bool), bool)
    check(assert_type(api.is_scalar(dtylike), bool), bool)
    check(assert_type(api.is_scalar(arr), bool), bool)
    check(assert_type(api.is_scalar(dframe), bool), bool)
    check(assert_type(api.is_scalar(ind), bool), bool)


def test_is_signed_integer_dtype() -> None:
    check(assert_type(api.is_signed_integer_dtype(arr), bool), bool)
    check(assert_type(api.is_signed_integer_dtype(nparr), bool), bool)
    check(assert_type(api.is_signed_integer_dtype(dtylike), bool), bool)
    check(
        assert_type(api.is_signed_integer_dtype(dframe), bool),
        bool,
    )
    check(assert_type(api.is_signed_integer_dtype(ind), bool), bool)
    # check(assert_type(api.is_signed_integer_dtype(ExtensionDtype), bool), bool) pandas GH 50923


def test_is_sparse() -> None:
    check(assert_type(api.is_sparse(arr), bool), bool)
    check(assert_type(api.is_sparse(nparr), bool), bool)
    check(assert_type(api.is_sparse(dframe), bool), bool)


def test_is_string_dtype() -> None:
    check(assert_type(api.is_string_dtype(arr), bool), bool)
    check(assert_type(api.is_string_dtype(nparr), bool), bool)
    check(assert_type(api.is_string_dtype(dtylike), bool), bool)
    check(
        assert_type(api.is_string_dtype(dframe), bool),
        bool,
    )
    check(assert_type(api.is_string_dtype(ind), bool), bool)
    check(assert_type(api.is_string_dtype(ExtensionDtype), bool), bool)


def test_is_timedelta64_dtype() -> None:
    check(assert_type(api.is_timedelta64_dtype(arr), bool), bool)
    check(assert_type(api.is_timedelta64_dtype(nparr), bool), bool)
    check(assert_type(api.is_timedelta64_dtype(dtylike), bool), bool)
    check(
        assert_type(api.is_timedelta64_dtype(dframe), bool),
        bool,
    )
    check(assert_type(api.is_timedelta64_dtype(ind), bool), bool)
    # check(assert_type(api.is_timedelta64_dtype(ExtensionDtype), bool), bool) pandas GH 50923


def test_is_timedelta64_ns_dtype() -> None:
    check(assert_type(api.is_timedelta64_ns_dtype(arr), bool), bool)
    check(assert_type(api.is_timedelta64_ns_dtype(nparr), bool), bool)
    check(assert_type(api.is_timedelta64_ns_dtype(dtylike), bool), bool)
    check(
        assert_type(api.is_timedelta64_ns_dtype(dframe), bool),
        bool,
    )
    check(assert_type(api.is_timedelta64_ns_dtype(ind), bool), bool)
    check(assert_type(api.is_timedelta64_ns_dtype(ExtensionDtype), bool), bool)


def test_is_unsigned_integer_dtype() -> None:
    check(assert_type(api.is_unsigned_integer_dtype(arr), bool), bool)
    check(assert_type(api.is_unsigned_integer_dtype(nparr), bool), bool)
    check(assert_type(api.is_unsigned_integer_dtype(dtylike), bool), bool)
    check(
        assert_type(
            api.is_unsigned_integer_dtype(dframe),
            bool,
        ),
        bool,
    )
    check(assert_type(api.is_unsigned_integer_dtype(ind), bool), bool)
    # check(assert_type(api.is_unsigned_integer_dtype(ExtensionDtype), bool), bool) pandas GH 50923


def test_pandas_dtype() -> None:
    check(assert_type(api.pandas_dtype(arr), DtypeObj), type(np.dtype("i8")))


def test_infer_dtype() -> None:
    check(assert_type(api.infer_dtype([1, 2, 3]), str), str)


def test_union_categoricals() -> None:
    to_union = [pd.Categorical([1, 2, 3]), pd.Categorical([3, 4, 5])]
    check(assert_type(api.union_categoricals(to_union), pd.Categorical), pd.Categorical)


def test_check_extension_dtypes() -> None:
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


def test_from_dataframe() -> None:
    # GH 712
    check(
        assert_type(pd.api.interchange.from_dataframe(dframe), pd.DataFrame),
        pd.DataFrame,
    )
