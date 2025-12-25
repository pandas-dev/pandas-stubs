from __future__ import annotations

from builtins import type as type_t
from collections.abc import (
    Callable,
    Iterable,
    MutableSequence,
    Sequence,
)
import decimal
import numbers
import sys
from typing import (
    Any,
    Concatenate,
    cast,
    overload,
)

import numpy as np
from pandas.api.extensions import (
    no_default,
    register_extension_dtype,
)
from pandas.api.types import (
    is_list_like,
    is_scalar,
)
from pandas.core import arraylike
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.indexers import check_array_indexer
from pandas.core.series import Series
from typing_extensions import Self

from pandas._typing import (
    ArrayLike,
    AstypeArg,
    Dtype,
    ListLike,
    ScalarIndexer,
    SequenceIndexer,
    SequenceNotStr,
    TakeIndexer,
)

from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
    is_dtype_equal,
    is_float,
    pandas_dtype,
)

from tests._typing import (
    np_1darray,
    np_1darray_bool,
    np_ndarray,
)


@register_extension_dtype
class DecimalDtype(ExtensionDtype):
    type = decimal.Decimal
    name = "decimal"
    _metadata = ("context",)

    @property
    def na_value(self) -> decimal.Decimal:
        return decimal.Decimal("NaN")

    def __init__(self, context: decimal.Context | None = None) -> None:
        self.context = context or decimal.getcontext()

    def __repr__(self) -> str:
        return f"DecimalDtype(context={self.context})"

    @classmethod
    def construct_array_type(cls) -> type_t[DecimalArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return DecimalArray

    @property
    def _is_numeric(self) -> bool:
        return True


class DecimalArray(OpsMixin, ExtensionArray):
    __array_priority__ = 1000

    def __init__(
        self,
        values: MutableSequence[decimal._DecimalNew] | np_ndarray | ExtensionArray,
        dtype: DecimalDtype | None = None,
        copy: bool = False,
        context: decimal.Context | None = None,
    ) -> None:
        for i, val in enumerate(values):
            if is_float(val):
                if np.isnan(val):
                    values[i] = DecimalDtype().na_value
                else:
                    fval = float(val)  # Handle numpy case
                    values[i] = DecimalDtype.type(fval)
            elif not isinstance(val, decimal.Decimal):
                raise TypeError("All values must be of type " + str(decimal.Decimal))
        values_np = np.asarray(values, dtype=object)

        self._data = values_np
        # Some aliases for common attribute names to ensure pandas supports
        # these
        self._items = self.data = self._data
        # those aliases are currently not working due to assumptions
        # in internal code (GH-20735)
        # self._values = self.values = self.data
        self._dtype = DecimalDtype(context)

    @property
    def dtype(self) -> DecimalDtype:
        return self._dtype

    @classmethod
    def _from_sequence(
        cls,
        scalars: list[decimal._DecimalNew] | np_ndarray | ExtensionArray,
        dtype: DecimalDtype | None = None,
        copy: bool = False,
    ) -> Self:
        return cls(scalars)

    @classmethod
    def _from_sequence_of_strings(
        cls,
        strings: SequenceNotStr[str],
        dtype: DecimalDtype | None = None,
        copy: bool = False,
    ) -> Self:
        return cls._from_sequence([decimal.Decimal(x) for x in strings], dtype, copy)

    @classmethod
    def _from_factorized(
        cls,
        values: list[decimal._DecimalNew] | np_ndarray | ExtensionArray,
        original: Any,
    ) -> Self:
        return cls(values)

    _HANDLED_TYPES = (decimal.Decimal, numbers.Number, np.ndarray)

    def to_numpy(
        self,
        dtype: np.typing.DTypeLike | None = None,
        copy: bool = False,
        na_value: object = no_default,
        decimals: int | None = None,
    ) -> np_ndarray:
        result = np.asarray(self, dtype=dtype)
        if decimals is not None:
            result = np.asarray([round(x, decimals) for x in result])
        return result

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        if not all(
            isinstance(t, self._HANDLED_TYPES + (DecimalArray,)) for t in inputs
        ):
            return NotImplemented

        if "out" in kwargs:
            return cast("Callable[Concatenate[Any, np.ufunc, str, ...], Any]", arraylike.dispatch_ufunc_with_out)(  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue]
                self, ufunc, method, *inputs, **kwargs
            )

        inputs = tuple(x._data if isinstance(x, DecimalArray) else x for x in inputs)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if method == "reduce":
            result = cast("Callable[Concatenate[Any, np.ufunc, str, ...], Any]", arraylike.dispatch_reduction_ufunc)(  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue]
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                return result

        def reconstruct(
            x: (
                decimal.Decimal
                | numbers.Number
                | list[decimal._DecimalNew]
                | np_ndarray
            ),
        ) -> decimal.Decimal | numbers.Number | DecimalArray:
            if isinstance(x, (decimal.Decimal, numbers.Number)):
                return x
            return DecimalArray._from_sequence(x)

        if ufunc.nout > 1:
            return tuple(reconstruct(x) for x in result)
        return reconstruct(result)

    def __getitem__(self, item: ScalarIndexer | SequenceIndexer) -> Any:
        if isinstance(item, numbers.Integral):
            return self._data[item]
        # array, slice.
        item = check_array_indexer(  # pyright: ignore[reportCallIssue,reportUnknownVariableType]
            self, item  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        )
        return type(self)(self._data[item])

    def take(
        self, indexer: TakeIndexer, *, allow_fill: bool = False, fill_value: Any = None
    ) -> DecimalArray:
        from pandas.api.extensions import take

        data = self._data
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        result = take(data, indexer, fill_value=fill_value, allow_fill=allow_fill)
        return self._from_sequence(result)

    def copy(self) -> DecimalArray:
        return type(self)(self._data.copy(), dtype=self.dtype)

    if sys.version_info >= (3, 11):

        @overload
        def astype(self, dtype: np.dtype, copy: bool = True) -> np_1darray: ...

    else:

        @overload
        def astype(self, dtype: np.dtype[Any], copy: bool = True) -> np_1darray: ...
    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = True) -> ExtensionArray: ...
    @overload
    def astype(self, dtype: AstypeArg, copy: bool = True) -> ArrayLike: ...

    def astype(self, dtype: Dtype, copy: bool = True):
        if is_dtype_equal(dtype, self._dtype):
            if not copy:
                return self
        dtype = pandas_dtype(dtype)
        if isinstance(dtype, type(self.dtype)):
            return type(self)(self._data, copy=copy, context=dtype.context)

        return super().astype(dtype, copy=copy)

    def __setitem__(
        self,
        key: int | slice[Any, Any, Any] | ListLike,
        value: decimal._DecimalNew | Sequence[decimal._DecimalNew],
    ) -> None:
        if is_list_like(value):
            assert isinstance(value, Iterable)
            if is_scalar(key):
                raise ValueError("setting an array element with a sequence.")
            value = [
                decimal.Decimal(v)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
                for v in value
            ]
        else:
            value = decimal.Decimal(value)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]

        key = check_array_indexer(self, key)
        self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, item: Any) -> bool | np.bool_:
        if not isinstance(item, decimal.Decimal):
            return False
        if item.is_nan():
            return self.isna().any()
        return super().__contains__(item)

    @property
    def nbytes(self) -> int:
        n = len(self)
        if n:
            return n * sys.getsizeof(self[0])
        return 0

    def isna(self) -> np_1darray_bool:
        if sys.version_info < (3, 11):
            return np.array([x.is_nan() for x in self._data], bool)  # type: ignore[return-value] # pyright: ignore[reportReturnType]
        return np.array([x.is_nan() for x in self._data], bool)

    @property
    def _na_value(self) -> decimal.Decimal:
        return decimal.Decimal("NaN")

    def _formatter(self, boxed: bool = False) -> Callable[..., str]:
        if boxed:
            return "Decimal: {}".format
        return repr

    @classmethod
    def _concat_same_type(cls, to_concat: Iterable[Self]) -> Self:
        return cls(np.concatenate([x._data for x in to_concat]))

    def _reduce(self, name: str, *, skipna: bool = True, **kwargs: Any) -> Any:
        if skipna:
            # If we don't have any NAs, we can ignore skipna
            if self.isna().any():
                other = self[~self.isna()]
                return other._reduce(name, **kwargs)

        if name == "sum" and len(self) == 0:
            # GH#29630 avoid returning int 0 or np.bool_(False) on old numpy
            return decimal.Decimal(0)

        try:
            op = getattr(self.data, name)
        except AttributeError as err:
            raise NotImplementedError(
                f"decimal does not support the {name} operation"
            ) from err
        return op(axis=0)

    def _cmp_method(
        self, other: Any, op: Callable[[Self, ExtensionArray | list[Any]], bool]
    ) -> np_1darray_bool:
        # For use with OpsMixin
        def convert_values(param: Any) -> ExtensionArray | list[Any]:
            if isinstance(param, ExtensionArray) or is_list_like(param):
                ovalues = param
            else:
                # Assume it's an object
                ovalues = [param] * len(self)
            return ovalues

        lvalues = self
        rvalues = convert_values(other)

        # If the operator is not defined for the underlying objects,
        # a TypeError should be raised
        res = [op(a, b) for (a, b) in zip(lvalues, rvalues)]

        return cast(np_1darray_bool, np.asarray(res, dtype=bool))

    def value_counts(self, dropna: bool = True) -> Series:
        from pandas.core.algorithms import (  # type: ignore[attr-defined] # isort: skip
            value_counts,  # pyright: ignore[reportAttributeAccessIssue,reportAttributeAccessIssue,reportUnknownVariableType]
        )

        return value_counts(
            self.to_numpy(), dropna=dropna
        )  # pyright: ignore[reportUnknownVariableType]

    @classmethod
    def _add_arithmetic_ops(cls) -> None: ...
    @classmethod
    def _add_comparison_ops(cls) -> None: ...
    @classmethod
    def _add_logical_ops(cls) -> None: ...


DecimalArray._add_arithmetic_ops()
