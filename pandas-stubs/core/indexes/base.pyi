from builtins import str as _str
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Sequence,
)
from datetime import (
    datetime,
    timedelta,
)
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    final,
    overload,
    type_check_only,
)

import numpy as np
from pandas import (
    DataFrame,
    DatetimeIndex,
    Interval,
    IntervalIndex,
    MultiIndex,
    Period,
    PeriodDtype,
    PeriodIndex,
    Series,
    TimedeltaIndex,
)
from pandas.core.arrays import ExtensionArray
from pandas.core.base import (
    IndexOpsMixin,
    NumListLike,
    _ListLike,
)
from pandas.core.strings.accessor import StringMethods
from typing_extensions import (
    Never,
    Self,
)

from pandas._libs.interval import _OrderableT
from pandas._typing import (
    C2,
    S1,
    T_COMPLEX,
    T_INT,
    AnyAll,
    ArrayLike,
    AxesData,
    DropKeep,
    Dtype,
    DtypeArg,
    DTypeLike,
    DtypeObj,
    GenericT,
    GenericT_co,
    HashableT,
    IgnoreRaise,
    Just,
    Label,
    Level,
    MaskType,
    NaPosition,
    ReindexMethod,
    Scalar,
    SequenceNotStr,
    SliceType,
    SupportsDType,
    TimedeltaDtypeArg,
    TimestampDtypeArg,
    np_1darray,
    np_ndarray_anyint,
    np_ndarray_bool,
    np_ndarray_complex,
    np_ndarray_float,
    np_ndarray_str,
    type_t,
)

class InvalidIndexError(Exception): ...

class Index(IndexOpsMixin[S1]):
    __hash__: ClassVar[None]  # type: ignore[assignment]
    # overloads with additional dtypes
    @overload
    def __new__(  # pyright: ignore[reportOverlappingOverload]
        cls,
        data: Sequence[bool | np.bool_] | IndexOpsMixin[bool] | np_ndarray_bool,
        *,
        dtype: Literal["bool"] | type_t[bool | np.bool_] = ...,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> Index[bool]: ...
    @overload
    def __new__(
        cls,
        data: Sequence[int | np.integer] | IndexOpsMixin[int] | np_ndarray_anyint,
        *,
        dtype: Literal["int"] | type_t[int | np.integer] = ...,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> Index[int]: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: Literal["int"] | type_t[int | np.integer],
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> Index[int]: ...
    @overload
    def __new__(
        cls,
        data: Sequence[float | np.floating] | IndexOpsMixin[float] | np_ndarray_float,
        *,
        dtype: Literal["float"] | type_t[float | np.floating] = ...,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> Index[float]: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: Literal["float"] | type_t[float | np.floating],
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> Index[float]: ...
    @overload
    def __new__(
        cls,
        data: (
            Sequence[complex | np.complexfloating]
            | IndexOpsMixin[complex]
            | np_ndarray_complex
        ),
        *,
        dtype: Literal["complex"] | type_t[complex | np.complexfloating] = ...,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> Index[complex]: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: Literal["complex"] | type_t[complex | np.complexfloating],
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> Index[complex]: ...
    # special overloads with dedicated Index-subclasses
    @overload
    def __new__(
        cls,
        data: Sequence[np.datetime64 | datetime] | IndexOpsMixin[datetime],
        *,
        dtype: TimestampDtypeArg = ...,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> DatetimeIndex: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: TimestampDtypeArg,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> DatetimeIndex: ...
    @overload
    def __new__(
        cls,
        data: Sequence[Period] | IndexOpsMixin[Period],
        *,
        dtype: PeriodDtype = ...,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> PeriodIndex: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: PeriodDtype,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> PeriodIndex: ...
    @overload
    def __new__(
        cls,
        data: Sequence[np.timedelta64 | timedelta] | IndexOpsMixin[timedelta],
        *,
        dtype: TimedeltaDtypeArg = ...,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> TimedeltaIndex: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: TimedeltaDtypeArg,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> TimedeltaIndex: ...
    @overload
    def __new__(
        cls,
        data: Sequence[Interval[_OrderableT]] | IndexOpsMixin[Interval[_OrderableT]],
        *,
        dtype: Literal["Interval"] = ...,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> IntervalIndex[Interval[_OrderableT]]: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: Literal["Interval"],
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> IntervalIndex[Interval[Any]]: ...
    # generic overloads
    @overload
    def __new__(
        cls,
        data: Iterable[S1] | IndexOpsMixin[S1],
        *,
        dtype: type[S1] = ...,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> Self: ...
    @overload
    def __new__(
        cls,
        data: AxesData = ...,
        *,
        dtype: type[S1],
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> Self: ...
    # fallback overload
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: Dtype = ...,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> Self: ...
    @property
    def str(
        self,
    ) -> StringMethods[  # pyrefly: ignore[bad-specialization]
        Self,
        MultiIndex,
        np_1darray[np.bool],
        Index[list[_str]],
        Index[int],
        Index[bytes],
        Index[_str],
        Index,
    ]: ...
    @final
    def is_(self, other) -> bool: ...
    def __len__(self) -> int: ...
    def __array__(
        self, dtype: _str | np.dtype = ..., copy: bool | None = ...
    ) -> np_1darray: ...
    def __array_wrap__(self, result, context=...): ...
    @property
    def dtype(self) -> DtypeObj: ...
    @final
    def ravel(self, order: _str = ...): ...
    def view(self, cls=...): ...
    def astype(self, dtype: DtypeArg, copy: bool = True) -> Index: ...
    def take(
        self,
        indices,
        axis: int = 0,
        allow_fill: bool = True,
        fill_value: Scalar | None = None,
        **kwargs,
    ): ...
    def repeat(self, repeats, axis=...): ...
    def copy(self, name: Hashable = ..., deep: bool = False) -> Self: ...
    @final
    def __copy__(self, **kwargs): ...
    @final
    def __deepcopy__(self, memo=...): ...
    def format(
        self, name: bool = ..., formatter: Callable | None = ..., na_rep: _str = ...
    ) -> list[_str]: ...
    def to_flat_index(self): ...
    def to_series(self, index=..., name: Hashable = ...) -> Series: ...
    def to_frame(self, index: bool = True, name=...) -> DataFrame: ...
    @property
    def name(self) -> Hashable | None: ...
    @name.setter
    def name(self, value: Hashable) -> None: ...
    @property
    def names(self) -> list[Hashable | None]: ...
    @names.setter
    def names(self, names: SequenceNotStr[Hashable | None]) -> None: ...
    def set_names(self, names, *, level=..., inplace: bool = ...): ...
    @overload
    def rename(self, name, *, inplace: Literal[False] = False) -> Self: ...
    @overload
    def rename(self, name, *, inplace: Literal[True]) -> None: ...
    @property
    def nlevels(self) -> int: ...
    def get_level_values(self, level: int | _str) -> Index: ...
    def droplevel(self, level: Level | list[Level] = 0): ...
    @property
    def is_monotonic_increasing(self) -> bool: ...
    @property
    def is_monotonic_decreasing(self) -> bool: ...
    @property
    def is_unique(self) -> bool: ...
    @property
    def has_duplicates(self) -> bool: ...
    @property
    def inferred_type(self) -> _str: ...
    def __reduce__(self): ...
    @property
    def hasnans(self) -> bool: ...
    @final
    def isna(self): ...
    isnull = ...
    @final
    def notna(self): ...
    notnull = ...
    def fillna(self, value=...): ...
    def dropna(self, how: AnyAll = "any") -> Self: ...
    def unique(self, level=...) -> Self: ...
    def drop_duplicates(self, *, keep: DropKeep = ...) -> Self: ...
    def duplicated(self, keep: DropKeep = "first") -> np_1darray[np.bool]: ...
    def __and__(self, other: Never) -> Never: ...
    def __rand__(self, other: Never) -> Never: ...
    def __or__(self, other: Never) -> Never: ...
    def __ror__(self, other: Never) -> Never: ...
    def __xor__(self, other: Never) -> Never: ...
    def __rxor__(self, other: Never) -> Never: ...
    def __neg__(self) -> Self: ...
    @final
    def __nonzero__(self) -> None: ...
    __bool__ = ...
    def union(
        self, other: list[HashableT] | Self, sort: bool | None = None
    ) -> Index: ...
    def intersection(
        self, other: list[S1] | Self, sort: bool | None = False
    ) -> Self: ...
    def difference(self, other: list | Self, sort: bool | None = None) -> Self: ...
    def symmetric_difference(
        self,
        other: list[S1] | Self,
        result_name: Hashable = ...,
        sort: bool | None = None,
    ) -> Self: ...
    def get_loc(self, key: Label) -> int | slice | np_1darray[np.bool]: ...
    def get_indexer(
        self, target, method: ReindexMethod | None = ..., limit=..., tolerance=...
    ): ...
    def reindex(
        self,
        target,
        method: ReindexMethod | None = ...,
        level=...,
        limit=...,
        tolerance=...,
    ): ...
    def join(
        self,
        other,
        *,
        how: _str = ...,
        level=...,
        return_indexers: bool = ...,
        sort: bool = ...,
    ): ...
    @property
    def values(self) -> np_1darray: ...
    @property
    def array(self) -> ExtensionArray: ...
    def memory_usage(self, deep: bool = False): ...
    def where(self, cond, other: Scalar | ArrayLike | None = None): ...
    def __contains__(self, key) -> bool: ...
    @final
    def __setitem__(self, key, value) -> None: ...
    @overload
    def __getitem__(
        self,
        idx: slice | np_ndarray_anyint | Sequence[int] | Index | MaskType,
    ) -> Self: ...
    @overload
    def __getitem__(self, idx: int | tuple[np_ndarray_anyint, ...]) -> S1: ...
    @overload
    def append(
        self: Index[C2], other: Index[C2] | Sequence[Index[C2]]
    ) -> Index[C2]: ...
    @overload
    def append(self, other: Index | Sequence[Index]) -> Index: ...
    def putmask(self, mask, value): ...
    def equals(self, other) -> bool: ...
    @final
    def identical(self, other) -> bool: ...
    @final
    def asof(self, label): ...
    def asof_locs(self, where, mask): ...
    def sort_values(
        self,
        *,
        return_indexer: bool = ...,
        ascending: bool = ...,
        na_position: NaPosition = ...,
        key: Callable[[Index], Index] | None = None,
    ): ...
    @final
    def sort(self, *args, **kwargs) -> None: ...
    def argsort(self, *args, **kwargs): ...
    def get_indexer_non_unique(self, target): ...
    @final
    def get_indexer_for(self, target, **kwargs): ...
    @final
    def groupby(self, values) -> dict[Hashable, np.ndarray]: ...
    def map(self, mapper, na_action=...) -> Index: ...
    def isin(self, values, level=...) -> np_1darray[np.bool]: ...
    def slice_indexer(
        self,
        start: Label | None = None,
        end: Label | None = None,
        step: int | None = None,
    ): ...
    def get_slice_bound(self, label, side): ...
    def slice_locs(
        self, start: SliceType = None, end: SliceType = None, step: int | None = None
    ): ...
    def delete(self, loc) -> Self: ...
    def insert(self, loc, item) -> Self: ...
    def drop(self, labels, errors: IgnoreRaise = "raise") -> Self: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    # Extra methods from old stubs
    def __eq__(self, other: object) -> np_1darray[np.bool]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __iter__(self) -> Iterator[S1]: ...
    def __ne__(self, other: object) -> np_1darray[np.bool]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __le__(self, other: Self | S1) -> np_1darray[np.bool]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __ge__(self, other: Self | S1) -> np_1darray[np.bool]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __lt__(self, other: Self | S1) -> np_1darray[np.bool]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __gt__(self, other: Self | S1) -> np_1darray[np.bool]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    @overload
    def __add__(self: Index[Never], other: _str) -> Never: ...
    @overload
    def __add__(self: Index[Never], other: complex | _ListLike | Index) -> Index: ...
    @overload
    def __add__(self, other: Index[Never]) -> Index: ...
    @overload
    def __add__(
        self: Index[bool],
        other: T_COMPLEX | Sequence[T_COMPLEX] | Index[T_COMPLEX],
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __add__(self: Index[bool], other: np_ndarray_bool) -> Index[bool]: ...
    @overload
    def __add__(self: Index[bool], other: np_ndarray_anyint) -> Index[int]: ...
    @overload
    def __add__(self: Index[bool], other: np_ndarray_float) -> Index[float]: ...
    @overload
    def __add__(self: Index[bool], other: np_ndarray_complex) -> Index[complex]: ...
    @overload
    def __add__(
        self: Index[int],
        other: (
            bool | Sequence[bool] | np_ndarray_bool | np_ndarray_anyint | Index[bool]
        ),
    ) -> Index[int]: ...
    @overload
    def __add__(
        self: Index[int],
        other: T_COMPLEX | Sequence[T_COMPLEX] | Index[T_COMPLEX],
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __add__(self: Index[int], other: np_ndarray_float) -> Index[float]: ...
    @overload
    def __add__(self: Index[int], other: np_ndarray_complex) -> Index[complex]: ...
    @overload
    def __add__(
        self: Index[float],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[T_INT]
        ),
    ) -> Index[float]: ...
    @overload
    def __add__(
        self: Index[float],
        other: T_COMPLEX | Sequence[T_COMPLEX] | Index[T_COMPLEX],
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __add__(self: Index[float], other: np_ndarray_complex) -> Index[complex]: ...
    @overload
    def __add__(
        self: Index[complex],
        other: (
            T_COMPLEX
            | Sequence[T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | np_ndarray_complex
            | Index[T_COMPLEX]
        ),
    ) -> Index[complex]: ...
    @overload
    def __add__(
        self: Index[_str],
        other: (
            np_ndarray_bool | np_ndarray_anyint | np_ndarray_float | np_ndarray_complex
        ),
    ) -> Never: ...
    @overload
    def __add__(
        self: Index[_str], other: _str | Sequence[_str] | np_ndarray_str | Index[_str]
    ) -> Index[_str]: ...
    @overload
    def __radd__(self: Index[Never], other: _str) -> Never: ...
    @overload
    def __radd__(self: Index[Never], other: complex | _ListLike | Index) -> Index: ...
    @overload
    def __radd__(
        self: Index[bool],
        other: T_COMPLEX | Sequence[T_COMPLEX] | Index[T_COMPLEX],
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __radd__(self: Index[bool], other: np_ndarray_bool) -> Index[bool]: ...
    @overload
    def __radd__(self: Index[bool], other: np_ndarray_anyint) -> Index[int]: ...
    @overload
    def __radd__(self: Index[bool], other: np_ndarray_float) -> Index[float]: ...
    @overload
    def __radd__(
        self: Index[int],
        other: (
            bool | Sequence[bool] | np_ndarray_bool | np_ndarray_anyint | Index[bool]
        ),
    ) -> Index[int]: ...
    @overload
    def __radd__(
        self: Index[int], other: T_COMPLEX | Sequence[T_COMPLEX] | Index[T_COMPLEX]
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __radd__(self: Index[int], other: np_ndarray_float) -> Index[float]: ...
    @overload
    def __radd__(
        self: Index[float],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[T_INT]
        ),
    ) -> Index[float]: ...
    @overload
    def __radd__(
        self: Index[float], other: T_COMPLEX | Sequence[T_COMPLEX] | Index[T_COMPLEX]
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __radd__(
        self: Index[complex],
        other: (
            T_COMPLEX
            | Sequence[T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[T_COMPLEX]
        ),
    ) -> Index[complex]: ...
    @overload
    def __radd__(
        self: Index[T_COMPLEX], other: np_ndarray_complex
    ) -> Index[complex]: ...
    @overload
    def __radd__(
        self: Index[_str],
        other: (
            np_ndarray_bool | np_ndarray_anyint | np_ndarray_float | np_ndarray_complex
        ),
    ) -> Never: ...
    @overload
    def __radd__(
        self: Index[_str], other: _str | Sequence[_str] | np_ndarray_str | Index[_str]
    ) -> Index[_str]: ...
    @overload
    def __sub__(self: Index[Never], other: DatetimeIndex) -> Never: ...
    @overload
    def __sub__(self: Index[Never], other: complex | _ListLike | Index) -> Index: ...
    @overload
    def __sub__(self, other: Index[Never]) -> Index: ...
    @overload
    def __sub__(
        self: Index[bool],
        other: Just[int] | Sequence[Just[int]] | np_ndarray_anyint | Index[int],
    ) -> Index[int]: ...
    @overload
    def __sub__(
        self: Index[bool],
        other: Just[float] | Sequence[Just[float]] | np_ndarray_float | Index[float],
    ) -> Index[float]: ...
    @overload
    def __sub__(
        self: Index[int],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | Index[bool]
            | Index[int]
        ),
    ) -> Index[int]: ...
    @overload
    def __sub__(
        self: Index[int],
        other: Just[float] | Sequence[Just[float]] | np_ndarray_float | Index[float],
    ) -> Index[float]: ...
    @overload
    def __sub__(
        self: Index[float],
        other: (
            float
            | Sequence[float]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[bool]
            | Index[int]
            | Index[float]
        ),
    ) -> Index[float]: ...
    @overload
    def __sub__(
        self: Index[complex],
        other: (
            T_COMPLEX
            | Sequence[T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[T_COMPLEX]
        ),
    ) -> Index[complex]: ...
    @overload
    def __sub__(
        self: Index[T_COMPLEX],
        other: (
            Just[complex]
            | Sequence[Just[complex]]
            | np_ndarray_complex
            | Index[complex]
        ),
    ) -> Index[complex]: ...
    @overload
    def __rsub__(self: Index[Never], other: DatetimeIndex) -> Never: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: Index[Never], other: complex | NumListLike | Index) -> Index: ...
    @overload
    def __rsub__(self, other: Index[Never]) -> Index: ...
    @overload
    def __rsub__(
        self: Index[bool],
        other: Just[int] | Sequence[Just[int]] | np_ndarray_anyint | Index[int],
    ) -> Index[int]: ...
    @overload
    def __rsub__(
        self: Index[bool],
        other: Just[float] | Sequence[Just[float]] | np_ndarray_float | Index[float],
    ) -> Index[float]: ...
    @overload
    def __rsub__(
        self: Index[int],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | Index[bool]
            | Index[int]
        ),
    ) -> Index[int]: ...
    @overload
    def __rsub__(
        self: Index[int],
        other: Just[float] | Sequence[Just[float]] | np_ndarray_float | Index[float],
    ) -> Index[float]: ...
    @overload
    def __rsub__(
        self: Index[float],
        other: (
            float
            | Sequence[float]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[bool]
            | Index[int]
            | Index[float]
        ),
    ) -> Index[float]: ...
    @overload
    def __rsub__(
        self: Index[complex],
        other: (
            T_COMPLEX
            | Sequence[T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[T_COMPLEX]
        ),
    ) -> Index[complex]: ...
    @overload
    def __rsub__(
        self: Index[T_COMPLEX],
        other: (
            Just[complex]
            | Sequence[Just[complex]]
            | np_ndarray_complex
            | Index[complex]
        ),
    ) -> Index[complex]: ...
    @overload
    def __mul__(
        self: Index[int] | Index[float], other: timedelta
    ) -> TimedeltaIndex: ...
    @overload
    def __mul__(
        self, other: float | Sequence[float] | Index[int] | Index[float]
    ) -> Self: ...
    def __rmul__(
        self, other: float | Sequence[float] | Index[int] | Index[float]
    ) -> Self: ...
    def __floordiv__(
        self, other: float | Sequence[float] | Index[int] | Index[float]
    ) -> Self: ...
    def __rfloordiv__(
        self, other: float | Sequence[float] | Index[int] | Index[float]
    ) -> Self: ...
    def __truediv__(
        self, other: float | Sequence[float] | Index[int] | Index[float]
    ) -> Self: ...
    def __rtruediv__(
        self, other: float | Sequence[float] | Index[int] | Index[float]
    ) -> Self: ...
    def infer_objects(self, copy: bool = True) -> Self: ...

@type_check_only
class _IndexSubclassBase(Index[S1], Generic[S1, GenericT_co]):
    @overload
    def to_numpy(  # pyrefly: ignore
        self,
        dtype: None = None,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs,
    ) -> np_1darray[GenericT_co]: ...
    @overload
    def to_numpy(
        self,
        dtype: np.dtype[GenericT] | SupportsDType[GenericT] | type[GenericT],
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs,
    ) -> np_1darray[GenericT]: ...
    @overload
    def to_numpy(
        self,
        dtype: DTypeLike,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs,
    ) -> np_1darray: ...
