from typing import (
    Callable,
    ClassVar,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import numpy as np
from pandas import (
    DataFrame,
    Series,
)
from pandas.core.arrays import ExtensionArray
from pandas.core.base import (
    IndexOpsMixin,
    PandasObject,
)
from pandas.core.strings import StringMethods

from pandas._typing import (
    T1,
    Dtype,
    DtypeArg,
    DtypeObj,
    Label,
    Level,
    Scalar,
    np_ndarray_bool,
    np_ndarray_int64,
    np_ndarray_str,
)

class InvalidIndexError(Exception): ...

_str = str

class Index(IndexOpsMixin, PandasObject):
    __hash__: ClassVar[None]  # type: ignore[assignment]

    def __new__(
        cls,
        data: Iterable = ...,
        dtype=...,
        copy: bool = ...,
        name=...,
        tupleize_cols: bool = ...,
        **kwargs,
    ) -> Index: ...
    def __init__(
        self,
        data: Iterable,
        dtype=...,
        copy: bool = ...,
        name=...,
        tupleize_cols: bool = ...,
    ): ...
    @property
    def str(self) -> StringMethods[Index]: ...
    @property
    def asi8(self) -> np_ndarray_int64: ...
    def is_(self, other) -> bool: ...
    def __len__(self) -> int: ...
    def __array__(self, dtype=...) -> np.ndarray: ...
    def __array_wrap__(self, result, context=...): ...
    @property
    def dtype(self) -> DtypeObj: ...
    def ravel(self, order: _str = ...): ...
    def view(self, cls=...): ...
    def astype(self, dtype: DtypeArg, copy: bool = ...) -> Index: ...
    def take(
        self, indices, axis: int = ..., allow_fill: bool = ..., fill_value=..., **kwargs
    ): ...
    def repeat(self, repeats, axis=...): ...
    def copy(self, name=..., deep: bool = ...) -> Index: ...
    def __copy__(self, **kwargs): ...
    def __deepcopy__(self, memo=...): ...
    def format(
        self, name: bool = ..., formatter: Optional[Callable] = ..., na_rep: _str = ...
    ) -> List[_str]: ...
    def to_native_types(self, slicer=..., **kwargs): ...
    def to_flat_index(self): ...
    def to_series(self, index=..., name=...) -> Series: ...
    def to_frame(self, index: bool = ..., name=...) -> DataFrame: ...
    @property
    def name(self): ...
    @name.setter
    def name(self, value) -> None: ...
    @property
    def names(self) -> List[_str]: ...
    @names.setter
    def names(self, names: List[_str]): ...
    def set_names(self, names, level=..., inplace: bool = ...): ...
    def rename(self, name, inplace: bool = ...): ...
    @property
    def nlevels(self) -> int: ...
    def sortlevel(self, level=..., ascending: bool = ..., sort_remaining=...): ...
    def get_level_values(self, level: Union[int, _str]) -> Index: ...
    def droplevel(self, level: Union[Level, List[Level]] = ...): ...
    @property
    def is_monotonic(self) -> bool: ...
    @property
    def is_monotonic_increasing(self) -> bool: ...
    @property
    def is_monotonic_decreasing(self) -> bool: ...
    def is_unique(self) -> bool: ...
    @property
    def has_duplicates(self) -> bool: ...
    def is_boolean(self) -> bool: ...
    def is_integer(self) -> bool: ...
    def is_floating(self) -> bool: ...
    def is_numeric(self) -> bool: ...
    def is_object(self) -> bool: ...
    def is_categorical(self) -> bool: ...
    def is_interval(self) -> bool: ...
    def is_mixed(self) -> bool: ...
    def holds_integer(self): ...
    def inferred_type(self): ...
    def is_all_dates(self) -> bool: ...
    def __reduce__(self): ...
    def hasnans(self) -> bool: ...
    def isna(self): ...
    isnull = ...
    def notna(self): ...
    notnull = ...
    def fillna(self, value=..., downcast=...): ...
    def dropna(self, how: _str = ...): ...
    def unique(self, level=...) -> Index: ...
    def drop_duplicates(
        self, keep: Literal["first", "last", False] = ...
    ) -> IndexOpsMixin: ...
    def duplicated(self, keep: _str = ...): ...
    def __add__(self, other) -> Index: ...
    def __radd__(self, other) -> Index: ...
    def __iadd__(self, other) -> Index: ...
    def __sub__(self, other) -> Index: ...
    def __rsub__(self, other) -> Index: ...
    def __and__(self, other) -> Index: ...
    def __or__(self, other) -> Index: ...
    def __xor__(self, other) -> Index: ...
    def __nonzero__(self) -> None: ...
    __bool__ = ...
    def union(self, other: Union[List[T1], Index], sort=...) -> Index: ...
    def intersection(
        self, other: Union[List[T1], Index], sort: bool = ...
    ) -> Index: ...
    def difference(self, other: Union[List[Union[T1, None]], Index]) -> Index: ...
    def symmetric_difference(
        self, other: Union[List[T1], Index], result_name=..., sort=...
    ) -> Index: ...
    def get_loc(self, key, method=..., tolerance=...): ...
    def get_indexer(self, target, method=..., limit=..., tolerance=...): ...
    def reindex(self, target, method=..., level=..., limit=..., tolerance=...): ...
    def join(
        self,
        other,
        how: _str = ...,
        level=...,
        return_indexers: bool = ...,
        sort: bool = ...,
    ): ...
    @property
    def values(self) -> np.ndarray: ...
    def array(self) -> ExtensionArray: ...
    def memory_usage(self, deep: bool = ...): ...
    def where(self, cond, other=...): ...
    def is_type_compatible(self, kind) -> bool: ...
    def __contains__(self, key) -> bool: ...
    def __setitem__(self, key, value) -> None: ...
    @overload
    def __getitem__(
        self, idx: Union[slice, np_ndarray_int64, Index, Series[bool]]
    ) -> Index: ...
    @overload
    def __getitem__(self, idx: Union[int, Tuple[np_ndarray_int64, ...]]) -> Scalar: ...
    def append(self, other): ...
    def putmask(self, mask, value): ...
    def equals(self, other) -> bool: ...
    def identical(self, other) -> bool: ...
    def asof(self, label): ...
    def asof_locs(self, where, mask): ...
    def sort_values(self, return_indexer: bool = ..., ascending: bool = ...): ...
    def sort(self, *args, **kwargs) -> None: ...
    def shift(self, periods: int = ..., freq=...) -> None: ...
    def argsort(self, *args, **kwargs): ...
    def get_value(self, series, key): ...
    def set_value(self, arr, key, value) -> None: ...
    def get_indexer_non_unique(self, target): ...
    def get_indexer_for(self, target, **kwargs): ...
    def groupby(self, values) -> Dict[Hashable, np.ndarray]: ...
    def map(self, mapper, na_action=...) -> Index: ...
    def isin(self, values, level=...) -> np_ndarray_bool: ...
    def slice_indexer(self, start=..., end=..., step=..., kind=...): ...
    def get_slice_bound(self, label, side, kind): ...
    def slice_locs(self, start=..., end=..., step=..., kind=...): ...
    def delete(self, loc): ...
    def insert(self, loc, item): ...
    def drop(self, labels, *, errors: _str = ...) -> Index: ...
    @property
    def shape(self) -> Tuple[int, ...]: ...
    # Extra methods from old stubs
    def __eq__(self, other: object) -> np_ndarray_bool: ...  # type: ignore[override]
    def __iter__(self) -> Iterator: ...
    def __ne__(self, other: object) -> np_ndarray_bool: ...  # type: ignore[override]
    def to_numpy(self) -> np.ndarray: ...

def ensure_index_from_sequences(
    sequences: Sequence[Sequence[Dtype]], names: List[str] = ...
) -> Index: ...
def ensure_index(index_like: Union[Sequence, Index], copy: bool = ...) -> Index: ...
def maybe_extract_name(name, obj, cls) -> Label: ...
