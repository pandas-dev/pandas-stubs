from __future__ import annotations

from typing import List

from pandas.core.arrays import ExtensionArray as ExtensionArray
from pandas.core.base import PandasObject as PandasObject

class Block(PandasObject):
    is_numeric: bool = ...
    is_float: bool = ...
    is_integer: bool = ...
    is_complex: bool = ...
    is_datetime: bool = ...
    is_datetimetz: bool = ...
    is_timedelta: bool = ...
    is_bool: bool = ...
    is_object: bool = ...
    is_categorical: bool = ...
    is_extension: bool = ...
    ndim = ...
    values = ...
    def __init__(self, values, placement, ndim=...) -> None: ...
    @property
    def is_view(self): ...
    @property
    def is_datelike(self): ...
    def is_categorical_astype(self, dtype): ...
    def external_values(self, dtype=...): ...
    def internal_values(self, dtype=...): ...
    def array_values(self) -> ExtensionArray: ...
    def get_values(self, dtype=...): ...
    def get_block_values(self, dtype=...): ...
    def to_dense(self): ...
    @property
    def fill_value(self): ...
    @property
    def mgr_locs(self): ...
    @mgr_locs.setter
    def mgr_locs(self, new_mgr_locs) -> None: ...
    @property
    def array_dtype(self): ...
    def make_block(self, values, placement=...) -> Block: ...
    def make_block_same_class(self, values, placement=..., ndim=...): ...
    def __len__(self) -> int: ...
    def getitem_block(self, slicer, new_mgr_locs=...): ...
    @property
    def shape(self): ...
    @property
    def dtype(self): ...
    @property
    def ftype(self): ...
    def merge(self, other): ...
    def concat_same_type(self, to_concat, placement=...): ...
    def iget(self, i): ...
    def set(self, locs, values) -> None: ...
    def delete(self, loc) -> None: ...
    def apply(self, func, **kwargs): ...
    def fillna(self, value, limit=..., inplace: bool = ..., downcast=...): ...
    def split_and_operate(self, mask, f, inplace: bool): ...
    def downcast(self, dtypes=...): ...
    def astype(self, dtype, copy: bool = ..., errors: str = ...): ...
    def convert(
        self,
        copy: bool = ...,
        datetime: bool = ...,
        numeric: bool = ...,
        timedelta: bool = ...,
        coerce: bool = ...,
    ): ...
    def to_native_types(self, slicer=..., na_rep: str = ..., quoting=..., **kwargs): ...
    def copy(self, deep: bool = ...): ...
    def replace(
        self,
        to_replace,
        value,
        inplace: bool = ...,
        filter=...,
        regex: bool = ...,
        convert: bool = ...,
    ): ...
    def setitem(self, indexer, value): ...
    def putmask(
        self,
        mask,
        new,
        align: bool = ...,
        inplace: bool = ...,
        axis: int = ...,
        transpose: bool = ...,
    ): ...
    def coerce_to_target_dtype(self, other): ...
    def interpolate(
        self,
        *,
        method: str = ...,
        axis: int = ...,
        index=...,
        inplace: bool = ...,
        limit=...,
        limit_direction: str = ...,
        limit_area=...,
        fill_value=...,
        downcast=...,
        **kwargs,
    ): ...
    def take_nd(self, indexer, axis, new_mgr_locs=..., fill_tuple=...): ...
    def diff(self, n: int, axis: int = ...) -> list[Block]: ...
    def shift(self, periods, axis: int = ..., fill_value=...): ...
    def where(
        self, other, cond, align=..., errors=..., try_cast: bool = ..., axis: int = ...
    ) -> list[Block]: ...
    def equals(self, other) -> bool: ...
    def quantile(self, qs, interpolation: str = ..., axis: int = ...): ...

class NonConsolidatableMixIn:
    def __init__(self, values, placement, ndim=...) -> None: ...
    @property
    def shape(self): ...
    def iget(self, i): ...
    def should_store(self, value): ...
    values = ...
    def set(self, locs, values, check: bool = ...) -> None: ...
    def putmask(
        self,
        mask,
        new,
        align: bool = ...,
        inplace: bool = ...,
        axis: int = ...,
        transpose: bool = ...,
    ): ...

class ExtensionBlock(NonConsolidatableMixIn, Block):
    is_extension: bool = ...
    def __init__(self, values, placement, ndim=...) -> None: ...
    @property
    def fill_value(self): ...
    @property
    def is_view(self): ...
    @property
    def is_numeric(self): ...
    def setitem(self, indexer, value): ...
    def get_values(self, dtype=...): ...
    def array_values(self) -> ExtensionArray: ...
    def to_dense(self): ...
    def to_native_types(self, slicer=..., na_rep: str = ..., quoting=..., **kwargs): ...
    def take_nd(self, indexer, axis: int = ..., new_mgr_locs=..., fill_tuple=...): ...
    def concat_same_type(self, to_concat, placement=...): ...
    def fillna(self, value, limit=..., inplace: bool = ..., downcast=...): ...
    def interpolate(
        self,
        *,
        method: str = ...,
        axis: int = ...,
        inplace: bool = ...,
        limit=...,
        fill_value=...,
        **kwargs,
    ): ...
    def diff(self, n: int, axis: int = ...) -> list[Block]: ...
    def shift(
        self, periods: int, axis: int = ..., fill_value=...
    ) -> list[ExtensionBlock]: ...
    def where(
        self, other, cond, align=..., errors=..., try_cast: bool = ..., axis: int = ...
    ) -> list[Block]: ...

class ObjectValuesExtensionBlock(ExtensionBlock):
    def external_values(self, dtype=...): ...

class NumericBlock(Block):
    is_numeric: bool = ...

class DatetimeLikeBlockMixin:
    @property
    def fill_value(self): ...
    def get_values(self, dtype=...): ...
    def iget(self, i): ...
    def shift(self, periods, axis: int = ..., fill_value=...): ...

class DatetimeBlock(DatetimeLikeBlockMixin, Block):
    is_datetime: bool = ...
    def __init__(self, values, placement, ndim=...) -> None: ...
    def astype(self, dtype, copy: bool = ..., errors: str = ...): ...
    def to_native_types(
        self, slicer=..., na_rep=..., date_format=..., quoting=..., **kwargs
    ): ...
    def should_store(self, value): ...
    def set(self, locs, values) -> None: ...
    def external_values(self): ...
    def array_values(self) -> ExtensionArray: ...

class DatetimeTZBlock(DatetimeBlock):
    is_datetimetz: bool = ...
    is_extension: bool = ...
    fill_value = ...
    @property
    def is_view(self): ...
    def get_values(self, dtype=...): ...
    def to_dense(self): ...
    def diff(self, n: int, axis: int = ...) -> list[Block]: ...
    def concat_same_type(self, to_concat, placement=...): ...
    def fillna(self, value, limit=..., inplace: bool = ..., downcast=...): ...
    def setitem(self, indexer, value): ...
    def equals(self, other) -> bool: ...
    def quantile(self, qs, interpolation: str = ..., axis: int = ...): ...

class BoolBlock(NumericBlock):
    is_bool: bool = ...
    def should_store(self, value): ...
    def replace(
        self,
        to_replace,
        value,
        inplace: bool = ...,
        filter=...,
        regex: bool = ...,
        convert: bool = ...,
    ): ...

class ObjectBlock(Block):
    is_object: bool = ...
    def __init__(self, values, placement=..., ndim: int = ...) -> None: ...
    @property
    def is_bool(self): ...
    def convert(
        self,
        copy: bool = ...,
        datetime: bool = ...,
        numeric: bool = ...,
        timedelta: bool = ...,
        coerce: bool = ...,
    ): ...
    def should_store(self, value): ...
    def replace(
        self,
        to_replace,
        value,
        inplace: bool = ...,
        filter=...,
        regex: bool = ...,
        convert: bool = ...,
    ): ...

class CategoricalBlock(ExtensionBlock):
    is_categorical: bool = ...
    def __init__(self, values, placement, ndim=...) -> None: ...
    @property
    def array_dtype(self): ...
    def to_dense(self): ...
    def to_native_types(self, slicer=..., na_rep: str = ..., quoting=..., **kwargs): ...
    def concat_same_type(self, to_concat, placement=...): ...
    def replace(
        self,
        to_replace,
        value,
        inplace: bool = ...,
        filter=...,
        regex: bool = ...,
        convert: bool = ...,
    ): ...

def get_block_type(values, dtype=...): ...
def make_block(values, placement, klass=..., ndim=..., dtype=...): ...
