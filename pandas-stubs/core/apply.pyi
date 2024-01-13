import abc
from abc import abstractmethod
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Sequence,
)
from typing import (
    Any,
    Generic,
    Literal,
    TypeVar,
)

import numpy as np
from pandas import (
    DataFrame,
    Index,
    Series,
)
from pandas.core.generic import NDFrame
from pandas.core.groupby import GroupBy
from pandas.core.resample import Resampler
from pandas.core.window.rolling import BaseWindow

from pandas._libs.lib import NoDefault
from pandas._typing import (
    AggFuncType,
    AggFuncTypeDict,
    Axis,
    AxisInt,
    Incomplete,
    NDFrameT,
    npt,
)
from pandas.util._decorators import cache_readonly

_AggObjT = TypeVar("_AggObjT", bound=NDFrame | GroupBy | BaseWindow | Resampler)
_AggGroupByObjT = TypeVar("_AggGroupByObjT", bound=GroupBy | BaseWindow | Resampler)
_AggResamplerWindowObjT = TypeVar(
    "_AggResamplerWindowObjT", bound=BaseWindow | Resampler
)

def frame_apply(
    obj: DataFrame,
    func: AggFuncType,
    axis: Axis = ...,
    raw: bool = ...,
    result_type: str | None = ...,
    by_row: Literal[False, "compat"] = ...,
    args=...,
    kwargs=...,
) -> FrameApply: ...

class Apply(Generic[_AggObjT], metaclass=abc.ABCMeta):
    axis: AxisInt
    obj: _AggObjT
    raw: bool
    by_row: Literal[False, "compat", "_compat"]
    args: Incomplete
    kwargs: Incomplete
    result_type: Literal["reduce", "broadcast", "expand"] | None
    func: AggFuncType
    def __init__(
        self,
        obj: _AggObjT,
        func: AggFuncType,
        raw: bool,
        result_type: Literal["reduce", "broadcast", "expand"] | None,
        *,
        by_row: Literal[False, "compat", "_compat"] = ...,
        args,
        kwargs,
    ) -> None: ...
    @abstractmethod
    def apply(self): ...
    @abstractmethod
    def agg_or_apply_list_like(self, op_name: Literal["agg", "apply"]): ...
    @abstractmethod
    def agg_or_apply_dict_like(self, op_name: Literal["agg", "apply"]): ...
    def agg(self): ...
    def transform(self): ...
    def transform_dict_like(self, func: AggFuncTypeDict) -> DataFrame: ...
    def transform_str_or_callable(self, func: str | Callable[..., Incomplete]): ...
    def agg_list_like(self): ...
    def compute_list_like(
        self,
        op_name: Literal["agg", "apply"],
        selected_obj: Series | DataFrame,
        kwargs: dict[str, Any],
    ) -> tuple[list[Hashable], list[Any]]: ...
    def wrap_results_list_like(
        self, keys: list[Hashable], results: list[Series | DataFrame]
    ): ...
    def agg_dict_like(self): ...
    def compute_dict_like(
        self,
        op_name: Literal["agg", "apply"],
        selected_obj: Series | DataFrame,
        selection: Hashable | Sequence[Hashable],
        kwargs: dict[str, Any],
    ) -> tuple[list[Hashable], list[Any]]: ...
    def wrap_results_dict_like(
        self,
        selected_obj: Series | DataFrame,
        result_index: list[Hashable],
        result_data: list,
    ) -> Series | DataFrame: ...
    def apply_str(self): ...
    def apply_list_or_dict_like(self): ...
    def normalize_dictlike_arg(
        self, how: str, obj: DataFrame | Series, func: AggFuncTypeDict
    ) -> AggFuncTypeDict: ...

class NDFrameApply(Apply[NDFrameT], metaclass=abc.ABCMeta):
    @property
    def index(self) -> Index: ...
    @property
    def agg_axis(self) -> Index: ...
    def agg_or_apply_list_like(self, op_name: Literal["agg", "apply"]): ...
    def agg_or_apply_dict_like(self, op_name: Literal["agg", "apply"]): ...

class FrameApply(NDFrameApply[DataFrame]):
    def __init__(
        self,
        obj: DataFrame,
        func: AggFuncType,
        raw: bool,
        result_type: Literal["reduce", "broadcast", "expand"] | None,
        *,
        by_row: Literal[False, "compat"] = ...,
        args,
        kwargs,
    ) -> None: ...
    @property
    @abstractmethod
    def result_index(self) -> Index: ...
    @property
    @abstractmethod
    def result_columns(self) -> Index: ...
    @property
    @abstractmethod
    def series_generator(self) -> Iterator[Series]: ...
    @abstractmethod
    def wrap_results_for_axis(self, results: dict[int, Any], res_index: Index): ...
    @property
    def res_columns(self) -> Index: ...
    @property
    def columns(self) -> Index: ...
    @cache_readonly
    def values(self): ...
    def apply(self): ...
    def agg(self): ...
    def apply_empty_result(self): ...
    def apply_raw(self): ...
    def apply_broadcast(self, target: DataFrame) -> DataFrame: ...
    def apply_standard(self): ...
    def apply_series_generator(self) -> tuple[dict[int, Any], Index]: ...
    def wrap_results(self, results: dict[int, Any], res_index: Index): ...
    def apply_str(self): ...

class FrameRowApply(FrameApply):
    @property
    def series_generator(self) -> Iterator[Series]: ...
    @property
    def result_index(self) -> Index: ...
    @property
    def result_columns(self) -> Index: ...
    def wrap_results_for_axis(self, results: dict[int, Any], res_index: Index): ...

class FrameColumnApply(FrameApply):
    def apply_broadcast(self, target: DataFrame) -> DataFrame: ...
    @property
    def series_generator(self) -> Iterator[Series]: ...
    @property
    def result_index(self) -> Index: ...
    @property
    def result_columns(self) -> Index: ...
    def wrap_results_for_axis(self, results: dict[int, Any], res_index: Index): ...
    def infer_to_same_shape(
        self, results: dict[int, Any], res_index: Index
    ) -> DataFrame: ...

class SeriesApply(NDFrameApply[Series]):
    by_row: Literal[False, "compat", "_compat"]
    convert_dtype: bool
    def __init__(
        self,
        obj: Series,
        func: AggFuncType,
        *,
        convert_dtype: bool | NoDefault = ...,
        by_row: Literal[False, "compat", "_compat"] = ...,
        args,
        kwargs,
    ) -> None: ...
    def apply(self): ...
    def agg(self): ...
    def apply_empty_result(self) -> Series: ...
    def apply_compat(self): ...
    def apply_standard(self): ...

class GroupByApply(Apply[_AggGroupByObjT]):
    def __init__(
        self, obj: _AggGroupByObjT, func: AggFuncType, *, args, kwargs
    ) -> None: ...
    def apply(self): ...
    def transform(self): ...
    def agg_or_apply_list_like(self, op_name: Literal["agg", "apply"]): ...
    def agg_or_apply_dict_like(self, op_name: Literal["agg", "apply"]): ...

class ResamplerWindowApply(GroupByApply[_AggResamplerWindowObjT]):
    def __init__(
        self, obj: _AggResamplerWindowObjT, func: AggFuncType, *, args, kwargs
    ) -> None: ...
    def apply(self): ...
    def transform(self): ...

def reconstruct_func(
    func: AggFuncType | None, **kwargs
) -> tuple[bool, AggFuncType, list[str] | None, npt.NDArray[np.intp] | None]: ...
def is_multi_agg_with_relabel(**kwargs) -> bool: ...
def normalize_keyword_aggregation(
    kwargs: dict,
) -> tuple[dict[str, list], list[str], npt.NDArray[np.intp]]: ...
def relabel_result(
    result: DataFrame | Series,
    func: dict[str, list[Callable | str]],
    columns: Iterable[Hashable],
    order: Iterable[int],
) -> dict[Hashable, Series]: ...
def reconstruct_and_relabel_result(result, func, **kwargs): ...
def maybe_mangle_lambdas(agg_spec: Any) -> Any: ...
def validate_func_kwargs(
    kwargs: dict,
) -> tuple[list[str], list[str | Callable[..., Any]]]: ...
def include_axis(
    op_name: Literal["agg", "apply"], colg: Series | DataFrame
) -> bool: ...
def warn_alias_replacement(obj, func: Callable, alias: str) -> None: ...
