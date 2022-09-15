from typing import (
    Any,
    Generic,
    overload,
)

import numpy as np
from pandas import (
    DataFrame,
    Series,
)
from pandas.core.generic import NDFrame
from pandas.core.window.rolling import BaseWindow

from pandas._typing import (
    AggFuncTypeBase,
    AggFuncTypeFrame,
    AggFuncTypeSeriesToFrame,
    Axis,
    NDFrameT,
    TimedeltaConvertibleTypes,
    WindowingEngine,
    WindowingEngineKwargs,
)

class ExponentialMovingWindow(BaseWindow[NDFrameT], Generic[NDFrameT]):
    com: Any = ...  # Incomplete
    span: Any = ...  # Incomplete
    halflife: Any = ...  # Incomplete
    alpha: Any = ...  # Incomplete
    adjust: Any = ...  # Incomplete
    ignore_na: Any = ...  # Incomplete
    times: Any = ...  # Incomplete
    def __init__(
        self,
        obj: NDFrameT,
        com: float | None = ...,
        span: float | None = ...,
        halflife: float | TimedeltaConvertibleTypes | None = ...,
        alpha: float | None = ...,
        min_periods: int | None = ...,
        adjust: bool = ...,
        ignore_na: bool = ...,
        axis: Axis = ...,
        times: str | np.ndarray | NDFrame | None = ...,
        method: str = ...,
        *,
        selection: Any | None = ...,
    ) -> None: ...
    @overload
    def aggregate(
        self: ExponentialMovingWindow[Series],
        func: AggFuncTypeBase,
        *args: Any,
        **kwargs: Any,
    ) -> Series: ...
    @overload
    def aggregate(
        self: ExponentialMovingWindow[Series],
        func: AggFuncTypeSeriesToFrame,
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def aggregate(
        self: ExponentialMovingWindow[DataFrame],
        func: AggFuncTypeFrame,
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    def mean(
        self,
        *,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
    ) -> NDFrameT: ...
    def sum(
        self,
        *,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
    ) -> NDFrameT: ...
    def std(self, bias: bool = ...) -> NDFrameT: ...
    def var(self, bias: bool = ...) -> NDFrameT: ...
    def cov(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        bias: bool = ...,
    ) -> NDFrameT: ...
    def corr(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
    ) -> NDFrameT: ...
