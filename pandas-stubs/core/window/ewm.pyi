from typing import (
    Any,
    Generic,
)

import numpy as np
from pandas import (
    DataFrame,
    Series,
)
from pandas.core.generic import NDFrame
from pandas.core.window.rolling import BaseWindow

from pandas._typing import (
    Axis,
    NDFrameT,
    TimedeltaConvertibleTypes,
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
    def online(self, engine: str = ..., engine_kwargs: Any | None = ...): ...
    def aggregate(self, func, *args, **kwargs) -> NDFrameT: ...
    agg = aggregate
    def mean(
        self,
        *args,
        engine: Any | None = ...,
        engine_kwargs: Any | None = ...,
        **kwargs,
    ) -> NDFrameT: ...
    def sum(
        self,
        *args,
        engine: Any | None = ...,
        engine_kwargs: Any | None = ...,
        **kwargs,
    ) -> NDFrameT: ...
    def std(self, bias: bool = ..., *args, **kwargs) -> NDFrameT: ...
    def vol(self, bias: bool = ..., *args, **kwargs) -> NDFrameT: ...
    def var(self, bias: bool = ..., *args, **kwargs) -> NDFrameT: ...
    def cov(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        bias: bool = ...,
        **kwargs,
    ) -> NDFrameT: ...
    def corr(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        **kwargs,
    ) -> NDFrameT: ...
