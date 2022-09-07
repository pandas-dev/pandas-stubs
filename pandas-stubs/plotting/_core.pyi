from typing import (
    Callable,
    Hashable,
    Literal,
    NamedTuple,
    Sequence,
    Union,
    overload,
)

from matplotlib.axes import Axes
from matplotlib.axes._subplots import AxesSubplot
from matplotlib.lines import Line2D
import numpy as np
from pandas.core.base import PandasObject
from pandas.core.frame import DataFrame
from scipy.stats.kde import gaussian_kde

from pandas._typing import (
    HashableT,
    npt,
)

class _BoxPlotT(NamedTuple):
    ax: Axes
    lines: dict[str, list[Line2D]]

_SingleColor = Union[
    str, list[float], tuple[float, float, float], tuple[float, float, float, float]
]
_PlotAccessorColor = Union[str, list[_SingleColor], dict[HashableT, _SingleColor]]

@overload
def boxplot(
    data: DataFrame,
    column: Hashable | list[HashableT] | None = ...,
    by: Hashable | list[HashableT] | None = ...,
    ax: Axes | None = ...,
    fontsize: float | str | None = ...,
    rot: float = ...,
    grid: bool = ...,
    figsize: tuple[float, float] | None = ...,
    layout: tuple[int, int] | None = ...,
    return_type: Literal["axes"] | None = ...,
    **kwargs,
) -> Axes: ...
@overload
def boxplot(
    data: DataFrame,
    column: Hashable | list[HashableT] | None = ...,
    by: Hashable | list[HashableT] | None = ...,
    ax: Axes | None = ...,
    fontsize: float | str | None = ...,
    rot: float = ...,
    grid: bool = ...,
    figsize: tuple[float, float] | None = ...,
    layout: tuple[int, int] | None = ...,
    *,
    return_type: Literal["dict"],
    **kwargs,
) -> dict[str, list[Line2D]]: ...
@overload
def boxplot(
    data: DataFrame,
    column: Hashable | list[HashableT] | None = ...,
    by: Hashable | list[HashableT] | None = ...,
    ax: Axes | None = ...,
    fontsize: float | str | None = ...,
    rot: float = ...,
    grid: bool = ...,
    figsize: tuple[float, float] | None = ...,
    layout: tuple[int, int] | None = ...,
    *,
    return_type: Literal["both"],
    **kwargs,
) -> _BoxPlotT: ...

class PlotAccessor(PandasObject):
    def __init__(self, data) -> None: ...
    def __call__(self, *args, **kwargs): ...
    @overload
    def line(
        self,
        x: Hashable | None = ...,
        y: Hashable | None = ...,
        color: _PlotAccessorColor = ...,
        subplots: Literal[False] | None = ...,
        **kwargs,
    ) -> AxesSubplot: ...
    @overload
    def line(
        self,
        x: Hashable | None = ...,
        y: Hashable | None = ...,
        color: _PlotAccessorColor = ...,
        subplots: Literal[True] = ...,
        **kwargs,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def bar(
        self,
        x: Hashable | None = ...,
        y: Hashable | None = ...,
        color: _PlotAccessorColor = ...,
        subplots: Literal[False] | None = ...,
        **kwargs,
    ) -> AxesSubplot: ...
    @overload
    def bar(
        self,
        x: Hashable | None = ...,
        y: Hashable | None = ...,
        color: _PlotAccessorColor = ...,
        subplots: Literal[True] = ...,
        **kwargs,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def barh(
        self,
        x: Hashable | None = ...,
        y: Hashable | None = ...,
        color: _PlotAccessorColor = ...,
        subplots: Literal[False] | None = ...,
        **kwargs,
    ) -> AxesSubplot: ...
    @overload
    def barh(
        self,
        x: Hashable | None = ...,
        y: Hashable | None = ...,
        color: _PlotAccessorColor = ...,
        subplots: Literal[True] = ...,
        **kwargs,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def box(
        self,
        by: Hashable | list[HashableT] | None = ...,
        subplots: Literal[False] | None = ...,
        **kwargs,
    ) -> AxesSubplot: ...
    @overload
    def box(
        self,
        by: Hashable | list[HashableT] | None = ...,
        subplots: Literal[True] = ...,
        **kwargs,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def hist(
        self,
        by: Hashable | list[HashableT] | None = ...,
        bins: int = ...,
        *,
        subplots: Literal[False] | None = ...,
        **kwargs,
    ) -> AxesSubplot: ...
    @overload
    def hist(
        self,
        by: Hashable | list[HashableT] | None = ...,
        bins: int = ...,
        *,
        subplots: Literal[True],
        **kwargs,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def kde(
        self,
        bw_method: Literal["scott", "silverman"]
        | float
        | Callable[[gaussian_kde], float]
        | None = ...,
        ind: npt.NDArray[np.float_] | int | None = ...,
        *,
        subplots: Literal[False] | None = ...,
        **kwargs,
    ) -> AxesSubplot: ...
    @overload
    def kde(
        self,
        bw_method: Literal["scott", "silverman"]
        | float
        | Callable[[gaussian_kde], float]
        | None = ...,
        ind: npt.NDArray[np.float_] | int | None = ...,
        *,
        subplots: Literal[True],
        **kwargs,
    ) -> AxesSubplot: ...
    @overload
    def area(
        self,
        x: Hashable | None = ...,
        y: Hashable | None = ...,
        stacked: bool = ...,
        *,
        subplots: Literal[False] | None = ...,
        **kwargs,
    ) -> AxesSubplot: ...
    @overload
    def area(
        self,
        x: Hashable | None = ...,
        y: Hashable | None = ...,
        stacked: bool = ...,
        *,
        subplots: Literal[True],
        **kwargs,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def pie(
        self, y: Hashable, *, subplots: Literal[False] | None = ..., **kwargs
    ) -> AxesSubplot: ...
    @overload
    def pie(
        self, y: Hashable, *, subplots: Literal[True], **kwargs
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def scatter(
        self,
        x: Hashable,
        y: Hashable,
        s: Hashable | Sequence[float] | None = ...,
        c: Hashable | list[str] = ...,
        *,
        subplots: Literal[False] | None = ...,
        **kwargs,
    ) -> AxesSubplot: ...
    @overload
    def scatter(
        self,
        x: Hashable,
        y: Hashable,
        s: Hashable | Sequence[float] | None = ...,
        c: Hashable | list[str] = ...,
        *,
        subplots: Literal[True],
        **kwargs,
    ) -> npt.NDArray[np.object_]: ...
    def hexbin(
        self,
        x: Hashable,
        y: Hashable,
        C: Hashable | None = ...,
        reduce_C_function: Callable[[list], float] | None = ...,
        gridsize: int | tuple[int, int] | None = ...,
        **kwargs,
    ) -> AxesSubplot: ...
    density = kde
