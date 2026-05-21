# Private types that are not used in tests

from collections.abc import (
    Callable,
    Hashable,
    Mapping,
    Sequence,
)
from datetime import timedelta
from typing import (
    Any,
    Generic,
    Literal,
    TypeAlias,
    overload,
    type_check_only,
)

import numpy as np
from pandas.core.base import T_INTERVAL_NP
from pandas.core.groupby.base import ReductionKernelType
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from typing_extensions import TypeVar

from pandas._libs.interval import Interval
from pandas._libs.tslibs.offsets import BaseOffset
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._libs.tslibs.timestamps import Timestamp
from pandas._typing import (
    S1,
    DTypeLike,
    GenericT,
    GenericT_co,
    Label,
    Scalar,
    ScalarT,
    SupportsDType,
    np_1darray,
)

T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

PivotAggCallable: TypeAlias = Callable[[Series], ScalarT]
PivotAggFunc: TypeAlias = (
    PivotAggCallable[ScalarT]
    | np.ufunc
    | ReductionKernelType
    | Literal[
        "ohlc",
        "quantile",
        "bfill",
        "cummax",
        "cummin",
        "cumprod",
        "cumsum",
        "diff",
        "ffill",
        "pct_change",
        "rank",
        "shift",
    ]
)
PivotAggFuncTypes: TypeAlias = (
    PivotAggFunc[ScalarT]
    | Sequence[PivotAggFunc[ScalarT]]
    | Mapping[Any, PivotAggFunc[ScalarT]]
)

PivotTableIndexTypes: TypeAlias = Label | Sequence[Hashable] | Series | Grouper | None
PivotTableColumnsTypes: TypeAlias = Label | Sequence[Hashable] | Series | Grouper | None
PivotTableValuesTypes: TypeAlias = Label | Sequence[Hashable] | None

PeriodAddSub: TypeAlias = (
    Timedelta | timedelta | np.timedelta64 | np.int64 | int | BaseOffset
)

OrderableScalars: TypeAlias = int | float
OrderableTimes: TypeAlias = Timestamp | Timedelta
Orderables: TypeAlias = OrderableScalars | OrderableTimes
OrderableScalarT = TypeVar("OrderableScalarT", bound=OrderableScalars)
OrderableTimesT = TypeVar("OrderableTimesT", bound=OrderableTimes)
OrderableT = TypeVar("OrderableT", bound=Orderables, default=Any)

@type_check_only
class IndexSubclassBase(Index[S1], Generic[S1, GenericT_co]):
    @overload
    def to_numpy(
        self: IndexSubclassBase[Interval],
        dtype: type[T_INTERVAL_NP],
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray: ...
    @overload
    def to_numpy(
        self,
        dtype: None = None,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray[GenericT_co]: ...
    @overload
    def to_numpy(
        self,
        dtype: np.dtype[GenericT] | SupportsDType[GenericT] | type[GenericT],
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray[GenericT]: ...
    @overload
    def to_numpy(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        dtype: DTypeLike,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray: ...
