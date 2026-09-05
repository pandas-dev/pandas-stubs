# Private types that are not used in tests

from collections.abc import (
    Callable,
    Hashable,
    Mapping,
    Sequence,
)
from datetime import (
    datetime,
    timedelta,
)
from typing import (
    Any,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    overload,
    type_check_only,
)

import numpy as np
from numpy import typing as npt
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.floating import FloatingArray
from pandas.core.arrays.integer import IntegerArray
from pandas.core.arrays.timedeltas import TimedeltaArray
from pandas.core.base import T_INTERVAL_NP
from pandas.core.groupby.base import ReductionKernelType
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.series import Series
from typing_extensions import (
    TypeVar,
    override,
)

from pandas._libs.interval import Interval
from pandas._libs.tslibs.offsets import BaseOffset
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._libs.tslibs.timestamps import Timestamp
from pandas._typing import (
    S1,
    S2,
    DTypeLike,
    GenericT,
    GenericT_co,
    Just,
    Label,
    Scalar,
    ScalarT,
    SupportsDType,
    np_1darray,
    np_ndarray,
    np_ndarray_anyint,
    np_ndarray_bool,
    np_ndarray_complex,
    np_ndarray_dt,
    np_ndarray_float,
    np_ndarray_td,
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

ScalarArrayIndexJustInt: TypeAlias = (
    Just[int]
    | np.integer
    | Sequence[Just[int] | np.integer]
    | np_ndarray_anyint
    | IntegerArray
    | Index[int]
)
ScalarArrayIndexSeriesJustInt: TypeAlias = ScalarArrayIndexJustInt | Series[int]
ScalarArrayIndexJustFloat: TypeAlias = (
    Just[float]
    | np.floating
    | Sequence[Just[float] | np.floating]
    | np_ndarray_float
    | FloatingArray
    | Index[float]
)
ScalarArrayIndexSeriesJustFloat: TypeAlias = ScalarArrayIndexJustFloat | Series[float]
ScalarArrayIndexJustComplex: TypeAlias = (
    Just[complex]
    | np.complexfloating
    | Sequence[Just[complex] | np.complexfloating]
    | np_ndarray_complex
    | Index[complex]
)
ScalarArrayIndexSeriesJustComplex: TypeAlias = (
    ScalarArrayIndexJustComplex | Series[complex]
)

NumpyRealScalar: TypeAlias = np.bool | np.integer | np.floating
IndexReal: TypeAlias = Index[bool] | Index[int] | Index[float]
ScalarArrayIndexReal: TypeAlias = (
    float
    | Sequence[float | NumpyRealScalar]
    | NumpyRealScalar
    | npt.NDArray[NumpyRealScalar]
    | ExtensionArray
    | IndexReal
)
SeriesReal: TypeAlias = Series[bool] | Series[int] | Series[float]
ScalarArrayIndexSeriesReal: TypeAlias = ScalarArrayIndexReal | SeriesReal

NumpyComplexScalar: TypeAlias = NumpyRealScalar | np.complexfloating
IndexComplex: TypeAlias = IndexReal | Index[complex]
ScalarArrayIndexComplex: TypeAlias = (
    complex
    | Sequence[complex | NumpyComplexScalar]
    | NumpyComplexScalar
    | npt.NDArray[NumpyComplexScalar]
    | ExtensionArray
    | IndexComplex
)
SeriesComplex: TypeAlias = SeriesReal | Series[complex]
ScalarArrayIndexSeriesComplex: TypeAlias = ScalarArrayIndexComplex | SeriesComplex

ArrayIndexBoolNoSeq: TypeAlias = np_ndarray_bool | Index[bool]

ArrayIndexBoolIntNoSeq: TypeAlias = (
    np_ndarray_bool | np_ndarray_anyint | Index[bool] | Index[int]
)

ArrayIndexTimedeltaNoSeq: TypeAlias = np_ndarray_td | TimedeltaArray | TimedeltaIndex
ScalarArrayIndexTimedelta: TypeAlias = (
    timedelta
    | np.timedelta64
    | Sequence[timedelta | np.timedelta64]
    | ArrayIndexTimedeltaNoSeq
)
ArrayIndexSeriesTimedeltaNoSeq: TypeAlias = ArrayIndexTimedeltaNoSeq | Series[Timedelta]
ScalarArrayIndexSeriesTimedelta: TypeAlias = (
    ScalarArrayIndexTimedelta | Series[Timedelta]
)

ArrayIndexDatetimeNoSeq: TypeAlias = np_ndarray_dt | DatetimeArray | DatetimeIndex
ScalarArrayIndexDatetime: TypeAlias = (
    datetime
    | np.datetime64
    | Sequence[datetime | np.datetime64]
    | ArrayIndexDatetimeNoSeq
)

NumListLike: TypeAlias = (  # TODO: pandas-dev/pandas-stubs#1474 deprecated, do not use
    ExtensionArray
    | np_ndarray_bool
    | np_ndarray_anyint
    | np_ndarray_float
    | np_ndarray_complex
    | dict[str, np_ndarray]
    | Sequence[complex]
)

OrderableScalars: TypeAlias = int | float
OrderableTimes: TypeAlias = Timestamp | Timedelta
Orderables: TypeAlias = OrderableScalars | OrderableTimes
OrderableScalarT = TypeVar("OrderableScalarT", bound=OrderableScalars)
OrderableTimesT = TypeVar("OrderableTimesT", bound=OrderableTimes)
OrderableT = TypeVar("OrderableT", bound=Orderables, default=Any)

@type_check_only
class ElementOpsMixin(Generic[S2]):
    @overload
    def _proto_add(
        self: ElementOpsMixin[bool], other: bool | np.bool_
    ) -> ElementOpsMixin[bool]: ...
    @overload
    def _proto_add(
        self: ElementOpsMixin[int], other: int | np.integer
    ) -> ElementOpsMixin[int]: ...
    @overload
    def _proto_add(
        self: ElementOpsMixin[float], other: float | np.floating
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_add(
        self: ElementOpsMixin[complex], other: complex | np.complexfloating
    ) -> ElementOpsMixin[complex]: ...
    @overload
    def _proto_add(self: ElementOpsMixin[str], other: str) -> ElementOpsMixin[str]: ...
    @overload
    def _proto_radd(
        self: ElementOpsMixin[bool], other: bool | np.bool_
    ) -> ElementOpsMixin[bool]: ...
    @overload
    def _proto_radd(
        self: ElementOpsMixin[int], other: int | np.integer
    ) -> ElementOpsMixin[int]: ...
    @overload
    def _proto_radd(
        self: ElementOpsMixin[float], other: float | np.floating
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_radd(
        self: ElementOpsMixin[complex], other: complex | np.complexfloating
    ) -> ElementOpsMixin[complex]: ...
    @overload
    def _proto_radd(self: ElementOpsMixin[str], other: str) -> ElementOpsMixin[str]: ...
    @overload
    def _proto_sub(
        self: ElementOpsMixin[int], other: int | np.integer
    ) -> ElementOpsMixin[int]: ...
    @overload
    def _proto_sub(
        self: ElementOpsMixin[float], other: float | np.floating
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_sub(
        self: ElementOpsMixin[complex], other: complex | np.complexfloating
    ) -> ElementOpsMixin[complex]: ...
    @overload
    def _proto_sub(
        self: ElementOpsMixin[Timedelta], other: timedelta | np.timedelta64 | Timedelta
    ) -> ElementOpsMixin[Timedelta]: ...
    @overload
    def _proto_sub(
        self: ElementOpsMixin[Timestamp],
        other: timedelta | np.timedelta64 | Timedelta | BaseOffset,
    ) -> ElementOpsMixin[Timestamp]: ...
    @overload
    def _proto_sub(
        self: ElementOpsMixin[Timestamp], other: datetime | np.datetime64 | Timestamp
    ) -> ElementOpsMixin[Timedelta]: ...
    @overload
    def _proto_rsub(
        self: ElementOpsMixin[int], other: int | np.integer
    ) -> ElementOpsMixin[int]: ...
    @overload
    def _proto_rsub(
        self: ElementOpsMixin[float], other: float | np.floating
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_rsub(
        self: ElementOpsMixin[complex], other: complex | np.complexfloating
    ) -> ElementOpsMixin[complex]: ...
    @overload
    def _proto_rsub(
        self: ElementOpsMixin[Timedelta], other: timedelta | np.timedelta64 | Timedelta
    ) -> ElementOpsMixin[Timedelta]: ...
    @overload
    def _proto_rsub(
        self: ElementOpsMixin[Timedelta], other: datetime | np.datetime64 | Timestamp
    ) -> ElementOpsMixin[Timestamp]: ...
    @overload
    def _proto_rsub(
        self: ElementOpsMixin[Timestamp], other: datetime | np.datetime64 | Timestamp
    ) -> ElementOpsMixin[Timedelta]: ...
    @overload
    def _proto_mul(
        self: ElementOpsMixin[bool], other: bool | np.bool_
    ) -> ElementOpsMixin[bool]: ...
    @overload
    def _proto_mul(
        self: ElementOpsMixin[int], other: int | np.integer
    ) -> ElementOpsMixin[int]: ...
    @overload
    def _proto_mul(
        self: ElementOpsMixin[float], other: float | np.floating
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_mul(
        self: ElementOpsMixin[complex], other: complex | np.complexfloating
    ) -> ElementOpsMixin[complex]: ...
    @overload
    def _proto_mul(
        self: ElementOpsMixin[Timedelta],
        other: Just[int] | Just[float] | np.integer | np.floating,
    ) -> ElementOpsMixin[Timedelta]: ...
    @overload
    def _proto_mul(
        self: ElementOpsMixin[str], other: Just[int] | np.integer
    ) -> ElementOpsMixin[str]: ...
    @overload
    def _proto_rmul(
        self: ElementOpsMixin[bool], other: bool | np.bool_
    ) -> ElementOpsMixin[bool]: ...
    @overload
    def _proto_rmul(
        self: ElementOpsMixin[int], other: int | np.integer
    ) -> ElementOpsMixin[int]: ...
    @overload
    def _proto_rmul(
        self: ElementOpsMixin[float], other: float | np.floating
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_rmul(
        self: ElementOpsMixin[complex], other: complex | np.complexfloating
    ) -> ElementOpsMixin[complex]: ...
    @overload
    def _proto_rmul(
        self: ElementOpsMixin[Timedelta],
        other: Just[int] | Just[float] | np.integer | np.floating,
    ) -> ElementOpsMixin[Timedelta]: ...
    @overload
    def _proto_rmul(
        self: ElementOpsMixin[str], other: Just[int] | np.integer
    ) -> ElementOpsMixin[str]: ...
    @overload
    def _proto_truediv(
        self: ElementOpsMixin[int], other: int | np.integer
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_truediv(
        self: ElementOpsMixin[float], other: float | np.floating
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_truediv(
        self: ElementOpsMixin[complex], other: complex | np.complexfloating
    ) -> ElementOpsMixin[complex]: ...
    @overload
    def _proto_truediv(
        self: ElementOpsMixin[Timedelta], other: timedelta | np.timedelta64 | Timedelta
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_rtruediv(
        self: ElementOpsMixin[int], other: int | np.integer
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_rtruediv(
        self: ElementOpsMixin[float], other: float | np.floating
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_rtruediv(
        self: ElementOpsMixin[complex], other: complex | np.complexfloating
    ) -> ElementOpsMixin[complex]: ...
    @overload
    def _proto_rtruediv(
        self: ElementOpsMixin[Timedelta], other: timedelta | np.timedelta64 | Timedelta
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_floordiv(
        self: ElementOpsMixin[int], other: int | np.integer
    ) -> ElementOpsMixin[int]: ...
    @overload
    def _proto_floordiv(
        self: ElementOpsMixin[float], other: float | np.floating
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_floordiv(
        self: ElementOpsMixin[Timedelta], other: timedelta | np.timedelta64 | Timedelta
    ) -> ElementOpsMixin[int]: ...
    @overload
    def _proto_rfloordiv(
        self: ElementOpsMixin[int], other: int | np.integer
    ) -> ElementOpsMixin[int]: ...
    @overload
    def _proto_rfloordiv(
        self: ElementOpsMixin[float], other: float | np.floating
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_rfloordiv(
        self: ElementOpsMixin[Timedelta], other: timedelta | np.timedelta64 | Timedelta
    ) -> ElementOpsMixin[int]: ...

@type_check_only
class Supports_ProtoAdd(Protocol[T_contra, S2]):
    def _proto_add(self, other: T_contra, /) -> ElementOpsMixin[S2]: ...

@type_check_only
class Supports_ProtoRAdd(Protocol[T_contra, S2]):
    def _proto_radd(self, other: T_contra, /) -> ElementOpsMixin[S2]: ...

@type_check_only
class Supports_ProtoSub(Protocol[T_contra, S2]):
    def _proto_sub(self, other: T_contra, /) -> ElementOpsMixin[S2]: ...

@type_check_only
class Supports_ProtoRSub(Protocol[T_contra, S2]):
    def _proto_rsub(self, other: T_contra, /) -> ElementOpsMixin[S2]: ...

@type_check_only
class Supports_ProtoMul(Protocol[T_contra, S2]):
    def _proto_mul(self, other: T_contra, /) -> ElementOpsMixin[S2]: ...

@type_check_only
class Supports_ProtoRMul(Protocol[T_contra, S2]):
    def _proto_rmul(self, other: T_contra, /) -> ElementOpsMixin[S2]: ...

@type_check_only
class Supports_ProtoTrueDiv(Protocol[T_contra, S2]):
    def _proto_truediv(self, other: T_contra, /) -> ElementOpsMixin[S2]: ...

@type_check_only
class Supports_ProtoRTrueDiv(Protocol[T_contra, S2]):
    def _proto_rtruediv(self, other: T_contra, /) -> ElementOpsMixin[S2]: ...

@type_check_only
class Supports_ProtoFloorDiv(Protocol[T_contra, S2]):
    def _proto_floordiv(self, other: T_contra, /) -> ElementOpsMixin[S2]: ...

@type_check_only
class Supports_ProtoRFloorDiv(Protocol[T_contra, S2]):
    def _proto_rfloordiv(self, other: T_contra, /) -> ElementOpsMixin[S2]: ...

@type_check_only
class IndexSubclassBase(Index[S1], Generic[S1, GenericT_co]):
    @overload
    @override
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
