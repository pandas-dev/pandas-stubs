# Series Arithmetic & Operator Type Algebra Matrix (AST-Generated)

## 1. Overview & Formal Typing Model

This matrix is **dynamically generated via AST introspection** of `pandas-stubs/core/series.pyi`. It represents the absolute source of truth for overload resolution in Series arithmetic, guaranteeing fidelity between the documentation and the codebase.

### Core TypeVar & Parameter Definitions
- `S1`: Generic element type of the left-hand Series.
- `SeriesDType`: Union of numeric, boolean, datetime, timedelta, period, interval, category, string, and extension types.

---

## 2. Binary Arithmetic Matrix

The following matrix defines the statically resolved return type based on exactly what is defined in the stubs:

| Left Operand (`self`) | Operator | Right Operand (`other`) | Resolved Return Type |
| :--- | :--- | :--- | :--- |
| `Any` | `__add__` | `Index[Never] \| Series[Never]` | `Series` |
| `Any` | `__floordiv__` | `np_ndarray_dt` | `Never` |
| `Any` | `__mul__` | `Index[Never] \| Series[Never]` | `Series` |
| `Any` | `__mul__` | `np_ndarray_dt` | `Never` |
| `Any` | `__radd__` | `Index[Never] \| Series[Never]` | `Series` |
| `Any` | `__rfloordiv__` | `np_ndarray_complex \| np_ndarray_dt` | `Never` |
| `Any` | `__rmul__` | `Index[Never] \| Series[Never]` | `Series` |
| `Any` | `__rmul__` | `np_ndarray_dt` | `Never` |
| `Any` | `__rsub__` | `Index[Never] \| Series[Never]` | `Series` |
| `Any` | `__rtruediv__` | `np_ndarray_dt` | `Never` |
| `Any` | `__sub__` | `Index[Never] \| Series[Never]` | `Series` |
| `Any` | `__truediv__` | `np_ndarray_dt` | `Never` |
| `SeriesComplex \| Series[Timedelta]` | `__truediv__` | `Index[Never] \| Series[Never]` | `Series` |
| `SeriesComplex` | `__rtruediv__` | `Index[Never] \| Series[Never]` | `Series` |
| `SeriesComplex` | `__rtruediv__` | `ScalarArrayIndexSeriesJustComplex` | `Series[complex]` |
| `SeriesComplex` | `__truediv__` | `ScalarArrayIndexSeriesJustComplex` | `Series[complex]` |
| `SeriesReal \| Series[Timedelta]` | `__floordiv__` | `Index[Never] \| Series[Never]` | `Series` |
| `SeriesReal \| Series[Timedelta]` | `__rfloordiv__` | `Index[Never] \| Series[Never]` | `Series` |
| `Series[BaseOffset]` | `__radd__` | `Period` | `Series[Period]` |
| `Series[BaseOffset]` | `__radd__` | `BaseOffset` | `Series[BaseOffset]` |
| `Series[Never]` | `__add__` | `_str` | `Series[_str]` |
| `Series[Never]` | `__add__` | `complex \| ListLike` | `Series` |
| `Series[Never]` | `__floordiv__` | `np_ndarray_td \| TimedeltaIndex` | `Never` |
| `Series[Never]` | `__floordiv__` | `ScalarArrayIndexSeriesReal` | `Series` |
| `Series[Never]` | `__mul__` | `complex \| NumListLike \| Index \| Series` | `Series` |
| `Series[Never]` | `__radd__` | `_str` | `Series[_str]` |
| `Series[Never]` | `__radd__` | `complex \| ListLike` | `Series` |
| `Series[Never]` | `__rfloordiv__` | `ScalarArrayIndexSeriesReal` | `Series` |
| `Series[Never]` | `__rmul__` | `complex \| NumListLike \| Index \| Series` | `Series` |
| `Series[Never]` | `__rsub__` | `complex \| datetime \| np.datetime64 \| np_ndarray_dt \| NumListLike \| Index[T_COMPLEX] \| Series[T_COMPLEX]` | `Series` |
| `Series[Never]` | `__rtruediv__` | `ScalarArrayIndexSeriesComplex \| ScalarArrayIndexSeriesTimedelta` | `Series` |
| `Series[Never]` | `__sub__` | `complex \| NumListLike \| Index[T_COMPLEX] \| Series[T_COMPLEX]` | `Series` |
| `Series[Never]` | `__truediv__` | `ScalarArrayIndexSeriesComplex` | `Series` |
| `Series[Never]` | `__truediv__` | `ArrayIndexTimedeltaNoSeq` | `Never` |
| `Series[Period]` | `__rsub__` | `Series[Period] \| Period` | `Series[BaseOffset]` |
| `Series[Period]` | `__sub__` | `Series[Period] \| Period` | `Series[BaseOffset]` |
| `Series[S2_NDT_contra]` | `__add__` | `Sequence[SupportsRAdd[S2_NDT_contra, S2]]` | `Series[S2]` |
| `Series[S2_NDT_contra]` | `__radd__` | `Sequence[SupportsAdd[S2_NDT_contra, S2]]` | `Series[S2]` |
| `Series[S2_contra]` | `__add__` | `SupportsRAdd[S2_contra, S2]` | `Series[S2]` |
| `Series[S2_contra]` | `__mul__` | `SupportsRMul[S2_contra, S2_NSDT] \| Sequence[SupportsRMul[S2_contra, S2_NSDT]]` | `Series[S2_NSDT]` |
| `Series[S2_contra]` | `__radd__` | `SupportsAdd[S2_contra, S2]` | `Series[S2]` |
| `Series[S2_contra]` | `__rmul__` | `SupportsMul[S2_contra, S2_NSDT] \| Sequence[SupportsMul[S2_contra, S2_NSDT]]` | `Series[S2_NSDT]` |
| `Series[T_COMPLEX]` | `__add__` | `np_ndarray_bool \| Index[bool] \| Series[bool]` | `Series[T_COMPLEX]` |
| `Series[T_COMPLEX]` | `__add__` | `np_ndarray_anyint \| Index[int] \| Series[int]` | `Series[T_COMPLEX]` |
| `Series[T_COMPLEX]` | `__add__` | `np_ndarray_float \| Index[float] \| Series[float]` | `Series[T_COMPLEX]` |
| `Series[T_COMPLEX]` | `__add__` | `np_ndarray_complex \| Index[complex] \| Series[complex]` | `Series[complex]` |
| `Series[T_COMPLEX]` | `__mul__` | `np_ndarray_bool \| Index[bool] \| Series[bool]` | `Series[T_COMPLEX]` |
| `Series[T_COMPLEX]` | `__mul__` | `np_ndarray_anyint \| Index[int] \| Series[int]` | `Series[T_COMPLEX]` |
| `Series[T_COMPLEX]` | `__mul__` | `np_ndarray_float \| Index[float] \| Series[float]` | `Series[T_COMPLEX]` |
| `Series[T_COMPLEX]` | `__mul__` | `np_ndarray_complex \| Index[complex] \| Series[complex]` | `Series[complex]` |
| `Series[T_COMPLEX]` | `__radd__` | `np_ndarray_bool \| Index[bool] \| Series[bool]` | `Series[T_COMPLEX]` |
| `Series[T_COMPLEX]` | `__radd__` | `np_ndarray_anyint \| Index[int] \| Series[int]` | `Series[T_COMPLEX]` |
| `Series[T_COMPLEX]` | `__radd__` | `np_ndarray_float \| Index[float] \| Series[float]` | `Series[T_COMPLEX]` |
| `Series[T_COMPLEX]` | `__radd__` | `np_ndarray_complex \| Index[complex] \| Series[complex]` | `Series[complex]` |
| `Series[T_COMPLEX]` | `__rmul__` | `np_ndarray_bool \| Index[bool] \| Series[bool]` | `Series[T_COMPLEX]` |
| `Series[T_COMPLEX]` | `__rmul__` | `np_ndarray_anyint \| Index[int] \| Series[int]` | `Series[T_COMPLEX]` |
| `Series[T_COMPLEX]` | `__rmul__` | `np_ndarray_float \| Index[float] \| Series[float]` | `Series[T_COMPLEX]` |
| `Series[T_COMPLEX]` | `__rmul__` | `np_ndarray_complex \| Index[complex] \| Series[complex]` | `Series[complex]` |
| `Series[T_COMPLEX]` | `__rsub__` | `Just[complex] \| Sequence[Just[complex]] \| np_ndarray_complex \| Index[complex] \| Series[complex]` | `Series[complex]` |
| `Series[T_COMPLEX]` | `__rtruediv__` | `ScalarArrayIndexSeriesJustFloat` | `Series[T_COMPLEX]` |
| `Series[T_COMPLEX]` | `__sub__` | `Just[complex] \| Sequence[Just[complex]] \| np_ndarray_complex \| Index[complex] \| Series[complex]` | `Series[complex]` |
| `Series[T_COMPLEX]` | `__truediv__` | `np_ndarray_td` | `Never` |
| `Series[T_COMPLEX]` | `__truediv__` | `ScalarArrayIndexSeriesJustFloat` | `Series[T_COMPLEX]` |
| `Series[Timedelta]` | `__add__` | `datetime \| np.datetime64 \| np_ndarray_dt \| DatetimeIndex \| Series[Timestamp]` | `Series[Timestamp]` |
| `Series[Timedelta]` | `__add__` | `timedelta \| np.timedelta64 \| np_ndarray_td \| TimedeltaIndex \| Series[Timedelta]` | `Series[Timedelta]` |
| `Series[Timedelta]` | `__floordiv__` | `np_ndarray_bool \| np_ndarray_complex` | `Never` |
| `Series[Timedelta]` | `__floordiv__` | `ScalarArrayIndexSeriesJustInt \| ScalarArrayIndexSeriesJustFloat` | `Series[Timedelta]` |
| `Series[Timedelta]` | `__floordiv__` | `ArrayIndexSeriesTimedeltaNoSeq` | `Series[int]` |
| `Series[Timedelta]` | `__mul__` | `np_ndarray_bool \| np_ndarray_complex` | `Never` |
| `Series[Timedelta]` | `__mul__` | `np_ndarray_anyint \| np_ndarray_float \| Index[int] \| Index[float] \| Series[int] \| Series[float]` | `Series[Timedelta]` |
| `Series[Timedelta]` | `__radd__` | `datetime \| np.datetime64 \| np_ndarray_dt \| DatetimeIndex \| Series[Timestamp]` | `Series[Timestamp]` |
| `Series[Timedelta]` | `__radd__` | `timedelta \| np.timedelta64 \| np_ndarray_td \| TimedeltaIndex \| Series[Timedelta]` | `Series[Timedelta]` |
| `Series[Timedelta]` | `__rfloordiv__` | `np_ndarray_num` | `Never` |
| `Series[Timedelta]` | `__rfloordiv__` | `ArrayIndexSeriesTimedeltaNoSeq` | `Series[int]` |
| `Series[Timedelta]` | `__rmul__` | `np_ndarray_bool \| np_ndarray_complex` | `Never` |
| `Series[Timedelta]` | `__rmul__` | `np_ndarray_anyint \| np_ndarray_float \| Index[int] \| Index[float] \| Series[int] \| Series[float]` | `Series[Timedelta]` |
| `Series[Timedelta]` | `__rsub__` | `datetime \| np.datetime64 \| np_ndarray_dt \| DatetimeIndex \| Series[Timestamp]` | `Series[Timestamp]` |
| `Series[Timedelta]` | `__rsub__` | `timedelta \| np.timedelta64 \| np_ndarray_td \| TimedeltaIndex \| Series[Timedelta]` | `Series[Timedelta]` |
| `Series[Timedelta]` | `__rtruediv__` | `ArrayIndexSeriesTimedeltaNoSeq` | `Series[float]` |
| `Series[Timedelta]` | `__sub__` | `np_ndarray_dt` | `Never` |
| `Series[Timedelta]` | `__sub__` | `timedelta \| np.timedelta64 \| np_ndarray_td \| TimedeltaIndex \| Series[Timedelta]` | `Series[Timedelta]` |
| `Series[Timedelta]` | `__truediv__` | `np_ndarray_bool \| np_ndarray_complex \| np_ndarray_dt` | `Never` |
| `Series[Timedelta]` | `__truediv__` | `ScalarArrayIndexSeriesJustInt \| ScalarArrayIndexSeriesJustFloat` | `Series[Timedelta]` |
| `Series[Timedelta]` | `__truediv__` | `ArrayIndexSeriesTimedeltaNoSeq` | `Series[float]` |
| `Series[Timestamp]` | `__add__` | `np_ndarray_dt` | `Never` |
| `Series[Timestamp]` | `__add__` | `timedelta \| np.timedelta64 \| np_ndarray_td \| TimedeltaIndex \| Series[Timedelta] \| BaseOffset` | `Series[Timestamp]` |
| `Series[Timestamp]` | `__mul__` | `np_ndarray` | `Never` |
| `Series[Timestamp]` | `__radd__` | `np_ndarray_dt` | `Never` |
| `Series[Timestamp]` | `__radd__` | `timedelta \| np.timedelta64 \| np_ndarray_td \| TimedeltaIndex \| Series[Timedelta] \| BaseOffset` | `Series[Timestamp]` |
| `Series[Timestamp]` | `__rmul__` | `np_ndarray` | `Never` |
| `Series[Timestamp]` | `__rsub__` | `np_ndarray_td` | `Never` |
| `Series[Timestamp]` | `__rsub__` | `datetime \| np.datetime64 \| np_ndarray_dt \| DatetimeIndex \| Series[Timestamp]` | `Series[Timedelta]` |
| `Series[Timestamp]` | `__sub__` | `datetime \| np.datetime64 \| np_ndarray_dt \| DatetimeIndex \| Series[Timestamp]` | `Series[Timedelta]` |
| `Series[Timestamp]` | `__sub__` | `timedelta \| np.timedelta64 \| np_ndarray_td \| TimedeltaIndex \| Series[Timedelta] \| BaseOffset` | `Series[Timestamp]` |
| `Series[_str]` | `__add__` | `np_ndarray_bool \| np_ndarray_anyint \| np_ndarray_float \| np_ndarray_complex` | `Never` |
| `Series[_str]` | `__add__` | `np_ndarray_str \| Index[_str] \| Series[_str]` | `Series[_str]` |
| `Series[_str]` | `__mul__` | `np_ndarray_bool \| np_ndarray_float \| np_ndarray_complex \| np_ndarray_dt \| np_ndarray_td` | `Never` |
| `Series[_str]` | `__mul__` | `np_ndarray_anyint \| Index[int] \| Series[int]` | `Series[_str]` |
| `Series[_str]` | `__radd__` | `np_ndarray_bool \| np_ndarray_anyint \| np_ndarray_float \| np_ndarray_complex` | `Never` |
| `Series[_str]` | `__radd__` | `np_ndarray_str \| Index[_str] \| Series[_str]` | `Series[_str]` |
| `Series[_str]` | `__rmul__` | `np_ndarray_bool \| np_ndarray_float \| np_ndarray_complex \| np_ndarray_dt \| np_ndarray_td` | `Never` |
| `Series[_str]` | `__rmul__` | `np_ndarray_anyint \| Index[int] \| Series[int]` | `Series[_str]` |
| `Series[_str]` | `__rtruediv__` | `Path` | `Series` |
| `Series[_str]` | `__truediv__` | `Path` | `Series` |
| `Series[bool] \| Series[complex]` | `__floordiv__` | `np_ndarray` | `Never` |
| `Series[bool] \| Series[complex]` | `__rfloordiv__` | `np_ndarray` | `Never` |
| `Series[bool] \| Series[complex]` | `__mul__` | `np_ndarray_td` | `Never` |
| `Series[bool] \| Series[complex]` | `__rmul__` | `np_ndarray_td` | `Never` |
| `Series[bool] \| Series[int]` | `__add__` | `np_ndarray_float \| Index[float] \| Series[float]` | `Series[float]` |
| `Series[bool] \| Series[int]` | `__radd__` | `np_ndarray_float \| Index[float] \| Series[float]` | `Series[float]` |
| `Series[bool] \| Series[int]` | `__floordiv__` | `np_ndarray_anyint \| Index[int] \| Series[int]` | `Series[int]` |
| `Series[bool] \| Series[int]` | `__rfloordiv__` | `np_ndarray_anyint \| Index[int] \| Series[int]` | `Series[int]` |
| `Series[bool] \| Series[int]` | `__mul__` | `np_ndarray_float \| Index[float] \| Series[float]` | `Series[float]` |
| `Series[bool] \| Series[int]` | `__rmul__` | `np_ndarray_float \| Index[float] \| Series[float]` | `Series[float]` |
| `Series[bool] \| Series[int]` | `__truediv__` | `ScalarArrayIndexSeriesJustInt` | `Series[float]` |
| `Series[bool] \| Series[int]` | `__truediv__` | `ScalarArrayIndexSeriesJustFloat` | `Series[float]` |
| `Series[bool] \| Series[int]` | `__rtruediv__` | `ScalarArrayIndexSeriesJustInt` | `Series[float]` |
| `Series[bool] \| Series[int]` | `__rtruediv__` | `ScalarArrayIndexSeriesJustFloat` | `Series[float]` |
| `Series[bool]` | `__add__` | `np_ndarray_anyint \| Index[int] \| Series[int]` | `Series[int]` |
| `Series[bool]` | `__mul__` | `np_ndarray_anyint \| Index[int] \| Series[int]` | `Series[int]` |
| `Series[bool]` | `__radd__` | `bool \| Sequence[bool]` | `Series[bool]` |
| `Series[bool]` | `__radd__` | `np_ndarray_anyint \| Index[int] \| Series[int]` | `Series[int]` |
| `Series[bool]` | `__rmul__` | `np_ndarray_anyint \| Index[int] \| Series[int]` | `Series[int]` |
| `Series[bool]` | `__rsub__` | `Just[int] \| Sequence[Just[int]] \| np_ndarray_anyint \| Index[int] \| Series[int]` | `Series[int]` |
| `Series[bool]` | `__rsub__` | `Just[float] \| Sequence[Just[float]] \| np_ndarray_float \| Index[float] \| Series[float]` | `Series[float]` |
| `Series[bool]` | `__sub__` | `Just[int] \| Sequence[Just[int]] \| np_ndarray_anyint \| Index[int] \| Series[int]` | `Series[int]` |
| `Series[bool]` | `__sub__` | `Just[float] \| Sequence[Just[float]] \| np_ndarray_float \| Index[float] \| Series[float]` | `Series[float]` |
| `Series[bool]` | `__truediv__` | `np_ndarray_bool` | `Never` |
| `Series[complex]` | `__radd__` | `float \| Sequence[float]` | `Series[complex]` |
| `Series[complex]` | `__rsub__` | `T_COMPLEX \| Sequence[T_COMPLEX] \| np_ndarray_bool \| np_ndarray_anyint \| np_ndarray_float \| Index[T_COMPLEX] \| Series[T_COMPLEX]` | `Series[complex]` |
| `Series[complex]` | `__rtruediv__` | `np_ndarray_bool \| np_ndarray_anyint \| Index[bool] \| Index[int] \| Series[bool] \| Series[int]` | `Series[complex]` |
| `Series[complex]` | `__sub__` | `T_COMPLEX \| Sequence[T_COMPLEX] \| np_ndarray_bool \| np_ndarray_anyint \| np_ndarray_float \| Index[T_COMPLEX] \| Series[T_COMPLEX]` | `Series[complex]` |
| `Series[complex]` | `__truediv__` | `np_ndarray_bool \| np_ndarray_anyint \| Index[bool] \| Index[int] \| Series[bool] \| Series[int]` | `Series[complex]` |
| `Series[float]` | `__floordiv__` | `np_ndarray_bool \| Index[bool] \| Series[bool]` | `Series[float]` |
| `Series[float]` | `__floordiv__` | `np_ndarray_anyint \| Index[int] \| Series[int]` | `Series[float]` |
| `Series[float]` | `__radd__` | `int \| Sequence[int]` | `Series[float]` |
| `Series[float]` | `__rfloordiv__` | `np_ndarray_bool \| Index[bool] \| Series[bool]` | `Series[float]` |
| `Series[float]` | `__rfloordiv__` | `np_ndarray_anyint \| Index[int] \| Series[int]` | `Series[float]` |
| `Series[float]` | `__rsub__` | `float \| Sequence[float] \| np_ndarray_bool \| np_ndarray_anyint \| np_ndarray_float \| Index[bool] \| Series[bool] \| Index[int] \| Series[int] \| Index[float] \| Series[float]` | `Series[float]` |
| `Series[float]` | `__rtruediv__` | `np_ndarray_bool \| np_ndarray_anyint \| Index[bool] \| Index[int] \| Series[bool] \| Series[int]` | `Series[float]` |
| `Series[float]` | `__sub__` | `float \| Sequence[float] \| np_ndarray_bool \| np_ndarray_anyint \| np_ndarray_float \| Index[bool] \| Series[bool] \| Index[int] \| Series[int] \| Index[float] \| Series[float]` | `Series[float]` |
| `Series[float]` | `__truediv__` | `np_ndarray_bool \| np_ndarray_anyint \| Index[bool] \| Index[int] \| Series[bool] \| Series[int]` | `Series[float]` |
| `Series[int] \| Series[float]` | `__floordiv__` | `np_ndarray_complex \| np_ndarray_td` | `Never` |
| `Series[int] \| Series[float]` | `__floordiv__` | `float \| Sequence[float] \| np_ndarray_float \| Index[float] \| Series[float]` | `Series[float]` |
| `Series[int] \| Series[float]` | `__rfloordiv__` | `np_ndarray_td` | `Never` |
| `Series[int] \| Series[float]` | `__rfloordiv__` | `float \| Sequence[float] \| np_ndarray_float \| Index[float] \| Series[float]` | `Series[float]` |
| `Series[int] \| Series[float]` | `__rfloordiv__` | `timedelta \| np.timedelta64 \| ArrayIndexSeriesTimedeltaNoSeq` | `Series[Timedelta]` |
| `Series[int] \| Series[float]` | `__rfloordiv__` | `Sequence[timedelta \| np.timedelta64]` | `Series` |
| `Series[int] \| Series[float]` | `__mul__` | `timedelta \| Sequence[timedelta] \| np.timedelta64 \| np_ndarray_td \| TimedeltaIndex \| Series[Timedelta]` | `Series[Timedelta]` |
| `Series[int] \| Series[float]` | `__rmul__` | `timedelta \| Sequence[timedelta] \| np.timedelta64 \| np_ndarray_td \| TimedeltaIndex \| Series[Timedelta]` | `Series[Timedelta]` |
| `Series[int] \| Series[float]` | `__rtruediv__` | `Sequence[timedelta \| np.timedelta64]` | `Series` |
| `Series[int] \| Series[float]` | `__rtruediv__` | `ScalarArrayIndexSeriesTimedelta` | `Series[Timedelta]` |
| `Series[int]` | `__floordiv__` | `np_ndarray_bool \| Index[bool] \| Series[bool]` | `Series[int]` |
| `Series[int]` | `__rfloordiv__` | `np_ndarray_bool \| Index[bool] \| Series[bool]` | `Series[int]` |
| `Series[int]` | `__rsub__` | `int \| Sequence[int] \| np_ndarray_bool \| np_ndarray_anyint \| Index[bool] \| Series[bool] \| Index[int] \| Series[int]` | `Series[int]` |
| `Series[int]` | `__rsub__` | `Just[float] \| Sequence[Just[float]] \| np_ndarray_float \| Index[float] \| Series[float]` | `Series[float]` |
| `Series[int]` | `__rtruediv__` | `np_ndarray_bool \| Index[bool] \| Series[bool]` | `Series[float]` |
| `Series[int]` | `__sub__` | `int \| Sequence[int] \| np_ndarray_bool \| np_ndarray_anyint \| Index[bool] \| Series[bool] \| Index[int] \| Series[int]` | `Series[int]` |
| `Series[int]` | `__sub__` | `Just[float] \| Sequence[Just[float]] \| np_ndarray_float \| Index[float] \| Series[float]` | `Series[float]` |
| `Series[int]` | `__truediv__` | `np_ndarray_bool \| Index[bool] \| Series[bool]` | `Series[float]` |
| `Supports_ProtoAdd[S2_contra, S2]` | `__add__` | `S2_contra \| Sequence[S2_contra]` | `Series[S2]` |
| `Supports_ProtoFloorDiv[T_contra, S2]` | `__floordiv__` | `T_contra \| Sequence[T_contra]` | `Series[S2]` |
| `Supports_ProtoMul[T_contra, S2]` | `__mul__` | `T_contra \| Sequence[T_contra]` | `Series[S2]` |
| `Supports_ProtoRAdd[S2_contra, S2]` | `__radd__` | `S2_contra \| Sequence[S2_contra]` | `Series[S2]` |
| `Supports_ProtoRFloorDiv[T_contra, S2]` | `__rfloordiv__` | `T_contra \| Sequence[T_contra]` | `Series[S2]` |
| `Supports_ProtoRMul[T_contra, S2]` | `__rmul__` | `T_contra \| Sequence[T_contra]` | `Series[S2]` |
| `Supports_ProtoRTrueDiv[T_contra, S2]` | `__rtruediv__` | `T_contra \| Sequence[T_contra]` | `Series[S2]` |
| `Supports_ProtoTrueDiv[T_contra, S2]` | `__truediv__` | `T_contra \| Sequence[T_contra]` | `Series[S2]` |

---
> **Note**: Overloads are evaluated by type checkers from top to bottom. The stubs use precise structural protocols and type restrictions (e.g. `Never` for invalid combinations) to enforce mathematical validity.
