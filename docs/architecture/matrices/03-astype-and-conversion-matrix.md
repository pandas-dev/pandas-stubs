# Series astype() & Dtype Conversion Type Algebra Matrix

## 1. Overview

The `.astype()` method on `Series[S1]` and `Index[S0]` is the primary mechanism for explicit type casting in pandas. Because runtime dtype specifiers can be Python types, NumPy dtypes, pandas ExtensionDtypes, or string aliases, the type stubs use a comprehensive overload cascade to resolve the target `Series[TargetType]`.

---

## 2. Dtype Specifier Resolution Matrix

| Target Dtype Specifier (`dtype: D`) | Category | Resolved Return Type | Example Specifiers | Key Provenance |
| :--- | :--- | :--- | :--- | :--- |
| `type[int]` \| `IntDtypeArg` | Integer Dtypes | `Series[int]` | `int`, `np.int64`, `"int32"`, `"int64"`, `Int64Dtype()` | pandas-dev/pandas-stubs#519, pandas-dev/pandas-stubs#756 |
| `type[float]` \| `FloatDtypeArg` | Floating Dtypes | `Series[float]` | `float`, `np.float64`, `"float64"`, `Float64Dtype()` | pandas-dev/pandas-stubs#519, pandas-dev/pandas-stubs#756 |
| `type[bool]` \| `BooleanDtypeArg` | Boolean Dtypes | `Series[bool]` | `bool`, `np.bool_`, `"boolean"`, `BooleanDtype()` | pandas-dev/pandas-stubs#519, pandas-dev/pandas-stubs#756 |
| `type[str]` \| `StrDtypeArg` | String Dtypes | `Series[str]` | `str`, `np.str_`, `"string"`, `StringDtype()` | pandas-dev/pandas-stubs#519, pandas-dev/pandas-stubs#756 |
| `type[complex]` \| `ComplexDtypeArg` | Complex Dtypes | `Series[complex]` | `complex`, `np.complex128`, `"complex128"` | pandas-dev/pandas-stubs#106, pandas-dev/pandas-stubs#519 |
| `TimestampDtypeArg` | Temporal Dtypes | `Series[Timestamp]` | `"datetime64[ns]"`, `DatetimeTZDtype(...)` | pandas-dev/pandas-stubs#519, pandas-dev/pandas-stubs#1274 |
| `TimedeltaDtypeArg` | Duration Dtypes | `Series[Timedelta]` | `"timedelta64[ns]"`, `TimedeltaDtype()` | pandas-dev/pandas-stubs#519, pandas-dev/pandas-stubs#1273 |
| `CategoryDtypeArg` | Categorical Dtypes | `Series[CategoricalDtype]` | `"category"`, `CategoricalDtype(...)` | pandas-dev/pandas-stubs#519 |
| `PeriodDtype` \| `"period"` | Period Dtypes | `Series[Period]` | `PeriodDtype("M")` | pandas-dev/pandas-stubs#519, pandas-dev/pandas-stubs#1151 |
| `IntervalDtype` \| `"interval"` | Interval Dtypes | `Series[Interval]` | `IntervalDtype("int64")` | pandas-dev/pandas-stubs#519, pandas-dev/pandas-stubs#1385 |
| `PyArrowIntDtypeArg` | PyArrow Dtypes | `Series[int]` | `pa.int64()`, `"int64[pyarrow]"` | pandas-dev/pandas-stubs#1909 |
| `PyArrowStrDtypeArg` | PyArrow String | `Series[str]` | `pa.string()`, `"string[pyarrow]"` | pandas-dev/pandas-stubs#1909 |
| `type[object]` \| `"object"` | Object Dtypes | `Series[object]` | `object`, `"O"` | pandas-dev/pandas-stubs#519 |
| `Dtype` (arbitrary / dynamic) | Dynamic / Unknown | `Series[Any]` | Dynamic variable `my_dtype` | pandas-dev/pandas-stubs#519 |

---

## 3. Overload Architecture Pattern

```python
@overload
def astype(self, dtype: IntDtypeArg, copy: bool = ..., errors: IgnoreRaise = ...) -> Series[int]: ...
@overload
def astype(self, dtype: FloatDtypeArg, copy: bool = ..., errors: IgnoreRaise = ...) -> Series[float]: ...
@overload
def astype(self, dtype: BooleanDtypeArg, copy: bool = ..., errors: IgnoreRaise = ...) -> Series[bool]: ...
@overload
def astype(self, dtype: StrDtypeArg, copy: bool = ..., errors: IgnoreRaise = ...) -> Series[str]: ...
@overload
def astype(self, dtype: TimestampDtypeArg, copy: bool = ..., errors: IgnoreRaise = ...) -> Series[Timestamp]: ...
@overload
def astype(self, dtype: TimedeltaDtypeArg, copy: bool = ..., errors: IgnoreRaise = ...) -> Series[Timedelta]: ...
@overload
def astype(self, dtype: AstypeArg, copy: bool = ..., errors: IgnoreRaise = ...) -> Series[Any]: ...
```

## 4. Default Values for Simple Parameters
For basic methods (including `astype`), a few PRs have historically added default values directly in the type stubs if they represent simple scalar values (e.g., `copy: bool = True`). This adheres to the standard practice described in the official typing guide ([Functions and Methods](https://typing.python.org/en/latest/guides/writing_stubs.html#functions-and-methods)). When designing stubs, if the default value is straightforward to represent and does not require complex object instantiation, it should be mirrored in the `.pyi` signature.
