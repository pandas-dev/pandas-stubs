# pandas-stubs Type Architecture

This document describes the architectural design, container hierarchy, and operator dispatch system used throughout `pandas-stubs`.

---

## 1. Core Architectural Principle: The Container Hierarchy

In Python and `pandas`, arithmetic and logical operations follow a strict container hierarchy:

```
+-------------------------------------------------------------+
| Level 3: 2D DataFrames (DataFrame)                          |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
| Level 2: 1D Series (Series[T])                              |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
| Level 1: 1D Indexes & Arrays (Index[T], ExtensionArray, np) |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
| Level 0: Scalars (int, float, Timestamp, Timedelta, Period) |
+-------------------------------------------------------------+
```

### Hierarchy Invariant

> **Lower-level tiers never include higher-level tiers in their forward operator signatures (`__op__`).**
> Interactions between differing tiers are always handled by the higher-level tier via its forward (`__op__`) and reverse (`__rop__`) methods.

```
                           Dispatch Flow
                           -------------
   Scalar + Scalar      ───►  Scalar.__add__(Scalar)           ──► Scalar
   Scalar + Index       ───►  Index.__radd__(Scalar)            ──► Index
   Index + Scalar       ───►  Index.__add__(Scalar)            ──► Index
   Index + Index        ───►  Index.__add__(Index)             ──► Index
   Index + Series       ───►  Series.__radd__(Index)           ──► Series
   Series + Index       ───►  Series.__add__(Index)            ──► Series
   Series + Series      ───►  Series.__add__(Series)           ──► Series
   Series + DataFrame   ───►  DataFrame.__radd__(Series)       ──► DataFrame
   DataFrame + Series   ───►  DataFrame.__add__(Series)        ──► DataFrame
```

---

## 2. Tier Specifications

### Tier 0: Scalars
- **Types**: `int`, `float`, `complex`, `str`, `Timestamp`, `Timedelta`, `Period`, `Interval`, `BaseOffset`
- **Scope**: Scalars only know how to operate on other scalars.
- **Rules**:
  - `Period.__add__(other: PeriodAddSub, /) -> Self`
  - Never accept `Sequence`, `np.ndarray`, `Index`, `Series`, or `DataFrame`.
  - When added to an `Index` or `Series`, the scalar operation fails/returns `NotImplemented`, allowing Python to fall back to `Index.__radd__` or `Series.__radd__`.

### Tier 1: 1D Indexes and Arrays
- **Types**: `Index[T]`, `DatetimeIndex`, `TimedeltaIndex`, `PeriodIndex`, `ExtensionArray`, `np.ndarray`
- **Scope**: Operate on scalars, scalar sequences, and other 1D structures (arrays/indexes).
- **Rules**:
  - `Index[Period].__add__(other: ScalarArrayIndexPeriod, /) -> Index[Period]`
  - `PeriodIndex.__add__(other: ScalarArrayIndexPeriod, /) -> PeriodIndex`
  - **Crucial Exclude**: Must **not** include `Series` in `ScalarArrayIndex*` operand types.
  - At runtime, `Index + Series` produces a `Series`. Therefore, `Index` must not claim to return `Index` when added to a `Series`. Python's dispatch delegates `Index + Series` to `Series.__radd__(Index)`.

### Tier 2: 1D Series
- **Types**: `Series[T]`
- **Scope**: Operates on scalars, scalar sequences, 1D arrays, `Index`, and other `Series`.
- **Rules**:
  - `Series[Period].__add__(other: ScalarArrayIndexSeriesPeriod, /) -> Series[Period]`
  - `ScalarArrayIndexSeriesPeriod` extends `ScalarArrayIndexPeriod` by including `Series[Timedelta]`, `Series[int]`, `Series[BaseOffset]`.
  - **Crucial Exclude**: Must **not** include `DataFrame` in `ScalarArrayIndexSeries*` operand types.

### Tier 3: 2D DataFrames
- **Types**: `DataFrame`
- **Scope**: Operates on scalars, 1D `Series`, and 2D `DataFrame` objects.

---

## 3. Type Alias System in `pandas.core.base`

To enforce this hierarchy systematically without duplication, `pandas-stubs` defines standard composite type aliases in `pandas/core/base.pyi`:

```
               Type Alias Construction Pattern
               -------------------------------

   [Primitives]  +  [Sequences]  +  [1D Arrays & Indexes]
                         │
                         ▼
             ScalarArrayIndex<Type>
             (Used by Index & Index subclasses)
                         │
                         ▼  + [Series<Compatible>]
             ScalarArrayIndexSeries<Type>
             (Used by Series)
```

### Type Alias Reference Table

| Category | Index Operand Alias (`ScalarArrayIndex*`) | Series Operand Alias (`ScalarArrayIndexSeries*`) |
| :--- | :--- | :--- |
| **Real Numbers** | `ScalarArrayIndexReal` | `ScalarArrayIndexSeriesReal` |
| **Complex Numbers** | `ScalarArrayIndexComplex` | `ScalarArrayIndexComplex` |
| **DateTime / Timestamp** | `ScalarArrayIndexDateTime` | `ScalarArrayIndexSeriesDateTime` |
| **Timedelta** | `ScalarArrayIndexTimedelta` | `ScalarArrayIndexSeriesTimedelta` |
| **Period** | `ScalarArrayIndexPeriod` | `ScalarArrayIndexSeriesPeriod` |

---

## 4. Method Signature Standards

### Positional-Only Parameters (`/`)
All dunder arithmetic and comparison methods (`__add__`, `__radd__`, `__mul__`, `__rmul__`, `__sub__`, `__rsub__`, `__eq__`, `__ne__`, etc.) must declare their operand parameter as positional-only:

```python
# Correct:
def __add__(self, other: ScalarArrayIndexPeriod, /) -> Self: ...

# Incorrect (allows invalid keyword calls like s.__add__(other=x)):
def __add__(self, other: ScalarArrayIndexPeriod) -> Self: ...
```

### Named Methods (`add`, `sub`, `mul`, `div`, etc.)
Named methods on `Series` and `DataFrame` accept optional keyword arguments (`level`, `fill_value`, `axis`):

```python
def add(
    self,
    other: ScalarArrayIndexSeriesPeriod,
    level: Level | None = None,
    fill_value: float | None = None,
    axis: int = 0,
) -> Series[Period]: ...
```

---

## 5. Protocol Unification (`SupportsAdd` / `SupportsRAdd`)

To avoid redundant combinatorial overload definitions when supporting custom objects that implement arithmetic protocols, `pandas-stubs` leverages type variables with upper bounds:

```python
# Forward addition:
@overload
def __add__(
    self: Supports_ProtoAdd[T_contra, S2], other: T_contra | Sequence[T_contra], /
) -> Series[S2]: ...

# Reverse addition:
@overload
def __add__(
    self: Series[S2_contra],
    other: (
        SupportsRAdd[S2_contra, S2_NSDT]
        | Sequence[SupportsRAdd[S2_contra, S2_NSDT]]
    ),
    /,
) -> Series[S2_NSDT]: ...
```

`S2_NSDT` (Non-Sequence / Non-DateTime / Non-Timedelta) prevents protocol overloads from prematurely capturing specialized temporal and sequence types whose return types require narrowing.

---

## 6. Subclass Overrides & Multi-Checker Diagnostics

Specialized Index subclasses (`DatetimeIndex`, `TimedeltaIndex`, `PeriodIndex`) narrow the generic return type of `Index` (e.g. from `Index[Period]` to concrete `PeriodIndex` / `Self`).

Because static type checkers check Liskov Substitution Principle (LSP) adherence against the broad base `Index` overloads, these narrowed signatures require explicit ignore annotations ordered by checker:

```python
@override
def __add__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override] # ty: ignore[invalid-method-override]
    self, other: ScalarArrayIndexPeriod, /
) -> Self: ...
```

### Ignore Comment Order
Always maintain the standard ignore order:
1. `mypy`: `# type: ignore[<code>]`
2. `pyright`: `# pyright: ignore[<code>]`
3. `pyrefly`: `# pyrefly: ignore[<code>]`
4. `ty`: `# ty: ignore[<code>]`

---

## 7. Testing Strategy and Combinatorial Coverage

Every arithmetic family must include comprehensive combinatorial test suites in `tests/`:

1. **Scalars**: Python primitives (`int`, `float`), temporal primitives (`datetime`, `timedelta`, `Timestamp`, `Timedelta`, `Period`, `DateOffset`).
2. **Sequences**: Python `list`, `tuple`, `Sequence`.
3. **NumPy Arrays**: `np.ndarray` with matching `dtype`s (`int64`, `float64`, `datetime64[ns]`, `timedelta64[ns]`, `object`).
4. **pandas Index**: Matching Index types (`Index[int]`, `DatetimeIndex`, `TimedeltaIndex`, `PeriodIndex`).
5. **pandas Series**: Matching Series types (`Series[int]`, `Series[Timestamp]`, `Series[Timedelta]`, `Series[Period]`).

### Dual Validation Pattern

- **Positive Tests**: Validated at both type-checking and runtime using `check(assert_type(expr, ExpectedType), ...)`:
  ```python
  check(assert_type(left + d, "pd.Series[pd.Period]"), pd.Series, pd.Period)
  ```
- **Negative Tests**: Invalid combinations guarded by `TYPE_CHECKING_INVALID_USAGE` to ensure type checkers catch illegal arithmetic without crashing pytest at runtime:
  ```python
  if TYPE_CHECKING_INVALID_USAGE:
      _0 = left + p  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
  ```
