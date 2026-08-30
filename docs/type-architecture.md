# Type Architecture

This document describes the architectural design, container hierarchy, and operator
dispatch system used throughout `pandas-stubs`. It is a **hand-maintained living design
guide**: it reflects the current `.pyi` stubs and is kept in sync with them by the
validator [`scripts/check_container_hierarchy.py`](../scripts/check_container_hierarchy.py)
(wired into the `architecture` CI job). It is grounded in the stubs themselves (linked
inline below) and complements the guidance in
[`docs/philosophy.md`](philosophy.md) and [`AGENTS.md`](../AGENTS.md).

---

## 1. Core Architectural Principle: The Container Hierarchy

In Python and `pandas`, arithmetic and logical operations follow a strict container
hierarchy:

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

> **Lower-level tiers never include higher-level tiers in their forward operator
> signatures (`__op__`).** Interactions between differing tiers are always handled by
> the higher-level tier via its forward (`__op__`) and reverse (`__rop__`) methods.

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

The invariant is what the CI drift guard asserts, on two levels:

- **Alias level** — every `ScalarArrayIndex*` alias (Tier 1) must not reference
  `Series`; every `ScalarArrayIndexSeries*` alias (Tier 2) must not reference
  `DataFrame`.
- **Forward-dunder level** — the forward arithmetic dunders of `Index` must not name
  `Series` in their `other` operand; the forward arithmetic dunders of `Series` must not
  name `DataFrame`.

Reverse dunders (`__r*__`) are excluded from the guard because they are, by design, the
place where a *lower* tier is the right operand of a *higher* tier. `Series.__matmul__`
takes `DataFrame` and is also excluded — see §2.

---

## 2. Tier Specifications

### Tier 0: Scalars

- **Types**: `int`, `float`, `complex`, `str`, `Timestamp`, `Timedelta`, `Period`,
  `Interval`, `BaseOffset`.
- **Scope**: Scalars only know how to operate on other scalars.
- **Rules**:
  - `Period.__add__(other: PeriodAddSub, /) -> Self`
  - Never accept `Sequence`, `np.ndarray`, `Index`, `Series`, or `DataFrame`.
  - When added to an `Index` or `Series`, the scalar operation fails or returns
    `NotImplemented`, allowing Python to fall back to `Index.__radd__` or
    `Series.__radd__`.

### Tier 1: 1D Indexes and Arrays

- **Types**: `Index[T]`, `DatetimeIndex`, `TimedeltaIndex`, `PeriodIndex`,
  `ExtensionArray`, `np.ndarray`.
- **Scope**: Operate on scalars, scalar sequences, and other 1D structures
  (arrays/indexes).
- **Rules**:
  - `Index[Period].__add__(other: …, /) -> Index[Period]` — the `other` operand is
    written as an inline union (see §3) that never names `Series`.
  - **Crucial Exclude**: `Index`'s forward arithmetic `other` must **not** include
    `Series`. At runtime `Index + Series` produces a `Series`, so `Index` must not claim
    to return an `Index` when its right operand is a `Series`. Python's dispatch
    delegates `Index + Series` to `Series.__radd__(Index)`.

### Tier 2: 1D Series

- **Types**: `Series[T]`.
- **Scope**: Operates on scalars, scalar sequences, 1D arrays, `Index`, and other
  `Series`.
- **Rules**:
  - The `ScalarArrayIndexSeries*` aliases extend their `ScalarArrayIndex*` counterparts
    by adding the compatible `Series[...]` types.
  - **Crucial Exclude**: `Series`'s forward arithmetic `other` must **not** include
    `DataFrame`. `Series + DataFrame` is a `DataFrame`, handled by
    `DataFrame.__radd__(Series)`.

### Tier 3: 2D DataFrames

- **Types**: `DataFrame`.
- **Scope**: Operates on scalars, 1D `Series`, and 2D `DataFrame` objects.
- **Reality check**: `DataFrame`'s forward arithmetic is mostly `other: Any` — only
  `__floordiv__` and `__truediv__` name a concrete union
  (`float | DataFrame | Series[int] | Series[float] | Sequence[float]` and
  `float | DataFrame | Series | Sequence[Any]` respectively). The tier is still where
  cross-tier `Series`/`DataFrame` interactions resolve; it just does not enumerate the
  alias system the way Tiers 1–2 do.

### A deliberate non-arithmetic exception

`Series.__matmul__(self, other: DataFrame, /) -> Series` names `DataFrame` as a forward
operand. `__matmul__` (`@`) is not an arithmetic operator, so it is **outside** the
container-hierarchy invariant: the guide documents it, and the validator deliberately
does not assert it (and never checks `__matmul__`).

---

## 3. Type Alias System in `pandas.core.base`

To enforce the hierarchy systematically without duplication, `pandas-stubs` defines
standard composite type aliases in
[`../pandas-stubs/core/base.pyi`](../pandas-stubs/core/base.pyi) (L169–234). The
construction pattern is:

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

The aliases that exist in the **current** stubs are:

| Category | Index Operand Alias (`ScalarArrayIndex*`) | Series Operand Alias (`ScalarArrayIndexSeries*`) |
| :--- | :--- | :--- |
| Just integers | `ScalarArrayIndexJustInt` | `ScalarArrayIndexSeriesJustInt` |
| Just floats | `ScalarArrayIndexJustFloat` | `ScalarArrayIndexSeriesJustFloat` |
| Just complex | `ScalarArrayIndexJustComplex` | `ScalarArrayIndexSeriesJustComplex` |
| Real numbers | `ScalarArrayIndexReal` | `ScalarArrayIndexSeriesReal` |
| Complex numbers | `ScalarArrayIndexComplex` | `ScalarArrayIndexSeriesComplex` |
| Timedelta | `ScalarArrayIndexTimedelta` | `ScalarArrayIndexSeriesTimedelta` |

> **Note on the alias table.** This table is exhaustive for the current stubs. There are
> **no** `ScalarArrayIndexDateTime` / `ScalarArrayIndexSeriesDateTime` or
> `ScalarArrayIndexPeriod` / `ScalarArrayIndexSeriesPeriod` aliases: datetime and
> `Period` operands are spelled **inline** in the affected overloads rather than via the
> alias system. The "Just*" family (`JustInt` / `JustFloat` / `JustComplex`, wrapping
> bare `int`/`float`/`complex` via the `Just` protocol) is the finer-grained split for
> the un-suffixed numeric operands.

### Where the aliases are (and are not) used

The alias system is not applied uniformly to every operator:

- `Index.__add__` ([`indexes/base.pyi`](../pandas-stubs/core/indexes/base.pyi)
  L642–697) and `Series.__add__` ([`series.pyi`](../pandas-stubs/core/series.pyi)
  L1951–2052) spell their `other` operands as **inline unions** (e.g.
  `complex | Period | ArrayLike | SequenceNotStr[S1] | Index` for `Index`, and
  `complex | ListLike` plus per-`dtype` unions for `Series`) rather than through
  `ScalarArrayIndex*`.
- The aliases are used in `__truediv__`/`__floordiv__` and in the **named** methods
  (`add`, `floordiv`, `sub`, `mul`, …), where the operand sets are stable enough to
  factor out.

The validator `scripts/check_container_hierarchy.py` asserts the hierarchy invariant
directly against these stubs, so this prose and the code cannot drift apart.

---

## 4. Method Signature Standards

### Positional-Only Parameters (`/`)

All dunder arithmetic and comparison methods (`__add__`, `__radd__`, `__mul__`,
`__rmul__`, `__sub__`, `__rsub__`, `__eq__`, `__ne__`, etc.) must declare their operand
parameter as positional-only, matching the CPython runtime protocol, which forbids
keyword calls to dunders:

```python
# Correct:
def __add__(self, other: ScalarArrayIndexReal, /) -> Self: ...

# Incorrect (allows invalid keyword calls like s.__add__(other=x)):
def __add__(self, other: ScalarArrayIndexReal) -> Self: ...
```

### Named Methods (`add`, `sub`, `mul`, `div`, etc.)

Named methods on `Series` and `DataFrame` accept optional keyword arguments (`level`,
`fill_value`, `axis`):

```python
def add(
    self,
    other: ScalarArrayIndexSeriesReal,
    level: Level | None = None,
    fill_value: float | None = None,
    axis: int = 0,
) -> Series[float]: ...
```

---

## 5. Protocol Unification (`SupportsAdd` / `SupportsRAdd`)

To avoid redundant combinatorial overload definitions when supporting custom objects
that implement arithmetic protocols, `pandas-stubs` leverages type variables with upper
bounds:

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

`S2_NSDT` (Non-Sequence / Non-DateTime / Non-Timedelta) prevents the protocol overloads
from prematurely capturing specialized temporal and sequence types whose return types
require narrowing.

---

## 6. Subclass Overrides & Multi-Checker Diagnostics

Specialized `Index` subclasses (`DatetimeIndex`, `TimedeltaIndex`, `PeriodIndex`) narrow
the generic return type of `Index` (e.g. from `Index[Period]` to the concrete
`PeriodIndex` / `Self`).

Because static type checkers check Liskov Substitution Principle (LSP) adherence against
the broad base `Index` overloads, these narrowed signatures require explicit ignore
annotations ordered by checker:

```python
@override
def __add__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override] # ty: ignore[invalid-method-override]
    self, other: ScalarArrayIndexTimedelta, /
) -> Self: ...
```

### Ignore Comment Order

Always maintain the standard ignore order (see
[`docs/philosophy.md`](philosophy.md#using-ignore-comments)):

1. `mypy`: `# type: ignore[<code>]`
2. `pyright`: `# pyright: ignore[<code>]`
3. `pyrefly`: `# pyrefly: ignore[<code>]`
4. `ty`: `# ty: ignore[<code>]`

---

## 7. Testing Strategy and Combinatorial Coverage

Every arithmetic family must include comprehensive combinatorial test suites in `tests/`
(see [`docs/philosophy.md`](philosophy.md#testing-the-type-stubs) and
[`AGENTS.md`](../AGENTS.md)):

1. **Scalars**: Python primitives (`int`, `float`), temporal primitives (`datetime`,
   `timedelta`, `Timestamp`, `Timedelta`, `Period`, `DateOffset`).
2. **Sequences**: Python `list`, `tuple`, `Sequence`.
3. **NumPy Arrays**: `np.ndarray` with matching `dtype`s (`int64`, `float64`,
   `datetime64[ns]`, `timedelta64[ns]`, `object`).
4. **pandas Index**: Matching Index types (`Index[int]`, `DatetimeIndex`,
   `TimedeltaIndex`, `PeriodIndex`).
5. **pandas Series**: Matching Series types (`Series[int]`, `Series[Timestamp]`,
   `Series[Timedelta]`, `Series[Period]`).

### Dual Validation Pattern

- **Positive Tests**: Validated at both type-checking and runtime using
  `check(assert_type(expr, ExpectedType), ...)`:

  ```python
  check(assert_type(left + d, "pd.Series[pd.Period]"), pd.Series, pd.Period)
  ```

- **Negative Tests**: Invalid combinations guarded by `TYPE_CHECKING_INVALID_USAGE` to
  ensure type checkers catch illegal arithmetic without crashing pytest at runtime:

  ```python
  if TYPE_CHECKING_INVALID_USAGE:
      _0 = left + p  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
  ```
