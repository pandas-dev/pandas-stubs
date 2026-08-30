---
status: accepted
date: 2022-07-10
deciders: [Dr-Irv, twoertwein, MarcoGorelli, loicdiridollou, cmp0xff]
consulted: [pandera maintainers, typing-sig]
informed: [pandas-stubs contributors]
---

# ADR-0002: Generic Series and TypeVar Bound Hierarchy

## Context and Problem Statement

In the pandas runtime, `pd.Series` is a non-generic class that holds a 1-dimensional array of homogeneous data (backed by NumPy arrays, ExtensionArrays, or PyArrow). At runtime, `pd.Series([1, 2, 3])` has `dtype: int64`, but the Python runtime class is simply `pandas.core.series.Series`.

Standard type annotations without generics (`def process(s: pd.Series) -> pd.Series`) lose all dtype-level type safety:
1. Operations on datetimes (e.g., subtracting two timestamp series) cannot be statically distinguished from invalid operations (adding two timestamp series).
2. Boolean series (e.g. masks resulting from comparisons `s < 3`) cannot be statically enforced when indexing into DataFrames.
3. String accessor methods (`s.str.split()`) return `Series[list[str]]`, which cannot be properly chained without element type tracking.

## Decision Drivers

- **Expressiveness**: Enable static type checkers to infer and validate the element type of Series across transformations.
- **Backward Compatibility**: Plain `Series` without type arguments must default to `Series[Any]` to preserve ergonomics for untyped user code.
- **Ecosystem Interoperability**: Avoid breaking external typing tools and runtime validators (e.g., `pandera`).
- **Soundness vs Permissiveness**: Constrain generic type arguments to valid pandas scalar/extension types rather than arbitrary Python objects.

## Considered Options

1. **Keep `Series` Non-Generic (Fidelity to Runtime)**:
   - *Pros*: Matches `pandas` runtime `isinstance(s, pd.Series)` behavior.
   - *Cons*: Severe loss of type safety; impossible to catch invalid datetime arithmetic or verify boolean mask indexing statically.
2. **Subclass-based Hierarchy (e.g. `IntSeries`, `TimestampSeries`, `OffsetSeries`)**:
   - *Pros*: Explicit types for specific domains.
   - *Cons*: Class explosion, inconsistent with standard Python typing paradigms, high maintenance burden (PR pandas-dev/pandas-stubs#844 and PR pandas-dev/pandas-stubs#1390 eventually deprecated `OffsetSeries` in favor of `Series[BaseOffset]`).
3. **Generic `Series[S1]` with `SeriesDType` Bound and PEP 696 Defaults** *(Chosen)*:
   - *Pros*: Clean typing model matching `list[T]` / `Sequence[T]`, allows rich operator overloads, supports PEP 696 `default=Any`.
   - *Cons*: Type inference limitations when Series are constructed dynamically from DataFrames (`df["col"]` resolves to `Series[Any]`).

## Decision Outcome

`Series` is declared as generic over `S1` (`class Series(IndexOpsMixin[S1], Generic[S1])`).

### Type Hierarchy and TypeVar Conventions

In `tests/_typing.py` and core stubs:

```python
SeriesDTypeNoStrDateTime: TypeAlias = (
    bytes | bool | int | float | complex | NpDtypeNoStr
    | ExtensionDtype | Period | Interval | CategoricalDtype | BaseOffset
)
SeriesDTypeNoDateTime: TypeAlias = (
    str | SeriesDTypeNoStrDateTime | type[str] | list[str]
)
SeriesDType: TypeAlias = (
    SeriesDTypeNoDateTime
    | datetime.date | datetime.time
    | datetime.datetime  # includes pd.Timestamp
    | datetime.timedelta  # includes pd.Timedelta
)

# Standard TypeVars with PEP 696 defaults
S0 = TypeVar("S0", bound=SeriesDType, default=Any)
S1 = TypeVar("S1", bound=SeriesDType, default=Any)
S2 = TypeVar("S2", bound=SeriesDType)
S2_contra = TypeVar("S2_contra", bound=SeriesDType, contravariant=True)

# Constrained TypeVar for operator overloads
C2 = TypeVar(
    "C2",
    str, bytes, datetime.date, datetime.time, bool, int, float, complex,
    Dtype, datetime.datetime, datetime.timedelta, Period, Interval,
    CategoricalDtype, BaseOffset,
)
```

### Key Rules
1. **Default Type Argument**: With PEP 696 adoption (PR pandas-dev/pandas-stubs#1232), `Series` is aliasable as both `Series` (implied `Series[Any]`) and `Series[T]`.
2. **Specialized Series Subtyping**: Subclasses like `OffsetSeries` were replaced by `Series[BaseOffset]` (PR pandas-dev/pandas-stubs#1390).
3. **Third-Party Compatibility**: Generic `Series` definition accommodates runtime schema libraries like `pandera` (PR pandas-dev/pandas-stubs#492).

## Consequences

- **Positive**: Type checkers catch arithmetic mismatch (e.g. `Series[Timestamp] + Series[Timestamp]` raises `[operator]` error).
- **Positive**: Preserves ergonomics for general code while offering progressive type safety for annotated code.
- **Negative / Neutral**: Dynamic DataFrame column indexing (`df["a"]`) yields `Series[Any]`, requiring progressive fallback rules (see ADR-0003).

## Historical References & Provenance

- **Primary Pull Requests**:
  - pandas-dev/pandas-stubs#130: Annotate Series `to_dict` and `to_list` with generics (pandas-dev/pandas-stubs@a3fdd9c1d80cfd1c0535718b7165548be01b7617)
  - pandas-dev/pandas-stubs#492: Fix issue for pandera allowing generic Series to work (pandas-dev/pandas-stubs@115dd5c57f22c25024776274cad07ce9bdd716c6)
  - pandas-dev/pandas-stubs#760: Make IndexOpsMixin (and Index) generic (pandas-dev/pandas-stubs@f7621f408f4cfed08453e1b906bf9a6a17b34b04)
  - pandas-dev/pandas-stubs#844: OffsetSeries inherits from `Series[BaseOffset]` (pandas-dev/pandas-stubs@146cf236be3f8a198d00d45371dfc5568f543d09)
  - pandas-dev/pandas-stubs#945: Add type parameters to generics in isna and notna (pandas-dev/pandas-stubs@ea8bdeeaf1f8880103afefa8a1d06048db7c70bb)
  - pandas-dev/pandas-stubs#1232: Use `default=Any` in TypeVar for Series and Index (pandas-dev/pandas-stubs@709d7458460df69c09b25084761fa9378d0fae41)
  - pandas-dev/pandas-stubs#1233: Remove redundant S1 TypeVars (pandas-dev/pandas-stubs@f8a329d51989e49ecba5c02982646021d3942bf8)
  - pandas-dev/pandas-stubs#1390: Drop OffsetSeries replacing it with `Series[BaseOffset]` (pandas-dev/pandas-stubs@10fe362f03bbcf36e01dfd4a263af2dee8e1b9ec)
  - pandas-dev/pandas-stubs#1542: Revert `Series[Any].__add__(str)` over-generalization (pandas-dev/pandas-stubs@33c462592f31e845d00dda52bc6b8b094d7b496f)
- **Upstream References**:
  - [PEP 696 – Python Type Defaults for Type Parameters](https://peps.python.org/pep-0696/)

## Controversies and Open Questions
The architecture around Generics and TypeVars continues to be an area of active debate. The primary controversies include:
1. **Variance Design**: Whether we should have used covariant and contravariant forms (`S0`, `S1`, `S2`, etc.) and dropped the invariant ones entirely.
2. **Bound vs. Enumeration**: Whether using a `bound` is mathematically sound, or if we should switch to the style `C2` (enumerating exact types) to prevent unexpected subclass drift.
3. **Defaulting Strategy**: Whether the default `Any` is actually safe, or if we should transition to `object` to enforce strict downstream checking when a user does not supply a generic parameter.
