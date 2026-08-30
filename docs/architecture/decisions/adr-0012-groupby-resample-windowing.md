---
status: accepted
date: 2022-07-23
deciders: [Dr-Irv, amotzop, danielroseman, chrisyeh96, cmp0xff]
consulted: [pandas-stubs community]
informed: [pandas-stubs contributors]
---

# ADR-0012: GroupBy, Resample, and Windowing Type Architecture

## Context and Problem Statement

Split-apply-combine operations (`df.groupby()`, `s.resample()`, `df.rolling()`) transform data structures into intermediate grouping objects (`DataFrameGroupBy`, `SeriesGroupBy`, `Resampler`, `Rolling`).

Key typing challenges:
1. Grouping keys can be column names, arrays, functions, mappings, or lists of groupings (`GroupByObject`).
2. Aggregation operations (`.agg()`, `.apply()`, `.transform()`) can change the dimensionality (collapsing rows into scalar summaries vs broadcasting).
3. Windowing engines allow Cython or Numba execution with engine-specific keyword configurations.

## Decision Drivers

- **Container Differentiation**: `DataFrameGroupBy` vs `SeriesGroupBy` must be distinct generic types.
- **Flexible Grouping Keys**: Support all valid grouping specifications via `GroupByObject`.
- **Typed Windowing Engines**: Provide static validation for Numba and Cython engine keyword dictionaries.

## Considered Options

1. **Single `GroupBy` Type returning `DataFrame | Series`**:
   - *Pros*: Simple hierarchy.
   - *Cons*: Callers must add type assertions after every groupby operation.
2. **Specialized Generic Classes (`SeriesGroupBy[S1]`, `DataFrameGroupBy`)** *(Chosen)*:
   - *Pros*: Accurate method chaining (e.g. `s.groupby(...).mean() -> Series[float]`, `s.groupby(...).count() -> Series[int]`).
   - *Cons*: Multiple intermediate stub classes.

## Decision Outcome

### 1. `GroupByObject` Union Definition
```python
GroupByObjectNonScalar: TypeAlias = (
    tuple[_HashableTa, ...] | list[_HashableTa] | Function
    | list[Function] | list[Series] | np_ndarray | list[np_ndarray]
    | Mapping[Label, Any] | list[Mapping[Label, Any]] | list[Index]
    | Grouper | list[Grouper]
)
GroupByObject: TypeAlias = (
    Scalar | Index | GroupByObjectNonScalar[_HashableTa] | Series
)
```

### 2. Windowing Engine Kwargs TypedDict
```python
class _WindowingNumbaKwargs(TypedDict, total=False):
    nopython: bool
    nogil: bool
    parallel: bool

WindowingEngine: TypeAlias = Literal["cython", "numba"] | None
WindowingEngineKwargs: TypeAlias = _WindowingNumbaKwargs | None
```

### 3. Aggregation Return Types
Methods like `.count()` on `GroupBy[Series]` return `Series[int]` (PR pandas-dev/pandas-stubs#966), while `.apply()` preserves inferred return types (PR pandas-dev/pandas-stubs#177).

## Consequences

- **Positive**: Seamless type chaining across split-apply-combine and rolling operations.
- **Positive**: IDE autocomplete offers exact aggregation methods for Series vs DataFrame groupbys.
- **Negative / Neutral**: Complex custom aggregation functions (`agg(dict)`) may require `Any` fallbacks for heterogeneous column mappings.

## Historical References & Provenance

- **Primary Pull Requests**:
  - pandas-dev/pandas-stubs#148: Groupby __iter__ fix types (pandas-dev/pandas-stubs@a6dd774bcb0cb43f209dd88e5adee05998824dd8)
  - pandas-dev/pandas-stubs#166: Align Groupby types (pandas-dev/pandas-stubs@927d4388775c829859e5caf4600b2f8ecf8e190d)
  - pandas-dev/pandas-stubs#173: Standardized aggregate functions typing (pandas-dev/pandas-stubs@8f9ba75f595b434987454881e8e016669ab45100)
  - pandas-dev/pandas-stubs#177: More specific types for GroupBy.apply (pandas-dev/pandas-stubs@02e1748becb97e485da6930ab4ed9fea382d8ed9)
  - pandas-dev/pandas-stubs#190: Added missing groupby methods and made SeriesGroupBy generic (pandas-dev/pandas-stubs@fed3be4c53250ad749f3f78ce7831bb6b27f909c)
  - pandas-dev/pandas-stubs#966: GroupBy[Series].count() return type Series[int] (pandas-dev/pandas-stubs@7e6aee4e41f8f60b4ce23df87ccfd4f39eb042ef)
