---
status: accepted
date: 2024-06-28
deciders: [JanEricNitschke, loicdiridollou, cmp0xff, MarcoGorelli]
consulted: [PyArrow team, pandas-stubs contributors]
informed: [pandas-stubs contributors]
---

# ADR-0013: Nullable Dtypes, NAType, and PyArrow Integration

## Context and Problem Statement

Modern pandas supports nullable data types (`Int64Dtype`, `Float64Dtype`, `StringDtype`, `BooleanDtype`) and PyArrow-backed storage (`ArrowDtype`). Null missing values are represented by `pd.NA` (`NAType`) or `pd.NaT` for temporal types.

Key typing challenges:
1. `pd.NA` has unique three-valued logic (e.g. `pd.NA == pd.NA` evaluates to `pd.NA`, not `True` or `False`).
2. `pd.NaT` comparisons with non-temporal objects return `False` or `True` depending on equality/inequality operators (PR pandas-dev/pandas-stubs#1915).
3. `pd.array()` constructs `ExtensionArray` or `ArrowExtensionArray` based on dtype arguments.

## Decision Drivers

- **Accurate Missing Value Semantics**: Accurately type `pd.NA` and `pd.NaT` comparison overloads.
- **PyArrow Engine Integration**: Provide overloads for PyArrow dtypes and engines in I/O and array constructors.
- **Generic Null Check Functions**: Support type parameter preservation in `pd.isna()` and `pd.notna()`.

## Considered Options

1. **Treat `pd.NA` as `Any`**:
   - *Pros*: Avoids complex overloads.
   - *Cons*: Silently allows invalid logic; loses three-valued logic awareness.
2. **Specialized `NAType` and `ExtensionDtype` Hierarchy** *(Chosen)*:
   - *Pros*: Precise equality/inequality signatures; full support for PyArrow array instantiation.
   - *Cons*: Requires extensive overload matrix for `pd.array()` and comparison dunders.

## Decision Outcome

### 1. `pd.NaT` Equality Overloads
Updated in PR pandas-dev/pandas-stubs#1915:
```python
# pd.NaT.__eq__ and __ne__ overloads:
@overload
def __eq__(self, other: NaTType, /) -> Literal[False]: ...
@overload
def __eq__(self, other: object, /) -> bool: ...
@overload
def __ne__(self, other: NaTType, /) -> Literal[True]: ...
@overload
def __ne__(self, other: object, /) -> bool: ...
```

### 2. `pd.array()` PyArrow Overloads
Added in PR pandas-dev/pandas-stubs#1909:
```python
@overload
def array(data: Sequence[Any], dtype: ArrowDtype, ...) -> ArrowExtensionArray: ...
@overload
def array(data: Sequence[Any], dtype: ExtensionDtype, ...) -> ExtensionArray: ...
```

### 3. Generic Null Checks
`pd.isna()` and `pd.notna()` preserve type arguments across scalar, Series, and DataFrame inputs (PR pandas-dev/pandas-stubs#945).

## Consequences

- **Positive**: Strict type safety when interacting with PyArrow backed tables and nullable dtypes.
- **Positive**: Correct boolean return types for `pd.NaT` equality checks.
- **Negative / Neutral**: Ongoing maintenance as pandas and PyArrow expand Arrow integration.

## Historical References & Provenance

- **Primary Pull Requests**:
  - pandas-dev/pandas-stubs#945: Add type parameters to generics in isna and notna (pandas-dev/pandas-stubs@ea8bdeeaf1f8880103afefa8a1d06048db7c70bb)
  - pandas-dev/pandas-stubs#1909: Add overloads to pd.array for pyarrow types (pandas-dev/pandas-stubs@763c1ea6d6c9695deb22f07fe2d8ff6eebe1bfbc)
  - pandas-dev/pandas-stubs#1915: Adjust pd.NaT.__eq__ and pd.NaT.__ne__ overloads (pandas-dev/pandas-stubs@06925fd587a999fa20eab383975ea28e65cfc4e4)
- **Primary Issues**:
  - pandas-dev/pandas-stubs#1907: pd.NaT.__eq__ and __ne__ can give more exact results
  - pandas-dev/pandas-stubs#1908: pd.array with Arrow dtypes not working
