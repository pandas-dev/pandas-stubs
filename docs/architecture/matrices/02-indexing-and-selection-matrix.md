# Indexing & Selection Type Algebra Matrix

## 1. Overview

Pandas data accessors (`.loc`, `.iloc`, `.at`, `.iat`, and `__getitem__`) map 1D Series and 2D DataFrame structures into scalar values, series slices, or sub-dataframes. The type system models these accessors via dedicated Generic Indexer classes (`_LocIndexer`, `_iLocIndexer`, `_AtIndexer`, `_iAtIndexer`).

---

## 2. Accessor Resolution Matrix

| Target Object | Accessor | Key Expression Type | Resolved Return Type | Structural Rule & Behavior | Key Provenance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Series[S1]` | `[key]` / `.loc[key]` | `Scalar` (matching index label) | `S1` | Single element scalar extraction | pandas-dev/pandas-stubs#760 |
| `Series[S1]` | `[key]` / `.loc[key]` | `slice` \| `SequenceNotStr[Scalar]` | `Series[S1]` | Preserves generic element type | pandas-dev/pandas-stubs#760, pandas-dev/pandas-stubs#1803 |
| `Series[S1]` | `[mask]` / `.loc[mask]` | `Series[bool]` \| `np_1darray_bool` | `Series[S1]` | Boolean mask filtering | pandas-dev/pandas-stubs#1311 |
| `Series[S1]` | `.iloc[i]` | `int` | `S1` | Pure integer position scalar lookup | pandas-dev/pandas-stubs#760 |
| `Series[S1]` | `.iloc[slice]` | `slice` \| `Sequence[int]` | `Series[S1]` | Positional subset selection | pandas-dev/pandas-stubs#760 |
| `Series[S1]` | `.iat[i]` / `.at[label]` | `int` / `Scalar` | `S1` | High-speed scalar only (no sequences permitted) | pandas-dev/pandas-stubs#760 |
| `DataFrame` | `[col_name]` | `Scalar` (str, int, etc.) | `Series[Any]` (or `Series[T]`) | Single column projection | pandas-dev/pandas-stubs#1803 |
| `DataFrame` | `[[col1, col2]]` | `SequenceNotStr[Scalar]` | `DataFrame` | Multi-column projection | pandas-dev/pandas-stubs#1803 |
| `DataFrame` | `.loc[row, col]` | `Scalar, Scalar` | `Any` (scalar value) | Single cell extraction | pandas-dev/pandas-stubs#1803 |
| `DataFrame` | `.loc[row, cols]` | `Scalar, SequenceNotStr[Scalar]` | `Series[Any]` | Single row across multiple columns | pandas-dev/pandas-stubs#1803 |
| `DataFrame` | `.loc[rows, col]` | `SequenceNotStr[Scalar], Scalar` | `Series[Any]` | Multiple rows for a single column | pandas-dev/pandas-stubs#1803 |
| `DataFrame` | `.loc[rows, cols]` | `SequenceNotStr[Scalar], SequenceNotStr[Scalar]` | `DataFrame` | 2D sub-matrix slicing | pandas-dev/pandas-stubs#1803 |
| `DataFrame` | `.iloc[r, c]` | `int, int` | `Any` (scalar value) | Positional cell extraction | pandas-dev/pandas-stubs#760 |
| `DataFrame` | `.iloc[r, cols]` | `int, slice` \| `int, Sequence[int]` | `Series[Any]` | Positional row projection | pandas-dev/pandas-stubs#760 |
| `DataFrame` | `.iloc[rows, cols]` | `slice, slice` | `DataFrame` | Positional 2D block | pandas-dev/pandas-stubs#760 |
| `DataFrame` | `.at[r, c]` / `.iat[r, c]`| `Scalar, Scalar` / `int, int` | `Any` (scalar value) | Strictly scalar; sequence keys trigger static error | pandas-dev/pandas-stubs#760 |

---

## 3. Disambiguation Rules: `SequenceNotStr` Protocol

In Python, `str` is an instance of `Sequence[str]`. Without disambiguation, an overload accepting `Sequence[Hashable]` would match a single string column name like `"age"`, leading to ambiguity between single-column Series extraction and multi-column DataFrame extraction.

### Solution: `SequenceNotStr`
The protocol `SequenceNotStr[T]` (defined in `pandas-stubs/_typing.pyi`) matches lists, tuples, sets, and 1D arrays while explicitly excluding `str` and `bytes` (citing PR pandas-dev/pandas-stubs#1803).

## 4. Collection Tradeoffs: `SequenceNotStr` vs `CovariantList`
When typing indexers that accept collections, the architecture utilizes both `SequenceNotStr` and `CovariantList` to prevent unexpected string unpacking while supporting list-like behaviors.
- **Limitations**: Both structures have inherent type-theoretic limitations in Python. See discussions linked in pandas-dev/pandas-stubs#1609 for a deep dive into why strict structural typing for list-likes remains challenging and how these two primitives are applied to mitigate indexing edge cases.
