---
status: accepted
date: 2022-06-23
deciders: [KianShah, twoertwein, Dr-Irv, geoffrey-eisenbarth, cmp0xff]
consulted: [pandas-stubs contributors]
informed: [pandas-stubs contributors]
---

# ADR-0011: Indexing and Selection Semantics for Accessors (.loc, .iloc, .at, .iat)

## Context and Problem Statement

Pandas indexing via `.loc`, `.iloc`, `.at`, `.iat`, and `__getitem__` is among the most versatile and complex parts of the API:
- Indexing with a single scalar returns a scalar or `Series`.
- Indexing with a slice or list of labels returns a `Series` or `DataFrame`.
- Indexing with boolean arrays or masks filters rows.
- MultiIndex selection supports tuples, partial slices, and cross-sections.

Statically typing these accessors requires balancing return type precision against exponential overload combinations.

## Decision Drivers

- **Return Type Precision**: Ensure `.iloc[0]` returns `Scalar` or `Series`, while `.iloc[0:5]` returns `DataFrame` or `Series`.
- **Distinguishing `.at`/`.iat` from `.loc`/`.iloc`**: `.at` and `.iat` access only single scalar values and should never return collections.
- **MultiIndex Tuple Resolution**: Accurately type tuple indexing into hierarchical multi-level indices.

## Considered Options

1. **Unified Accessor returning `Any`**:
   - *Pros*: Avoids massive overload lists.
   - *Cons*: Loses all static safety in core data selection workflows.
2. **Dedicated Accessor Indexer Classes with Overload Matrices** *(Chosen)*:
   - *Pros*: High precision for `.loc`, `.iloc`, `.at`, `.iat`; handles scalar vs slice vs sequence indexing cleanly.
   - *Cons*: Significant stub size for accessor classes (`_LocIndexer`, `_iLocIndexer`).

## Decision Outcome

Implement dedicated generic indexer classes with discrete overloads for scalar, slice, sequence, and boolean mask keys.

### Indexing Type Aliases (in `tests/_typing.py`)
```python
IndexingInt: TypeAlias = (
    int | np.int_ | np.integer | np.unsignedinteger | np.signedinteger | np.int8
)
AxesData: TypeAlias = Mapping[S0, Any] | Axes | KeysView[S0]
```

### Accessor Rules
1. **`.at` / `.iat`**: Strictly return scalar values (`ScalarT`).
2. **`.loc`**: Overloaded for `(Scalar, SequenceNotStr[Scalar])` (PR pandas-dev/pandas-stubs#1803) and slice indexing.
3. **`Index` Hierarchy**: Generic `Index[S0]` and `MultiIndex` operations share `IndexOpsMixin` (PR pandas-dev/pandas-stubs#760).

## Consequences

- **Positive**: Type checkers infer exact scalar vs DataFrame types when slicing and indexing.
- **Positive**: Prevents invalid indexing operations (e.g. non-integer indexing in `.iloc`).
- **Negative / Neutral**: Heavy accessor stub definitions require ongoing maintenance.

## Historical References & Provenance

- **Primary Pull Requests**:
  - pandas-dev/pandas-stubs#39: Fix to_dict and from_dict type stubs (pandas-dev/pandas-stubs@8fbe101a4b28335cac7391d3630288553e01ed5b)
  - pandas-dev/pandas-stubs#760: Make IndexOpsMixin and Index generic (pandas-dev/pandas-stubs@f7621f408f4cfed08453e1b906bf9a6a17b34b04)
  - pandas-dev/pandas-stubs#1803: Allow df.loc[Scalar, SequenceNotStr[Scalar]] (pandas-dev/pandas-stubs@af53c5a3839a334519c96bc67d5fe4255db59fa1)
