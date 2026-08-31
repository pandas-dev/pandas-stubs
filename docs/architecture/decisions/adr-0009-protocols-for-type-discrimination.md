---
status: accepted
date: 2025-08-19
deciders: [cmp0xff, Dr-Irv, loicdiridollou, MarcoGorelli]
consulted: [typing-sig]
informed: [pandas-stubs contributors]
---

# ADR-0009: Structural and Nominal Protocols for Type Discrimination

## Context and Problem Statement

In Python's typing system, `bool` is a subtype of `int` (`issubclass(bool, int) is True`). Consequently, an overload accepting `int` will automatically match `bool` unless explicitly prevented.

In pandas, however, operations on `bool` have radically different semantics from operations on `int`:
1. `pd.Series([1, 2]) - pd.Series([3, 4])` is valid subtraction yielding `Series[int]`.
2. `pd.Series([True]) - pd.Series([False])` raises a NumPy `TypeError` (boolean subtraction not supported).
3. Parameter annotations expecting read-only lists (`list[T]`) are invariant by default, rejecting valid covariant subtypes (`list[SubType]`).

## Decision Drivers

- **Type Discrimination**: Disambiguate `bool` from numeric types (`int`, `float`, `complex`) in overloads.
- **Covariance for Sequences**: Allow parameter annotations to accept covariant lists without making stubs mutable or invariant.
- **Division Protocol Modeling**: Decouple division operand resolution into structural protocols (`SupportsTrueDiv`, `SupportsRTrueDiv`).

## Considered Options

1. **Runtime Type Guards**:
   - *Pros*: Simple at runtime.
   - *Cons*: Cannot be expressed purely in static stubs without type checker support.
2. **`Just[T]` and Structural Protocols** *(Chosen)*:
   - *Pros*: `Just[T]` uses a nominal/structural property trick to match *only* exact `T` without matching subtypes like `bool`. `CovariantList[T]` provides a read-only list protocol.
   - *Cons*: Requires ignore comments for property overrides across some checkers (`mypy`, `pyrefly`).

## Decision Outcome

Implement specialized protocols in `tests/_typing.py`:

### 1. The `Just[T]` Protocol (Exact Type Matching)
Used to match exact types like `int` or `float` without matching `bool`:

```python
class Just(Protocol, Generic[T]):
    @property
    @override
    def __class__(self, /) -> type[T]: ...
    @__class__.setter
    @override
    def __class__(self, t: type[T], /) -> None: ...
```

### 2. `CovariantList[_T_co]` Protocol
Used for read-only parameter annotations that accept lists of subtypes:

```python
class CovariantList(Protocol[_T_co]):
    __hash__: ClassVar[None]
    @property
    @override
    def __class__(self) -> type[list[Any]]: ...
    def __iter__(self) -> Iterator[_T_co]: ...
```

### 3. Division Protocols
```python
class SupportsTrueDiv(Protocol[_T_contra, _T_co]):
    def __truediv__(self, x: _T_contra, /) -> _T_co: ...

class SupportsRTrueDiv(Protocol[_T_contra, _T_co]):
    def __rtruediv__(self, x: _T_contra, /) -> _T_co: ...
```

## Consequences

- **Positive**: Overloads for integer arithmetic no longer accidentally accept boolean series.
- **Positive**: Read-only function parameters can safely accept covariant list arguments.
- **Negative / Neutral**: Requires careful protocol maintenance to work around type checker limitations (`python/mypy#15900`, `astral-sh/ty#4150`).

## Historical References & Provenance

- **Primary Pull Requests**:
  - pandas-dev/pandas-stubs#1312: Arithmetic truediv and subtraction protocols (pandas-dev/pandas-stubs@5459aa73eb07e7ab5049ace65de4d4dd61d01b5a)
  - pandas-dev/pandas-stubs#1745: Read-only (covariant) list parameter annotations (pandas-dev/pandas-stubs@5c8669de44d960b37d32dc044e41afd1542f3cbc)
  - pandas-dev/pandas-stubs#1836: Strict typing modes with protocol support (pandas-dev/pandas-stubs@95d5808d03c95029de73faf5595edaeaed9f2a24)
  - pandas-dev/pandas-stubs#1845: Enable ty with array protocols (pandas-dev/pandas-stubs@576c492e3b8700725db67f225d91388124aa8da0)
