---
status: accepted
date: 2025-08-19
deciders: [cmp0xff, Dr-Irv, loicdiridollou, MarcoGorelli]
consulted: [pandas-stubs community]
informed: [pandas-stubs contributors]
---

# ADR-0003: Progressive Arithmetic Typing and Deterministic Fallbacks

## Context and Problem Statement

Mathematical and logical operations in pandas (`+`, `-`, `*`, `/`, `//`, `%`, `**`, `&`, `|`, `^`) exhibit extensive polymorphic behavior depending on operands:
1. Arithmetic between numeric series (e.g. `Series[int] + Series[float] -> Series[float]`).
2. Temporal arithmetic between timestamps and timedeltas (e.g. `Series[Timestamp] - Series[Timestamp] -> Series[Timedelta]`, while `Series[Timestamp] + Series[Timestamp]` is invalid at runtime and should be rejected at type-check time).
3. Arithmetic on unannotated or DataFrame-derived series (`df["col"] -> Series[Any]`).

When a column is extracted from a `DataFrame`, static type checkers treat it as `Series[Any]`. A naive typing system would either:
- Disallow all operations on `Series[Any]`, breaking untyped user workflows.
- Allow everything and return `Series[Any]`, losing all subsequent type safety even when the outcome is unambiguous.

## Decision Drivers

- **Deterministic Return Typing**: When an operation with a specific right-hand operand has exactly *one valid outcome type*, return that concrete type even if the left-hand operand is `Series[Any]`.
- **Sound Error Detection**: Prevent nonsensical operations (e.g. adding two `Timestamp` series or subtracting boolean series) statically.
- **Progressive Narrowing**: Allow users to opt into higher type safety progressively without breaking existing untyped DataFrame manipulation.
- **Checker Parity**: Overload definitions must behave consistently across mypy, pyright, pyrefly, and ty.

## Considered Options

1. **Strict Type Matching Only (`Series[Any]` Returns `Series[Any]`)**:
   - *Pros*: Simple stub overloads.
   - *Cons*: Erases types in chained operations (e.g., `df["ts"] - pd.Timestamp(...)` becomes `Series[Any]` instead of `TimedeltaSeries`).
2. **Permissive Fallback with Progressive Specialization** *(Chosen)*:
   - *Pros*: If there is only one valid outcome (e.g. subtracting `pd.Timestamp` from `Series[Any]` can only validly yield `TimedeltaSeries`, or adding `str` can only validly yield `Series[str]`), provide that single valid outcome. When multiple valid outcomes exist (e.g. numeric addition), fall back to `Series[Any]`.
   - *Cons*: Requires intricate overload sequencing in stub definitions to avoid overlap warnings in strict type checkers.

## Decision Outcome

Adopt progressive arithmetic typing across Series and Index dunder methods (`__add__`, `__sub__`, `__mul__`, `__truediv__`, `__radd__`, etc.).

### Overload Design Pattern

```python
# Example: Progressive subtraction pattern in Series
@overload
def __sub__(self: Series[Timestamp], other: Series[Timestamp] | Timestamp) -> Series[Timedelta]: ...
@overload
def __sub__(self: Series[Timedelta], other: Series[Timedelta] | Timedelta) -> Series[Timedelta]: ...
@overload
def __sub__(self: Series[Timestamp], other: Series[Timedelta] | Timedelta) -> Series[Timestamp]: ...
# Progressive fallback: subtracting Timestamp from Series[Any] can only be TimedeltaSeries
@overload
def __sub__(self: Series[Any], other: Timestamp) -> Series[Timedelta]: ...
@overload
def __sub__(self: Series[S1], other: Any) -> Series[Any]: ...
```

### Boolean and Division Rules
1. **Boolean Subtraction Prohibition**: Boolean addition is permitted (`Series[bool] + Series[bool] -> Series[int]`), but boolean subtraction raises a runtime `TypeError` in NumPy/pandas and is rejected by the stubs (PR pandas-dev/pandas-stubs#1311, PR pandas-dev/pandas-stubs#1312).
2. **True Division Return Types**: True division (`/`) between integer or float series always produces `Series[float]`, while true division by timedeltas produces `Series[float]` (ratio) or `Series[Timedelta]` (rate).

## Consequences

- **Positive**: Code operating on DataFrame columns (`frame["timestamp"] - pd.Timestamp(...)`) gains accurate downstream type hints (`Series[Timedelta]`) without explicit casting.
- **Positive**: Type checkers reject invalid operations like `Timestamp + Timestamp` at type-check time.
- **Negative / Neutral**: Complex overload cascades require exhaustive matrix testing in `tests/series/arithmetic/`.

## Historical References & Provenance

- **Primary Pull Requests**:
  - pandas-dev/pandas-stubs#106: Allow complex types in series arithmetic operations (pandas-dev/pandas-stubs@9d80790c5bde23d597663eee7d5f5a3cfbbbde6b)
  - pandas-dev/pandas-stubs#1275: Arithmetic addition implementation for series (pandas-dev/pandas-stubs@845f9c593227f75e3fe8b33feb8c7a94d5edaaca)
  - pandas-dev/pandas-stubs#1311: Addition for boolean series (pandas-dev/pandas-stubs@67755efd3432ed285ebd8e650e7bd09f134ac15a)
  - pandas-dev/pandas-stubs#1312: Arithmetic truediv and subtraction (pandas-dev/pandas-stubs@5459aa73eb07e7ab5049ace65de4d4dd61d01b5a)
  - pandas-dev/pandas-stubs#1343: Fix arithmetic overloads for `Series[Any]` (pandas-dev/pandas-stubs@669a2585c794505da7d0b6cd80edac3fa875972d)
  - pandas-dev/pandas-stubs#1542: Revert over-broad `Series[Any].__add__(str)` (pandas-dev/pandas-stubs@33c462592f31e845d00dda52bc6b8b094d7b496f)
- **Primary Issues**:
  - pandas-dev/pandas-stubs#1098: Addition of Series with complex returns unknown
  - pandas-dev/pandas-stubs#1541: String concatenation on Series[Any]
