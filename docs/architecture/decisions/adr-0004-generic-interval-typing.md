---
status: accepted
date: 2022-07-30
deciders: [amotzop, Dr-Irv, twoertwein, cmp0xff]
consulted: [pandas-stubs community]
informed: [pandas-stubs contributors]
---

# ADR-0004: Generic Interval Typing and Endpoint Restrictions

## Context and Problem Statement

`pd.Interval` represents bounded mathematical intervals (e.g. $[0, 5)$ or $[2020-01-01, 2020-01-05]$). In pandas runtime, intervals can have integer, float, timestamp, or timedelta endpoints.

Untyped intervals lead to subtle runtime errors:
1. Adding a `Timestamp` to a time-based `Interval` (`Interval[Timestamp] + Timestamp`) is invalid at runtime, but without generic endpoint tracking, static type checkers cannot detect it.
2. Arithmetic on integer intervals (`Interval[int] + 5 -> Interval[int]`) is valid and preserves the endpoint dtype.
3. Combining incompatible intervals in `IntervalIndex` or `IntervalDtype` must be caught at analysis time.

## Decision Drivers

- **Endpoint Type Fidelity**: Track the scalar endpoint type of `Interval` using `Interval[T]`.
- **Invalid Arithmetic Elimination**: Reject adding `Timestamp` to timestamp-based intervals while allowing numeric addition on numeric intervals.
- **Type Checker Parser Robustness**: Work around type checker limitations in handling complex union bounds in `Interval[...]`.

## Considered Options

1. **Non-Generic `Interval`**:
   - *Pros*: Matches runtime class definition.
   - *Cons*: Cannot distinguish `Interval[Timestamp]` from `Interval[int]`; misses runtime type errors.
2. **Generic `Interval[T]` with Restricted Operators** *(Chosen)*:
   - *Pros*: Precise operator support; rejects invalid datetime addition at type-check time.
   - *Cons*: Type checkers like `ty` encounter edge cases with union bounds `Interval[int | float | Timestamp | Timedelta]`, requiring explicit single-type overloads (PR pandas-dev/pandas-stubs#1845).

## Decision Outcome

`pd.Interval` is typed as `Interval[T]`.

### Implementation Pattern

```python
# In pandas-stubs Interval stub:
class Interval(Generic[_T_co]):
    @property
    def left(self) -> _T_co: ...
    @property
    def right(self) -> _T_co: ...
    @property
    def closed(self) -> IntervalClosedType: ...
    
    # Arithmetic: numeric intervals support translation
    @overload
    def __add__(self: Interval[int], other: int) -> Interval[int]: ...
    @overload
    def __add__(self: Interval[float], other: float) -> Interval[float]: ...
    @overload
    def __add__(self: Interval[Timestamp], other: Timedelta) -> Interval[Timestamp]: ...
    # Timestamp addition is deliberately omitted -> rejected at type-check time!
```

## Consequences

- **Positive**: Adding a `Timestamp` to `Interval[Timestamp]` correctly triggers a type checker error (`[operator]`).
- **Positive**: `IntervalIndex` inherits endpoint typing, ensuring typed `.left` and `.right` properties.
- **Negative / Neutral**: Requires explicit ignore comments in negative test suites for type checker verification.

## Historical References & Provenance

- **Primary Pull Requests**:
  - pandas-dev/pandas-stubs#174: Fixed typing on IntervalIndex functions (pandas-dev/pandas-stubs@a8bc6c63a66f984c4163e491e0b202bbcb2f1c6d)
  - pandas-dev/pandas-stubs#1845: Enable ty in interval and array tests (pandas-dev/pandas-stubs@576c492e3b8700725db67f225d91388124aa8da0)
- **Primary Issues**:
  - pandas-dev/pandas-stubs#1799: Investigate why ty does not accept Interval unions
