---
status: accepted
date: 2025-08-19
deciders: [cmp0xff, Dr-Irv, loicdiridollou]
consulted: [pandas-stubs contributors]
informed: [pandas-stubs contributors]
---

# ADR-0017: Prohibition of Boolean Subtraction in Static Type Stubs

## Context and Problem Statement
In Python runtime, `isinstance(True, int)` is `True`. Many static type systems implicitly treat `bool` as assignable to `int`.

However, NumPy and pandas raise a runtime `TypeError` on boolean subtraction (`s1 - s2` where `s1, s2: Series[bool]`). Standard integer overloads would permit this invalid operation without static warnings.

## Decision Drivers
- Statically reject invalid boolean subtraction.
- Permit valid boolean addition (`s1 + s2 -> Series[int]`).
- Disambiguate exact `int` from `bool` using nominal protocols.

## Decision Outcome
1. Explicitly omit `__sub__` overloads for `Series[bool]`.
2. Use the `Just[T]` protocol in `tests/_typing.py` to prevent `bool` from matching integer subtraction overloads (PR pandas-dev/pandas-stubs#1312).

## Consequences
- **Positive**: Type checkers report `[operator]` errors if user code attempts boolean subtraction.
- **Positive**: Eliminates runtime type errors in production pipelines.

## Historical References & Provenance
- pandas-dev/pandas-stubs#1311: feat(series): addition for bools (pandas-dev/pandas-stubs@67755efd3432ed285ebd8e650e7bd09f134ac15a)
- pandas-dev/pandas-stubs#1312: feat(series): arithmetic truediv and sub (pandas-dev/pandas-stubs@5459aa73eb07e7ab5049ace65de4d4dd61d01b5a)
