---
status: accepted
date: 2023-07-21
deciders: [ramvikrams, randolf-scholz, Dr-Irv, twoertwein]
consulted: [pandas-stubs community]
informed: [pandas-stubs contributors]
---

# ADR-0016: Series astype() Dtype Specifier Overload Architecture

## Context and Problem Statement
`Series.astype()` converts series data types. Callers supply target types using literals (`"int"`, `"float"`, `"str"`, `"category"`), NumPy string dtypes (`"int64"`, `"float32"`), PyArrow strings (`"int64[pyarrow]"`), or extension dtype instances (`CategoricalDtype`).

In PR pandas-dev/pandas-stubs#519 (84 comments), maintainers struggled with strict literal unions rejecting custom extension dtypes vs broad `str` arguments losing all return type specificity.

## Decision Drivers
- Preserve exact return type inference for common string literals (e.g. `s.astype("category") -> Series[Categorical]`).
- Accept arbitrary `str` and `DtypeObj` for user-defined and NumPy extension dtypes.

## Decision Outcome
In PR pandas-dev/pandas-stubs#756, maintainers adopted an overload cascade that evaluates specific literal string matches first, followed by extension dtype classes, and finally falls back to `Series[Any]`.

## Consequences
- **Positive**: IDE autocompletion infers accurate return dtypes for standard astype operations.
- **Positive**: Zero false positives when users pass dynamic or runtime-constructed dtype objects.

## Historical References & Provenance
- pandas-dev/pandas-stubs#519: gh-372 :  Fixing Series.astype() (pandas-dev/pandas-stubs@c6815aa22ab8d6f510afdfdee8e3c252ee2d4d5c)
- pandas-dev/pandas-stubs#756: added pyarrow/numpy dtype literals and allowed `str` | `DtypeObj` as input for `Series.astype` (pandas-dev/pandas-stubs@490914f32ee048d6f0da7cb8899221081154ab73)
