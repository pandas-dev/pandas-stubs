---
status: accepted
date: 2024-06-22
deciders: [cmp0xff, twoertwein, Dr-Irv, MarcoGorelli]
consulted: [NumPy typing team]
informed: [pandas-stubs contributors]
---

# ADR-0018: NumPy 2.0 Migration and Dtype Compatibility Layer

## Context and Problem Statement
NumPy 2.0 introduced major changes to scalar type hierarchies, removing deprecated type aliases (e.g. `np.float_`, `np.int_`) and altering promotion rules.

## Decision Drivers
- Support both NumPy 1.x and NumPy 2.x environments.
- Use `numpy-typing-compat` and conditional type aliases in `tests/_typing.py`.

## Decision Outcome
Adopted compatibility type aliases for NumPy integer and floating types and pinned test configurations in PR pandas-dev/pandas-stubs#1785.

## Consequences
- **Positive**: Seamless type checking across NumPy 1.26 and NumPy 2.0+.
- **Negative / Neutral**: Required capping dependencies in specific CI test matrices.

## Historical References & Provenance
- pandas-dev/pandas-stubs#317: MAINT: Bump pandas to 1.5.0 (pandas-dev/pandas-stubs@d1f00f3c1576a9e64f9729c7daa7612fcfa0ed63)
- pandas-dev/pandas-stubs#756: added pyarrow/numpy dtype literals and allowed `str` | `DtypeObj` as input for `Series.astype` (pandas-dev/pandas-stubs@490914f32ee048d6f0da7cb8899221081154ab73)
- pandas-dev/pandas-stubs#1785: BUG: update `numpy-typing-compat` and cap `numpy` (pandas-dev/pandas-stubs@a2024ed9c5ed53457443a97fa4a7c6cd9c664556)
