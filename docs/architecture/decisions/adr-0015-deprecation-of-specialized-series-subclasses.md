---
status: accepted
date: 2025-07-13
deciders: [cmp0xff, Dr-Irv, loicdiridollou, MarcoGorelli]
consulted: [pandas-stubs community]
informed: [pandas-stubs contributors]
---

# ADR-0015: Deprecation and Removal of Specialized Series Subclasses

## Context and Problem Statement
Earlier versions of pandas-stubs introduced nominal subclasses of Series (`TimestampSeries`, `TimedeltaSeries`, and `OffsetSeries`) to track specialized element types. While this initially simplified arithmetic returns, it created profound architectural issues:
1. Subclassing broke covariance and required thousands of redundant method overrides across generic.pyi.
2. User code typed as `Series[Timestamp]` was rejected by functions expecting `TimestampSeries`.
3. Maintenance burden exploded as new Series methods had to be implemented on four separate classes.

## Decision Drivers
- Unify all 1D series under standard generic `Series[S1]`.
- Eliminate subclass maintenance duplication.
- Align with standard Python typing practices.

## Decision Outcome
Formally deprecate and remove all specialized Series subclasses in favor of parameterized `Series[T]`:
- `TimedeltaSeries` removed in PR pandas-dev/pandas-stubs#1273.
- `TimestampSeries` removed in PR pandas-dev/pandas-stubs#1274 (79 discussion threads).
- `OffsetSeries` removed in PR pandas-dev/pandas-stubs#1390 in favor of `Series[BaseOffset]`.

## Consequences
- **Positive**: Clean, unified type hierarchy where `Series[Timestamp]` and `Series[Timedelta]` share all standard Series methods.
- **Positive**: Simplified overload definitions in DataFrame indexing.
- **Negative / Neutral**: Required updating legacy test files that referenced `TimestampSeries`.

## Historical References & Provenance
- pandas-dev/pandas-stubs#844: OffsetSeries inherits from Series[BaseOffset] (pandas-dev/pandas-stubs@146cf236be3f8a198d00d45371dfc5568f543d09)
- pandas-dev/pandas-stubs#1273: refactor(series)!: ⏱️ drop TimedeltaSeries (pandas-dev/pandas-stubs@7ac98f279dacad533bbfba01ca523c44964b66ee)
- pandas-dev/pandas-stubs#1274: refactor(series)!: 🕰️ drop TimestampSeries (pandas-dev/pandas-stubs@57682145f30d654cd9379d36efd4e3e85033e9d4)
- pandas-dev/pandas-stubs#1390: GH1379 Drop OffsetSeries replacing it with Series[BaseOffset] (pandas-dev/pandas-stubs@10fe362f03bbcf36e01dfd4a263af2dee8e1b9ec)
