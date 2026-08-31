---
status: accepted
date: 2026-08-20
deciders: [MarcoGorelli, cmp0xff, loicdiridollou]
consulted: [Pyrefly maintainers]
informed: [pandas-stubs contributors]
---

# ADR-0019: Adoption of Pyrefly for Type-Coverage and Strict Validation

## Context and Problem Statement
Tracking stub type coverage across thousands of functions was previously done using Pyright. Pyrefly (from Meta) provides high-performance static analysis with fine-grained error diagnostic tracking.

## Decision Drivers
- High-speed type coverage reporting in CI.
- Verification against emerging type checker ecosystems.

## Decision Outcome
Integrated `pyrefly` in PR pandas-dev/pandas-stubs#1875 and transitioned primary type-coverage tracking to Pyrefly in PR pandas-dev/pandas-stubs#1910.

## Consequences
- **Positive**: Faster CI coverage feedback.
- **Positive**: Uncovered subtle lambda typing edge cases addressed in PR pandas-dev/pandas-stubs#1878 and PR pandas-dev/pandas-stubs#1879.

## Historical References & Provenance
- pandas-dev/pandas-stubs#1875: TST: #1801 enable `pyrefly_dist` (pandas-dev/pandas-stubs@5a33cb7dbbf8cb9b98c0dcafd62a909bca53ac00)
- pandas-dev/pandas-stubs#1878: BUG: relocate `to_offset` and other `pyrefly`-inspired changes (pandas-dev/pandas-stubs@b0e70149eacc63fcf053c7838cca0509f13d6008)
- pandas-dev/pandas-stubs#1879: CLN: #1878 disable `pyright` `reportUnknownLambdaType` and `pyrefly` `implicity-any-lambda` (pandas-dev/pandas-stubs@eea8472261b5e5d029b7088173374f9264cac258)
- pandas-dev/pandas-stubs#1910: Use Pyrefly for type-coverage instead of pyright (pandas-dev/pandas-stubs@ad98b58e51efb316105c4f58c4efbc7280e27694)
