---
status: accepted
date: 2026-07-28
deciders: [cmp0xff, loicdiridollou, MarcoGorelli]
consulted: [Astral ty maintainers]
informed: [pandas-stubs contributors]
---

# ADR-0020: Integration of Astral Ty Rust-Based Type Checker into CI

## Context and Problem Statement
Astral introduced `ty`, an extremely fast Rust-based Python type checker. Ensuring `pandas-stubs` compatibility with `ty` provides future-proof support for Astral toolchains.

## Decision Drivers
- Expand stub compatibility to next-generation type checkers.
- Identify and report parser and union-bound edge cases to upstream maintainers.

## Decision Outcome
Integrated `ty` across Index, Series, and Array test suites in PR pandas-dev/pandas-stubs#1836, PR pandas-dev/pandas-stubs#1845, and PR pandas-dev/pandas-stubs#1867.

## Consequences
- **Positive**: Sub-second type checking feedback.
- **Positive**: Proactive reporting of upstream issues (`astral-sh/ty#4150`, `astral-sh/ty#4161`).

## Historical References & Provenance
- pandas-dev/pandas-stubs#1836: TST: "strict" modes for `mypy`, `pyrefly` and `ty` (pandas-dev/pandas-stubs@95d5808d03c95029de73faf5595edaeaed9f2a24)
- pandas-dev/pandas-stubs#1845: TST: #1799 enable `ty` in some of `tests/arrays/**/*` (pandas-dev/pandas-stubs@576c492e3b8700725db67f225d91388124aa8da0)
- pandas-dev/pandas-stubs#1867: CLN: #1836 merge `pyrefly_strict` into `pyrefly`, make `ty_all` and `pyrefly_all` passing (pandas-dev/pandas-stubs@9b5b668636819a2a998af2d4c8dacd56961203ef)
- pandas-dev/pandas-stubs#1874: CLN: bump `ty` (pandas-dev/pandas-stubs@6ce1934be1582424ef882881ee998c6560e27323)
- pandas-dev/pandas-stubs#1877: TST: add Python versions to type checkers astral-sh/ty#4161 facebook/pyrefly#4416 facebook/pyrefly#4422 (pandas-dev/pandas-stubs@e86cf5ff34f30504a609f0b688301a6c40729709)
