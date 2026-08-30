---
status: accepted
date: 2025-05-29
deciders: [MarcoGorelli, loicdiridollou, cmp0xff, Dr-Irv]
consulted: [pandas-stubs contributors]
informed: [pandas-stubs contributors]
---

# ADR-0014: Shared Typing Architecture and Test Typing Module

## Context and Problem Statement

A complex typing repository requires numerous internal type aliases, TypeVars, and protocols:
1. Distributing identical TypeVar definitions across dozens of individual `.pyi` stub files leads to naming collisions and maintenance drift.
2. In multi-type-checker environments, strict mode flags (`strict = true`) flag missing or ambiguous type exports.
3. Test suites need access to internal aliases and protocols without coupling test types to the public stub wheel distribution.

## Decision Drivers

- **Centralized Alias Repository**: Provide a single authoritative source of shared type aliases and TypeVars.
- **Decoupled Test Suite Types**: Isolate test-specific helper types in `tests/_typing.py` while keeping distributed package stubs clean under `pandas-stubs/`.
- **Cyclic Import Elimination**: Prevent circular type import cycles between stub modules.

## Considered Options

1. **Inline Type Definitions in Every `.pyi` File**:
   - *Pros*: Self-contained stub files.
   - *Cons*: Massive code duplication; inconsistent bounds across files.
2. **Single Distributed Internal Module (`pandas._typing`)**:
   - *Pros*: Centralized aliases.
   - *Cons*: Internal aliases leak into public user autocompletion.
3. **Decoupled Architecture: Package Stubs + `tests/_typing.py`** *(Chosen)*:
   - *Pros*: Clean public distribution under `pandas-stubs/`; comprehensive shared type aliases, TypeVars, and protocols in `tests/_typing.py` for test harnesses.
   - *Cons*: Requires maintaining synchronization when public aliases evolve.

## Decision Outcome

1. **Package Stub Structure**: Public stubs reside in `pandas-stubs/`, exposing clean interfaces and PEP 561 compliance.
2. **Centralized Test Typing (`tests/_typing.py`)**: Defines shared TypeVars (`S0`, `S1`, `S2_contra`, `C2`, `ByT`), data aliases (`SeriesDType`, `AxesData`, `IntoColumn`), and structural protocols (`Just`, `CovariantList`, `SupportsTrueDiv`).
3. **Explicit Export Control**: `tests/_typing.py` utilizes explicit `__all__` and scoped imports to avoid polluting global namespace.

## Consequences

- **Positive**: Zero circular import errors across type checker test runs.
- **Positive**: Central place to update TypeVar bounds (e.g. PEP 696 defaults) in a single commit.
- **Negative / Neutral**: Changes to core TypeVars must be tested against both stub files and test typing definitions.

## Historical References & Provenance

- **Primary Pull Requests**:
  - pandas-dev/pandas-stubs#1151: Timestamp type aliases and pyright-strict compliance (pandas-dev/pandas-stubs@69b833cc8343055b47c12b1db8cad7fce3fe26a7)
  - pandas-dev/pandas-stubs#1232: Use default in TypeVar for Series and Index (pandas-dev/pandas-stubs@709d7458460df69c09b25084761fa9378d0fae41)
  - pandas-dev/pandas-stubs#1783: Refactor string method typing (pandas-dev/pandas-stubs@bf3d649b611025881948017823834ba34a07bacf)
  - pandas-dev/pandas-stubs#1836: Strict modes for mypy, pyrefly, and ty (pandas-dev/pandas-stubs@95d5808d03c95029de73faf5595edaeaed9f2a24)
  - pandas-dev/pandas-stubs#1921: Style typing ignores across test typing module (pandas-dev/pandas-stubs@cebd95490ba9c7b855051e803193f78602371fb5)
