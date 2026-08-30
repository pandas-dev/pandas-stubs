---
status: accepted
date: 2022-05-01
deciders: [Dr-Irv, twoertwein, MarcoGorelli, Microsoft Pyright Team]
consulted: [pandas-dev core team]
informed: [pandas-stubs contributors]
---

# ADR-0001: Separation of Type Stubs from Pandas Runtime and Public API Scope

## Context and Problem Statement

Pandas is a foundational data manipulation library in the Python ecosystem with a vast dynamic API, extensive Cython internals, and complex runtime polymorphism. Early attempts to embed full static typing directly inside the pandas runtime faced significant challenges:
1. Pandas source code typing focuses on internal implementation safety and Cython boundaries rather than public API ergonomics.
2. The runtime API is heavily dynamic (e.g. methods returning scalar vs `Series` vs `DataFrame` depending on argument values and axes).
3. The original stub baseline was developed by Microsoft in the `python-type-stubs` repository to support Pyright/VS Code.

The community needed a dedicated home for type stubs that could evolve independently, support multiple type checkers, and model the recommended public API conventions.

## Decision Drivers

- **Public API Focus**: Type hints must reflect how users should consume pandas, not how pandas is implemented internally.
- **Independent Release Cadence**: Type stub fixes and enhancements need to be released rapidly without waiting for major pandas core release cycles.
- **PEP 561 Compliance**: Deliver stubs as a standalone stub-only package (`pandas-stubs`) containing a `py.typed` marker file and `.pyi` interface definitions.
- **Cross-Type-Checker Compatibility**: Ensure compatibility across all major static analysis tools (`mypy`, `pyright`, `pyrefly`, `ty`).

## Considered Options

1. **Inline Type Annotations in pandas core repository (`pandas-dev/pandas`)**:
   - *Pros*: Single repository, type annotations updated alongside runtime code.
   - *Cons*: Slow release cadence, strict coupling to pandas runtime release schedules, friction with complex Cython/C internals, internal implementation details leaking into public signatures.
2. **Typeshed inclusion (`python/typeshed`)**:
   - *Pros*: Centralized repository for standard library and third-party stubs.
   - *Cons*: Typeshed rules limit rapid experimentation with advanced generic models (e.g. Generic `Series[S1]`), and pandas is too large/fast-moving for typeshed maintainers.
3. **Dedicated PEP 561 Stub Repository (`pandas-dev/pandas-stubs`)** *(Chosen)*:
   - *Pros*: Dedicated issue tracker, independent versioning matching pandas releases (e.g., `pandas-stubs 2.x.x`), freedom to design high-fidelity generic extensions (like generic `Series` and `Interval`), and unified testing harness.
   - *Cons*: Potential drift between runtime pandas changes and stub updates (mitigated by nightly CI tests against pandas development builds).

## Decision Outcome

The project operates as an independent PEP 561 stub package under the `pandas-dev` GitHub organization.

Key architectural boundaries:
- **Public API Target**: Stubs strictly model public, documented pandas APIs. Internal classes and helper methods annotated during Microsoft's initial extraction that do not affect public consumption are maintained for stub integrity or pruned when redundant.
- **Distribution**: Stubs are packaged under the top-level `pandas-stubs/` directory with root `__init__.pyi` and `py.typed` marker.
- **Nightly Synchronization**: CI continuously tests stubs against upstream pandas nightly releases to catch upstream API breaking changes before public releases.

## Consequences

- **Positive**: Users gain rich, IDE-agnostic static type checking for pandas without runtime performance overhead.
- **Positive**: The stub repository can move faster and implement expressive typing models (e.g., generic `Series[Timestamp]`) that are not present in pandas core.
- **Negative / Neutral**: Maintenance requires active tracking of pandas core deprecations and API changes.

## Historical References & Provenance

- **Foundational Pull Requests**:
  - pandas-dev/pandas-stubs#6: Clean up with black and tests (pandas-dev/pandas-stubs@1af190411028fd05f7fceaa7043a5c811d864e33)
  - pandas-dev/pandas-stubs#10: Test the stubs with mypy and pyright (pandas-dev/pandas-stubs@63d03bc9297357715eda7a41b8f694b91b51395e)
  - pandas-dev/pandas-stubs#24: Stub packaging and PEP 561 compliance (pandas-dev/pandas-stubs@2eba4d4e512421927a4ba2e6d0ac7bbd4e934afe)
  - pandas-dev/pandas-stubs#183: Stubtest configuration and ignore-missing-stub handling (pandas-dev/pandas-stubs@1ec0bb9f5dd4714e50143e1067e3b6addba6cd78)
  - pandas-dev/pandas-stubs#1128: Prune unnecessary definitions in `generic.pyi` (pandas-dev/pandas-stubs@2acecd7181711c08046d1826ee6888d60ca2aa45)
- **Upstream & Standards References**:
  - [PEP 561 – Distributing and Packaging Type Information](https://peps.python.org/pep-0561/)
  - [Pandas Typing Guidelines](https://pandas.pydata.org/docs/development/contributing_codebase.html?highlight=typing#type-hints)
