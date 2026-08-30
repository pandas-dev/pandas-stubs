---
status: accepted
date: 2026-08-24
deciders: [cmp0xff, Dr-Irv, loicdiridollou, MarcoGorelli]
consulted: [pandas-stubs contributors]
informed: [pandas-stubs contributors]
---

# ADR-0010: Adoption of PEP 570 Positional-Only Parameters for Dunder and Public Methods

## Context and Problem Statement

Special dunder methods (e.g. `__add__`, `__sub__`, `__eq__`, `__getitem__`, `__contains__`) in Python CPython execution are invoked with positional-only arguments. For example, `s.__add__(other)` in CPython does not allow calling `s.__add__(other=3)`.

Furthermore, many pandas C/Cython extension functions and public APIs enforce positional-only semantics. Without PEP 570 positional-only syntax (`/`), stub signatures allow keyword calls that fail at runtime.

## Decision Drivers

- **CPython Dunder Fidelity**: Accurately reflect positional-only requirements on all operator and container dunder methods.
- **Prevent Invalid Keyword Invocations**: Catch invalid keyword calls (e.g. `s.add(other=...)` vs `s + other`) at type-check time.
- **PEP 570 Standardization**: Adopt standard `/` syntax supported in Python 3.8+.

## Considered Options

1. **Standard Named Parameters Without `/`**:
   - *Pros*: Simple parameter list.
   - *Cons*: Permits invalid keyword arguments that fail at runtime or violate the dunder protocol contract.
2. **PEP 570 Positional-Only Syntax (`/`)** *(Chosen)*:
   - *Pros*: Precise contract, prevents keyword binding on dunders and positional methods.
   - *Cons*: Requires updating legacy signatures across stubs.

## Decision Outcome

All dunder methods and designated positional C-extensions are annotated with PEP 570 positional-only syntax (`/`).

### Example Signature
```python
def __add__(self: Series[S1], other: Series[S2], /) -> Series[Any]: ...
def __eq__(self, other: object, /) -> Series[bool]: ...
def __getitem__(self, key: Any, /) -> Any: ...
```

## Consequences

- **Positive**: Strict alignment with Python data model specifications and CPython runtime behavior.
- **Positive**: Cleaner documentation and IDE parameter hints without distracting pseudo-keyword names.
- **Negative / Neutral**: Requires systematic migration across the codebase.

## Historical References & Provenance

- **Primary Pull Requests**:
  - pandas-dev/pandas-stubs#1312: Positional-only dunder definitions for arithmetic (pandas-dev/pandas-stubs@5459aa73eb07e7ab5049ace65de4d4dd61d01b5a)
  - pandas-dev/pandas-stubs#1914: Align __add__ family of Index and Series with positional parameters
  - pandas-dev/pandas-stubs#1917: Mark dunder method arguments as positional-only
  - pandas-dev/pandas-stubs#1923: Standardize fixture style for arithmetic tests
- **Standards References**:
  - [PEP 570 – Python Positional-Only Parameters](https://peps.python.org/pep-0570/)
