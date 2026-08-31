---
status: accepted
date: 2022-07-10
deciders: [Dr-Irv, twoertwein, MarcoGorelli, loicdiridollou, cmp0xff]
consulted: [pandas-stubs community]
informed: [pandas-stubs contributors]
---

# ADR-0005: Argument Narrowing vs. Widening Policy

## Context and Problem Statement

A critical challenge when designing type stubs for pandas functions and methods is calibrating the breadth of parameter annotations ("just right"):
- **Too Narrow**: Annotating an argument as `list[str]` when pandas accepts any `Sequence[str]`, `Iterable[str]`, or 1D array rejects valid user code.
- **Too Wide**: Annotating an argument as `Sequence[Any]` or `Any` when pandas internally assumes indexable sequences allows invalid arguments (e.g. infinite iterators or unhashable objects) without static warnings.

## Decision Drivers

- **Ergonomics for Callers**: Accept general Python structures (`Sequence`, `Iterable`, `Mapping`) on input arguments where pandas safely consumes them.
- **Precision on Return Types**: Keep return types as narrow and concrete as possible.
- **Distinguishing Strings from Sequences**: Strings in Python are `Sequence[str]`, which can cause subtle typing bugs when APIs expect a sequence of column names vs a single column name.

## Considered Options

1. **Strict Concrete Types (`list`, `dict`, `set`)**:
   - *Pros*: Simple signatures.
   - *Cons*: Rejects tuples, custom sequences, and numpy arrays.
2. **Broad `Any` / `Iterable[Any]`**:
   - *Pros*: Rarely triggers false positives.
   - *Cons*: Defeats static type checking; passes buggy code silently.
3. **Calibrated Structural & Union Types (`SequenceNotStr`, `ScalarArg`, `AxesData`)** *(Chosen)*:
   - *Pros*: Disambiguates single string vs list of strings; accepts arrays and sequences; enforces valid arguments.
   - *Cons*: Requires custom type aliases in `tests/_typing.py`.

## Decision Outcome

Adopt calibrated argument typing rules:
1. **Disambiguating Strings and Sequences**: Use `SequenceNotStr[T]` or overloads (`str | Sequence[str]`) to prevent a `str` from accidentally matching a `Sequence[str]` overload.
2. **Matrix Multiplication (`dot`)**: Accept lists and sequences as `other` in `DataFrame.dot` (PR pandas-dev/pandas-stubs#1885).
3. **Frequency Arguments**: Accept `datetime.timedelta` alongside string aliases for rounding and frequency operations (PR pandas-dev/pandas-stubs#1911).
4. **CSV Parsing**: Calibrate `usecols` to accept callables, lists of ints, or lists of strings (PR pandas-dev/pandas-stubs#65).

## Consequences

- **Positive**: Clean separation between scalar column access and multi-column list access in `.loc` / `__getitem__`.
- **Positive**: High compatibility with NumPy and standard library sequence types.
- **Negative / Neutral**: Requires careful overload ordering so that specific types (e.g. `list[str]`) match before generic iterables.

## Historical References & Provenance

- **Primary Pull Requests**:
  - pandas-dev/pandas-stubs#65: Added type hint to read_csv usecols (pandas-dev/pandas-stubs@a4860495b0a7852719ad7503899d5a10befd8066)
  - pandas-dev/pandas-stubs#124: Allow subtypes of List[Scalar] via ScalarArg (pandas-dev/pandas-stubs@d2f32648340c29bf57f73753f06ab9c6ad18a98a)
  - pandas-dev/pandas-stubs#1803: Allow df.loc[Scalar, SequenceNotStr[Scalar]] (pandas-dev/pandas-stubs@af53c5a3839a334519c96bc67d5fe4255db59fa1)
  - pandas-dev/pandas-stubs#1811: Change signature of is_number and is_hashable (pandas-dev/pandas-stubs@14e697f1256ff67695bc4b1e25e06499290944e2)
  - pandas-dev/pandas-stubs#1885: Allow list as other in DataFrame.dot (pandas-dev/pandas-stubs@93b776f7d6af62abccee6081e44d86479b859b4b)
  - pandas-dev/pandas-stubs#1911: Accept timedelta as freq for round, floor, and ceil (pandas-dev/pandas-stubs@6528be476d3583bb8f56cf3a062a1d4ec32a15ba)
