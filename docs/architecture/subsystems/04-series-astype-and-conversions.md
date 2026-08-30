**Target Module**: [pandas-stubs/core/series.pyi](../../../pandas-stubs/core/series.pyi)

# Subsystem: Series.astype() and Dtype Conversion Specification

## 1. Overview & Architectural Role

`Series.astype()` is one of the most heavily used and dynamic conversion methods in pandas. Callers can specify target dtypes using Python builtins (`int`, `str`, `float`), NumPy strings (`"int64"`, `"float32"`), PyArrow types (`"int64[pyarrow]"`), or extension dtype instances (`CategoricalDtype`).

## 2. The Great astype() Debate (PR #519 & PR #756)

In PR pandas-dev/pandas-stubs#519: gh-372 :  Fixing Series.astype() (pandas-dev/pandas-stubs@c6815aa22ab8d6f510afdfdee8e3c252ee2d4d5c) (84 discussions across `ramvikrams`, `Dr-Irv`, `twoertwein`), maintainers debated whether `astype()` should use a closed set of literal strings vs broad `str`. An overly narrow literal union broke custom extension dtypes, while broad `str` lost return type inference.

In PR pandas-dev/pandas-stubs#756: added pyarrow/numpy dtype literals and allowed `str` | `DtypeObj` as input for `Series.astype` (pandas-dev/pandas-stubs@490914f32ee048d6f0da7cb8899221081154ab73), `randolf-scholz` added PyArrow and NumPy dtype string literals and allowed `str | DtypeObj` input, creating an overload matrix that preserved precise return types for common string literals (`astype("category") -> Series[Categorical]`) while providing a graceful fallback.

## 3. Key Pull Requests & Commits

- pandas-dev/pandas-stubs#519: gh-372 :  Fixing Series.astype() (pandas-dev/pandas-stubs@c6815aa22ab8d6f510afdfdee8e3c252ee2d4d5c)
- pandas-dev/pandas-stubs#756: added pyarrow/numpy dtype literals and allowed `str` | `DtypeObj` as input for `Series.astype` (pandas-dev/pandas-stubs@490914f32ee048d6f0da7cb8899221081154ab73)
