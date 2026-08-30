**Target Module**: [pandas-stubs/core/arrays/base.pyi](../../../pandas-stubs/core/arrays/base.pyi)

# Subsystem: Extension Arrays, Nullable Dtypes, NAType & PyArrow

## 1. Overview & Architectural Role

Modern pandas relies heavily on Extension Arrays, nullable scalar indicators (`pd.NA` / `NAType`), and PyArrow-backed storage (`ArrowDtype`).

## 2. Historical Debates & Evolution

### Null Check Preservation
In PR pandas-dev/pandas-stubs#945: Add type parameters to generics in isna and notna (pandas-dev/pandas-stubs@ea8bdeeaf1f8880103afefa8a1d06048db7c70bb), `JanEricNitschke` added generic type parameters to `isna()` and `notna()` so input types are preserved across checks.

### PyArrow Engine and Array Overloads
In PR pandas-dev/pandas-stubs#1909: GH1908 Add overloads to pd.array for pyarrow types (pandas-dev/pandas-stubs@763c1ea6d6c9695deb22f07fe2d8ff6eebe1bfbc), `loicdiridollou` added overloads to `pd.array()` for PyArrow dtypes (addressing issue pandas-dev/pandas-stubs#1908).

### `pd.NaT` Equality Semantics
In PR pandas-dev/pandas-stubs#1915: GH1907 Adjust pd.NaT.__eq__ and pd.NaT.__ne__ overloads (pandas-dev/pandas-stubs@06925fd587a999fa20eab383975ea28e65cfc4e4), `loicdiridollou` refined `pd.NaT.__eq__` and `pd.NaT.__ne__` overloads (resolving issue pandas-dev/pandas-stubs#1907).

### Pandas 3.0 Arrow Transition
In PR pandas-dev/pandas-stubs#1643: GH1641 Pandas 3.0 support (pandas-dev/pandas-stubs@62435ddc336443f22ac54508c2e9b2ab70325efa), `loicdiridollou` led the monumental effort to support Pandas 3.0 (60 discussion threads), preparing stubs for default string storage backed by PyArrow.

## 3. Key Pull Requests & Commits

- pandas-dev/pandas-stubs#945: Add type parameters to generics in isna and notna (pandas-dev/pandas-stubs@ea8bdeeaf1f8880103afefa8a1d06048db7c70bb)
- pandas-dev/pandas-stubs#1643: GH1641 Pandas 3.0 support (pandas-dev/pandas-stubs@62435ddc336443f22ac54508c2e9b2ab70325efa)
- pandas-dev/pandas-stubs#1909: GH1908 Add overloads to pd.array for pyarrow types (pandas-dev/pandas-stubs@763c1ea6d6c9695deb22f07fe2d8ff6eebe1bfbc)
- pandas-dev/pandas-stubs#1915: GH1907 Adjust pd.NaT.__eq__ and pd.NaT.__ne__ overloads (pandas-dev/pandas-stubs@06925fd587a999fa20eab383975ea28e65cfc4e4)
