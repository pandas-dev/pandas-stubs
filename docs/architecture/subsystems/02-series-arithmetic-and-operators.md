**Target Module**: [pandas-stubs/core/series.pyi](../../../pandas-stubs/core/series.pyi)

# Subsystem: Series Arithmetic, Operator Algebra & Symmetries

## 1. Overview & Architectural Role

Pandas series support rich operator polymorphism across unary, binary, reverse, and in-place dunders. The operator typing architecture enforces static mathematical correctness while permitting progressive type inference for untyped data.

## 2. Historical Struggles & Debates

### Boolean Operator Restrictions
In NumPy and Python, `bool` is an integer subclass. However, `pd.Series([True]) - pd.Series([False])` raises a runtime `TypeError`. In PR pandas-dev/pandas-stubs#1311: feat(series): addition for bools (pandas-dev/pandas-stubs@67755efd3432ed285ebd8e650e7bd09f134ac15a) and PR pandas-dev/pandas-stubs#1312: feat(series): arithmetic truediv and sub (pandas-dev/pandas-stubs@5459aa73eb07e7ab5049ace65de4d4dd61d01b5a), `cmp0xff` and `Dr-Irv` restructured the arithmetic hierarchy to allow boolean addition (`Series[bool] + Series[bool] -> Series[int]`) while strictly omitting boolean subtraction.

### Complex Number Operations
In PR pandas-dev/pandas-stubs#106: Allow `num` to be a `complex` type to support `Series` operations. (pandas-dev/pandas-stubs@9d80790c5bde23d597663eee7d5f5a3cfbbbde6b), `aholmes` added complex support across series arithmetic operations, resolving long-standing issues where complex arithmetic defaulted to `Any`.

### Symmetrical Operator Alignment & Positional Arguments
In PR pandas-dev/pandas-stubs#1275: feat(series): #1098 arithmetic addition (pandas-dev/pandas-stubs@845f9c593227f75e3fe8b33feb8c7a94d5edaaca) and PR pandas-dev/pandas-stubs#1312: feat(series): arithmetic truediv and sub (pandas-dev/pandas-stubs@5459aa73eb07e7ab5049ace65de4d4dd61d01b5a), the entire arithmetic family (`__add__`, `__sub__`, `__mul__`, `__truediv__`, `__floordiv__`, `__mod__`, `__pow__`) was standardized. Later, PR pandas-dev/pandas-stubs#1914: TYP: #1378 Align `__add__` family of Index and Series with `__mul__` and PR pandas-dev/pandas-stubs#1917: TYP: #1914 mark dunder method arguments as positional-only aligned the `__add__` family of `Index` and `Series` with `__mul__`, marking all dunder arguments as PEP 570 positional-only (`/`) to adhere to CPython runtime protocol standards.

### Progressive Typing on Series[Any]
When a series is extracted from a DataFrame (`df["ts"]`), it is typed as `Series[Any]`. In PR pandas-dev/pandas-stubs#1343: fix(series): arithmetic for Series[Any] (pandas-dev/pandas-stubs@669a2585c794505da7d0b6cd80edac3fa875972d), maintainers introduced single-outcome progressive typing: subtracting `Timestamp` from `Series[Any]` returns `Series[Timedelta]`, because no other valid outcome is mathematically possible.

## 3. Structural Protocols & Implementation Pattern

```python
class SupportsTrueDiv(Protocol[_T_contra, _T_co]):
    def __truediv__(self, x: _T_contra, /) -> _T_co: ...

class SupportsRTrueDiv(Protocol[_T_contra, _T_co]):
    def __rtruediv__(self, x: _T_contra, /) -> _T_co: ...
```

## 4. Key Pull Requests & Commits

- pandas-dev/pandas-stubs#106: Allow `num` to be a `complex` type to support `Series` operations. (pandas-dev/pandas-stubs@9d80790c5bde23d597663eee7d5f5a3cfbbbde6b)
- pandas-dev/pandas-stubs#378: added_int_bitwise_operator (pandas-dev/pandas-stubs@c5d66489a6de952a4ae8c3fc313da0f560578338)
- pandas-dev/pandas-stubs#432: added np.timedelta64 for series arithmetic methods (pandas-dev/pandas-stubs@b7163c25f2b1a986078e3787c5110913054088f0)
- pandas-dev/pandas-stubs#1275: feat(series): #1098 arithmetic addition (pandas-dev/pandas-stubs@845f9c593227f75e3fe8b33feb8c7a94d5edaaca)
- pandas-dev/pandas-stubs#1311: feat(series): addition for bools (pandas-dev/pandas-stubs@67755efd3432ed285ebd8e650e7bd09f134ac15a)
- pandas-dev/pandas-stubs#1312: feat(series): arithmetic truediv and sub (pandas-dev/pandas-stubs@5459aa73eb07e7ab5049ace65de4d4dd61d01b5a)
- pandas-dev/pandas-stubs#1343: fix(series): arithmetic for Series[Any] (pandas-dev/pandas-stubs@669a2585c794505da7d0b6cd80edac3fa875972d)
- pandas-dev/pandas-stubs#1542: GH1541 Revert Series[Any].__add__(str) (pandas-dev/pandas-stubs@33c462592f31e845d00dda52bc6b8b094d7b496f)
- pandas-dev/pandas-stubs#1914: TYP: #1378 Align `__add__` family of Index and Series with `__mul__`
- pandas-dev/pandas-stubs#1917: TYP: #1914 mark dunder method arguments as positional-only
- pandas-dev/pandas-stubs#1923: TST: #1914 standardise fixture style for left operands in arithmetic tests
