# Era 3: NumPy 2.0, PyArrow Integration & Dropping Specialized Series (2024–2025)

## Overview
The 2024–2025 era was marked by the retirement of specialized series subclasses (`TimestampSeries`, `TimedeltaSeries`, `OffsetSeries`), the adoption of PEP 696 typevar defaults, and preparation for Pandas 3.0 / PyArrow defaults.

## Key Maintainer Debates & Breakthroughs

1. **Dropping TimestampSeries & TimedeltaSeries**: `cmp0xff` and `Dr-Irv` consolidated series into unified `Series[Timestamp]` and `Series[Timedelta]` in PR pandas-dev/pandas-stubs#1273: refactor(series)!: ⏱️ drop TimedeltaSeries (pandas-dev/pandas-stubs@7ac98f279dacad533bbfba01ca523c44964b66ee) and PR pandas-dev/pandas-stubs#1274: refactor(series)!: 🕰️ drop TimestampSeries (pandas-dev/pandas-stubs@57682145f30d654cd9379d36efd4e3e85033e9d4) (79 discussion threads).
2. **Operator Algebra & Symmetries**: `cmp0xff` overhauled arithmetic addition, subtraction, and true division in PR pandas-dev/pandas-stubs#1275: feat(series): #1098 arithmetic addition (pandas-dev/pandas-stubs@845f9c593227f75e3fe8b33feb8c7a94d5edaaca), PR pandas-dev/pandas-stubs#1311: feat(series): addition for bools (pandas-dev/pandas-stubs@67755efd3432ed285ebd8e650e7bd09f134ac15a), PR pandas-dev/pandas-stubs#1312: feat(series): arithmetic truediv and sub (pandas-dev/pandas-stubs@5459aa73eb07e7ab5049ace65de4d4dd61d01b5a), and PR pandas-dev/pandas-stubs#1343: fix(series): arithmetic for Series[Any] (pandas-dev/pandas-stubs@669a2585c794505da7d0b6cd80edac3fa875972d).
3. **PEP 696 Adoption**: `MarcoGorelli` simplified `Series` and `Index` TypeVars with `default=Any` in PR pandas-dev/pandas-stubs#1232: Use `default` in `TypeVar` so `Series` defaults to `Series[Any]`, and `Index` to `Index[Any]` (pandas-dev/pandas-stubs@709d7458460df69c09b25084761fa9378d0fae41).
4. **Test Framework Modernization**: `loicdiridollou` migrated tests to a new modular framework in PR pandas-dev/pandas-stubs#1093: GH1089 Migrate frame/series tests to new framework.

## Key PRs
- pandas-dev/pandas-stubs#945: Add type parameters to generics in isna and notna (pandas-dev/pandas-stubs@ea8bdeeaf1f8880103afefa8a1d06048db7c70bb)
- pandas-dev/pandas-stubs#960: Fix series.map overloads
- pandas-dev/pandas-stubs#966: GroupBy[Series].count() return type should be Series[int] (pandas-dev/pandas-stubs@7e6aee4e41f8f60b4ce23df87ccfd4f39eb042ef)
- pandas-dev/pandas-stubs#1075: GH1074 Add type hint Series[list[str]] for Series.str.split with expand=False (pandas-dev/pandas-stubs@109dc86010a6e47067cf1e60ac108d0b99932250)
- pandas-dev/pandas-stubs#1093: GH1089 Migrate frame/series tests to new framework
- pandas-dev/pandas-stubs#1128: remove several unnecessary definitions in generic.pyi (pandas-dev/pandas-stubs@2acecd7181711c08046d1826ee6888d60ca2aa45)
- pandas-dev/pandas-stubs#1146: Introduce UnknownSeries and UnknownIndex, type `core.strings.pyi` using them (pandas-dev/pandas-stubs@2b0279e8c8f9985a7eff5b1c387444299b2c2813)
- pandas-dev/pandas-stubs#1151: make tslibs strptime, timedeltas, and timestamps pass with pyright-strict (pandas-dev/pandas-stubs@69b833cc8343055b47c12b1db8cad7fce3fe26a7)
- pandas-dev/pandas-stubs#1232: Use `default` in `TypeVar` so `Series` defaults to `Series[Any]`, and `Index` to `Index[Any]` (pandas-dev/pandas-stubs@709d7458460df69c09b25084761fa9378d0fae41)
- pandas-dev/pandas-stubs#1233: Remove some unnecessary S1 TypeVars (pandas-dev/pandas-stubs@f8a329d51989e49ecba5c02982646021d3942bf8)
- pandas-dev/pandas-stubs#1242: GH456 First attempt GroupBy.transform improved typing (pandas-dev/pandas-stubs@b12c28d7a987e9b67a13ad0e3335f531973c9114)
- pandas-dev/pandas-stubs#1273: refactor(series)!: ⏱️ drop TimedeltaSeries (pandas-dev/pandas-stubs@7ac98f279dacad533bbfba01ca523c44964b66ee)
- pandas-dev/pandas-stubs#1274: refactor(series)!: 🕰️ drop TimestampSeries (pandas-dev/pandas-stubs@57682145f30d654cd9379d36efd4e3e85033e9d4)
- pandas-dev/pandas-stubs#1275: feat(series): #1098 arithmetic addition (pandas-dev/pandas-stubs@845f9c593227f75e3fe8b33feb8c7a94d5edaaca)
- pandas-dev/pandas-stubs#1293: Add defaults for parameters (pandas-dev/pandas-stubs@ace58eabaedf5af7640b01704927d4e9af783d10)
- pandas-dev/pandas-stubs#1311: feat(series): addition for bools (pandas-dev/pandas-stubs@67755efd3432ed285ebd8e650e7bd09f134ac15a)
- pandas-dev/pandas-stubs#1312: feat(series): arithmetic truediv and sub (pandas-dev/pandas-stubs@5459aa73eb07e7ab5049ace65de4d4dd61d01b5a)
- pandas-dev/pandas-stubs#1343: fix(series): arithmetic for Series[Any] (pandas-dev/pandas-stubs@669a2585c794505da7d0b6cd80edac3fa875972d)
- pandas-dev/pandas-stubs#1390: GH1379 Drop OffsetSeries replacing it with Series[BaseOffset] (pandas-dev/pandas-stubs@10fe362f03bbcf36e01dfd4a263af2dee8e1b9ec)
- pandas-dev/pandas-stubs#1437: Fix: proper return types for MultiIndex.swaplevel & MultiIndex.union (pandas-dev/pandas-stubs@143bab4291b054ea9ec6c6b30a95b32f485e0373)
- pandas-dev/pandas-stubs#1542: GH1541 Revert Series[Any].__add__(str) (pandas-dev/pandas-stubs@33c462592f31e845d00dda52bc6b8b094d7b496f)
