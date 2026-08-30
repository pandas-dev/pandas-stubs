# Era 4: The Four-Checker Frontier: Pyrefly, Astral Ty & Pandas 3.0 (2026)

## Overview
In 2026, `pandas-stubs` expanded CI validation across four concurrent type checkers: Mypy, Pyright, Meta's Pyrefly, and Astral's Ty, while completing Pandas 3.0 compatibility.

## Key Maintainer Debates & Breakthroughs

1. **Pandas 3.0 Support**: `loicdiridollou` spearheaded full Pandas 3.0 typing support in PR pandas-dev/pandas-stubs#1643: GH1641 Pandas 3.0 support (pandas-dev/pandas-stubs@62435ddc336443f22ac54508c2e9b2ab70325efa) (60 discussion threads).
2. **Pyrefly & Ty Integration**: `cmp0xff` integrated `ty` and `pyrefly` in PR pandas-dev/pandas-stubs#1836: TST: "strict" modes for `mypy`, `pyrefly` and `ty` (pandas-dev/pandas-stubs@95d5808d03c95029de73faf5595edaeaed9f2a24), PR pandas-dev/pandas-stubs#1845: TST: #1799 enable `ty` in some of `tests/arrays/**/*` (pandas-dev/pandas-stubs@576c492e3b8700725db67f225d91388124aa8da0), PR pandas-dev/pandas-stubs#1867: CLN: #1836 merge `pyrefly_strict` into `pyrefly`, make `ty_all` and `pyrefly_all` passing (pandas-dev/pandas-stubs@9b5b668636819a2a998af2d4c8dacd56961203ef), PR pandas-dev/pandas-stubs#1875: TST: #1801 enable `pyrefly_dist` (pandas-dev/pandas-stubs@5a33cb7dbbf8cb9b98c0dcafd62a909bca53ac00), and PR pandas-dev/pandas-stubs#1877: TST: add Python versions to type checkers astral-sh/ty#4161 facebook/pyrefly#4416 facebook/pyrefly#4422 (pandas-dev/pandas-stubs@e86cf5ff34f30504a609f0b688301a6c40729709).
4. **Canonical Ignore Sequence**: `cmp0xff` standardized the multi-checker ignore sequence in PR pandas-dev/pandas-stubs#1921: CLN: #1916 style typing ignores (pandas-dev/pandas-stubs@cebd95490ba9c7b855051e803193f78602371fb5).

## Key PRs
- pandas-dev/pandas-stubs#1643: GH1641 Pandas 3.0 support (pandas-dev/pandas-stubs@62435ddc336443f22ac54508c2e9b2ab70325efa)
- pandas-dev/pandas-stubs#1745: Read-only (covariant) list parameter annotations (pandas-dev/pandas-stubs@5c8669de44d960b37d32dc044e41afd1542f3cbc)
- pandas-dev/pandas-stubs#1748: GH1415 Enhance typing of Series[Categorical] (pandas-dev/pandas-stubs@a0736699abf028a1f842e9a9f2175d0416735039)
- pandas-dev/pandas-stubs#1765: Remove `pyrefly: ignore-errors` in `test_io.py` (pandas-dev/pandas-stubs@ba78c4b331b02316cf6e3eb6d9a82af2c083750a)
- pandas-dev/pandas-stubs#1780: Resolve `pyrefly: ignore-errors` in arithmetic test files (pandas-dev/pandas-stubs@d8539f6a8aa4aa4413b8270d5ea7958655b06f76)
- pandas-dev/pandas-stubs#1783: TYP: refactor string methods (pandas-dev/pandas-stubs@bf3d649b611025881948017823834ba34a07bacf)
- pandas-dev/pandas-stubs#1803: Allow `df.loc[Scalar, SequenceNotStr[Scalar]]` (pandas-dev/pandas-stubs@af53c5a3839a334519c96bc67d5fe4255db59fa1)
- pandas-dev/pandas-stubs#1811: ENH: Change signature of `is_number` and `is_hashable` (EDIT: originally included also "is_array_like") (pandas-dev/pandas-stubs@14e697f1256ff67695bc4b1e25e06499290944e2)
- pandas-dev/pandas-stubs#1819: type `JsonReader.__init__`, `AbstractHolidayCalendar` attributes, `Holiday` attributes (pandas-dev/pandas-stubs@6c036a9f0cc108465109e4b971587bbd53bc9997)
- pandas-dev/pandas-stubs#1836: TST: "strict" modes for `mypy`, `pyrefly` and `ty` (pandas-dev/pandas-stubs@95d5808d03c95029de73faf5595edaeaed9f2a24)
- pandas-dev/pandas-stubs#1845: TST: #1799 enable `ty` in some of `tests/arrays/**/*` (pandas-dev/pandas-stubs@576c492e3b8700725db67f225d91388124aa8da0)
- pandas-dev/pandas-stubs#1867: CLN: #1836 merge `pyrefly_strict` into `pyrefly`, make `ty_all` and `pyrefly_all` passing (pandas-dev/pandas-stubs@9b5b668636819a2a998af2d4c8dacd56961203ef)
- pandas-dev/pandas-stubs#1874: CLN: bump `ty` (pandas-dev/pandas-stubs@6ce1934be1582424ef882881ee998c6560e27323)
- pandas-dev/pandas-stubs#1875: TST: #1801 enable `pyrefly_dist` (pandas-dev/pandas-stubs@5a33cb7dbbf8cb9b98c0dcafd62a909bca53ac00)
- pandas-dev/pandas-stubs#1877: TST: add Python versions to type checkers astral-sh/ty#4161 facebook/pyrefly#4416 facebook/pyrefly#4422 (pandas-dev/pandas-stubs@e86cf5ff34f30504a609f0b688301a6c40729709)
- pandas-dev/pandas-stubs#1878: BUG: relocate `to_offset` and other `pyrefly`-inspired changes (pandas-dev/pandas-stubs@b0e70149eacc63fcf053c7838cca0509f13d6008)
- pandas-dev/pandas-stubs#1879: CLN: #1878 disable `pyright` `reportUnknownLambdaType` and `pyrefly` `implicity-any-lambda` (pandas-dev/pandas-stubs@eea8472261b5e5d029b7088173374f9264cac258)
- pandas-dev/pandas-stubs#1885: TYP: Allow list as other in DataFrame.dot (pandas-dev/pandas-stubs@93b776f7d6af62abccee6081e44d86479b859b4b)
- pandas-dev/pandas-stubs#1895: CLN: #1880 disable `pyright` `reportUnknownArgumentType` (pandas-dev/pandas-stubs@1b0c5374ac7dd7ccd1f61a943c98f44ba63dd28a)
- pandas-dev/pandas-stubs#1909: GH1908 Add overloads to pd.array for pyarrow types (pandas-dev/pandas-stubs@763c1ea6d6c9695deb22f07fe2d8ff6eebe1bfbc)
- pandas-dev/pandas-stubs#1910: Use Pyrefly for type-coverage instead of pyright (pandas-dev/pandas-stubs@ad98b58e51efb316105c4f58c4efbc7280e27694)
- pandas-dev/pandas-stubs#1911: TYP: accept datetime.timedelta as freq for round, floor, and ceil (pandas-dev/pandas-stubs@6528be476d3583bb8f56cf3a062a1d4ec32a15ba)
- pandas-dev/pandas-stubs#1915: GH1907 Adjust pd.NaT.__eq__ and pd.NaT.__ne__ overloads (pandas-dev/pandas-stubs@06925fd587a999fa20eab383975ea28e65cfc4e4)
- pandas-dev/pandas-stubs#1921: CLN: #1916 style typing ignores (pandas-dev/pandas-stubs@cebd95490ba9c7b855051e803193f78602371fb5)
