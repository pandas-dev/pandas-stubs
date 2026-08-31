**Target Module**: [pandas-stubs/__init__.pyi](../../../pandas-stubs/__init__.pyi)

# Subsystem: Multi-Type-Checker Matrix, Toolchain Quirks & Ignore Standards

## 1. Overview & Architectural Role

`pandas-stubs` is tested continuously against four distinct static type checkers:
- **Mypy**: Reference checker
- **Pyright**: VS Code language server
- **Pyrefly**: Meta's high-speed type checker
- **Ty**: Astral's Rust-based type checker

## 2. The Multi-Checker Frontier (2025–2026)

### Pyrefly Integration
In PR pandas-dev/pandas-stubs#1875: TST: #1801 enable `pyrefly_dist` (pandas-dev/pandas-stubs@5a33cb7dbbf8cb9b98c0dcafd62a909bca53ac00), `cmp0xff` enabled `pyrefly_dist`. In PR pandas-dev/pandas-stubs#1910: Use Pyrefly for type-coverage instead of pyright (pandas-dev/pandas-stubs@ad98b58e51efb316105c4f58c4efbc7280e27694), `MarcoGorelli` transitioned type-coverage metrics to Pyrefly.

### Astral Ty Integration
In PR pandas-dev/pandas-stubs#1836: TST: "strict" modes for `mypy`, `pyrefly` and `ty` (pandas-dev/pandas-stubs@95d5808d03c95029de73faf5595edaeaed9f2a24), PR pandas-dev/pandas-stubs#1845: TST: #1799 enable `ty` in some of `tests/arrays/**/*` (pandas-dev/pandas-stubs@576c492e3b8700725db67f225d91388124aa8da0), and PR pandas-dev/pandas-stubs#1867: CLN: #1836 merge `pyrefly_strict` into `pyrefly`, make `ty_all` and `pyrefly_all` passing (pandas-dev/pandas-stubs@9b5b668636819a2a998af2d4c8dacd56961203ef), `cmp0xff` integrated `ty` into the testing matrix, navigating parser differences in union bounds and strict modes.

### Standardized Ignore Sequence
In PR pandas-dev/pandas-stubs#1921: CLN: #1916 style typing ignores (pandas-dev/pandas-stubs@cebd95490ba9c7b855051e803193f78602371fb5), `cmp0xff` standardized the canonical ignore ordering across all test and stub files:
`# type: ignore[...] # pyright: ignore[...] # pyrefly: ignore[...] # ty: ignore[...]`

## 3. Key Pull Requests & Commits

- pandas-dev/pandas-stubs#59: TYP/CI: enable more pyright checks (pandas-dev/pandas-stubs@1118b791e4cee09bf3129ad216c658a3c6dc9df0)
- pandas-dev/pandas-stubs#83: CI: run style checks on CI (pandas-dev/pandas-stubs@3c7e0f65b6c8c78b8095ae8435e7bd1f7102f4c4)
- pandas-dev/pandas-stubs#1765: Remove `pyrefly: ignore-errors` in `test_io.py` (pandas-dev/pandas-stubs@ba78c4b331b02316cf6e3eb6d9a82af2c083750a)
- pandas-dev/pandas-stubs#1780: Resolve `pyrefly: ignore-errors` in arithmetic test files (pandas-dev/pandas-stubs@d8539f6a8aa4aa4413b8270d5ea7958655b06f76)
- pandas-dev/pandas-stubs#1836: TST: "strict" modes for `mypy`, `pyrefly` and `ty` (pandas-dev/pandas-stubs@95d5808d03c95029de73faf5595edaeaed9f2a24)
- pandas-dev/pandas-stubs#1845: TST: #1799 enable `ty` in some of `tests/arrays/**/*` (pandas-dev/pandas-stubs@576c492e3b8700725db67f225d91388124aa8da0)
- pandas-dev/pandas-stubs#1867: CLN: #1836 merge `pyrefly_strict` into `pyrefly`, make `ty_all` and `pyrefly_all` passing (pandas-dev/pandas-stubs@9b5b668636819a2a998af2d4c8dacd56961203ef)
- pandas-dev/pandas-stubs#1874: CLN: bump `ty` (pandas-dev/pandas-stubs@6ce1934be1582424ef882881ee998c6560e27323)
- pandas-dev/pandas-stubs#1875: TST: #1801 enable `pyrefly_dist` (pandas-dev/pandas-stubs@5a33cb7dbbf8cb9b98c0dcafd62a909bca53ac00)
- pandas-dev/pandas-stubs#1877: TST: add Python versions to type checkers astral-sh/ty#4161 facebook/pyrefly#4416 facebook/pyrefly#4422 (pandas-dev/pandas-stubs@e86cf5ff34f30504a609f0b688301a6c40729709)
- pandas-dev/pandas-stubs#1878: BUG: relocate `to_offset` and other `pyrefly`-inspired changes (pandas-dev/pandas-stubs@b0e70149eacc63fcf053c7838cca0509f13d6008)
- pandas-dev/pandas-stubs#1879: CLN: #1878 disable `pyright` `reportUnknownLambdaType` and `pyrefly` `implicity-any-lambda` (pandas-dev/pandas-stubs@eea8472261b5e5d029b7088173374f9264cac258)
- pandas-dev/pandas-stubs#1895: CLN: #1880 disable `pyright` `reportUnknownArgumentType` (pandas-dev/pandas-stubs@1b0c5374ac7dd7ccd1f61a943c98f44ba63dd28a)
- pandas-dev/pandas-stubs#1910: Use Pyrefly for type-coverage instead of pyright (pandas-dev/pandas-stubs@ad98b58e51efb316105c4f58c4efbc7280e27694)
- pandas-dev/pandas-stubs#1921: CLN: #1916 style typing ignores (pandas-dev/pandas-stubs@cebd95490ba9c7b855051e803193f78602371fb5)
