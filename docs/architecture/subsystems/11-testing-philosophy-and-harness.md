**Target Module**: [pandas-stubs/__init__.pyi](../../../pandas-stubs/__init__.pyi)

# Subsystem: Testing Philosophy, Static Validation & Nightly CI

## 1. Overview & Architectural Role

Unlike runtime test suites, `pandas-stubs` tests must simultaneously verify that valid code passes multiple static type checkers and executes cleanly in `pytest`, while ensuring invalid operations are rejected at type-check time.

## 2. The Evolution of the Test Harness

### From check_XXX to assert_type
In PR pandas-dev/pandas-stubs#7: use assert_type instead of check_XXX in some tests (pandas-dev/pandas-stubs@17b4423e65abcf8a0b403bf9fe445f06d3ff2236) and PR pandas-dev/pandas-stubs#8: use assert_type throughout, remove check_ functions (pandas-dev/pandas-stubs@51aeba53ceb106492a167d87dd117c2922c5d147), `Dr-Irv` eradicated custom `check_XXX` inspection helpers across the entire test suite, replacing them with standard `assert_type()`.

### Nightly CI against pandas development builds
In PR pandas-dev/pandas-stubs#238: run pytest against nightly (pandas-dev/pandas-stubs@b0728425368ed472f4812995d10bace7fa560e20), `twoertwein` introduced automated nightly CI test runs against pandas development builds (46 discussion threads), catching upstream API changes before official pandas releases.

### Test Framework Migration
In PR pandas-dev/pandas-stubs#1093: GH1089 Migrate frame/series tests to new framework, `loicdiridollou` migrated legacy series and frame tests to a modern, modular testing framework.

### Negative Testing with `TYPE_CHECKING_INVALID_USAGE`
Invalid operations are protected by `if TYPE_CHECKING_INVALID_USAGE:`, preventing runtime crashes while verifying static type checker rejection.

## 3. Key Pull Requests & Commits

- pandas-dev/pandas-stubs#7: use assert_type instead of check_XXX in some tests (pandas-dev/pandas-stubs@17b4423e65abcf8a0b403bf9fe445f06d3ff2236)
- pandas-dev/pandas-stubs#8: use assert_type throughout, remove check_ functions (pandas-dev/pandas-stubs@51aeba53ceb106492a167d87dd117c2922c5d147)
- pandas-dev/pandas-stubs#114: assert types at runtime (pandas-dev/pandas-stubs@2fd9697fe7f75e54845c0926f22a6c2df6d9f219)
- pandas-dev/pandas-stubs#238: run pytest against nightly (pandas-dev/pandas-stubs@b0728425368ed472f4812995d10bace7fa560e20)
- pandas-dev/pandas-stubs#1093: GH1089 Migrate frame/series tests to new framework
- pandas-dev/pandas-stubs#1877: TST: add Python versions to type checkers astral-sh/ty#4161 facebook/pyrefly#4416 facebook/pyrefly#4422 (pandas-dev/pandas-stubs@e86cf5ff34f30504a609f0b688301a6c40729709)
- pandas-dev/pandas-stubs#1921: CLN: #1916 style typing ignores (pandas-dev/pandas-stubs@cebd95490ba9c7b855051e803193f78602371fb5)
