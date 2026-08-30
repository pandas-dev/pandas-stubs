---
status: accepted
date: 2022-05-08
deciders: [Dr-Irv, twoertwein, MarcoGorelli, loicdiridollou, cmp0xff]
consulted: [Mypy, Pyright, Pyrefly, and Ty maintainers]
informed: [pandas-stubs contributors]
---

# ADR-0006: Multi-Checker Testing Harness and Assert Type Policy

## Context and Problem Statement

Type stubs are only as good as their validation. Unlike regular Python code that can be tested with unit tests alone, type stubs must satisfy two distinct criteria:
1. **Runtime Execution**: Code in test files must execute without runtime exceptions under `pytest`.
2. **Static Type Soundness**: Types inferred by static analysis tools must match expectations across different type checker implementations.

Early tests used custom inspection functions (`check_XXX()`) and `reveal_type()`, which were fragile, produced noisy log output, and could not be verified in automated CI without manual string matching.

## Decision Drivers

- **Cross-Checker Validation**: Stubs must be verified concurrently against all major Python type checkers: `mypy`, `pyright`, `pyrefly`, and `ty`.
- **Standardized Static Assertions**: Use standard `assert_type()` (PEP 681 / Python 3.11 / `typing_extensions.assert_type`) to assert inferred static types.
- **Single Command Orchestration**: Developers must be able to run all checks via a single Poetry task (`poetry run poe test_all`).

## Considered Options

1. **Test with Mypy Only**:
   - *Pros*: Simpler configuration.
   - *Cons*: Misses pyright, pyrefly, and ty specific issues; poor experience for VS Code and Astral toolchain users.
2. **Comment-based `reveal_type` scraping**:
   - *Pros*: Native to all type checkers.
   - *Cons*: Highly brittle; breaks on formatting differences between checkers.
3. **Multi-Checker Matrix with `assert_type()` and Poe Tasks** *(Chosen)*:
   - *Pros*: Standardized, portable assertions; validates strict compliance across all four major type checkers in CI.
   - *Cons*: Requires maintaining compatible ignore comments when type checkers differ on edge cases (see ADR-0008).

## Decision Outcome

The testing pipeline enforces dual validation:
1. **Runtime Execution**: Run all test files via `pytest` to ensure code runs without runtime errors.
2. **Static Assertion Suite**: Test files use `assert_type(expr, ExpectedType)` from `typing_extensions`.
3. **Checker Suite**:
   - `mypy`: Standard reference checker.
   - `pyright`: Language server engine used in VS Code / Pylance.
   - `pyrefly`: High-speed type checker from Meta (PR pandas-dev/pandas-stubs#1875, PR pandas-dev/pandas-stubs#1910).
   - `ty`: Rust-based type checker from Astral (PR pandas-dev/pandas-stubs#1845, PR pandas-dev/pandas-stubs#1874, PR pandas-dev/pandas-stubs#1877).

### Orchestration Command
```bash
poetry run poe test_all
```

## Consequences

- **Positive**: Eliminates runtime/typing drift and ensures stubs work seamlessly across diverse developer toolchains.
- **Positive**: `assert_type` enforces exact type equality rather than loose assignability.
- **Negative / Neutral**: When a new type checker is integrated or upgraded, discrepancies require temporary suppression comments until upstream bug reports are resolved.

## Historical References & Provenance

- **Primary Pull Requests**:
  - pandas-dev/pandas-stubs#7: Use assert_type instead of check_XXX in some tests (pandas-dev/pandas-stubs@17b4423e65abcf8a0b403bf9fe445f06d3ff2236)
  - pandas-dev/pandas-stubs#8: Use assert_type throughout, remove check_ functions (pandas-dev/pandas-stubs@51aeba53ceb106492a167d87dd117c2922c5d147)
  - pandas-dev/pandas-stubs#10: Test the stubs with mypy and pyright (pandas-dev/pandas-stubs@63d03bc9297357715eda7a41b8f694b91b51395e)
  - pandas-dev/pandas-stubs#59: Enable more pyright checks in CI (pandas-dev/pandas-stubs@1118b791e4cee09bf3129ad216c658a3c6dc9df0)
  - pandas-dev/pandas-stubs#83: Run style checks in CI (pandas-dev/pandas-stubs@3c7e0f65b6c8c78b8095ae8435e7bd1f7102f4c4)
  - pandas-dev/pandas-stubs#114: Assert types at runtime (pandas-dev/pandas-stubs@2fd9697fe7f75e54845c0926f22a6c2df6d9f219)
  - pandas-dev/pandas-stubs#1765: Remove pyrefly ignore-errors in test_io.py (pandas-dev/pandas-stubs@ba78c4b331b02316cf6e3eb6d9a82af2c083750a)
  - pandas-dev/pandas-stubs#1780: Resolve pyrefly ignore-errors in arithmetic test files (pandas-dev/pandas-stubs@d8539f6a8aa4aa4413b8270d5ea7958655b06f76)
  - pandas-dev/pandas-stubs#1836: Strict modes for mypy, pyrefly, and ty (pandas-dev/pandas-stubs@95d5808d03c95029de73faf5595edaeaed9f2a24)
  - pandas-dev/pandas-stubs#1875: Enable pyrefly_dist in testing (pandas-dev/pandas-stubs@5a33cb7dbbf8cb9b98c0dcafd62a909bca53ac00)
  - pandas-dev/pandas-stubs#1877: Add Python versions to type checkers (pandas-dev/pandas-stubs@e86cf5ff34f30504a609f0b688301a6c40729709)
  - pandas-dev/pandas-stubs#1910: Use Pyrefly for type-coverage instead of pyright (pandas-dev/pandas-stubs@ad98b58e51efb316105c4f58c4efbc7280e27694)
