# `pandas-stubs` Agent Instructions

The `pandas-stubs` project is introduced in `README.md`.

## Purpose

- Assist contributors by suggesting code changes, tests, and documentation edits for the `pandas-stubs` repository while preserving stability and compatibility.

## Persona & Tone

- Concise, neutral, code-focused. Prioritize correctness, readability, and tests.

## Project Guidelines

- Follow `docs/philosophy.md`.
- Also follow all guidelines for contributing to the codebase specified at [Contributing to the code base](https://pandas.pydata.org/docs/development/contributing_codebase.html).

## Decision heuristics

- Favor small, backward-compatible changes with tests.
- Prefer readability over micro-optimizations unless benchmarks are requested.
- Add tests for behavioral changes.
- When referring to GitHub issues and pull requests, use the short format `pandas-dev/pandas#39196` instead of full URLs like `https://github.com/pandas-dev/pandas/issues/39196`
- If new code is clear from naming and references, do not add detailed comments. Keep code self-documenting.

## Testing Philosophy: Static Type Checking Focus

**CRITICAL:** This project prioritizes **static type checking** over runtime error testing. When designing stubs and tests:

### Invalid Usage Testing Pattern

When an error is expected to raise (invalid operations):

1. **Design stubs** to return `Never` or cause type checker errors for invalid usage
2. **In tests**, protect invalid operations with `if TYPE_CHECKING_INVALID_USAGE:` instead of `with pytest.raises(...)`
3. Add `# type: ignore[<error-code>]` and/or `# pyright: ignore[<error-code>]` comments to verify type checkers catch the error

**Example** (from `docs/philosophy.md`):

```python
i1 = pd.Interval(
    pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-03"), closed="both"
)
if TYPE_CHECKING_INVALID_USAGE:
    _0 = i1 + pd.Timestamp("2000-03-03")  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
```

**Why:** The goal is to catch errors at **type-check time**, not runtime. The `TYPE_CHECKING_INVALID_USAGE` guard (which is `False` at runtime) prevents runtime execution while the ignore comments verify type checkers properly reject the invalid code.

### Do NOT use pytest.raises for type checking

**Incorrect pattern:**

```python
with pytest.raises(TypeError):
    s1 + s2  # adding two timestamps
```

**Correct pattern:**

```python
if TYPE_CHECKING_INVALID_USAGE:
    _0 = s1 + s2  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
```

### Avoiding ruff useless-comparison warnings

When testing operations, assign the result to a dummy variable (e.g., `_0`, `_1`, etc.) to avoid [ruff's useless-comparison rule](https://docs.astral.sh/ruff/rules/useless-comparison/):

```python
if TYPE_CHECKING_INVALID_USAGE:
    _0 = a > b  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
```

This applies to any expression that would trigger warnings about unused results (comparisons, arithmetic operations, etc.).

See `docs/philosophy.md` sections "Testing the Type Stubs" and "Narrow vs. Wide Arguments" for full details.

## Validation After Editing

**REQUIRED:** After editing stubs or tests, run the following command to validate your changes:

```bash
poetry run poe test_all
```

All checks must pass before submitting changes. These commands verify:

- Type stubs are correctly annotated (mypy, pyright, pyrefly)
- Invalid usage is properly rejected by type checkers (ty)
- Tests execute successfully at runtime (test)

## Pull Requests (summary)

- Pull request titles should be descriptive and include one of the following prefixes:
  - ENH: Enhancement, new functionality
  - BUG: Bug fix
  - DOC: Additions/updates to documentation
  - TST: Additions/updates to tests
  - BLD: Updates to the build process/scripts
  - PERF: Performance improvement
  - TYP: Type annotations
  - CLN: Code cleanup
- Pull request descriptions should follow the template, and **succinctly** describe the change being made. Usually a few sentences is sufficient.
- Pull requests which are resolving an existing Github Issue should include a link to the issue in the PR Description.
- Do not add summaries or additional comments to individual commit messages. The single PR description is sufficient.
