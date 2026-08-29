# `pandas-stubs` Agent Instructions

The `pandas-stubs` project is introduced in `README.md`.

## Purpose

- Assist contributors by suggesting code changes, tests, and documentation edits for the `pandas-stubs` repository while preserving stability and compatibility.

## Persona & Tone

- Concise, neutral, code-focused. Prioritize correctness, readability, and tests.

## Project Guidelines

- Follow `docs/philosophy.md`.
- Also follow all guidelines for contributing to the codebase specified at [Contributing to the code base](https://pandas.pydata.org/docs/development/contributing_codebase.html).

## Citation & Reference Formatting Rules (Important)

When generating documentation or writing PR descriptions, format all GitHub references as plain text so that GitHub renders them as native autolinks.
- Avoid wrapping PRs, Issues, or Commits in backticks.
- **Pull Requests / Issues**: pandas-dev/pandas-stubs#1911 (not `` `pandas-dev/pandas-stubs#1911` ``).
- **Commits**: pandas-dev/pandas-stubs@cebd954 (not `` `cebd954` ``).
- **External issues**: pandas-dev/pandas#39196 (not a full URL).

## PR and Commit Conventions

- **Commit signatures**: When AI generates commits, add a `Co-authored-by:` trailer naming the actual model or tool. Finding official emails can be tricky, so use these standard templates for popular agents:
  - `Co-authored-by: Antigravity AI <bot@antigravity.dev>`
  - `Co-authored-by: DeepCode <bot@deepcode.ai>`
  - `Co-authored-by: GitHub Copilot <noreply@github.com>`
  - `Co-authored-by: OpenAI Codex <noreply@openai.com>`
- **Emojis in PR Titles**: Consider adding an emoji to the PR title (e.g., 🤖 or 🪄) as a fun, immediate signal to maintainers that AI assisted with the PR.
- **PR body — visible and collapsible text**:
  - *Visible text* (for human reviewers): concise summary, checklist, links. Humans read the rendered page.
  - *Collapsible text* (for AI agents): place comprehensive implementation plans or technical notes inside a `<details><summary>AI Implementation Plan</summary>` block at the bottom of the body. This keeps the PR clean for humans but accessible if they want to read it, while agents can read it via the raw body.
  - Rules: never put machine-only detail in the visible text; never hide information a human reviewer needs.
- **Splitting exceptionally large PRs**: If a PR is massive, split it into small, individually reviewable sub-PRs. Each sub-PR body should start with `- [x] Towards pandas-dev/pandas-stubs#<parent>`.

## Decision Heuristics

- Favor small, backward-compatible changes with tests.
- Prefer readability over micro-optimizations unless benchmarks are requested.
- Add tests for behavioral changes.
- If new code is clear from naming and references, do not add detailed comments. Keep code self-documenting.

## Testing Philosophy: Static Type Checking Focus (The 4-Checker Paradigm)

This project prioritizes **static type checking** over runtime error testing and enforces a 4-checker CI pipeline: `mypy`, `pyright`, `pyrefly`, and `ty`.

When designing stubs and tests:

### Invalid Usage Testing Pattern

When an error is expected to raise (invalid operations):

1. **Design stubs** to return `Never` or cause type checker errors for invalid usage.
2. **In tests**, protect invalid operations with `if TYPE_CHECKING_INVALID_USAGE:` instead of `with pytest.raises(...)`.
3. Add the canonical multi-checker ignore comment sequence.

**Example 1: Standard invalid operations** (tested in `tests/scalars/timedelta/test_arithmetic.py`):

```python
if TYPE_CHECKING_INVALID_USAGE:
    _0 = a * b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
```

**Example 2: Guarding `Never` assertions**
When a stub returns `Never`, attempting to assign it or use it will trigger a type error. Because `Never` represents an operation that shouldn't happen, placing it in standard execution flow can cause runtime crashes. Guard these inside an uncalled function:

```python
if TYPE_CHECKING_INVALID_USAGE:
    def _test_never():
        # This function is never called at runtime, safely allowing type checkers to evaluate the Never return
        _0 = s.dt.tz_convert("UTC")  # type: ignore[call-overload] # pyright: ignore[reportCallIssue] # pyrefly: ignore[no-matching-overload] # ty: ignore[missing-overload]
```

**Why:** The goal is to catch errors at **type-check time**, not runtime. The `TYPE_CHECKING_INVALID_USAGE` guard (which is `False` at runtime) and uncalled functions prevent runtime execution while the ignore comments verify `mypy`, `pyright`, `pyrefly`, and `ty` properly reject the invalid code.

**Note on `ty`**: `ty` is being gradually integrated into `pandas-stubs`, fully parallel with the other three type checkers. It is currently in beta; we actively report bugs to `astral-sh/ty` (and similarly to `mypy`, `pyright`, and `pyrefly`).

### Do NOT use pytest.raises for type checking

**Incorrect pattern:**

```python
with pytest.raises(TypeError):
    s1 + s2  # adding two timestamps
```

**Correct pattern:**

```python
if TYPE_CHECKING_INVALID_USAGE:
    _0 = s1 + s2  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
```

### Avoiding ruff useless-comparison warnings

When testing operations, assign the result to a dummy variable (e.g., `_0`, `_1`, etc.) to avoid [ruff's useless-comparison rule](https://docs.astral.sh/ruff/rules/useless-comparison/):

```python
if TYPE_CHECKING_INVALID_USAGE:
    _0 = a > b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
```

This applies to any expression that would trigger warnings about unused results (comparisons, arithmetic operations, etc.).

See `docs/philosophy.md` sections "Testing the Type Stubs" and "Narrow vs. Wide Arguments" for full details.

## Validation After Editing

**REQUIRED:** After editing stubs or tests, run the following command to validate your changes:

```bash
poetry run poe test_all
```

All checks must pass before submitting changes. These commands verify:

- Type stubs are correctly annotated (`mypy`, `pyright`, `pyrefly`, `ty`)
- Tests execute successfully at runtime (`pytest`)

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
