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
- Avoid wrapping PRs, Issues, or Commits in backticks; GitHub will only render these as clickable links if they are left as plain text.
- **Pull Requests / Issues**: pandas-dev/pandas-stubs#1911 (not `` `pandas-dev/pandas-stubs#1911` ``).
- **Commits**: pandas-dev/pandas-stubs@cebd954 (not `` `cebd954` ``).
- **External issues**: pandas-dev/pandas#39196 (not a full URL).

## Pull Requests and Commits

- **PR titles**: descriptive, and include one of the following prefixes:
  - ENH: Enhancement, new functionality
  - BUG: Bug fix
  - DOC: Additions/updates to documentation
  - TST: Additions/updates to tests
  - BLD: Updates to the build process/scripts
  - PERF: Performance improvement
  - TYP: Type annotations
  - CLN: Code cleanup
- **PR descriptions**: follow the template; keep the visible text succinct (usually a few sentences). PRs resolving an existing issue should include a link to it in the description.
- **PR body — concise for humans, an index for agents**:
  - Humans read the rendered page: lead with a concise summary, checklist, and links.
  - Put a collapsible `<details><summary>Accountability Index</summary>` at the bottom. It is an index, not the record: a commit table (each commit with a one-line what and why), links to the discussion-thread comments where review decisions were made, and a short caveats note. The full rationale lives in commit message bodies and the discussion thread.
  - Rules: never put machine-only detail in the visible text; never duplicate in the index a rationale that already exists in a commit body or thread comment — link to it instead. Update the index after each push and review round.
- **Commit messages**: the subject line states what changed, using the prefix convention above. The body states why, for that commit: motivation, alternatives rejected, and evidence (tests run, verification). AI-authored commits must include this body. Do not write a body that merely restates the diff.
- **Commit signatures**: When AI generates commits, add a `Co-authored-by:` trailer naming the actual model or tool. Prefer the exact model name when known; use the tool name only when the model is not disclosed. Use the provider's official no-reply address:
  - `Co-authored-by: deepseek-v4-pro <noreply@deepseek.com>`
  - `Co-authored-by: gpt-5.6-terra <noreply@openai.com>`
  - `Co-authored-by: claude-opus-4-20250514 <noreply@anthropic.com>`
  - `Co-authored-by: GitHub Copilot <noreply@github.com>`
  - `Co-authored-by: Gemini 3.1 Pro <noreply@google.com>`
- **Why a fixed list**: models cannot reliably report their own name or version (they frequently hallucinate or are unaware of their exact identifier), so a canonical list lets an agent choose the closest matching identity deterministically; when the model isn't listed, use the tool name.
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

1. **Design stubs** to cause type checker errors for invalid usage. Return `Never` only when a direct error cannot be expressed.
2. **In tests**, protect invalid operations with `if TYPE_CHECKING_INVALID_USAGE:` instead of `with pytest.raises(...)`.
3. Add the canonical multi-checker ignore comment sequence.

**Example 1: Standard invalid operations** (tested in `tests/scalars/timedelta/test_arithmetic.py`):

```python
a = pd.Timedelta("1 day")
b = True
if TYPE_CHECKING_INVALID_USAGE:
    _0 = a * b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
```

**Example 2: Asserting a `Never` return from arithmetic**
Arithmetic operators that return `Never` do not stop type checkers from checking the rest of the scope, so assert the return type directly without an uncalled-function wrapper (tested in `tests/indexes/test_floordiv.py`):

```python
from tests import TYPE_CHECKING_INVALID_USAGE
from typing import Never, assert_type

import numpy as np
import pandas as pd

left_i = pd.MultiIndex.from_arrays([[1, 2, 3]]).levels[0]  # pd.Index[int]
c = np.array([1.1j, 2.2j, 4.1j], np.complex128)
if TYPE_CHECKING_INVALID_USAGE:
    assert_type(left_i // c, Never)
```

**Why:** The goal is to catch errors at **type-check time**, not runtime. The `TYPE_CHECKING_INVALID_USAGE` guard (which is `False` at runtime) prevents runtime execution while `assert_type(expr, Never)` verifies the stub really returns `Never` — no ignore comments needed.

**Example 3: Guarding `Never`-returning calls with an uncalled function**
Some `Never`-returning calls make everything after them unreachable, so type checkers stop checking the rest of the block. Confine such an assertion to an uncalled function (tested in `tests/indexes/test_indexes.py`):

```python
def test_multiindex_from_product_forbid_strings() -> None:
    """Test that passing strings directly to `MultiIndex.from_product` is forbidden."""
    if TYPE_CHECKING_INVALID_USAGE:

        def _0() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(pd.MultiIndex.from_product(["12", "34"]), Never)
```

`# pyright: ignore[reportUnusedFunction]` silences pyright's unused-function diagnostic. When in doubt, the wrapper is always safe.

**Note on `ty`**: `ty` is being gradually integrated into `pandas-stubs`, fully parallel with the other three type checkers. It is currently in beta; we actively report bugs to `astral-sh/ty` (and similarly to `mypy`, `pyright`, and `pyrefly`).

### Do NOT use pytest.raises for type checking

**Incorrect pattern:**

```python
with pytest.raises(TypeError):
    s1 + s2  # adding two timestamps
```

**Correct pattern:**

```python
import pandas as pd

s1 = pd.Series([pd.Timestamp("2000-01-01")])
s2 = pd.Series([pd.Timestamp("2000-01-02")])
if TYPE_CHECKING_INVALID_USAGE:
    _0 = s1 + s2  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
```

### Avoiding ruff useless-comparison warnings

When testing operations, assign the result to a dummy variable (e.g., `_0`, `_1`, etc.) to avoid [ruff's useless-comparison rule](https://docs.astral.sh/ruff/rules/useless-comparison/):

```python
import datetime as dt

import pandas as pd

date_obj = dt.date(2023, 1, 1)
if TYPE_CHECKING_INVALID_USAGE:
    _0 = date_obj > pd.NaT  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
```

This applies to any expression that would trigger warnings about unused results (comparisons, arithmetic operations, etc.).

### Version-conditional runtime warnings

Use the `pytest_warns_bounded` / `pytest_warns_conditioned` helpers from `tests/__init__.py` when a runtime warning only exists in a certain pandas (or Python) version range, so tests pass on all supported versions (pandas-dev/pandas-stubs#1927, pandas-dev/pandas-stubs#1802):

```python
with pytest_warns_bounded(UserWarning, match="foo", lower="1.2.99", upper="1.5.99"):
    ...
```

See `docs/philosophy.md` sections "Testing the Type Stubs" and "Narrow vs. Wide Arguments" for full details.

## Validation After Editing

**REQUIRED:** After editing stubs or tests, run the following command to validate your changes:

```bash
poetry run poe test_all
```

All checks must pass before submitting changes. These commands verify:

- Type stubs are correctly annotated (`mypy`, `pyright`, `pyrefly`, `ty`)
- Tests execute successfully at runtime (`pytest`)
