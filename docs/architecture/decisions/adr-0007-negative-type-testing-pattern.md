---
status: accepted
date: 2022-07-08
deciders: [twoertwein, Dr-Irv, cmp0xff, loicdiridollou]
consulted: [pandas-stubs contributors]
informed: [pandas-stubs contributors]
---

# ADR-0007: Negative Type Testing Pattern with TYPE_CHECKING_INVALID_USAGE

## Context and Problem Statement

Testing that valid code passes type checking is only half the requirement. A robust type stub suite must also verify **negative cases**: ensuring that invalid operations (e.g. adding two `Timestamp` series or passing wrong arguments) are **rejected by static type checkers**.

Traditional runtime error testing uses:
```python
with pytest.raises(TypeError):
    s1 + s2
```
However, `pytest.raises` only verifies runtime exceptions. If stubs are too wide (annotating `s1 + s2` as `Any`), `pytest.raises` still passes at runtime while the static type checker silently accepts the invalid code. Furthermore, type checkers often consider code inside `with pytest.raises` as valid unless an explicit ignore comment is asserted.

## Decision Drivers

- **Static Error Verification**: Verify that static type checkers report errors on invalid operations.
- **Prevent Runtime Test Crashes**: Prevent invalid expressions from executing and failing during `pytest` runs.
- **Unused Ignore Detection**: Ensure that if a stub is accidentally widened, the unused `# type: ignore` comment triggers an immediate CI failure.
- **Linter Hygiene**: Prevent `ruff` useless-comparison and unused expression warnings on test lines.

## Considered Options

1. **`with pytest.raises(...)`**:
   - *Pros*: Standard pytest idiom.
   - *Cons*: Cannot test static rejection; permits overly wide annotations.
2. **`if TYPE_CHECKING_INVALID_USAGE:` Block with Inline Ignores** *(Chosen)*:
   - *Pros*: `TYPE_CHECKING_INVALID_USAGE` is `False` at runtime, skipping runtime execution, while static type checkers analyze the block and require `# type: ignore` comments to pass.
   - *Cons*: Requires dummy variable assignment (`_0 = ...`) to satisfy ruff linters.

## Decision Outcome

Negative typing tests MUST use the `if TYPE_CHECKING_INVALID_USAGE:` pattern.

### Standard Template

```python
from pandas._typing import TYPE_CHECKING_INVALID_USAGE

i1 = pd.Interval(pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-03"), closed="both")

if TYPE_CHECKING_INVALID_USAGE:
    # Adding a Timestamp to an Interval is invalid:
    _0 = i1 + pd.Timestamp("2000-03-03")  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
```

### Rule on Dummy Variables
Assign expressions to dummy variables (`_0`, `_1`, etc.) to prevent `ruff` rule `B015` / useless-comparison warnings (e.g. `_0 = a > b`).

## Consequences

- **Positive**: Type checkers are configured with `warn_unused_ignores = true`. If a stub change makes an invalid operation valid, CI fails immediately.
- **Positive**: Runtime test suite runs cleanly and quickly without exception handling overhead.
- **Negative / Neutral**: Contributors must follow this pattern instead of using `pytest.raises`.

## Historical References & Provenance

- **Primary Pull Requests**:
  - pandas-dev/pandas-stubs#8: Standardize assert_type and static error testing (pandas-dev/pandas-stubs@51aeba53ceb106492a167d87dd117c2922c5d147)
  - pandas-dev/pandas-stubs#114: Assert types at runtime vs static analysis (pandas-dev/pandas-stubs@2fd9697fe7f75e54845c0926f22a6c2df6d9f219)
  - pandas-dev/pandas-stubs#1877: Add Python versions to type checkers (pandas-dev/pandas-stubs@e86cf5ff34f30504a609f0b688301a6c40729709)
  - pandas-dev/pandas-stubs#1921: Style typing ignores across tests (pandas-dev/pandas-stubs@cebd95490ba9c7b855051e803193f78602371fb5)
