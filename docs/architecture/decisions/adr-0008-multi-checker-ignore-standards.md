---
status: accepted
date: 2026-08-26
deciders: [cmp0xff, loicdiridollou, MarcoGorelli, Dr-Irv]
consulted: [pandas-stubs contributors]
informed: [pandas-stubs contributors]
---

# ADR-0008: Multi-Checker Ignore Comment Standardization and Ordering

## Context and Problem Statement

Supporting four distinct static type checkers (`mypy`, `pyright`, `pyrefly`, and `ty`) means dealing with different diagnostic error codes, bug workarounds, and comment syntax:
1. `mypy` uses `# type: ignore[<error-code>]`
2. `pyright` uses `# pyright: ignore[<rule-name>]`
3. `pyrefly` uses `# pyrefly: ignore[<error-code>]`
4. `ty` uses `# ty: ignore[<error-code>]`

Without a strict convention, comment annotations in stubs and tests become disorderly, hard to read, and difficult for automated tools to parse or clean up.

## Decision Drivers

- **Predictable Ordering**: Establish a single canonical order for multiple ignore comments on a single line.
- **Specific Error Codes**: Prohibit blanket `# type: ignore` without specific error codes.
- **Upstream Bug Attribution**: Reference upstream type checker issue numbers when an ignore comment is added as a temporary workaround.

## Considered Options

1. **Ad-hoc Comment Placement**:
   - *Pros*: No contributor friction.
   - *Cons*: Messy diffs, duplicate ignores, unmaintainable codebase.
2. **Canonical Multi-Checker Ignore Order** *(Chosen)*:
   - *Pros*: Clear, clean, and easily validated by pre-commit linters and style checks (PR pandas-dev/pandas-stubs#1921).
   - *Cons*: Requires contributors to adhere to exact ordering.

## Decision Outcome

All multi-checker ignore comments must follow this exact canonical sequence:

```python
# Canonical Sequence:
# type: ignore[<code-1>] # pyright: ignore[<code-2>] # pyrefly: ignore[<code-3>] # ty: ignore[<code-4>]
```

### Formatting Rules
1. **Always specify error codes**: Never use `# type: ignore` or `# pyright: ignore` without brackets.
2. **Order**: `mypy` first $\rightarrow$ `pyright` second $\rightarrow$ `pyrefly` third $\rightarrow$ `ty` fourth.
3. **Upstream Bug References**: When an ignore is required due to an upstream checker defect, include a comment referencing the upstream issue:
   ```python
   # TODO: python/mypy#15900 we did use explicit override but mypy does not see it
   # TODO: astral-sh/ty#4150 python/mypy#21795
   ```

## Consequences

- **Positive**: Uniform comment formatting across all test and stub files.
- **Positive**: Facilitates automated audit scripts and clean removal when upstream checker bugs are fixed.
- **Negative / Neutral**: Pre-commit CI checks enforce style compliance.

## Historical References & Provenance

- **Primary Pull Requests**:
  - pandas-dev/pandas-stubs#1878: Relocate to_offset and pyrefly-inspired changes (pandas-dev/pandas-stubs@b0e70149eacc63fcf053c7838cca0509f13d6008)
  - pandas-dev/pandas-stubs#1879: Disable pyright reportUnknownLambdaType and pyrefly implicity-any-lambda (pandas-dev/pandas-stubs@eea8472261b5e5d029b7088173374f9264cac258)
  - pandas-dev/pandas-stubs#1895: Disable pyright reportUnknownArgumentType (pandas-dev/pandas-stubs@1b0c5374ac7dd7ccd1f61a943c98f44ba63dd28a)
  - pandas-dev/pandas-stubs#1921: Style typing ignores across test files (pandas-dev/pandas-stubs@cebd95490ba9c7b855051e803193f78602371fb5)
