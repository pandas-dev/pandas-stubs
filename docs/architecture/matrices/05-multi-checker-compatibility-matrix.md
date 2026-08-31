# Multi-Type-Checker Compatibility Matrix & Ignore Standards

## 1. Supported Type Checkers Matrix

pandas-stubs continuously tests all stub signatures and test suites across four primary type checker engines:

| Type Checker | Engine Type | Vendor / Maintainer | Primary Role | Strict Flag Configuration | Key Provenance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mypy** | Python AST Interpreter | Python Core / Dropbox | Reference standard checker | `--strict`, `--warn-unused-ignores` | pandas-dev/pandas-stubs#10, pandas-dev/pandas-stubs#1836 |
| **Pyright** | TypeScript / Node.js AST | Microsoft | VS Code / Pylance Language Server | `typeCheckingMode: "strict"` | pandas-dev/pandas-stubs#10, pandas-dev/pandas-stubs#59, pandas-dev/pandas-stubs#1895 |
| **Pyrefly** | Rust High-Performance Engine | Meta | Fast type-coverage auditing | `pyrefly_dist` CI harness | pandas-dev/pandas-stubs#1875, pandas-dev/pandas-stubs#1910 |
| **Ty** | Rust Fast Type Checker | Astral | Fast next-generation type analysis | `ty check --strict` | pandas-dev/pandas-stubs#1836, pandas-dev/pandas-stubs#1845, pandas-dev/pandas-stubs#1867 |

---

## 2. Canonical Ignore Comment Sequence Standard

When a line requires multi-checker suppression due to divergent diagnostic rules or upstream compiler defects, ignores must be formatted in this exact canonical order (citing PR pandas-dev/pandas-stubs#1921):

```python
# Canonical Sequence:
# type: ignore[<mypy-code>] # pyright: ignore[<pyright-rule>] # pyrefly: ignore[<pyrefly-code>] # ty: ignore[<ty-code>]
```

### Common Diagnostic Error Codes
- **Mypy**: `[operator]`, `[assignment]`, `[arg-type]`, `[return-value]`, `[override]`, `[type-var]`, `[misc]`
- **Pyright**: `[reportGeneralTypeIssues]`, `[reportUnknownMemberType]`, `[reportPrivateUsage]`, `[reportArgumentType]`
- **Pyrefly**: `[unsupported-operation]`, `[incompatible-type]`, `[implicity-any-lambda]`
- **Ty**: `[unsupported-operator]`, `[invalid-argument-type]`, `[unknown-type]`

---

## 3. Negative Testing Pattern: `TYPE_CHECKING_INVALID_USAGE`

Negative static test cases (verifying that invalid code triggers type checker errors without failing during runtime `pytest` execution) must follow this standard idiom:

```python
from pandas._typing import TYPE_CHECKING_INVALID_USAGE

if TYPE_CHECKING_INVALID_USAGE:
    # Invalid: subtracting two booleans
    _0 = s_bool - s_bool  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
```
- **Rule**: Expressions must be assigned to dummy variables (`_0`, `_1`) to satisfy `ruff` rule `B015` (useless-expression checks).
- **Rule**: `warn_unused_ignores = true` ensures that if a stub is accidentally widened, CI fails immediately.
