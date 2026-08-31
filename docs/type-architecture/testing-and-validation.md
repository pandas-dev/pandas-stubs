# Testing and validation

Type changes need focused tests that exercise the intended static result and, where
appropriate, the corresponding runtime behavior. See
[`docs/philosophy.md`](../philosophy.md#testing-the-type-stubs) for the project-wide
testing approach.

## Type-test patterns

For a valid expression, pair `assert_type` with the repository's `check` helper when a
runtime value also needs confirmation:

```python
check(assert_type(left + right, "pd.Series[int]"), pd.Series, int)
```

For invalid usage, guard the expression with `TYPE_CHECKING_INVALID_USAGE` and use the
canonical four-checker ignore sequence. When an overload deliberately returns `Never`,
assert that return type directly; use an uncalled function if the checkers would treat
the rest of the block as unreachable.

## Hierarchy checker

Run the structural checker with:

```console
python3 scripts/check_container_hierarchy.py
```

It parses every `.pyi` file to resolve `TypeAlias` references and inspects binary
dunders declared directly on `Index` and `Series`, including whether each `other`
parameter is positional-only. Its unit tests create temporary stub trees for positive,
direct-reference, transitive-alias, positional-parameter, bitwise/comparison, and
matrix-exception cases.

The checker proves only that its alias and operand restrictions hold. It does not check
runtime dispatch, return annotations, reflected dunders, overload selection, or all
pandas container relationships. A passing result is therefore a focused regression
signal, not a general proof of type-architecture correctness.

## Verification

Run the focused checker tests while iterating:

```console
poetry run pytest tests/test_check_container_hierarchy.py
```

Before submitting a change that touches stubs or tests, run the full project suite:

```console
poetry run poe test_all
```
