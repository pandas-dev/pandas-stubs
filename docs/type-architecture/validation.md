# Validation

General type-test patterns live in
[`docs/philosophy.md`](../philosophy.md#testing-the-type-stubs); this page covers the
structural checker for the [operand hierarchy](operand-hierarchy.md).

## Operand-hierarchy checker

Run the structural checker with:

```console
poetry run python scripts/check_operand_hierarchy.py
```

The script needs no dependencies beyond the standard library, but `poetry run` is the
portable way to name a Python interpreter: the bare command differs by platform
(`python` on Windows, `python3` on macOS), and CI supplies its own.

It parses every `.pyi` file to resolve `TypeAlias` references and inspects binary
dunders declared directly on `Index`, `MultiIndex`, and `Series`. Which dunders it
scans, and how it reads their operand, is documented in
[Operator signatures](operator-signatures.md#dunder-parameters). Its unit tests create
temporary stub trees for positive, direct-reference, transitive-alias,
bitwise/comparison, and matrix-exception cases.

Run it when a change touches either of the two things it reads:

- a **forward binary dunder** on `Index`, `MultiIndex`, or `Series` — adding,
  removing, or editing the signature, including renaming its `other` operand; or
- a **`ScalarArrayIndex*` alias** in `pandas-stubs/core/base.pyi`, or any `TypeAlias`
  those aliases reference transitively.

Nothing else in the tree can change its result, so other stub work does not need it.

The checker proves only that its alias and operand restrictions hold. It does not check
runtime dispatch, return annotations, reflected dunders, overload selection, or all
pandas operand relationships. A passing result is therefore a focused regression
signal, not a general proof of type-architecture correctness.

## Verification

Run the focused checker tests while iterating:

```console
poetry run pytest tests/test_check_operand_hierarchy.py
```

Before submitting a change that touches stubs or tests, run the full project suite:

```console
poetry run poe test_all
```

See [`docs/tests.md`](../tests.md) for the full set of `poe` tasks and what each one
covers.
