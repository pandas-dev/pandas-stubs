# Validation

General type-test patterns live in
[`docs/philosophy.md`](../philosophy.md#testing-the-type-stubs); this page covers the
structural checker for the [operand hierarchy](operand-hierarchy.md).

## Operand-hierarchy checker

Run the structural checker with:

```console
python -S scripts/check_operand_hierarchy.py
```

The script needs no dependencies beyond the standard library; `-S` skips `site`
initialization, so it runs on a bare interpreter.

It parses every `.pyi` file to resolve `TypeAlias` references and inspects forward binary
dunders declared directly on the scanned classes: the tier-0 scalars, the tier-1
array-like `ExtensionArray`, `Index` with every `Index` subclass it discovers through base
chains, and `Series` and `DataFrame`. Which dunders it scans, and how it reads their
operand, is documented in
[Operator signatures](operator-signatures.md#dunder-parameters). Its unit tests build
synthetic stub trees for the alias, operand, canonicalization, conditional, count and
documented-example cases, and one test scans this repository's shipped registry.

Run it when a change touches one of the things it reads:

- a **forward binary dunder** on a scanned class — the tier-0 scalars, `ExtensionArray`,
  `Index` and its subclasses, `Series`, or `DataFrame` — adding, removing, or editing the
  signature, including renaming its `other` operand;
- **any `TypeAlias`** reachable, directly or transitively, from such an `other`
  annotation, including a `ScalarArrayIndex*` alias in
  `pandas-stubs/_stubs_only/__init__.pyi`;
- a **new `Index` subclass or base chain**: subclasses are discovered through base
  declarations in every stub file, and the class declaration must be in a
  `REQUIRED_STUB_FILES` entry or the scan fails with "could not find class"; or
- the **model tables or required-file list** when the checker itself changes.

A change outside these surfaces does not need the check.

The checker proves only that its alias and operand restrictions hold, and — whenever it
validates the shipped registry, both as a script and through
`test_repository_matches_allowances` — that every exception key is exercised by the tree it
scans. A passing run prints the exceptions file it consulted and the number of
combinations and occurrences observed. It does not check runtime dispatch, return
annotations, reflected dunders, overload selection, or all pandas operand relationships. A
passing result is therefore a focused regression signal, not a general proof of
type-architecture correctness.

## Verification

Run the checker's focused tests with:

```console
poetry run poe pytest_architecture
```

Before submitting a change that touches stubs or tests, run the full project suite; see
[`docs/tests.md`](../tests.md) for the full set of `poe` tasks and what each one covers.
