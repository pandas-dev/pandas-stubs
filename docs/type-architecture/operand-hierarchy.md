# Operand hierarchy

The stubs model a useful direction for binary operators: a type should not claim a
higher-tier type as a normal forward operand when that higher tier owns the relevant
result shape. This is a signature-design constraint, not a description of Python's full
runtime method-resolution rules. It is about operand scope, not about containment: the
tiers span scalars and array-like values that are not containers in the `typing.Container`
sense.

## Tiers

| Tier | Examples | Normal operand scope |
| --- | --- | --- |
| 0 | Scalars such as `int`, `Timestamp`, and `Timedelta` | Scalar values |
| 1 | Array-like values such as extension arrays and NumPy arrays | Scalars and array-like values |
| 2 | `Index` and `MultiIndex` | Scalars, array-like values, and `Index` |
| 3 | `Series` | Scalars, array-like values, `Index`, and `Series` |
| 4 | `DataFrame` | Scalars, lower-dimensional types, and `DataFrame` |

The tiers describe the ownership convention used in the current operator annotations;
they do not classify every pandas object or every method.

`MultiIndex` is an `Index` subclass in the stubs and declares no forward binary dunders of
its own, so it shares `Index`'s operand scope. Its multidimensional labels are data
semantics, not operator ownership; the checker scans `MultiIndex` directly as a guard
against future higher-tier forward operands.

## Cross-tier lookup examples

When reviewing a cross-tier expression, start with the operand type whose result shape is
being represented, then inspect its forward and reflected overloads and a focused test.
For example:

| Expression to review | Relevant stub surface |
| --- | --- |
| Scalar and `Index` | `Index` operator overloads |
| `Index` and `Series` | `Series` operator overloads |
| `Series` and `DataFrame` | `DataFrame` operator overloads |

These are lookup examples for the stubs, not a claim about the exact runtime dispatch
sequence for every operand pair.

## Structural invariant

The checker enforces three restrictions in the current stubs:

1. A `ScalarArrayIndex*` alias must not directly or transitively reference `Series` or
   `DataFrame`.
2. A `ScalarArrayIndexSeries*` alias must not directly or transitively reference
   `DataFrame`.
3. Every forward binary dunder declared directly on `Index`, `MultiIndex`, or `Series`
   must declare an `other` operand and must not directly or transitively reference that
   class's higher tier: `Series` or `DataFrame` for `Index` and `MultiIndex`, and
   `DataFrame` for `Series`.

This includes arithmetic, bitwise, comparison, and matrix-multiplication dunders. The
checker deliberately excludes reflected dunders such as `__radd__`; those signatures
need their own review and focused tests. See
[Operator signatures](operator-signatures.md#dunder-parameters) for which dunders the
checker scans and how it reads the `other` operand, and
[Reflected dunders](operator-signatures.md#reflected-dunders) for why the excluded ones
are still reviewed.

## Matrix multiplication

`Series.__matmul__(DataFrame)` is the sole current exception. It is represented in
`FORWARD_DUNDER_EXCEPTIONS` in the checker rather than omitted from the scan: the stubs
return a `Series` for this matrix-multiplication overload. Pandas supports the expression
at runtime, and `tests/series/test_series.py::test_types_dot` exercises the forward
operation. Removing the overload would therefore regress valid code unless pandas changes
which operand owns this runtime operation.

Adding an exception requires all three of the following changes in the same review:

1. Add a rationale and documentation entry here.
2. Add the explicit registry entry in `scripts/check_operand_hierarchy.py`.
3. Update the exact-registry and declared-exception tests in
   `tests/test_check_operand_hierarchy.py`.
