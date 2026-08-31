# Container hierarchy

The stubs model a useful direction for binary operators: a container should not claim a
higher-tier container as a normal forward operand when that higher tier owns the relevant
result shape. This is a signature-design constraint, not a description of Python's full
runtime method-resolution rules.

## Tiers

| Tier | Examples | Normal operand scope |
| --- | --- | --- |
| 0 | Scalars such as `int`, `Timestamp`, and `Timedelta` | Scalar values |
| 1 | `Index`, extension arrays, and NumPy arrays | Scalars and one-dimensional values |
| 2 | `Series` | Scalars, one-dimensional values, and `Series` |
| 3 | `DataFrame` | Scalars, one-dimensional values, and `DataFrame` |

The tiers describe the ownership convention used in the current operator annotations;
they do not classify every pandas object or every method.

## Cross-tier lookup examples

When reviewing a cross-tier expression, start with the container whose result shape is
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
3. Every forward binary dunder declared directly on `Index` or `Series` with an `other`
   parameter must not directly or transitively reference that class's higher tier:
   `Series` or `DataFrame` for `Index`, and `DataFrame` for `Series`.

This includes arithmetic, bitwise, comparison, and matrix-multiplication dunders. The
checker deliberately excludes reflected dunders such as `__radd__`; those signatures
need their own review and focused tests.

## Matrix multiplication

`Series.__matmul__(DataFrame)` is the sole current exception. It is represented in
`FORWARD_DUNDER_EXCEPTIONS` in the checker rather than omitted from the scan: the stubs
return a `Series` for this matrix-multiplication overload.

Adding an exception requires all three of the following changes in the same review:

1. Add a rationale and documentation entry here.
2. Add the explicit registry entry in `scripts/check_container_hierarchy.py`.
3. Update the exact-registry and declared-exception tests in
   `tests/test_check_container_hierarchy.py`.
