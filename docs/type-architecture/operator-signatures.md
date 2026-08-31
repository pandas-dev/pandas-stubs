# Operator signatures

The operator declarations live in the relevant `.pyi` files; this page explains common
patterns for editing them. Confirm an intended change with focused type and runtime
tests rather than treating an example here as a complete overload recipe.

## Operand aliases

`pandas-stubs/core/base.pyi` defines `ScalarArrayIndex*` aliases for operands shared by
`Index` operations. Their `ScalarArrayIndexSeries*` counterparts add compatible
`Series` forms for `Series` operations. The numeric, complex, and timedelta families
use this pattern where it keeps overloads readable.

The names describe cumulative operand scopes: `Scalar` contains scalar values,
`ScalarArray` conceptually adds array-like values, `ScalarArrayIndex` adds `Index`, and
`ScalarArrayIndexSeries` adds `Series`. `ScalarArray` is vocabulary for the conceptual
level, not a current alias family in `pandas-stubs/core/base.pyi`.

Not every operator uses one of these aliases. Several additions and temporal operations
spell their operand unions inline, and named methods may have a different set of
parameters from the matching dunder. Check the declaration you are editing instead of
assuming alias coverage.

## Dunder parameters

Binary dunders use a positional-only `other` parameter:

```python
def __add__(self, other: ScalarArrayIndexReal, /) -> Self: ...
```

The `/` records the dunder calling convention in the stubs. Named methods such as
`add`, `sub`, and `mul` can additionally expose keyword parameters such as `axis`,
`level`, or `fill_value` when their own declarations support them.

## Protocol overloads

Protocol-based overloads and constrained type variables are the preferred direction when
an operand family has uniform return behavior. They avoid repeating the same combinations
for custom operand types. Keep explicit overloads where operand validity or return types
differ, and preserve specialized temporal overloads when changing those families: a
broader protocol can otherwise capture a case that needs a distinct result.

This distinction matters for bool-sensitive operations. For example, subtraction and true
division cannot blindly generalize from supported scalar protocols when `bool` minus
`bool` or `bool` divided by `bool` is invalid. The structural hierarchy checker cannot
infer those semantic exceptions, so focused type tests remain necessary.

## Subclass overrides

Specialized `Index` subclasses can narrow a broad `Index` return type. Their dunder
overrides use the repository's ordered multi-checker ignore comments when a checker
reports an intentional override incompatibility. Keep the standard order documented in
[`docs/philosophy.md`](../philosophy.md#using-ignore-comments): mypy, pyright, pyrefly,
then ty.
