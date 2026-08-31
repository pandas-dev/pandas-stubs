# Operator signatures

The operator declarations live in the relevant `.pyi` files; this page explains common
patterns for editing them. Confirm an intended change with focused type and runtime
tests rather than treating an example here as a complete overload recipe.

## Operand aliases

`pandas-stubs/core/base.pyi` defines `ScalarArrayIndex*` aliases for operands shared by
`Index` operations. Their `ScalarArrayIndexSeries*` counterparts add compatible
`Series` forms for `Series` operations. The numeric, complex, and timedelta families
use this pattern where it keeps overloads readable.

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

Some overload families use arithmetic protocols and constrained type variables to avoid
repeating the same combinations for custom operand types. Preserve the existing type
variable bounds and specialized temporal overloads when changing those families: a
broader protocol can otherwise capture a case that needs a distinct return type.

## Subclass overrides

Specialized `Index` subclasses can narrow a broad `Index` return type. Their dunder
overrides use the repository's ordered multi-checker ignore comments when a checker
reports an intentional override incompatibility. Keep the standard order documented in
[`docs/philosophy.md`](../philosophy.md#using-ignore-comments): mypy, pyright, pyrefly,
then ty.
