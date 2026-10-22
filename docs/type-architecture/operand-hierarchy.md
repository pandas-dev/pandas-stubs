# Operand hierarchy

## The rule

A class accepts operands at its own tier or below in the annotated `other` parameter of a
forward binary dunder. Higher-tier operands are rejected unless a counted allowance records
them. The check validates signature structure only, not Python dispatch, runtime behavior,
return types or overload correctness; type-checker and runtime tests remain necessary.

## Tiers and scanned methods

| Tier | Classes |
| --- | --- |
| 0 | `Timedelta`, `Timestamp`, `Period`, `Interval`, `NAType` |
| 1 | `ExtensionArray` |
| 2 | `Index`, `MultiIndex`, and Index subclasses found through base chains |
| 3 | `Series` |
| 4 | `DataFrame` |

Scanned: arithmetic (`__add__` through `__pow__`, including `__matmul__` and `__divmod__`),
bitwise (`__lshift__`, `__rshift__`, `__and__`, `__or__`, `__xor__`) and comparisons
(`__lt__`, `__le__`, `__eq__`, `__ne__`, `__gt__`, `__ge__`). Reflected and in-place methods
are not scanned, so `__radd__` is outside the check while `__rshift__` is scanned.

`CLASS_TIERS` in `scripts/operand_hierarchy/model.py` tiers every registered class; Index
subclasses are discovered through their bases, not listed. `REQUIRED_STUB_FILES` lists every
file that may declare a checked class, so moving a class to a new file requires updating it.

## Cross-tier lookup examples

When reviewing a cross-tier expression, start with the operand type whose result shape is
being represented, then inspect its forward and reflected overloads and a focused test.

| Expression to review | Relevant stub surface |
| --- | --- |
| Scalar and `Index` | `Index` operator overloads, plus the scalar's own forward overloads |
| Scalar and `Series` | `Series` operator overloads, plus the scalar's own forward overloads |
| `Index` and `Series` | `Series` operator overloads, plus `IntervalIndex`'s comparison overloads |
| `Series` and `DataFrame` | `DataFrame` operator overloads |

The scalar rows are not one-sided: a scalar such as `Timedelta` declares its own forward
overloads for the array-like operands it broadcasts over, and those overloads are part of
the surface to review. [Scalar and Index subclass overloads](#scalar-and-index-subclass-overloads)
states why the checker permits them and where the temporary list that records them lives.

For example, `IntervalIndex`'s comparison overloads against `Series` return `Series[bool]`,
so the result shape belongs to `Series`.

These are lookup examples for the stubs, not a claim about the exact runtime dispatch
sequence for every operand pair.

## A rejected signature, traced

```python
class Series:
    def __sub__(self, other: DataFrame, /) -> Series: ...
```

1. `main()` calls `check_operand_hierarchy`, which calls `StubTree.load`. That parses every
   stub, collecting aliases and bases everywhere and classes from the required files only.
2. `check_operand_hierarchy` calls `check_class("Series", observed)`. `DataFrame` (tier 4)
   is above `Series` (tier 3), so it is in the higher-tier map.
3. `_other_parameter` returns `other`, and `StubTree.references` matches `DataFrame` by
   terminal name. The counter records `("Series", "__sub__", "DataFrame")` once.
4. With no allowance, expected is 0 and observed is 1. `check_operand_hierarchy` prints to
   stderr and returns False; `main()` exits 1.

```text
ERROR: Series.__sub__ higher-tier operand DataFrame: expected 0 overload(s), observed 1; review the signature or update scripts/operand_hierarchy/exceptions.py.
```

`test_guide_traced_example` runs this example, so the trace above cannot drift from the checker.

## Scalar and Index subclass overloads

`FORWARD_DUNDER_EXCEPTIONS` in `scripts/operand_hierarchy/exceptions.py` maps
`(class, operator, canonical higher-tier operand)` to a positive count. There are no wildcards.

- Each declaration counts once per distinct higher tier it references.
- Index subclasses canonicalize to `Index`, so `Index | MultiIndex | TimedeltaIndex` counts
  once, even through repeated aliases. `Index | Series` on a scalar counts once per operand.
- Both branches of a conditional count, even when a type checker selects only one.
- Reordering declarations or using an equivalent alias does not change a count.
- Observed and recorded counts must match exactly. New combinations, increases, decreases
  and unused entries all fail.

The goal is to shrink this registry towards empty; pandas-dev/pandas-stubs#2023 tracks the
steps. A PR may remove a key or lower a count. Adding a key or raising a count needs the
signature's purpose explained in the PR and its typing and runtime behavior tested as
appropriate. Never raise a count merely to silence a failure.

### Why the allowances exist

The allowances are not a claim that the lower tier owns dispatch. At runtime the tier-0
scalars and `IntervalIndex` return `NotImplemented` for a `Series` or `Index` operand, and
the higher tier's reflected dunder produces the result. Each key therefore records a
cross-tier declaration the stubs currently carry, not a runtime entry point.

The measured four-checker fallback is recorded in pandas-dev/pandas-stubs#2023. Deleting
`Timedelta.__lt__(Series[Timedelta])` and `Timedelta.__floordiv__(Series[Timedelta])`
keeps `Series[bool]` and `Series[int]` in mypy, pyright, pyrefly and ty through the
reflected dunder; that issue records the measurement and the per-class steps.

`__eq__` and `__ne__` are the exception. The temporal and interval classes pair their
higher-tier overload with an `object` catch-all; once the specific overload is gone, the
catch-all absorbs the operand and the reflected method never runs. This was measured for
`Timedelta`, whose catch-all returns `Literal[False]` in all four checkers, so the lower
tier keeps declaring the result for these two names. Check each class before assuming the
same fallback: `NAType` declares no `object` overload.

This rationale is scoped to the operand pairs measured in pandas-dev/pandas-stubs#2023. It
is not a general claim that the higher tier's reflected dunder suffices for every pair.

## Matrix multiplication

`Series.__matmul__(DataFrame)` returns a `Series`, and pandas supports the expression at
runtime; `tests/series/test_series.py::test_types_dot` exercises the forward operation.
The result shape belongs to `DataFrame`, so the corresponding design is a
`DataFrame.__rmatmul__(Series)` overload: it lets the `Series` side keep its operand scope
while the tier that owns the shape declares the result. pandas-dev/pandas-stubs#2023
tracks the step.

## Limits

- `other` must be annotated, in positional, keyword-only or variadic form.
- Names match by terminal name (`pd.DataFrame` is `DataFrame`). Unions, qualified names and
  generic arguments are searched syntactically.
- Only `TypeAlias`-annotated aliases are expanded, transitively. Aliases resolve globally by
  terminal name, not by scope, so same-named aliases contribute. Duplicates are kept and
  cycles terminate.
- Tier 1 is registered through the `ExtensionArray` ABC, so an operand is read at tier 1
  when the annotation spells `ExtensionArray`, or a `TypeAlias` chain such as `ArrayLike`
  that reaches it. A concrete array-like such as `Categorical` is not read
  at tier 1: the checker inherits through base classes for `Index` subclasses only, so an
  annotation that names `Categorical` without an alias chain to `ExtensionArray` is an
  unknown spelling.
- `ScalarArrayIndex*` aliases must not reference `Series` or `DataFrame`. `ScalarArrayIndexSeries*`
  aliases may reference `Series` but not `DataFrame`. No exceptions.
- Both branches of class-level and module-level `if` statements are scanned without
  evaluating conditions. Nested functions and classes are not entered. Inherited methods
  are not resolved.
- Unsupported: quoted annotations, implicit aliases, PEP 695 `type` aliases, TypeVar bounds,
  import renaming, async methods, callable class attributes and other class families.
- Passing is not proof that a signature is correct.

See [Validation](validation.md) for when a change needs the check.

## Running the checks

- `poetry run poe pytest_architecture` runs the synthetic cases, anchor validation and a
  strict check of the real stubs. CI runs it in a dedicated Ubuntu job on Python 3.11.
  `poe test` and `poe test_all` include it once; ordinary `poe pytest` excludes it.
- `python -S scripts/check_operand_hierarchy.py` is the stdlib-only check. It prints
  combination and occurrence counts. Exit 0 is success, 1 is a contract violation. A missing
  required file is a violation; syntax errors surface as exceptions.
