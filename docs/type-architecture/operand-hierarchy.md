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

For example, one `Series.__matmul__(DataFrame)` declaration requires
`("Series", "__matmul__", "DataFrame"): 1`.

To change a count, explain the signature's purpose in the PR, test its typing and runtime
behavior as appropriate, then update the count. Never raise a count merely to silence a
failure. Allowances are a review boundary, not a promise that each will disappear.

## Limits

- `other` must be annotated, in positional, keyword-only or variadic form.
- Names match by terminal name (`pd.DataFrame` is `DataFrame`). Unions, qualified names and
  generic arguments are searched syntactically.
- Only `TypeAlias`-annotated aliases are expanded, transitively. Aliases resolve globally by
  terminal name, not by scope, so same-named aliases contribute. Duplicates are kept and
  cycles terminate.
- `ScalarArrayIndex*` aliases must not reference `Series` or `DataFrame`. `ScalarArrayIndexSeries*`
  aliases may reference `Series` but not `DataFrame`. No exceptions.
- Both branches of class-level and module-level `if` statements are scanned without
  evaluating conditions. Nested functions and classes are not entered. Inherited methods
  are not resolved.
- Unsupported: quoted annotations, implicit aliases, PEP 695 `type` aliases, TypeVar bounds,
  import renaming, async methods, callable class attributes and other class families.
- Passing is not proof that a signature is correct.

## Running the checks

- `poetry run poe pytest_architecture` runs the synthetic cases, anchor validation and a
  strict check of the real stubs. CI runs it in a dedicated Ubuntu job on Python 3.11.
  `poe test` and `poe test_all` include it once; ordinary `poe pytest` excludes it.
- `python -S scripts/check_operand_hierarchy.py` is the stdlib-only check. It prints
  combination and occurrence counts. Exit 0 is success, 1 is a contract violation. A missing
  required file is a violation; syntax errors surface as exceptions.
