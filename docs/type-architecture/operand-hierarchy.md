# Operand hierarchy

The operand hierarchy is a structural convention for binary operator signatures:
a class normally accepts operands at its own tier or below in its annotated `other`
parameter. Counted allowances record signatures that intentionally depart from it.
This check validates signature structure, not Python dispatch, runtime support,
return types, or overload correctness. Type-checker and runtime tests remain necessary.

## Tiers and scanned classes

| Tier | Classes |
| --- | --- |
| 0 | `Timedelta`, `Timestamp`, `Period`, `Interval`, `NAType` |
| 1 | `ExtensionArray` |
| 2 | `Index`, `MultiIndex`, and classes discovered through Index base chains |
| 3 | `Series` |
| 4 | `DataFrame` |

The tables in `scripts/operand_hierarchy/model.py` define the registered classes,
canonical operand names and required stub files. Base classes and aliases are discovered
throughout the stub tree, but checked class declarations must occur in registered files.
A missing required file or registered/discovered class is an error. Moving a class to a new file
requires updating that registry. Other scalar or array families are not discovered.

The scanner checks these methods, including each overload declaration:

- Arithmetic: `__add__`, `__sub__`, `__mul__`, `__matmul__`, `__truediv__`,
  `__floordiv__`, `__mod__`, `__divmod__`, `__pow__`.
- Bitwise: `__lshift__`, `__rshift__`, `__and__`, `__or__`, `__xor__`.
- Comparisons: `__lt__`, `__le__`, `__eq__`, `__ne__`, `__gt__`, `__ge__`.

Reflected, in-place and other methods are outside the scan. In particular, `__rshift__`
is a forward operator and is scanned; `__radd__` is reflected and is not.
Both branches of class-level `if` statements are scanned recursively without evaluating
conditions. Nested functions and nested classes are not entered when scanning methods.
Inherited methods are not resolved. Module-level conditional class declarations are also
collected from both branches.

## Supported syntax

Each scanned ordinary `def` method must have an annotated parameter named `other`, including positional,
keyword-only and variadic spellings. Names, qualified names, unions and generic arguments
are searched syntactically. A qualified name is matched by its terminal name:
`pd.DataFrame` is treated as `DataFrame`.

```python
class Series:
    def __add__(self, other: int | Index | Series, /) -> Series: ...  # accepted
    def __sub__(self, other: DataFrame, /) -> Series: ...  # rejected without allowance
    def __mul__(self, right: int, /) -> Series: ...  # rejected: no annotated other
```

Aliases explicitly annotated with `TypeAlias` (including `typing.TypeAlias`) are expanded
transitively. All definitions of a name are retained, including duplicates across files
or version branches. Alias cycles terminate. Names are resolved globally by terminal name,
not by import or lexical scope, so unrelated aliases with the same name can contribute
references. Base discovery also uses terminal names, including subscripted bases.

```python
Higher: TypeAlias = pd.DataFrame
class Series:
    def __add__(self, other: Higher, /) -> Series: ...  # rejected without allowance
```

Aliases whose names start with `ScalarArrayIndex` must not reference `Series` or
`DataFrame`; the more specific `ScalarArrayIndexSeries` prefix permits `Series` but
still forbids `DataFrame`. These alias rules have no exceptions.

```python
ScalarArrayIndexOperand: TypeAlias = int | Index  # accepted
ScalarArrayIndexSeriesOperand: TypeAlias = int | Series  # accepted
ScalarArrayIndexBad: TypeAlias = Series  # rejected
ScalarArrayIndexSeriesBad: TypeAlias = DataFrame  # rejected
```

Quoted annotations are not parsed, including quoted portions of unions. Implicit aliases,
PEP 695 `type` aliases, TypeVar bounds and constraints, import renaming, inherited methods
and additional class-family discovery are unsupported. Async methods and callable class
attributes are not scanned. The check does not infer semantic equivalence or validate
whether generic nesting represents an actual binary operand. Passing this check is not
proof that a signature is correct.

## Scalar and Index subclass overloads

`FORWARD_DUNDER_EXCEPTIONS` in `scripts/operand_hierarchy/exceptions.py` maps
`(class, operator, canonical higher-tier operand)` to a positive overload count.
There are no wildcard keys. Each declaration counts once for each distinct higher tier
it references. All Index subclasses canonicalize to `Index`, so an overload naming
`Index | MultiIndex | TimedeltaIndex` counts once, even through repeated aliases.
An overload naming `Index | Series` on a scalar counts once under each operand.
Both conditional branches count, even when a type checker selects only one branch.

Observed and recorded counts must match exactly. New combinations, count increases,
count decreases and unused entries all fail with expected-versus-observed diagnostics.
For example, a single `Series.__matmul__(DataFrame)` declaration requires
`("Series", "__matmul__", "DataFrame"): 1`; another declaration requires reviewing the
signature and deliberately updating that count. Reordering declarations or using an
equivalent alias does not change the count.

Allowances include existing scalar/Index subclass overloads and legitimate signatures
such as `Series.__matmul__(DataFrame)`. They are a review boundary, not a promise that
every allowance can eventually disappear. When changing one, explain the signature's
purpose in the PR, test its typing and runtime behavior as appropriate, and update the
count to match. Do not raise a count merely to silence a failure.

## Running the checks

Run `poetry run poe pytest_architecture` for the synthetic cases, documentation link
validation and strict check of the real stub tree. The dedicated Ubuntu architecture
job uses Python 3.11. Local `poe test` and `poe test_all` include it once; ordinary
`poe pytest`, including nightly runs, excludes this test file.

Run `python -S scripts/check_operand_hierarchy.py` for the standalone standard-library
check. It reports the number of combinations and occurrences and exits with status 0
on success or 1 on a contract violation. A missing required file is a contract violation; syntax errors and other read failures
remain visible exceptions.
