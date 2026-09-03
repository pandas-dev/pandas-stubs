---
status: accepted
date: 2026-09-03
deciders: [Dr-Irv, twoertwein, MarcoGorelli, loicdiridollou, cmp0xff]
consulted: [typing-sig]
informed: [pandas-stubs contributors]
---

# ADR-0023: Protocol-Typed `self` Overloads Need Explicit `Never` Exclusions

## Context and Problem Statement

The `Index` arithmetic operator families (`__add__`/`__radd__`, `__sub__`/`__rsub__`,
`__mul__`/`__rmul__`, `__truediv__`/`__rtruediv__`, `__floordiv__`/`__rfloordiv__`)
consolidate their scalar/sequence overloads onto a protocol-typed `self`, e.g.:

```python
@overload
def __sub__(
    self: Supports_ProtoSub[T_contra, S2], other: T_contra | Sequence[T_contra], /
) -> Index[S2]: ...
```

`Supports_ProtoSub` is satisfied structurally via `ElementOpsMixin._proto_sub`, one
`_proto_*` method per operator per direction, declared in
`pandas-stubs/_stubs_only/__init__.pyi:202-394`. **Deliberately omitting an element
type from a `_proto_*` family is how the stubs declare an operation illegal** — for
example `_proto_sub` covers `int`/`float`/`complex`/`Timedelta`/`Timestamp` but not
`bool` (boolean subtraction is prohibited, ADR-0017) and not `str`. Structurally,
`Index[bool]` should therefore fail to satisfy `Supports_ProtoSub[bool, S2]`, and mypy,
pyright, and pyrefly should all reject `Index[bool] - bool`.

python/mypy#20061 ("Issue when `self` is typed as a Protocol") breaks that expectation
for mypy only: mypy matches the Protocol-typed `self` overload anyway, so it silently
**accepts** `Index[bool] - bool` — a false negative. pyright and pyrefly reject it
correctly.

### Why this is very blocking

1. **No ignore comment can fix it.** `docs/philosophy.md` "Using ignore comments"
   prescribes an ignore for "when type checkers have bugs themselves" — but that
   assumes a *false positive*. This is a **false negative**: mypy reports no error to
   suppress. With `warn_unused_ignores = true` (pyproject.toml:293) and
   `reportUnnecessaryTypeIgnoreComment = true` (pyproject.toml:334), adding
   `# type: ignore[operator]` anyway itself errors as `unused-ignore` (verified
   empirically). The canonical multi-checker ignore sequence (`AGENTS.md` "Invalid
   Usage Testing Pattern") also *requires* mypy's ignore first, which is impossible
   here.
2. **It silently erodes the project's core value.** The only visible symptom is an
   *asymmetric* pyright-only ignore on an otherwise-identical test, which is easily
   misread as a pyright quirk rather than as lost mypy coverage.
3. **It taxes the consolidation itself.** Every deliberately-excluded element type
   needs a hand-written `Never` overload, partially defeating the de-duplication the
   protocol idiom was adopted for (ADR-0009) — across the whole
   `__add__`/`__sub__`/`__mul__`/`__truediv__`/`__floordiv__` migration
   (pandas-dev/pandas-stubs#1378, pandas-dev/pandas-stubs#1474).

## Decision Drivers

- Restore three-checker agreement (mypy, pyright, pyrefly) on operations that are
  supposed to be statically illegal.
- Keep the `_proto_*` idiom's "omission means illegal" invariant enforceable, without
  abandoning the protocol consolidation that removed large amounts of duplicated
  overloads.
- Avoid ignore comments where the underlying problem is a checker false negative, not a
  false positive.
- Preserve the existing `assert_type(expr, Never)` negative-testing style (ADR-0007)
  with zero ignore comments.

## Considered Options

1. **Ignore comments on the affected test expressions** — infeasible: mypy emits no
   error to ignore (false negative), and `warn_unused_ignores` /
   `reportUnnecessaryTypeIgnoreComment` turn a speculative ignore into its own error.
2. **Wait for upstream mypy to fix python/mypy#20061** — blocks the entire
   protocol-based operator-family migration indefinitely and leaves `Index[bool] -
   bool` silently accepted by mypy in the meantime.
3. **Add explicit narrower `-> Never` overloads immediately before the
   `Supports_Proto*` overload** — chosen. Relies on ordinary first-match-wins
   top-to-bottom overload resolution, which is unaffected by #20061 because the
   matching overload is no longer Protocol-typed.
4. **Abandon the protocol consolidation and return to one overload per concrete
   element-type combination** — sound under all checkers, but reverses the
   de-duplication ADR-0009 and the #1378/#1474 migration were adopted for, at a scale
   (five operator families) that would be a large regression.

## Decision Outcome

Adopt option 3: for every element type deliberately excluded from a `_proto_*` family,
add an explicit narrower `-> Never` overload **immediately before** the
`Supports_Proto*` overload it would otherwise be shadowed by, e.g.:

```python
# Must precede `Supports_ProtoSub`: mypy matches it for `bool` (python/mypy#20061)
@overload
def __sub__(self: Index[bool], other: bool | SequenceNotStr[bool], /) -> Never: ...
@overload
def __sub__(
    self: Supports_ProtoSub[T_contra, S2], other: T_contra | Sequence[T_contra], /
) -> Index[S2]: ...
```

With the narrower overload listed first, mypy, pyright, and pyrefly all resolve
`Index[bool] - bool` to the `Never` overload, and the test needs **no** ignore
comments — `assert_type(expr, Never)` is sufficient (`AGENTS.md` Example 2). This is
consistent with `AGENTS.md`'s "Invalid Usage Testing Pattern": *"Return `Never` only
when a direct error cannot be expressed"* — for mypy under #20061, a direct
`[operator]` error genuinely cannot be expressed, so `Never` is the correct fallback,
not a workaround of convenience.

This is not a new pattern for `base.pyi`: 18 of the 20 existing operator `Never`
overloads already sit *before* their corresponding protocol overload. The two
exceptions — `__add__` (base.pyi:688) and `__radd__` (base.pyi:742), both guarding
`Index[_str]` against numeric ndarrays — are safe precisely *because* `_proto_add`
**does** carry a `str` overload, so no protocol-matching ambiguity exists there for
mypy to mis-resolve. The invariant this decision makes explicit is: **wherever a
`_proto_*` gap is what makes an operation illegal, the corresponding `Never` overload
must precede the protocol overload**, not just exist somewhere in the file.

### Two empirical gotchas

Both cost real debugging time while implementing the `__sub__`/`__rsub__` guards in
pandas-dev/pandas-stubs#1938:

1. **Use `SequenceNotStr[T]`, not `Sequence[T]`, in the new overload's parameter
   type.** With `Sequence[bool]`, mypy's overload-fallback recovery mis-resolved an
   unrelated expression, `Index[Any] - str`, in `tests/indexes/test_sub.py` to
   `Never`, producing a spurious `Need type annotation for "_0" [var-annotated]`
   instead of the expected result.
2. **The new overload can trigger `overload-overlap`.** The fix belongs on the
   *neighbouring* `Index[Never]` overload, as
   `# type: ignore[overload-overlap]`, not on the newly-added `Never` overload itself.

### Verified blast radius

`_proto_*` gaps — the element types deliberately missing from each family
(`_stubs_only/__init__.pyi:202-394`), i.e. every place the bug can bite:

| family | deliberately missing element types |
| :--- | :--- |
| `_proto_add` / `_proto_radd` | Timedelta, Timestamp |
| `_proto_sub` / `_proto_rsub` | **bool**, **str** |
| `_proto_mul` / `_proto_rmul` | Timestamp |
| `_proto_truediv` / `_proto_rtruediv` | **bool**, **str**, Timestamp |
| `_proto_floordiv` / `_proto_rfloordiv` | **bool**, **complex**, **str**, Timestamp |

Guard status in `base.pyi` for the scalar/sequence forms, and whether a regression
test exists at all:

| operation | `Never` guard | test status |
| :--- | :--- | :--- |
| `Index[bool] -/- bool` | **yes** (base.pyi:762, :802) | clean `assert_type(…, Never)`, fixed in pandas-dev/pandas-stubs#1938 |
| `Index[bool] / bool`, `bool / Index[bool]` | no (only `np_ndarray_bool`, base.pyi:987) | 4 sites, pyright-only guard + bare `# TODO: python/mypy#20061` |
| `Series[bool] / bool` (+ `.truediv`/`.div`/`.rtruediv`/`.rdiv`) | no | 8 sites, pyright-only guard + bare `# TODO: python/mypy#20061` |
| `Index[bool] // bool`, `Index[complex] // complex` | no (ndarray-only, base.pyi:1103, :1145) | **no test file exists at all** |
| `Index[str] - str`, `/ str`, `// str` | no | **no test file exists at all** |

Two further structural facts:

- **`Series` is affected too, differently from `Index`.** `series.pyi` does not import
  `Supports_ProtoSub` / `Supports_ProtoRSub` at all, so `Series` bool *subtraction* is
  unaffected while `Series` bool *true-division* is (`series.pyi` uses the protocol
  idiom for 8 of the 10 families). `Series` applies the idiom to both dunders and named
  methods (`.truediv`/`.div`/`.rtruediv`/`.rdiv`, 16 use sites total), so the eventual
  fix surface for true-division is larger on `Series` than on `Index`.
- **This is a three-checker problem, not four.** `pyproject.toml:435-447` excludes `ty`
  from `tests/**/*{add,sub,truediv,mul,floordiv}.py` entirely, and sets
  `respect-type-ignore-comments = false` for `ty`. Do not add `# ty: ignore` for this
  issue, and do not claim `ty` validates any of the guards above. Separately, pyrefly
  has `unused-ignore = false` (pyproject.toml:423), so a stale pyrefly ignore on this
  code would go unflagged rather than erroring like mypy's / pyright's would.

## Consequences

- **Positive**: mypy, pyright, and pyrefly agree again on operations the `_proto_*`
  gaps are meant to prohibit, restoring the project's three-checker guarantee for those
  expressions.
- **Positive**: the negative-test style stays the clean `assert_type(expr, Never)` form
  with no ignore comments, matching `AGENTS.md` Example 2 and ADR-0007.
- **Negative / Neutral**: every deliberately-excluded element type in a `_proto_*`
  family now needs a hand-written `Never` overload per operator per direction,
  partially offsetting the de-duplication the protocol consolidation (ADR-0009,
  pandas-dev/pandas-stubs#1378, pandas-dev/pandas-stubs#1474) was adopted for.
- **Negative / Neutral**: the "omission from `_proto_*` means illegal" invariant is no
  longer self-enforcing under mypy — it now depends on someone remembering to add the
  matching `Never` overload in the right position. Every new `_proto_*` family added in
  the future needs an explicit audit against python/mypy#20061 rather than trusting the
  protocol alone.

### Follow-ups (deferred; document now, fix later)

- Apply this decision to `__truediv__`/`__rtruediv__`: 4 `Index` sites plus 8 `Series`
  sites, including the named methods (`.truediv`/`.div`/`.rtruediv`/`.rdiv`).
- Add the missing `__floordiv__`/`__rfloordiv__` guards for `Index[bool]` and
  `Index[complex]`, **and write the test files for them — none exist yet**.
- Cover the `str` gap across `-`, `/`, and `//` — also untested today.
  pandas-dev/pandas-stubs#1938 fixed only the `bool` case for `__sub__`/`__rsub__`, not
  `str`.

## Historical References & Provenance

- **Primary / related pull requests & issues**:
  - python/mypy#20061: Issue when `self` is typed as a Protocol (the upstream root
    cause).
  - pandas-dev/pandas-stubs#1938: `Index[bool]` subtraction `Never` guards (the fix
    that surfaced this decision).
  - pandas-dev/pandas-stubs#1474, pandas-dev/pandas-stubs#1378: the `Supports_Proto*`
    operator-family consolidation this decision is a corollary of.
  - pandas-dev/pandas-stubs#1926: umbrella tracking issue for the architecture-records
    effort this ADR was written under.
- **Cross references**:
  - ADR-0017 — Prohibition of Boolean Subtraction in Static Type Stubs (the underlying
    rule `_proto_sub`'s missing `bool` overload encodes).
  - ADR-0009 — Protocols for Type Discrimination (the `Supports_Proto*` idiom this
    decision patches rather than reverses).
  - ADR-0008 — Multi-Checker Ignore Standards (why an ignore comment cannot express a
    false negative).
  - ADR-0005 — Argument Narrowing vs. Widening (the overload-ordering discipline this
    decision extends).
  - ADR-0006, ADR-0007 — Multi-Checker Testing Harness and Negative Type Testing
    Pattern (the `assert_type(expr, Never)` style this decision preserves without
    ignore comments).

## Controversies and Open Questions

1. **Whether to keep the protocol consolidation if upstream never fixes #20061.**
   The per-element-type `Never` tax accumulates with every new `_proto_*` family; if
   the number of required exclusions keeps growing, revisiting option 4 (concrete
   overloads, no protocol) may become worth the duplication it reintroduces.
2. **Whether pyrefly or `ty` could regress the same way in a future release.**
   pyrefly currently rejects the false negative correctly, but nothing pins that
   behavior beyond the current test suite; `ty` cannot currently catch this at all in
   the excluded test files, so a `ty`-only regression here would go unnoticed until
   `ty`'s exclusion (pyproject.toml:435-447) is lifted.
3. **Whether the `Never`-overload tax outweighs the consolidation benefit for large
   operator families.** `__floordiv__`/`__rfloordiv__` alone need four exclusions
   (bool, complex, str, Timestamp) across two directions; whether that volume still
   nets positive versus concrete per-type overloads is not yet settled and is left to
   the follow-up work above.
