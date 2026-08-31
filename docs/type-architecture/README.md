# Type architecture

This guide records the current type-container structure used when changing pandas-stubs
operator signatures. It is an explanation of the checked-in stubs, not an independent
specification of pandas runtime behavior.

## Source of truth

The relevant `.pyi` declarations and focused type/runtime tests are the source of truth.
This guide helps contributors navigate those artifacts and state the local constraints
that the hierarchy checker verifies. When they disagree, update the guide only after the
stubs and tests establish the intended behavior.

The checker at
[`scripts/check_container_hierarchy.py`](../../scripts/check_container_hierarchy.py)
tests a narrow structural contract. It does not prove result types, runtime dispatch,
the completeness of overloads, or that this documentation is exhaustive.

## Pages

- [Container hierarchy](container-hierarchy.md) — tiers, cross-tier lookup examples,
  structural invariant, and the matrix-multiplication exception.
- [Operator signatures](operator-signatures.md) — operand aliases, protocols,
  positional-only parameters, and subclass overrides.
- [Testing and validation](testing-and-validation.md) — focused type-test patterns,
  checker diagnostics, and the exact scope of validation.

## Adding a page

Keep a new page focused on one stable technical topic. Link it here, identify the stubs
and tests it explains, and distinguish checked facts from contributor guidance. Record
rollout history and review decisions in the relevant pull request rather than adding a
repository-wide process diary.
