# Type architecture

This guide records the current operand hierarchy used when changing pandas-stubs
operator signatures. It is an explanation of the checked-in stubs, not an independent
specification of pandas runtime behavior.

## Source of truth

The relevant `.pyi` declarations and focused type/runtime tests are the source of truth.
This guide helps contributors navigate those artifacts and state the local constraints
that the hierarchy checker verifies. When they disagree, update the guide only after the
stubs and tests establish the intended behavior.

The checker entry point at
[`scripts/check_operand_hierarchy.py`](../../scripts/check_operand_hierarchy.py)
(the implementation is in `scripts/operand_hierarchy/`) tests a narrow structural
contract. It does not prove result types, runtime dispatch, the completeness of overloads,
or that this documentation is exhaustive.

## Pages

- [Operand hierarchy](operand-hierarchy.md) — tiers and scanned methods, cross-tier
  lookup examples, a traced rejection, the cross-tier overload policy and the
  exception list with its empty-set target, the matrix-multiplication exception, and the
  scope limits.
- [Operator signatures](operator-signatures.md) — operand aliases, protocols,
  positional-only parameters, subclass overrides, and the dunder set the checker scans.
- [Validation](validation.md) — the operand-hierarchy checker, when to run it, and the
  exact scope of validation.

## Adding a page

Keep a new page focused on one stable technical topic. Link it here, identify the stubs
and tests it explains, and distinguish checked facts from contributor guidance. Record
rollout history and review decisions in commit messages and the discussion thread,
indexed in the pull request body, rather than adding a repository-wide process diary.
