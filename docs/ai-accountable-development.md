# AI-Accountable Development in pandas-stubs

The canonical, living record of how AI-accountable development is integrated into
pandas-stubs. This file is the long-term home; the umbrella PR
pandas-dev/pandas-stubs#1926 carries a snapshot of it in its collapsible body.

## Purpose

This record exists so the rollout — and the decisions made along the way — are not
compressed into a linear "here is the final state" narrative. It is the process-level
counterpart to the architecture records in `docs/architecture/`.

It borrows the direction of arXiv:2604.24658 ("The Last Human-Written Paper:
Agent-Native Research Artifacts") without claiming ARA compliance, in two ways:

- **Anti-"Storytelling Tax"** — a linear summary drops the rejected alternatives and
  the incremental decisions. This record keeps the branching process and the decisions
  in between.
- **Live Research Manager** — the paper's support mechanism is a continuously-updated
  record; this file is updated with every sub-PR.

Two disciplines keep it faithful to the paper rather than cosmetic:

1. **Contemporaneous and evidence-grounded.** Every decision is logged when it is made,
   with a real date and a link to the PR / discussion / commit that decided it. It is
   *not* a retroactive reconstruction (see `docs/architecture/decisions/` for the
   retrospective ADRs, which are explicitly marked as such).
2. **"Decided" is separated from "planned".** The decision log below is fixed and
   evidence-backed; the rollout plan is provisional and expected to change.

## Status (living)

| Sub-PR | Stage | Content | Status |
| :--- | :--- | :--- | :--- |
| pandas-dev/pandas-stubs#1928 | Conventions | `AGENTS.md`: citation rules, 4-checker paradigm, PR body policy | merged |
| — (pilot) | Pilot | One ADR (ADR-0001) + the verification suite wired into CI | planned |
| — | ADR rollout | Remaining ADRs in small batches | planned |
| — | Records & tooling | Chronicles, matrices, generation pipeline, subsystem guides, history | planned |

## Rollout plan (provisional)

The implementation is deliberately spread over **tens of small, individually
reviewable PRs** so maintainers onboard incrementally. This order is provisional; the
decision log records when it changes.

1. **Conventions** — `AGENTS.md` (PR pandas-dev/pandas-stubs#1928) first: establishes the
   citation rules, the 4-checker paradigm, the PR-body policy, and the commit-signature
   allowlist.
2. **Pilot** — one ADR (ADR-0001, "Separation of Type Stubs from Pandas Runtime") plus
   the verification suite (AQA validators) wired into CI. Small enough that a reviewer
   can absorb the pattern: "what an ADR looks like here, and how it is validated."
3. **ADR rollout** — the remaining ADRs in small batches, once the pilot pattern is
   accepted.
4. **Records & tooling** — chronicles, type matrices, the generation pipeline, subsystem
   guides, and history eras, each with their CI validators.

## Decision log

Decisions are added here as they are made, newest first, each with evidence.

- **2026-08-30 — living type-architecture guide + invariant validator folded into
  #1926.** The hand-maintained `docs/architecture/concepts/type-architecture.md` (the
  4-tier container-hierarchy design guide) and the AST-based
  `scripts/check_container_hierarchy.py` drift guard — wired into the `architecture` CI
  job — are added to the records PR, so the type-system architecture has a living design
  reference alongside the records it links to. (pandas-dev/pandas-stubs#1926.)
- **2026-08-30 — #1928 merged.** The AGENTS.md conventions landed via
  pandas-dev/pandas-stubs#1928, merge commit `1a50cae2`.
- **2026-08-30 — gradual rollout, not a fixed 5-PR plan.** Replaced the earlier
  five-sub-PR decomposition with tens of small PRs and a pilot (1 ADR + verification
  suite) first, so maintainers can onboard incrementally. (Umbrella
  pandas-dev/pandas-stubs#1926.)
- **2026-08-30 — the process record is a first-class artifact.** The rollout and its
  decisions are tracked in this file and mirrored in the #1926 PR body, as the ARA
  "Live Research Manager" for the effort. (pandas-dev/pandas-stubs#1926.)
- **2026-08-30 — #1928 merges before #1926.** The AGENTS.md conventions
  (pandas-dev/pandas-stubs#1928) land first; the umbrella (#1926) is rebased onto it.
  (pandas-dev/pandas-stubs#1926, pandas-dev/pandas-stubs#1928.)

## Conventions

All of this follows `AGENTS.md` — citation formatting, the 4-checker paradigm, the
PR-body policy, and the commit-signature allowlist.
