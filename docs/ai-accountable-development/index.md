# AI-Accountable Development in pandas-stubs

The canonical, living record of how AI-accountable development is integrated into
pandas-stubs. It is split into two files linked from this index:

- [Rollout plan](rollout-plan.md) — the provisional implementation order.
- [Decision log](decision-log.md) — decisions as they are made, newest first.

## Purpose

This record exists so the rollout — and the decisions made along the way — are not
compressed into a linear "here is the final state" narrative. It borrows the direction
of arXiv:2604.24658 ("The Last Human-Written Paper: Agent-Native Research Artifacts")
without claiming ARA compliance, in two ways:

- **Anti-"Storytelling Tax"** — a linear summary drops the rejected alternatives and
  the incremental decisions. This record keeps the branching process and the decisions
  in between.
- **Live Research Manager** — the paper's support mechanism is a continuously-updated
  record; this record is updated with every sub-PR.

Two disciplines keep it faithful to the paper rather than cosmetic:

1. **Contemporaneous and evidence-grounded.** Every decision is logged when it is made,
   with a real date and a link to the PR / discussion / commit that decided it. It is
   *not* a retroactive reconstruction.
2. **"Decided" is separated from "planned".** The decision log is fixed and
   evidence-backed; the rollout plan is provisional and expected to change.

## Status (living)

| Sub-PR | Stage | Content | Status |
| :--- | :--- | :--- | :--- |
| pandas-dev/pandas-stubs#1928 | Conventions | `AGENTS.md`: citation rules, 4-checker paradigm, PR body policy | merged |
| — (pilot) | Pilot | One ADR (ADR-0001) + the verification suite wired into CI | planned |
| — | ADR rollout | Remaining ADRs in small batches | planned |
| — | Records & tooling | Chronicles, matrices, generation pipeline, subsystem guides, history | planned |

## Conventions

All of this follows `AGENTS.md` — citation formatting, the 4-checker paradigm, the
PR-body policy, and the commit-signature allowlist.
