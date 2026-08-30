# Rollout plan (provisional)

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
