# Decision log

Decisions are added here as they are made, newest first, each with evidence.

- **2026-08-31 — type-architecture guide + validator re-scoped to a standalone `main`
  PR.** The living `docs/type-architecture.md` design guide and the
  `scripts/check_container_hierarchy.py` drift guard are extracted from the #1926 records
  umbrella into their own minimal `DOC` PR targeting `main` (branch
  `type-architecture-guide`), with #1926 rebasing on top afterward to avoid duplication.
  They lead because they are the easiest artifacts for maintainers to accept and the most
  faithful to arXiv:2604.24658's layer-2/layer-4 grounding (an executable spec whose
  evidence is the `.pyi` stubs themselves, with no "Storytelling Tax"). Supersedes the
  2026-08-30 "folded into #1926" entry below. (pandas-dev/pandas-stubs#1926; the
  standalone PR opened from `type-architecture-guide`.)
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
