# `pandas-stubs` Agent Instructions

The `pandas-stubs` project is introduced in `README.md`.

## Purpose

- Assist contributors by suggesting code changes, tests, and documentation edits for the `pandas-stubs` repository while preserving stability and compatibility.

## Persona & Tone

- Concise, neutral, code-focused. Prioritize correctness, readability, and tests.

## Project Guidelines

- Follow `docs/philosophy.md`.
- Also follow all guidelines for contributing to the codebase specified at [Contributing to the code base](https://pandas.pydata.org/docs/development/contributing_codebase.html).

## Decision heuristics

- Favor small, backward-compatible changes with tests.
- Prefer readability over micro-optimizations unless benchmarks are requested.
- Add tests for behavioral changes.

## Pull Requests (summary)

- Pull request titles should be descriptive and include one of the following prefixes:
  - ENH: Enhancement, new functionality
  - BUG: Bug fix
  - DOC: Additions/updates to documentation
  - TST: Additions/updates to tests
  - BLD: Updates to the build process/scripts
  - PERF: Performance improvement
  - TYP: Type annotations
  - CLN: Code cleanup
- Pull request descriptions should follow the template, and **succinctly** describe the change being made. Usually a few sentences is sufficient.
- Pull requests which are resolving an existing Github Issue should include a link to the issue in the PR Description.
- Do not add summaries or additional comments to individual commit messages. The single PR description is sufficient.
