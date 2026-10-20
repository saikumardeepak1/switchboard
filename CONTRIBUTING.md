# Contributing

## Workflow

1. Read `docs/ARCHITECTURE.md` for the contract of the package you are touching, and
   `docs/DECISIONS.md` for why it is shaped that way.
2. Write a failing test first. Run it and see it fail for the right reason.
3. Write the smallest code that passes. Refactor with the suite green.
4. Before committing: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest`.
   All four must pass.

## Rules

- Python 3.12, `mypy --strict` clean, ruff clean. Public functions and classes have type hints.
- Never read wall time or sleep directly. Take a `switchboard.core.clock.Clock`.
- Never use `eval`, `exec`, or pickle on untrusted input.
- Randomness takes a `random.Random` passed in, so simulation runs are reproducible per seed.
- Pure logic stays pure: policy checks, claim extraction, and endpointing math have no I/O.
- Import direction follows the package order in ARCHITECTURE.md. `core` imports nothing internal.
- Comments explain why, not what. Keep them sparse.
- No em dash characters anywhere: code, comments, docs, commit messages.

## Commits

Imperative subject under 72 characters, a blank line, then a short body saying what changed and
why. One logical change per commit. No tool or assistant attribution trailers.
