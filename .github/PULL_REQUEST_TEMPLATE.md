## What

<!-- One feature / component per PR. What does this add, and which phase
     and session of plan.md §9 does it belong to? -->

## Checklist

- [ ] Scoped to a single phase session; no scaffolding ahead of the current phase
- [ ] New core dependency? Written justification included (plan.md §14). New provider? MockTransport test + zero core changes.
- [ ] `Protocol` added only because a second implementation now exists
- [ ] Unit tests added; deterministic, offline, no API key
- [ ] `CHANGELOG.md` updated in this PR (Keep a Changelog format)
- [ ] `README.md` / `ROADMAP.md` updated if capabilities or status changed
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run mypy src` passes
- [ ] `uv run pytest -q` passes
