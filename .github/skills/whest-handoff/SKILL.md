---
name: whest-handoff
description: "Use when: context is getting large during WHest estimator work, benchmarking, submission prep, or debugging and the next session needs a compact restart packet. Captures current state, commands, decisions, gotchas, and next steps under .tmp/handoff/."
argument-hint: "handoff topic or current WHest task"
---

# WHest Handoff

This skill preserves the current working state across context clears or new sessions. It has two modes based on whether `.tmp/handoff/briefing.md` exists.

## Mode Detection

```bash
test -f .tmp/handoff/briefing.md && echo read || echo write
```

## Write Mode

Use write mode when the session has accumulated decisions, benchmark results, or partial implementation details that should not be re-derived.

1. Check state with `git status --short`, `git branch --show-current`, and `git log --oneline -5`.
2. Identify the active task: estimator optimization, validation failure, submission prep, docs work, or benchmark analysis.
3. Capture recent verified commands and distinguish them from unverified commands.
4. Summarize score evidence using the fields that matter for the task.
5. Write `.tmp/handoff/briefing.md` and `.tmp/handoff/tasks.md`.
6. Tell the user the next session can resume by asking for the WHest handoff.

### `briefing.md` Template

````markdown
# WHest Handoff Briefing

**Date**: YYYY-MM-DD HH:MM
**Branch**: <branch-name>
**Task**: <short task name>
**WIP commit**: <short SHA or none>
**Primary files**: <files that matter>

## What's Done
- <completed work and evidence>

## What's Next
- <ordered next actions>

## Key Decisions
- <decision plus rationale>

## Score Evidence
- <command, seed/dataset, runner mode, key score fields, or none>

## Verified Commands
```bash
<commands that ran cleanly>
```

## Provisional Commands
```bash
<commands that may need adjustment>
```

## Gotchas
- <failed approaches, budget traps, local/subprocess mismatches, packaging notes>

## Files Changed
<git status --short output>
````

### `tasks.md` Template

````markdown
# Tasks

- [ ] <remaining task>
- [~] <in-progress task>
- [x] <completed task>
````

## Read Mode

Use read mode when `.tmp/handoff/briefing.md` exists.

1. Read `.tmp/handoff/briefing.md` and `.tmp/handoff/tasks.md`.
2. Verify current git state against the briefing.
3. Restate the next action in one short paragraph.
4. Continue from `What's Next`; do not rerun expensive benchmarks unless the briefing says the result was provisional.

## Safety Notes

- Do not create a WIP commit unless the user asks for it.
- Do not stage all files blindly; if committing is requested, stage named files only.
- Keep `.tmp/handoff/` ephemeral. Promote durable findings to `bench_logs/` or docs.