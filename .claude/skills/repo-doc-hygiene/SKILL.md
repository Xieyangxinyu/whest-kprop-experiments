---
name: repo-doc-hygiene
description: "Use when: tidying WHest docs, finding orphaned markdown, checking stale references, reviewing docs navigation, or deciding whether a docs file should be linked, redirected, kept, or deleted."
argument-hint: "docs area or hygiene goal"
---

# Repo Doc Hygiene

Use this skill for periodic documentation cleanup. It is a review workflow, not a CI gate.

## Scope

Check Markdown under:

- `docs/`
- `examples/README.md`
- `README.md`
- `.github/skills/`
- durable benchmark notes in `bench_logs/` when relevant

Treat dated benchmark logs and active plans as intentionally standalone unless the user asks for archival cleanup.

## Workflow

1. List candidate Markdown files.
2. For each candidate, count inbound references by path, filename, and title phrase.
3. Classify files as linked, low-reference, orphaned, generated/ephemeral, or intentionally standalone.
4. Review content before deleting anything.
5. Choose one action per orphan: link, redirect, merge, keep, or delete.

## Reference Scan

Prefer `rg` when available. If not, use `git grep`.

```bash
git ls-files '*.md'
git grep -n --fixed-strings '<relative/path.md>' -- '*.md'
git grep -n --fixed-strings '<filename.md>' -- '*.md'
```

When checking `.github/skills/`, also search skill names and reference filenames because skills often link by relative path.

## Review Questions

- Is this file reachable from a parent `README.md` or an appropriate skill?
- Is the content still accurate for the estimator contract, scoring model, or current CLI?
- Does it duplicate another doc that should be canonical?
- Is it a benchmark note that belongs in `bench_logs/` rather than docs navigation?
- Would linking it improve a common workflow, or is it intentionally archival?

## Actions

- **Link** valuable orphaned docs from the nearest parent page.
- **Redirect** stale docs with a short pointer when external references may exist.
- **Merge** duplicated content into the canonical doc and remove the duplicate.
- **Keep** dated plans, logs, and intentionally standalone artifacts when they still carry useful context.
- **Delete** only after confirming the content is obsolete and unreferenced.

## Anti-Patterns

- Deleting benchmark learnings because they have few references.
- Treating low-reference docs as dead without reading them.
- Updating docs navigation without checking that linked paths render from GitHub.
- Moving skill reference files without updating the parent `SKILL.md`.