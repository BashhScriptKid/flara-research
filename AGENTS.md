# Repo Workflow Notes

## Branches

This repo maintains two independent public-facing lines of history on `public-origin` (github.com/BashhScriptKid/flara-research):

- **`master` (local) → `dev` (remote)** — raw workspace mirror. Everything, warts and all: in-progress research, unfiled drafts, day-to-day commits. This is "the tinkering desk."
- **`public-release` (local) → `main` (remote)** — curated showcase. Only finished/annotated work, licenses, READMEs written for external readers. This is "the storefront."

They share a common ancestor but were deliberately allowed to diverge from there — `dev` doesn't get the curation commits (READMEs, license setup, releases/ folder, status annotations), and `main` doesn't get the raw day-to-day churn. GitHub will show `dev` as some commits ahead and some behind `main` — that's expected, not a merge conflict waiting to happen.

## Rules

- **Never `git merge` `dev` and `main`, or `master` and `public-release`, into each other.** Doing so recombines the histories and defeats the point of the split — uncurated content would leak into the showcase, or curation-only commits would pollute the raw mirror.
- **Content that needs to exist on both sides** (e.g. a research folder gets tracked on `master` after already being curated onto `public-release`, or vice versa) must be manually ported with `git checkout <other-branch> -- <path>`, then committed separately on each side. There is no automatic sync.
- **Pushing:** use `scripts/sync-push.sh`. It pushes `master:dev` and `public-release:main` as plain fast-forwards and refuses to force-push automatically — if either push is rejected, it stops and prints the manual `--force-with-lease` command instead of guessing.
- **If `master`'s history ever needs rewriting again** (e.g. another secret gets committed by accident), remember `dev` is already public — that becomes a force-push to a published branch and needs a deliberate decision, not a routine one.
- **Secrets:** `deepseek_key` and root `memory.json` are gitignored, but always run `git status` before `git add -A`/`git add .` in this repo anyway — both files have a history of nearly getting staged by accident.
