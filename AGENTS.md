# Repo Workflow Notes

## Branches

This repo maintains two independent public-facing lines of history on `public-origin` (github.com/BashhScriptKid/flara-research):

- **`master` (local) → `dev` (remote)** — raw workspace mirror. Everything, warts and all: in-progress research, unfiled drafts, day-to-day commits. This is "the tinkering desk."
- **`public-release` (local) → `main` (remote)** — curated showcase. Only finished/annotated work, licenses, READMEs written for external readers. This is "the storefront."

They share a common ancestor but were deliberately allowed to diverge from there — `dev` doesn't get the curation commits (READMEs, license setup, releases/ folder, status annotations), and `main` doesn't get the raw day-to-day churn. GitHub will show `dev` as some commits ahead and some behind `main` — that's expected, not a merge conflict waiting to happen.

## Rules

- **Never `git merge` `dev` and `main`, or `master` and `public-release`, into each other.** Doing so recombines the histories and defeats the point of the split — uncurated content would leak into the showcase, or curation-only commits would pollute the raw mirror.
- **Content that needs to exist on both sides** (e.g. a research folder gets tracked on `master` after already being curated onto `public-release`, or vice versa) must be manually ported with `git checkout <other-branch> -- <path>`, then committed separately on each side. There is no automatic sync.
- **Pushing:** either push privately (`dev` only — raw work-in-progress, fine to push alone) or push both (`dev` AND `main` together, if `main` is getting anything). Never push `main` alone without `dev` also being in sync — `main` getting ahead of `dev` with content `dev` doesn't have defeats the "dev has everything" property the split relies on. Default to `scripts/sync-push.sh`, which pushes `master:dev` and `public-release:main` together as plain fast-forwards and refuses to force-push automatically — if either push is rejected, it stops and prints the manual `--force-with-lease` command instead of guessing.
- **If `master`'s history ever needs rewriting again** (e.g. another secret gets committed by accident), remember `dev` is already public — that becomes a force-push to a published branch and needs a deliberate decision, not a routine one.
- **Secrets:** `deepseek_key` and root `memory.json` are gitignored, but always run `git status` before `git add -A`/`git add .` in this repo anyway — both files have a history of nearly getting staged by accident.

## Known exception: 2026-07-08 merge

On 2026-07-08, `main` was deliberately merged into `dev` (commit `e1e1679`), against the "never merge" rule above — a user-authorized one-time override, not an accident, done because `dev` had fallen behind `main` by 13 commits and diverging further wasn't worth it. `dev`'s history from this point on legitimately contains `main`'s curation commits; that's expected, not something to "fix" by reverting. The rule above still holds going forward — this was a one-time reset, not a change of policy.

Side effect: the merge and a follow-up README commit were made on a separately-created local `dev` branch rather than the canonical local `master`, so local `master` is currently behind `public-origin/dev` and will need reconciling (`git merge origin dev-work-branch` or equivalent) before `scripts/sync-push.sh` will cleanly fast-forward `master:dev` again.
