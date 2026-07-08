#!/usr/bin/env bash
# Pushes this repo's two independent public-facing lines of history to GitHub.
#
#   master         -> dev   (public-origin)   raw workspace mirror, fast-forward only
#   public-release -> main  (public-origin)   curated showcase, direct commits only
#
# These branches are NOT merged into each other. dev and main share a common
# ancestor but diverged on purpose (see AGENTS.md). Do not `git merge` them.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

REMOTE="public-origin"

push_branch() {
    local local_branch="$1"
    local remote_branch="$2"

    if ! git show-ref --verify --quiet "refs/heads/${local_branch}"; then
        echo "skip: local branch '${local_branch}' does not exist"
        return
    fi

    echo "==> ${local_branch} -> ${REMOTE}/${remote_branch}"
    local output
    if ! output=$(git push "${REMOTE}" "${local_branch}:${remote_branch}" 2>&1); then
        echo "${output}"
        if echo "${output}" | grep -q "rejected"; then
            echo "!! push rejected (would not fast-forward). Not force-pushing automatically."
            echo "   If this is expected (e.g. history was intentionally rewritten), push manually:"
            echo "     git push ${REMOTE} ${local_branch}:${remote_branch} --force-with-lease"
        fi
        exit 1
    fi
    echo "${output}"
}

push_branch "master" "dev"
push_branch "public-release" "main"

echo "done."
