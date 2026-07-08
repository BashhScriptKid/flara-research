# PseudoClaude Run

Real experimental run data backing `releases/PseudoClaude_Paper.md` — testing whether the behavioral philosophy in Claude's constitution can be transplanted onto a non-Anthropic model (DeepSeek V4 Pro) via a staged compression pipeline, rather than copied verbatim.

## Files

- `JOURNAL.md` — phase-by-phase run journal: what was sent, what came back, what was decided at each step. All calls are real API calls to `deepseek-v4-pro`, not simulated.
- `deepseek_call.py`, `deepseek_chat.py` — the API call scripts used to drive the run (read the key from a local `deepseek_key` file, never hardcoded)
- `scratch_branch/` — raw request/response artifacts for every phase (system prompts, user turns, model outputs, both content and reasoning traces), numbered in run order

See `releases/PseudoClaude_Paper.md` for the write-up and conclusions; this folder is the primary-source evidence for it.
