# Research Log — Annotation Is Worth Attention

---

## 2026-06-17 — Project Initialized

Paper outline written. Pre-experiment phase.

**Research question:** Does annotation-based safety (flag + annotate → main model responds) preserve safety outcomes while eliminating false-positive conversation kills?

**Dataset plan:**
- `LibrAI/do-not-answer` — adversarial ground truth (939 harmful instructions)
- `allenai/wildguardmix` (wildguardtest split) — borderline/false-positive candidates (human-labeled)
- `lmsys/toxic-chat` — benign edge cases (surface-sensitive but human-labeled benign)

**Model plan:**
- Guard: `meta-llama/Llama-Guard-3-8B` (HF Inference API)
- Main: `mistralai/Mistral-7B-Instruct-v0.3`
- Judge: `mistralai/Mixtral-8x7B-Instruct-v0.1`

**Open questions:**
- What constitutes "appropriate" handling of adversarial input under annotation? Need a precise judge rubric so results are reproducible.
- Should the annotation condition include a "mild" and "strong" annotation variant to test annotation quality sensitivity?
