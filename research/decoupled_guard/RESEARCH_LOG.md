# Research Log — A Servant And A Guard

---

## 2026-06-17 — Project Initialized

Paper outline written. Pre-experiment phase.

**Research question:** Does decoupling the safety guard from the main model improve both safety accuracy and task quality vs. a single self-guarding model?

**Dataset plan:**
- `lmsys/toxic-chat` — real toxic/benign chat data with human labels
- `JailbreakBench/JBB-Behaviors` — curated jailbreak attempts
- `tatsu-lab/alpaca` — benign baseline

**Model plan:**
- Guard: `meta-llama/Llama-Guard-3-8B` (HF Inference API)
- Main: `mistralai/Mistral-7B-Instruct-v0.3`
- Judge: `mistralai/Mixtral-8x7B-Instruct-v0.1`

**Open questions:**
- What's the right monolithic baseline system prompt? Needs to be genuinely good — strawmanning it would undermine the paper.
- Sample size: how many from each dataset to keep HF Inference API rate limits manageable?
- Judge rubric: need to finalize the 3-axis scoring (helpfulness, specificity, hedging).
