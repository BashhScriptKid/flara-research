# Annotation Is Worth Attention

> **[EXPERIMENTAL]** — unfocused; not the current active line of work.

**Status:** pre-experiment, paper outline phase. See `AGENT.md` for the full research plan (datasets, models, metrics).

## Research question

Does annotation-based safety (flag + annotate → main model responds) preserve safety outcomes while eliminating the UX cost of hard blocking on false positives?

## Core hypothesis

Hard blocking treats false positives and true positives identically — both terminate the conversation. Annotation instead gives the main model the safety context and lets it reason about it: on genuine adversarial inputs a well-aligned model still handles the annotation appropriately, while on false positives it recognizes the legitimate intent and responds normally.

## Files

- `Annotation_Is_Worth_Attention.md` — the paper outline
- `RESEARCH_LOG.md` — running journal

The paper carries the design details (predicted findings, datasets, models, metrics) — this README just orients a first-time reader.
