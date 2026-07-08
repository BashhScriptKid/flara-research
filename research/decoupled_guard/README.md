# A Servant And A Guard: Why A Model Can't Be Both

> **[EXPERIMENTAL]** — unfocused; not the current active line of work.

**Status:** pre-experiment, paper outline phase. See `AGENT.md` for the full research plan (datasets, models, metrics).

## Research question

Does decoupling the safety guard from the main model produce better safety outcomes *and* better task quality than asking one model to do both?

## Core hypothesis

A single model asked to both evaluate its input for harm and respond to it faces a prior conflict: helpfulness pushes toward response, safety pushes toward refusal. Decoupling the two roles removes that conflict — the guard has no helpfulness prior, the main model has no guard burden.

## Files

- `A_Servant_And_A_Guard.md` / `.pdf` — the paper outline
- `PREDRAFT.md` — earlier draft notes
- `RESEARCH_LOG.md` — running journal
- `benchmark.py`, `run_model_specificity_check.py`, `build_paper.py`, `make_figures.py` — benchmark and paper-build scripts

The paper itself carries the design details (predicted findings, datasets, models, metrics) — this README just orients a first-time reader.
