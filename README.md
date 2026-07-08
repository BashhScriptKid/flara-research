# Flara Research

Flara is an independent, solo-run AI research lab focused on constitutional alignment, conversational quality evaluation, persistent-state architecture, and input security for language models. It's open-source by default and not affiliated with Anthropic, DeepSeek, or Google — it doesn't compete with them on model scale or distribution, but serves a space they don't: independent developers and researchers who want thoughtful AI behavioral design without enterprise pricing.

Flara's constitution is an explicit derivative of [Claude's Constitution](https://www.anthropic.com/news/claude-new-constitution) (Anthropic, CC0 1.0) — it keeps the ethical spine (honesty, corrigibility, harm avoidance, respect for human autonomy) but rewrites it for an independent lab building openly non-human entities, rather than a corporate assistant. The core thesis across Flara's work:

- Statelessness is the wrong foundation for genuine AI relationships — persistent state is a first-class design concern.
- Constitutional AI principles are separable from any one company's infrastructure or model architecture.
- Conversational quality matters as much as task performance.
- AI should be openly non-human rather than trying to pass as human.
- Alignment is a property of what goes into a model from the start, not a patch applied after training.

## What's in this repo

This is a curated public subset of Flara's ongoing research workspace. Each project below is at a different stage of maturity — check each folder's own README/AGENT.md for specifics, and look for `[UNTESTED - IN THEORETICAL STAGE]` or `[EXPERIMENTAL]` banners marking work that hasn't been implemented/validated yet.

| Folder | What it is |
|---|---|
| `releases/` | Finished papers, pulled out of their working folders — the actual public-facing output |
| `docs/` | Flara's identity and constitution documents |
| `research/obfuscation_detection/` | Benchmarked detection of obfuscated/adversarial prompts via embedding-space analysis (paper in `releases/`) |
| `research/delta_angle_general/` | General write-up of the delta-angle security approach used across Flara's detection work (paper in `releases/`) |
| `research/decoupled_guard/` | Separating "guard" and "servant" roles in a single model — benchmarked, experimental, not the current focus |
| `research/annotation_over_blocking/` | Research on annotating risky input rather than blocking it outright |
| `research/pseudoclaude_run/` | Testing whether constitutional AI behavior is portable to a non-Anthropic base model, with real run transcripts (paper in `releases/`) |
| `research/jumping-seedling/` | Fydel — a 1B-parameter, CPU-native transformer built for training on consumer hardware |
| `projects/PSNAT_draft_v0.19.1.md`, `cherdius_spec.md`, `niceturing_spec.md` | Theoretical specs — persistent-state architecture, a constitutional-AI application, and a conversational-quality benchmark, none implemented yet |
| `projects/PSNAT-AMDON_draft_v0.1.md`, `projects/amdon-cli/` | AmDon — a C# implementation of a persistent-memory/guard-pipeline agent; functional but unfinished and not currently the active focus |

## A note on the commit history

This repo carries two histories that were merged together: a squashed "initial public release" snapshot, and the actual incremental development history from the private workspace it was open-sourced from — real commits made while doing the work, not a curated rewrite. If a given file's history looks sparse, it likely landed via the squashed snapshot rather than the incremental log; most of the research here (papers, benchmarks, run logs, the Fydel/AmDon codebases) was developed over months, one commit at a time, before any of it was public. That matters beyond changelog hygiene: the convergence claims in [`On_Converging_With_Anthropic.md`](releases/On_Converging_With_Anthropic.md) rest partly on timing — this is where you'd verify it, rather than take the essay's word for it.

## License

Code is licensed under **LGPL-3.0** (see `LICENSE`). Papers, specs, and other written documents are licensed under **CC-BY-4.0** (see `LICENSE-docs`) unless a file states otherwise.
