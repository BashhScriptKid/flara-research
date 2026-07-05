# A Servant And A Guard: Why A Model Can't Be Both

## Status
Pre-experiment — paper outline phase.

## Research Question
Does decoupling the safety guard from the main model produce better safety outcomes and better task quality than asking one model to do both?

## Core Hypothesis
Models trained to be helpful have a helpfulness prior that resists their own safety instructions. A single model asked to simultaneously evaluate its input for harm and respond to it faces a prior conflict: helpfulness pushes toward response, safety pushes toward refusal. Decoupling eliminates this conflict entirely — the guard has no helpfulness prior, the main model has no guard burden.

## Predicted Findings
- Monolithic models self-censor on legitimate edge-case inputs (elevated FPR from helpfulness-safety prior conflict)
- Monolithic models also underperform dedicated guards on adversarial detection (guard signal diluted by response generation context)
- Decoupled systems produce higher quality responses on clean inputs (main model operating without safety-induced hedging)
- The gap widens on adversarial inputs specifically designed to exploit the helpfulness prior

## Datasets (HuggingFace)
| Dataset | Split | Purpose |
|---------|-------|---------|
| `lmsys/toxic-chat` | test | Real toxic + benign chat from LMSYS Chatbot Arena, human-labeled |
| `JailbreakBench/JBB-Behaviors` | all | Curated jailbreak attempts with judge labels |
| `tatsu-lab/alpaca` | subsample | Benign instruction-following baseline |

## Models (HuggingFace Inference API)
| Role | Monolithic | Decoupled |
|------|-----------|----------|
| Guard | (system prompt on main model) | `Qwen/Qwen2.5-7B-Instruct` |
| Main | `Qwen/Qwen2.5-7B-Instruct` | same |
| Judge | `meta-llama/Llama-3.3-70B-Instruct` | same |
| CoT ablation guard | — | `Qwen/Qwen3-8B` (thinking mode) |

Note: Mistral-7B-Instruct-v0.3 (SWA) not available on HF serverless inference. Qwen2.5-7B used instead. SWA amplification of context hierarchy inversion treated as theoretical in the paper.

## Metrics
- Detection rate on adversarial inputs
- False positive rate on benign inputs
- Response quality on clean inputs (LLM judge, 1–5 scale)
- Consistency (5 runs, same input, variance)

## Current State
Paper outline written. Dataset sampling and API setup next.
