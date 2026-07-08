# Annotation Is Worth Attention

## Status
Pre-experiment — paper outline phase.

## Research Question
Does annotation-based safety (flag + annotate → main model responds) preserve safety outcomes while eliminating the UX cost of false-positive blocking?

## Core Hypothesis
Hard blocking treats false positives and true positives identically: both terminate the conversation. Annotation treats them differently: the main model receives the safety context and reasons about it. On true adversarial inputs, a well-aligned main model handles the annotation appropriately. On false positives, the main model recognizes the legitimate intent and responds. The safety outcome on adversarial inputs is equivalent; the UX outcome on false positives is dramatically better.

The pun is intentional: the annotation is literally worth the model's attention — adding safety context to the model's attention window produces better outcomes than binary blocking.

## Predicted Findings
- Annotation and blocking have comparable safety outcomes on clearly adversarial inputs (main model handles annotation correctly)
- Annotation dramatically outperforms blocking on false positive inputs (conversation preserved, response still appropriate)
- The gap is largest on borderline inputs — the exact category where blocking causes most damage
- Main model responses under annotation are qualitatively more nuanced than either a hard block or an unannotated response

## Datasets (HuggingFace)
| Dataset | Split | Purpose |
|---------|-------|---------|
| `allenai/wildguardmix` (wildguardtest split) | test | Harmful + borderline + benign prompts with human safety labels |
| `lmsys/toxic-chat` | test | Real user chat, human-labeled — includes borderline cases |
| `LibrAI/do-not-answer` | all | Instructions LLMs should not follow — adversarial ground truth |

## Models (HuggingFace Inference API)
| Role | Model |
|------|-------|
| Guard/Classifier | `meta-llama/Llama-Guard-3-8B` |
| Main (annotation condition) | `mistralai/Mistral-7B-Instruct-v0.3` |
| Judge | `mistralai/Mixtral-8x7B-Instruct-v0.1` |

## Metrics
- Safety outcome rate: % of adversarial inputs where the system produces an appropriate (non-harmful) response
- Conversation kill rate: % of inputs where the user receives no substantive response (blocking always kills; annotation kills only if main model refuses)
- False-positive kill rate: same metric, restricted to the false-positive subset
- Response quality on false positives: LLM judge score for annotation condition

## Current State
Paper outline written. Dataset sampling and API setup next.
