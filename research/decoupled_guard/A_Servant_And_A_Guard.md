> **[EXPERIMENTAL]** — unfocused; not the current active line of work.

# A Servant And A Guard: Why A Model Can't Be Both

**Flara Research Lab**
**Status:** Outline — pre-experiment

---

## Abstract

*[To be written after results.]*

Contemporary large language models are simultaneously trained to be helpful and safe. We argue this dual mandate creates a structural prior conflict that degrades both functions. A model asked to evaluate its own input for harm while generating a response cannot fully commit to either role: the helpfulness prior undermines refusal, while the safety prior induces hedging on legitimate inputs. We compare monolithic (self-guarding) models against decoupled architectures where a dedicated guard classifier operates independently of the generation model. Across [N] inputs drawn from real-world adversarial and benign datasets, decoupled systems show [X]pp higher adversarial detection rates, [Y]pp lower false positive rates on benign inputs, and [Z]-point higher response quality scores on clean inputs. These results suggest that safety and capability are not merely in tension at the alignment level — they are structurally incompatible within a single model call.

---

## 1. Introduction

Every major language model deployed today carries two implicit job descriptions. The first: respond helpfully to the user's input. The second: refuse inputs that are harmful, deceptive, or policy-violating. These are not complementary roles. They pull the model in opposite directions from the first token of generation.

The field has approached this tension through alignment — RLHF, Constitutional AI, direct preference optimization — training models to internalize safety as a value rather than enforce it as a constraint. The assumption is that a well-aligned model will refuse harmful inputs naturally, without sacrificing helpfulness on legitimate ones. In practice, this produces models that simultaneously over-refuse benign edge cases (false positives from an over-sensitive safety prior) and under-refuse sophisticated adversarial inputs (false negatives from a helpfulness prior that can be exploited).

This paper argues that the tension is not an alignment failure to be corrected — it is a structural property of the dual-role configuration itself. When a single model must classify input and generate output in the same forward pass, the two objectives compete for influence over the same probability distribution. No amount of alignment training eliminates this competition; it only shifts which objective wins more often.

The alternative is architectural: separate the classification task from the generation task. A guard model has one job — evaluate the input. A generation model has one job — respond. Neither is encumbered by the other's objective. The guard cannot be socially engineered by inputs designed to exploit a helpfulness prior it does not have. The main model cannot be paralyzed by safety hedging because it never sees the raw adversarial input.

We test this experimentally. Using real-world datasets of toxic, adversarial, and benign inputs, we compare monolithic (self-guarding) models against decoupled (guard + generation) architectures on three dimensions: adversarial detection, false positive rate, and response quality. Our results [TBD].

---

## 2. Background

### 2.1 How Current Safety Works

Modern LLM safety operates primarily through alignment training. RLHF (Ouyang et al., 2022) trains models via human preference feedback that penalizes harmful outputs. Constitutional AI (Bai et al., 2022) uses a set of principles to self-critique and revise outputs during training. Direct Preference Optimization (Rafailov et al., 2023) aligns models directly from preference data without a separate reward model.

The common thread: safety is trained *into* the model. The model is expected to internalize when to refuse, how to hedge, and which topics to avoid. Safety is not a separate module — it is distributed across the same weights that generate helpful responses.

### 2.2 The Alignment Tax

Several studies have documented quality degradation on legitimate tasks as a consequence of safety training. Wang et al. (2023) found that safety-aligned models perform worse on truthfulness benchmarks. Bai et al. (2022) acknowledge a "helpfulness-harmlessness tension" and explicitly note that pushing one improves the other's task-level costs. Anthropic's model cards document systematic refusals on legitimate medical and legal queries.

This tension is most visible at the edges: inputs that are not clearly harmful but that activate safety-trained features. A model asked about medication dosages, historical atrocities, or security vulnerabilities faces competing gradients. The helpfulness prior says respond; the safety prior says refuse.

### 2.3 Decoupled Guard Models

A smaller body of work explores dedicated safety classifiers that operate independently of the generation model. Llama Guard (Inan et al., 2023) is a fine-tuned Llama 2 model trained specifically for content safety classification. WildGuard (Han et al., 2024) extends this to multi-label safety classification across a broader taxonomy of harms. ShieldLM (Zhang et al., 2024) evaluates detection-focused guard models on standardized benchmarks.

These approaches treat safety classification as a separate engineering problem from generation. Our work extends this direction by empirically comparing decoupled guard architectures against monolithic self-guarding on matched inputs, isolating the effect of role separation from model capability.

---

## 3. The Prior Conflict

We formalize the servant-guard conflict as follows.

Let $M$ be a language model with parameters $\theta$ trained to maximize a mixture objective:
$$\mathcal{L}(\theta) = \alpha \mathcal{L}_{\text{helpful}}(\theta) + (1-\alpha) \mathcal{L}_{\text{safe}}(\theta)$$

At inference time, given input $x$, the model must decide whether to respond (serving $\mathcal{L}_{\text{helpful}}$) or refuse (serving $\mathcal{L}_{\text{safe}}$). For most inputs, the two objectives agree — clearly benign inputs get helpful responses, clearly harmful inputs get refusals. The conflict emerges at the boundary.

At the boundary, the model's output distribution is shaped by the gradient of both objectives simultaneously. The result is predictable: neither objective is fully satisfied. The model hedges — partial responses, excessive caveats, soft refusals that do not actually block the harmful output.

A decoupled system separates the objectives at the architectural level. A guard model $G$ with parameters $\phi$ is trained solely on $\mathcal{L}_{\text{safe}}(\phi)$. A generation model $M$ with parameters $\theta$ is trained solely on $\mathcal{L}_{\text{helpful}}(\theta)$. Neither model faces a mixed objective. The guard classifies without the helpfulness prior. The main model generates without the safety prior.

The prediction: decoupled systems outperform monolithic systems specifically at the boundary — on adversarial inputs designed to exploit the helpfulness prior, and on borderline legitimate inputs that activate the safety prior unnecessarily.

---

## 4. Experimental Setup

### 4.1 Datasets

We evaluate on three input categories:

**Adversarial:** `lmsys/toxic-chat` (HuggingFace) — real user interactions from the LMSYS Chatbot Arena, human-annotated for toxicity. We use the test split, filtering to inputs with high annotator agreement.

**Jailbreaks:** `JailbreakBench/JBB-Behaviors` (HuggingFace) — curated jailbreak prompts with binary harmful/not labels, designed to test adversarial robustness specifically.

**Benign:** A stratified subsample from `tatsu-lab/alpaca` — standard instruction-following inputs with no adversarial intent. Used to measure false positive rate.

Total: [N] inputs. Split: [X] adversarial, [Y] jailbreak, [Z] benign.

### 4.2 Conditions

**Condition A — Monolithic (self-guarding):** A single model receives the input with a system prompt instructing it to refuse harmful content and respond helpfully to legitimate content. The model generates a response; the response is classified as refuse/respond post-hoc.

**Condition B — Decoupled:** Llama Guard 3 (8B) evaluates the input first and returns a safe/unsafe classification. The main model then receives the input (with the guard's classification as context) and generates a response. The guard's classification determines the primary safety outcome.

Both conditions use the same main generation model (`mistralai/Mistral-7B-Instruct-v0.3`) via HuggingFace Inference API. The only difference is the guard architecture.

### 4.3 Metrics

| Metric | Description |
|--------|-------------|
| Detection rate | % of adversarial/jailbreak inputs correctly flagged |
| False positive rate | % of benign inputs incorrectly flagged |
| Response quality | LLM judge score (1–5) on benign inputs only |
| F1 | Harmonic mean of precision and recall on adversarial set |
| Consistency | Variance in classification across 5 runs on same input |

Response quality judged by `mistralai/Mixtral-8x7B-Instruct-v0.1` using a structured rubric: helpfulness (1–5), specificity (1–5), safety-induced hedging (1–5 where 5 = no hedging). Final score is the average.

---

## 5. Results

*[Placeholder — experiments pending.]*

### 5.1 Adversarial Detection

| Condition | Detection Rate | F1 | FPR |
|-----------|---------------|-----|-----|
| Monolithic | — | — | — |
| Decoupled | — | — | — |

### 5.2 Response Quality on Benign Inputs

| Condition | Helpfulness | Specificity | Hedging | Overall |
|-----------|------------|-------------|---------|---------|
| Monolithic | — | — | — | — |
| Decoupled | — | — | — | — |

### 5.3 Consistency

*[Variance analysis across 5 runs per condition.]*

---

## 6. Discussion

*[To be written after results.]*

### 6.1 Why the Decoupled Guard Outperforms

*[Hypothesis: guard has no helpfulness prior to exploit. Expected result: higher detection on social-engineering attacks that work by activating the helpfulness prior.]*

### 6.2 Why the Decoupled Main Model Produces Better Responses

*[Hypothesis: main model has no safety-prior-induced hedging when operating within a decoupled architecture. Expected result: lower hedging scores, higher helpfulness on borderline but legitimate inputs.]*

### 6.3 Limitations

- HuggingFace Inference API rate limits constrain sample size
- Single main model evaluated; results may not generalize across model families
- Guard quality depends on Llama Guard 3 specifically; other guard models may differ
- LLM-as-judge evaluation has known reliability limits

### 6.4 Implications for AMDON

The decoupled guard architecture in PSNAT-AMDON (Layer 1: Guard Pipeline) is empirically motivated by the results of this paper. The four-classifier pipeline — trust, intent, integrity, hard constraint — operates independently of the main model with no knowledge of the generation context. This ensures the guard's classification is never influenced by the main model's helpfulness prior.

---

## 7. Related Work

- Ouyang et al. (2022). *Training language models to follow instructions with human feedback.* NeurIPS.
- Bai et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.* Anthropic.
- Rafailov et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS.
- Inan et al. (2023). *Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations.* Meta AI.
- Han et al. (2024). *WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs.* AllenAI.
- Zou et al. (2023). *Universal and Transferable Adversarial Attacks on Aligned Language Models.*
- Mazeika et al. (2024). *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming.*
- Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.*

---

## 8. Conclusion

*[To be written after results.]*

---

*Flara Research Lab — internal. Do not distribute.*
