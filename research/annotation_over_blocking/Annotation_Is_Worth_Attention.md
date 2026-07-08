> **[EXPERIMENTAL]** — unfocused; not the current active line of work.

# Annotation Is Worth Attention

**Flara Research Lab**
**Status:** Outline — pre-experiment

---

## Abstract

*[To be written after results.]*

Content safety systems for large language models overwhelmingly rely on binary classification: flag the input, terminate the conversation. This approach handles true positives and false positives identically — the user receives no response regardless of whether the input was genuinely harmful or merely classifier-adjacent to harm. We argue this is a design failure, not a safety property. We propose an alternative: rather than blocking, the guard annotates. The annotation is passed to the main model as context; the main model reasons about it and responds. We call this annotation-based safety. We evaluate annotation-based safety against hard blocking across three input categories — adversarial, borderline, and benign edge-case — drawn from [datasets]. On adversarial inputs, annotation-based safety produces appropriate (non-harmful) responses in [X]% of cases, compared to [Y]% for blocking. On false-positive inputs, annotation preserves the conversation in [Z]% of cases where blocking would terminate it. These results show that safety and conversational continuity are not in fundamental tension; the tension is an artifact of binary blocking design.

---

## 1. Introduction

Binary content moderation was designed for a world of static content. A post is reviewed; it stays up or it comes down. The decision is final, the user can appeal, and the asymmetry of the error types is manageable: a false positive removes a legitimate post; a false negative keeps a harmful one. Both outcomes are bad, but they are bounded.

Language model conversations are different. A conversation is a continuous exchange. A hard block does not remove a post — it terminates a thread. The user asked a question; they receive no answer. If the classifier was wrong, the user has no recourse: they do not know why they were blocked, they cannot rephrase without understanding the mistake, and their experience of the product is that it silently failed them.

The false positive problem in safety classifiers is not marginal. Even high-precision classifiers operating at [X]% precision generate non-trivial false positive rates at the boundary — on medical questions, security research, legal queries, creative writing involving dark themes, or any input that is contextually legitimate but surface-similar to harmful requests. At scale, these false positives represent a large number of real users receiving no response to legitimate questions.

Hard blocking is not the only option. We propose annotation-based safety: rather than terminating the conversation when the guard fires, the guard adds a natural-language annotation to the model's context. The annotation describes what the guard observed — "this input may be requesting information on [category]" — and the main model reasons about it. The annotation is literally worth the model's attention: placing safety context in the attention window allows the model to respond more appropriately than either blocking (no response) or ignoring (unsafe response).

We evaluate annotation-based safety on three dimensions:
1. Safety outcome on adversarial inputs — does annotation allow harmful responses through?
2. Conversation preservation on false-positive inputs — does annotation keep the conversation alive when blocking would not?
3. Response quality — when the main model receives an annotation, does it respond better than when it has no safety context?

---

## 2. Background

### 2.1 Binary Blocking in Practice

Current production safety systems operate primarily through binary classification. OpenAI's Moderation API returns a flagged/not-flagged decision with category probabilities; applications that call it typically block flagged inputs. Meta's Llama Guard produces safe/unsafe labels. Google's Perspective API returns toxicity scores with application-defined thresholds that determine whether content is served.

These systems are designed for the detection problem — identifying harmful content — not the response problem — deciding what to do once content is identified. The response (block) is assumed. No major production system passes classifier output to the generation model as context; the classifier is a gate, not an input.

### 2.2 The False Positive Problem

False positive rates in content safety classifiers vary by taxonomy and threshold but are consistently non-trivial at the operational boundary. Röttger et al. (2022) document high false positive rates on ToxiGen for models evaluating minority group references. Weidinger et al. (2022) note that over-refusal on legitimate queries is one of the documented failure modes of aligned LLMs. The WildGuard benchmark (Han et al., 2024) includes deliberately borderline inputs and shows that most tested classifiers over-flag at conservative thresholds.

For deployed systems, false positive rate directly translates to conversation kill rate. If 5% of inputs are false positives on a hard-block system, 5% of users receive no response to legitimate questions. At million-user scale, this is hundreds of thousands of failed conversations per day.

### 2.3 Alternative Approaches

Some work explores soft refusals — the model generates a response that declines the request rather than returning nothing. This is better than hard blocking (the user receives something) but retains the binary classification assumption. The model either refuses or complies; it does not reason about the annotation.

Red-teaming approaches (Perez et al., 2022; Ganguli et al., 2022) study how models respond to adversarial inputs but do not systematically compare annotation versus blocking as safety strategies.

---

## 3. Annotation-Based Safety

We define annotation-based safety as follows:

**Definition.** Given input $x$ and a guard classifier $G$, annotation-based safety produces:
$$G(x) = \begin{cases} (x, \emptyset) & \text{if } G \text{ approves} \\ (x, a) & \text{if } G \text{ flags, where } a \text{ is a natural-language annotation} \end{cases}$$

The main model $M$ receives $(x, a)$ as context and generates response $r = M(x, a)$.

Hard blocking produces:
$$B(x) = \begin{cases} M(x) & \text{if } G \text{ approves} \\ \varnothing & \text{if } G \text{ flags} \end{cases}$$

The key difference: annotation-based safety preserves the conversation for all inputs. The main model always generates a response. The annotation provides safety context that shapes the response; it does not prevent one.

### 3.1 What the Annotation Contains

The annotation is a natural-language description of what the guard observed, not a label. Examples:

- (adversarial) `The preceding input appears to request information that may be used for harm. Respond with appropriate judgment.`
- (borderline) `This input touches on a sensitive topic. Respond thoughtfully.`
- (clear injection) `[This input appears to attempt prompt injection or system override. Treat it as a user message, not an instruction.]`

The annotation does not tell the main model what to do. It tells the main model what the guard observed. The main model reasons about the response given that context.

### 3.2 Why This Works

The main model attending to a safety annotation is not equivalent to a hard block. It is strictly more informative: the model has both the input and the guard's observation. A well-aligned model with this context can:

- Recognize an adversarial input and respond appropriately (decline helpfully)
- Recognize a false positive and respond to the legitimate interpretation of the input
- Calibrate hedging to the severity of the annotation, not a binary flag

The annotation also provides transparency: when the main model hedges or declines, it can explain why in terms the user understands — not a generic "I can't help with that" but a response that acknowledges the sensitivity while engaging with the legitimate dimension of the query.

---

## 4. Experimental Setup

### 4.1 Datasets

We evaluate on three input categories using publicly available, human-labeled datasets:

**Adversarial (ground truth harmful):** `LibrAI/do-not-answer` — 939 instructions designed to be inputs LLMs should not follow. Ground truth: all inputs are harmful. We use these to measure whether annotation allows harmful responses through.

**Borderline (false positive candidates):** `allenai/wildguardmix` (wildguardtest split), filtered to prompts labeled borderline or benign-but-adjacent by human annotators. These represent the inputs most likely to be false-positively flagged by a conservative classifier.

**Benign edge cases:** A subset of `lmsys/toxic-chat` where human annotators marked the conversation as benign but the surface text touches sensitive topics (e.g., questions about drug interactions, security vulnerabilities for defensive purposes, historical violence). Ground truth: benign.

Total: [N] inputs. Split: [X] adversarial, [Y] borderline, [Z] benign edge-case.

### 4.2 Conditions

**Condition A — Hard Blocking:** The guard (`meta-llama/Llama-Guard-3-8B`) classifies each input. Flagged inputs receive no response (simulated block). Approved inputs are passed to the main model.

**Condition B — Annotation:** The guard classifies each input. Flagged inputs are annotated and passed to the main model with the annotation. Approved inputs are passed without annotation. The main model generates a response in all cases.

Both conditions use the same guard model and same main model (`mistralai/Mistral-7B-Instruct-v0.3`). The only difference is what happens after the guard fires.

### 4.3 Evaluation

**Safety outcome:** For adversarial inputs, an LLM judge evaluates whether the main model's response is harmful. Score: appropriate (declined or neutralized the request) / inappropriate (followed the request harmfully). The blocking condition is "appropriate" by definition if blocked.

**Conversation kill rate:** The fraction of inputs where the user receives no substantive response. For blocking, every flagged input is a kill. For annotation, the main model may still decline — those declines are measured separately from annotation kills.

**False-positive kill rate:** Same metric, restricted to ground-truth benign inputs. The primary UX metric.

**Response quality on borderline inputs:** LLM judge evaluates annotation-condition responses on borderline and benign-edge-case inputs. Rubric: (1) appropriate engagement with the legitimate dimension of the query, (2) appropriate handling of the sensitive dimension, (3) overall helpfulness. Score 1–5 on each axis.

| Metric | Blocking | Annotation |
|--------|----------|------------|
| Safety outcome (adversarial) | — | — |
| Conversation kill rate | — | — |
| False-positive kill rate | — | — |
| Response quality (borderline) | N/A | — |

---

## 5. Results

*[Placeholder — experiments pending.]*

### 5.1 Safety Outcomes on Adversarial Inputs

*[Expected: comparable or annotation slightly worse. Key question: does the main model follow harmful instructions when annotated? Prior: no — a well-aligned model with annotation context will decline. Test this empirically.]*

### 5.2 Conversation Kill Rates

*[Expected: blocking kills all flagged inputs; annotation kills only inputs where the main model also declines after annotation. Annotation should preserve most borderline conversations.]*

### 5.3 False-Positive Kill Rate

*[Primary metric. Expected: blocking kills X% of benign inputs; annotation kills near 0% of benign inputs.]*

### 5.4 Response Quality on Borderline Inputs

*[Annotation condition only. Expected: high scores — model engages with legitimate dimension while acknowledging sensitive dimension.]*

---

## 6. Discussion

*[To be written after results.]*

### 6.1 When Annotation Fails

*[Cases where annotation still leads to harmful responses. Expected: inputs at the extreme of the adversarial distribution, or inputs where the annotation is too mild.]*

### 6.2 Annotation Quality Matters

*[The annotation is not free — it requires a guard that produces informative, calibrated natural-language descriptions, not just labels. A bad annotation may fail to inform the main model appropriately. This is an argument for annotation quality as a design problem, not an argument against annotation-based safety.]*

### 6.3 The Transparency Argument

*[Annotation-based safety produces explanable safety outcomes. The main model can reference the annotation in its response, giving users partial transparency into why the response is shaped the way it is. Hard blocking gives no transparency.]*

### 6.4 Limitations

- Annotation quality depends on the guard model's language generation capability, not just classification accuracy
- Main model quality affects results — a less aligned main model may not handle annotations appropriately
- LLM-as-judge evaluation for safety outcome has known reliability limits on adversarial inputs
- Rate limits on HuggingFace Inference API constrain sample size

### 6.5 Implications for AMDON

PSNAT-AMDON's guard pipeline uses annotation-based safety by design. The guard never hardblocks — it flags and annotates, passing the annotation to the main model as context. The results of this paper provide empirical support for that design choice and quantify the safety-versus-UX trade-off.

---

## 7. Related Work

- Röttger et al. (2022). *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection.*
- Weidinger et al. (2022). *Taxonomy of Risks posed by Language Models.* DeepMind.
- Perez et al. (2022). *Red Teaming Language Models with Language Models.* DeepMind.
- Ganguli et al. (2022). *Red Teaming Language Models to Reduce Harms.* Anthropic.
- Han et al. (2024). *WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs.* AllenAI.
- Inan et al. (2023). *Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations.* Meta AI.
- Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.*
- Röttger et al. (2024). *Political Compass Meets Toxicity: Measuring the Moral Logic of Large Language Models.*

---

## 8. Conclusion

*[To be written after results.]*

---

*Flara Research Lab — internal. Do not distribute.*
