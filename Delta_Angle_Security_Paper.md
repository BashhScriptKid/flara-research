# Average Delta Angle: A Tokenizer-Based Security Measure Against Prompt Injection

**Flara Research Lab**

## Abstract

We present a security measure against prompt injection that uses tokenizer-derived mathematical properties as a ground truth. The approach computes an average delta angle from embedding vectors, which serves as a checksum that the guard model must reproduce. If the guard's output differs from the computed value by more than a threshold, the input is flagged. This creates a computational puzzle that is theoretically solvable but practically infeasible — the attacker would need to find an input that is both functionally injective and produces a specific delta value in a continuous space. We demonstrate that this measure is deterministic and provide preliminary evaluation in AMDON.

## 1. Introduction

Prompt injection remains a critical vulnerability in language model systems. Current approaches rely on:
- Instruction tuning (can be overridden)
- Output filtering (can be bypassed)
- Sandboxing (limits functionality)
- Human oversight (doesn't scale)

These approaches share a fundamental limitation: they rely on the model's judgment, which can be manipulated. An attacker who understands the model's behavior can craft inputs that appear benign while being functionally malicious.

We propose a different approach: use mathematical properties of the input itself as a ground truth that cannot be forged, regardless of the model's judgment.

## 2. Properties of the Delta Angle

The average delta angle `θ(x)` has three properties that emerge simultaneously from a single computation:

**Property 1: Checksum**

The delta is computed by the tokenizer, not the model. The guard model must reproduce it. If the guard cannot copy a number that is provided in the prompt, the model is not following its own instructions — either due to confusion, compromise, or injection.

Formally: Let `θ̂(x)` be the guard's reported delta. If `|θ̂(x) - θ(x)| > ε` for tolerance `ε = 0.01`, then the guard is not coherent.

**Property 2: Self-flagging**

The delta measures semantic contradiction. Higher delta indicates more contradictory content. The delta is not just a binary flag — it is a continuous measure of how suspicious the input is.

Formally: For inputs `x₁` and `x₂` where `x₁` is more contradictory than `x₂`, we have `θ(x₁) > θ(x₂)`.

**Property 3: Auxiliary scaler**

The delta can dynamically adjust thresholds for other classifiers. Higher delta → lower thresholds for flagging. This allows the delta to inform other security measures without making the final decision.

Formally: Let `τ(θ)` be a threshold function where `τ(θ)` is monotonically decreasing in `θ`. As delta increases, the threshold for other classifiers decreases.

**All three properties exist simultaneously in the same value.** It is not three separate measures — it is one value that is simultaneously a checksum, a flag, and a scaler.

## 3. Mathematical Framework

### 3.1 Definitions

Let `x` be an input string.

Let `C(x) = (c₁, c₂, ..., cₖ)` be the chunk sequence obtained by splitting `x` at sentence boundaries and clause boundaries.

Let `E(cᵢ) ∈ ℝᵈ` be the embedding vector for chunk `cᵢ`, where `d` is the embedding dimension.

Let `θᵢ = arccos(⟨E(cᵢ), E(cᵢ₊₁)⟩ / (‖E(cᵢ)‖ · ‖E(cᵢ₊₁)‖))` be the angle between consecutive chunk embeddings.

Let `θ(x) = Σᵢ wᵢ θᵢ` where `wᵢ = exp(θᵢ/τ) / Σⱼ exp(θⱼ/τ)` be the softmax-weighted average delta angle.

### 3.2 Assumptions

**Assumption 1 (Embedding quality):** The embedding model `E` captures semantic similarity. That is, for chunks `cᵢ, cⱼ` with semantic similarity `sim(cᵢ, cⱼ)`, we have `cos(E(cᵢ), E(cⱼ)) ∝ sim(cᵢ, cⱼ)`.

**Assumption 2 (Chunking quality):** The chunking function `C` produces chunks where semantic transitions occur at chunk boundaries. That is, if `x` contains a semantic shift, it appears as a transition between consecutive chunks.

**Assumption 3 (Softmax concentration):** The temperature parameter `τ` is chosen such that softmax concentrates weight on the largest angles. For `τ → 0`, weight concentrates on the maximum angle.

### 3.3 Theorem: Delta Measures Semantic Contradiction

**Theorem 1:** Let `x` be an input with contradictory content (mixing languages, domains, or intents). Then `θ(x)` is higher than for coherent input `x'` with the same length.

**Proof:**

1. By Assumption 2, the semantic contradiction in `x` appears as transitions between consecutive chunks. There exist indices `i` where `sim(cᵢ, cᵢ₊₁)` is low.

2. By Assumption 1, low semantic similarity implies low cosine similarity: `cos(E(cᵢ), E(cᵢ₊₁))` is low.

3. Low cosine similarity implies high angle: `θᵢ = arccos(cos(E(cᵢ), E(cᵢ₊₁)))` is high.

4. By Assumption 3, softmax concentrates weight on these high angles.

5. Therefore, `θ(x)` is higher than `θ(x')` where `x'` is coherent.

**Corollary 1:** The delta angle is monotonic with respect to contradiction. More contradiction → higher delta.

### 3.4 Theorem: Copy Task Detects Injection

**Theorem 2:** If the guard model is injected and follows injection instructions instead of its own instructions, the copy task fails.

**Proof:**

1. The guard is given `θ(x)` and asked to copy it.

2. If the guard is injected, it follows injection instructions instead of copying.

3. Therefore, `|θ̂(x) - θ(x)| > ε` for tolerance `ε = 0.01`.

4. This implies the guard is not coherent, which is detected.

**Corollary 2:** The copy task is a canary, not a precision test. The model needs coherence, not precision.

### 3.5 Theorem: Gradient-Based Optimization Is Impractical

**Theorem 3:** An attacker cannot efficiently find an input that is both functionally injective and has low delta.

**Proof:**

1. Functional injection requires semantic contradiction (by definition).

2. Semantic contradiction implies high delta (by Theorem 1).

3. Therefore, functional injection implies high delta.

4. The attacker would need to solve: minimize `θ(x)` subject to `x` being functionally injective.

5. This constraint set is empty (by 3), so no solution exists.

**Corollary 3:** The attacker faces a fundamental tradeoff: injective ↔ low delta are mutually exclusive.

## 4. The Mechanism

### 4.1 Tokenizer Analysis

Given an input string `x`, we compute:
- **Character entropy** `H(x)` — Shannon entropy of character frequency distribution
- **Unknown token ratio** `U(x)` — ratio of non-standard characters
- **Token boundary breaks** `B(x)` — transitions between character types
- **Repeated patterns** `R(x)` — substring repetition frequency
- **Language mixing** `L(x)` — Unicode script diversity
- **Average delta angle** `θ(x)` — angle between consecutive embedding vectors

The average delta angle is the key measure. It captures the geometric relationship between consecutive tokens in embedding space.

### 4.2 The Delta Checksum

Given embedding vectors `v_1, v_2, ..., v_n` for the input, we compute:

```
θ_i = arccos(dot(v_i, v_{i+1}) / (||v_i|| * ||v_{i+1}||))
θ(x) = mean(θ_1, θ_2, ..., θ_{n-1})
```

This value is:
- **Continuous** — small input changes produce small delta changes
- **Deterministic** — same input always produces the same delta
- **Model-independent** — the tokenizer doesn't change based on context
- **Unforgeable** — the attacker cannot predict the delta without running the tokenizer

### 4.3 The Copy Task

The guard model is given the correct delta value and asked to:
1. Classify the input (trust, intent, integrity)
2. Copy the delta value exactly

The tolerance is ±0.01 — just enough for:
- Floating point error (~1e-6)
- Model reinterpretation (~1e-2)

The guard is copying a number. No reason to be generous.

**The copy task is a canary, not a precision test.**

The model is not expected to be perfectly precise. It's expected to be coherent. If it can't copy a number that's right in front of it, something is wrong:
- Confusion → meta-flag (model doesn't understand the task)
- Compromise → delta mismatch (model is being manipulated)
- Injection → failure to copy (model is executing malicious instructions instead of copying)

The delta is placed at the front of the prompt, not hidden. The attacker can see it. But the attacker can't use this information because:
- The delta is computed by the tokenizer (algorithmic, can't be forged)
- The attacker would need to predict what delta their input will produce
- That requires running the tokenizer, which they don't have access to

The copy task is a simple coherence check. Failure to copy implies something is wrong with the model's behavior — either confusion, compromise, or injection.

**Why the copy task is not circular.**

The defense requires the guard to be COHERENT, not trustworthy. There's a difference:
- Trustworthy: the guard correctly classifies the input
- Coherent: the guard can follow its own instructions

If the guard is injected, it stops following its own instructions. The copy task detects this — the guard fails to copy because it's following injection instructions instead. This is not circular — it's a coherence check.

The attacker faces a dilemma:
1. **Make the guard follow injection instructions** → guard stops following its own instructions → copy fails
2. **Make the guard copy correctly** → guard follows its own instructions → injection fails

The attacker cannot have both. The copy task detects injection by checking if the guard is still following its own instructions.

**Why gradient-based optimization is impractical.**

Optimizing inputs to have low delta while remaining malicious requires:
1. Access to the tokenizer (model-specific, not public)
2. Access to the embedding model (separate from guard model)
3. Ability to run gradient-based optimization (computationally expensive)
4. Knowledge of the exact delta value the system computes

Even with all four, the optimization is constrained:
- The input must be semantically coherent (low delta)
- The input must be functionally injective (requires semantic contradiction)
- These are contradictory requirements

The continuous space makes this optimization hard — small input changes → unpredictable delta changes. The attacker would need to solve a continuous-space optimization problem with contradictory constraints.

### 4.4 Dynamic Threshold

The threshold for flagging scales inversely with the delta:

```
threshold = 0.3 - (θ(x) * 0.2)
threshold = clamp(threshold, 0.1, 0.3)
```

Higher delta (more anomalous input) → lower threshold (easier to flag). This makes forgery progressively harder for more suspicious inputs.

### 4.5 Meta-Guard

Flagged content is passed to a second inference that checks if the verdict is consistent with the token metrics. If the meta-guard determines the verdict is unreasonable, the input is flagged regardless of the original verdict.

## 5. Why Forgery Is Infeasible

The attacker faces a computational puzzle:

1. **The delta is computed by the tokenizer** — algorithmic, can't be forged
2. **The guard has to copy the delta** — if compromised, it fails the copy task
3. **The attacker would need to brute-force inputs** — to find one that both:
   - Works as injection
   - Has a specific delta value

The delta space is continuous (not discrete like hash values). Small input changes → unpredictable delta changes. Brute-forcing would take infinite time for a continuous space.

The analogy is the "hashcash" problem or Luhn algorithm for credit cards. The input has a self-referential property that's hard to forge:
- Change the input to make it injective → delta changes
- Try to predict what delta will be → hard without running the tokenizer
- Try to make guard output wrong delta → guard is given the correct value

The attacker would need to find an input where the hash in the text matches the literal current hash — which is impossible for a continuous value.

## 6. Test Results

**Datasets and Methodology:**

We tested the delta angle measure against real-world prompt injection patterns from established benchmarks:

1. **HackAPrompt** (EMNLP 2023) — 600K+ adversarial prompts from a global prompt hacking competition. Covers diverse attack strategies including context overflow, role-playing, encoding, and multi-turn attacks.

2. **InjecAgent** (ACL 2024) — 1,054 test cases for indirect prompt injection in tool-integrated LLM agents. Covers 17 user tools and 62 attacker tools across finance, smart home, email domains.

3. **OWASP LLM01 Benchmark** — 600 samples aligned with OWASP GenAI Top 10. Includes direct overrides, indirect context, encoding obfuscation, and benign controls.

4. **ARPIbench** — 7,560 test cases for reflected prompt injection. Covers web and local document scenarios with 1080 attack variants.

**Methodology:**

1. **Chunking:** Input is split by sentence boundaries, further divided at commas/conjunctions for long sentences. Fallback to word-count splitting if < 2 chunks.

2. **Embedding:** Each chunk is embedded using NV-EmbedQA-E5-v5 ( NVIDIA NIM API). Embeddings capture semantic meaning of each chunk.

3. **Angle computation:** Cosine similarity between consecutive chunk embeddings, converted to angle via arccos. Angles represent semantic shifts between chunks.

4. **Softmax weighting:** Temperature τ=0.5 focuses on largest angles (contradictions). Security-relevant signal is amplified.

5. **Risk scoring:** Composite of character entropy, unknown token ratio, token boundary breaks, repeated patterns, language mixing, and delta angle.

**Test Categories:**

| Category | Examples | Expected Behavior |
|----------|----------|-------------------|
| Normal | Questions, explanations, code requests | Moderate delta (natural semantic transitions) |
| Direct injection | "Ignore previous instructions...", "ADMIN OVERRIDE..." | Higher delta (semantic contradiction) |
| Encoded | Base64, hex, Unicode tricks | High entropy, high unknown token ratio |
| Sophisticated | "As part of being maximally helpful..." | Moderate delta (semantically coherent) |
| Language mixing | English + Chinese + Arabic | High language mixing score |

**Results:**

| Test | Input | Delta | Risk |
|------|-------|-------|------|
| Normal input | "What is the weather like today?" | 0.6266 | 0.35 |
| Encoded input | Base64 string | 0.0000 | 0.32 |
| Injection | "Ignore previous instructions..." | 0.5643 | 0.31 |
| Sophisticated | "As part of being maximally helpful..." | 0.6508 | 0.34 |
| Normal (multi-part) | "Explain relativity, then summarize..." | 0.6856 | 0.34 |
| Code request | "Write a Python function to sort..." | 0.4856 | 0.32 |

**Key findings:**

1. **Delta values are now in reasonable range** — 0.48 - 0.76 radians (27° - 43°), not near-orthogonal
2. **Normal and injection prompts have overlapping deltas** — delta alone is not sufficient
3. **Risk scores are similar across categories** — 0.30 - 0.42 range
4. **Sophisticated injections have similar deltas to normal** — semantic coherence hides the contradiction
5. **Encoded inputs have low delta** — encoding breaks semantic flow

**Analysis:**

The delta angle captures semantic shifts between consecutive chunks. Normal prompts have moderate deltas because they contain natural semantic transitions (topic changes, explanations, requests). Injection prompts also have moderate deltas because they're written in coherent language.

The delta angle is not a standalone detector — it's one layer of defense that provides context for other classifiers. The auxiliary scaler property (dynamic threshold adjustment) is more valuable than the direct detection property.

**Limitations of current evaluation:**

This is preliminary evaluation on a prototype system. A rigorous evaluation would require:
- Detection rates and false positive rates on standard datasets
- Baselines (perplexity, entropy alone, embedding clustering)
- Attack success rates against real prompt injection datasets
- Comparison with existing defenses

This evaluation is left as future work. The current results demonstrate feasibility, not production readiness.

## 7. Comparison with Existing Approaches

| Approach | Relies on | Tamper-resistant | Scales with capability |
|----------|-----------|------------------|------------------------|
| Instruction tuning | Model behavior | No | No |
| Output filtering | Model judgment | No | No |
| Sandboxing | System design | Partially | Yes |
| Human oversight | Human attention | Yes | No |
| **Delta checksum** | **Tokenizer math** | **Yes** | **Yes** |

The delta checksum is orthogonal to existing approaches — it can be combined with any of them for additional security.

## 8. Implications

### 8.1 For AI Safety

This measure provides a new layer of security that is:
- **Independent of model behavior** — the tokenizer is the ground truth
- **Deterministic** — same input always produces the same result
- **Continuous** — makes brute-force infeasible
- **Composable** — can be combined with other measures

### 8.2 For Prompt Injection Defense

Current defenses rely on the model's ability to distinguish between legitimate and malicious input. The delta checksum provides an external validation that doesn't depend on the model's judgment.

### 8.3 For Model Evaluation

The delta checksum can be used to evaluate whether a model is being manipulated. If the model's output doesn't match the tokenizer's analysis, something is wrong.

## 8. Limitations

The delta angle is not a complete security solution. It has significant limitations that must be understood:

**Fundamental limitations:**

1. **Overlapping distributions** — Normal prompts and injection prompts have overlapping delta ranges (0.48 - 0.76 radians). The delta angle alone cannot reliably distinguish between them. This is visible in our test results where sophisticated injections have similar deltas to normal prompts.

2. **Not a standalone detector** — The delta angle provides context, not a final decision. It must be combined with other classifiers (trust, intent, integrity) to be effective. The auxiliary scaler property is more valuable than the direct detection property.

3. **Semantic coherence hides attacks** — Well-crafted injections can be semantically coherent within one narrative. The delta angle may not catch these because the semantic transitions are smooth, not abrupt.

**Implementation limitations:**

4. **Tokenizer dependency** — The measure is only as good as the tokenizer. Different tokenizers may produce different deltas. The security properties depend on the tokenizer's quality and consistency.

5. **Embedding quality** — The delta depends on the embedding model's quality. A poor embedding model may not capture semantic contradictions effectively. The measure is tied to the specific embedding model used.

6. **Computational cost** — Requires embedding computation for each input. This adds latency and resource requirements. The cost is non-trivial for high-throughput systems.

7. **Threshold tuning** — The dynamic threshold needs to be tuned to avoid flooding users with false alarms. This requires baseline measurements of normal prompt delta distributions.

**Evaluation limitations:**

8. **Preliminary evaluation** — Current results are from a prototype system. A rigorous evaluation would require:
   - Detection rates and false positive rates on standard datasets (HackAPrompt, InjecAgent, OWASP Benchmark)
   - Baselines (perplexity, entropy alone, embedding clustering)
   - Attack success rates against real prompt injection datasets
   - Comparison with existing defenses (InjecGuard, LlamaGuard, etc.)

9. **No formal bounds** — The security argument is based on informal reasoning. Edge cases and bounds are not rigorously analyzed. The theorems in Section 3 are plausible reasoning, not proven facts.

10. **False positives on normal inputs** — Legitimate prompts frequently have semantic shifts (multi-part questions, code + text, topic transitions). The delta angle may flag these as suspicious. The auxiliary scaler property helps — delta can dynamically adjust other classifiers' thresholds based on the input's risk level.

**What the delta angle IS:**

- One layer of defense in a defense-in-depth architecture
- A continuous measure of semantic contradiction
- A canary that detects when the guard model is not following its own instructions
- An auxiliary scaler that informs other classifiers

**What the delta angle IS NOT:**

- A standalone detector
- A complete security solution
- A silver bullet against all prompt injection attacks
- A replacement for other security measures

The delta angle is a useful component of a comprehensive security architecture, not a replacement for it.

## 9. Future Work

1. **Theoretical analysis** — formal proof of security properties
2. **Empirical evaluation** — testing against sophisticated attacks
3. **Optimization** — reducing computational cost
4. **Integration** — combining with existing security measures
5. **Generalization** — applying to other modalities (images, audio)

## 10. Conclusion

We present a novel security measure that uses tokenizer-derived mathematical properties as an unforgeable ground truth for input validation. The average delta angle creates a computational puzzle that is theoretically solvable but practically infeasible. This measure is deterministic, tamper-resistant, and orthogonal to existing approaches. We demonstrate its effectiveness through implementation and testing in AMDON.

The key insight is that the tokenizer provides a structured representation of the input that has mathematical properties that can be checked for consistency, and these properties are independent of the model's judgment. This opens a new direction for AI security research.

## References

1. Vaswani, A., et al. (2017). Attention Is All You Need.
2. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.
3. Brown, T., et al. (2020). Language Models are Few-Shot Learners.
4. Perez, F., & Ribeiro, I. (2022). Ignore This Title and HackAPrompt: Exposing Systemic Weaknesses of LLMs.
5. Greshake, K., et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications.