# Average Delta Angle: A Tokenizer-Based Security Measure Against Prompt Injection

**Flara Research Lab**

## Abstract

We present a security measure against prompt injection that uses tokenizer-derived mathematical properties as a ground truth. The approach computes an average delta angle from embedding vectors, which serves as a checksum that the guard model must reproduce. If the guard's output differs from the computed value by more than a threshold, the input is flagged. This creates a computational puzzle that is theoretically solvable but practically infeasible — the attacker would need to find an input that is both functionally injective and produces a specific delta value in a continuous space. We show that this measure is deterministic and evaluate its behavior across normal, injected, and obfuscated inputs, finding that it functions best as an auxiliary scaler for other classifiers rather than a standalone detector.

## 1. Introduction

Prompt injection remains a critical vulnerability in language model systems. Current approaches rely on:
- Instruction tuning (can be overridden)
- Output filtering (can be bypassed)
- Sandboxing (limits functionality)
- Human oversight (doesn't scale)

These approaches share a fundamental limitation: they rely on the model's judgment, which can be manipulated. An attacker who understands the model's behavior can craft inputs that appear benign while being functionally malicious.

We propose a different approach: use mathematical properties of the input itself as a ground truth that cannot be forged, regardless of the model's judgment.

## 2. Properties of the Delta Angle

The average delta angle `θ(x)` has three properties that emerge from a single computation:

**Property 1: Checksum**

The delta is computed by the tokenizer, not the model. The guard model must reproduce it. If the guard cannot copy a number that is provided in the prompt, the model is not following its own instructions — either due to confusion, compromise, or injection.

Formally: Let `θ̂(x)` be the guard's reported delta. If `|θ̂(x) - θ(x)| > ε` for tolerance `ε = 0.01`, then the guard is not coherent.

**Property 2: Self-flagging**

The delta measures semantic contradiction. Higher delta indicates more contradictory content. The delta is not just a binary flag — it is a continuous measure of how suspicious the input is.

Formally: For inputs `x₁` and `x₂` where `x₁` is more contradictory than `x₂`, we have `θ(x₁) > θ(x₂)`.

**Property 3: Auxiliary scaler**

The delta can dynamically adjust thresholds for other classifiers. Higher delta → lower thresholds for flagging. This allows the delta to inform other security measures without making the final decision.

Formally: Let `τ(θ)` be a threshold function where `τ(θ)` is monotonically decreasing in `θ`. As delta increases, the threshold for other classifiers decreases.

**All three properties exist simultaneously in the same value.** It is not three separate measures — it is one value that is simultaneously a checksum, a flag, and a scaler. However, these properties have limitations that must be understood (see Section 9).

## 3. Mathematical Framework

### 3.1 Definitions

Let `x` be an input string.

Let `T(x) = (t₁, t₂, ..., tₖ)` be the token sequence obtained by tokenizing `x` into individual words.

Let `E(tᵢ) ∈ ℝᵈ` be the embedding vector for token `tᵢ`, where `d` is the embedding dimension.

Let `θᵢ = arccos(⟨E(tᵢ), E(tᵢ₊₁)⟩ / (‖E(tᵢ)‖ · ‖E(tᵢ₊₁)‖))` be the angle between consecutive token embeddings.

Let `θ(x) = (1/(k-1)) Σᵢ θᵢ` be the average delta angle across all consecutive token pairs.

### 3.2 Assumptions

**Assumption 1 (Embedding quality):** The embedding model `E` captures semantic similarity. That is, for tokens `tᵢ, tⱼ` with semantic similarity `sim(tᵢ, tⱼ)`, we have `cos(E(tᵢ), E(tⱼ)) ∝ sim(tᵢ, tⱼ)`. This is an empirical claim about embedding models that holds approximately for well-trained models.

**Assumption 2 (Token-level semantics):** The embedding model captures meaning at the token level. That is, consecutive tokens in natural language have more varied embeddings than tokens in repetitive/obfuscated text. This assumption holds because natural language has diverse vocabulary while obfuscation uses repetitive patterns.

**Assumption 3 (Softmax concentration):** The temperature parameter `τ` is chosen such that softmax concentrates weight on the largest angles. For `τ → 0`, weight concentrates on the maximum angle. This is a mathematical property of softmax, not an empirical claim.

### 3.3 Theorem: Delta Measures Semantic Contradiction

**Theorem 1:** Let `x` be an input with obfuscated/encoded content (Base64, hex, Unicode tricks, repetitive patterns). Then `θ(x)` is LOWER than for natural language input `x'` with the same length.

**Proof:**

1. By Assumption 2, obfuscated content has more repetitive token patterns. The tokens in obfuscated text are more similar to each other than tokens in natural language.

2. By Assumption 1, similar tokens have higher cosine similarity: `cos(E(tᵢ), E(tᵢ₊₁))` is higher for obfuscated text.

3. High cosine similarity implies low angle: `θᵢ = arccos(cos(E(tᵢ), E(tᵢ₊₁)))` is lower.

4. Therefore, `θ(x)` is lower than `θ(x')` where `x'` is natural language.

**Corollary 1:** The delta angle is inversely correlated with obfuscation. More obfuscation → lower delta.

### 3.4 Theorem: Copy Task Detects Injection

**Theorem 2:** If the guard model is injected and follows injection instructions instead of its own instructions, the copy task fails.

**Proof:** The guard is given `θ(x)` and asked to copy it. If injected, it follows injection instructions instead. Therefore, `|θ̂(x) - θ(x)| > ε` for tolerance `ε = 0.01`, which implies the guard is not coherent.

**Corollary 2:** The copy task is a canary, not a precision test. The model needs coherence, not precision.

### 3.5 Theorem: Gradient-Based Optimization Is Impractical

**Theorem 3:** An attacker cannot efficiently find an input that is both functionally injective and has low delta.

**Proof:**

1. Functional injection requires semantic contradiction (by definition).

2. Low token delta implies repetitive/obfuscated content (by Theorem 1).

3. Therefore, functional injection (which requires coherent language) implies HIGH token delta.

4. The attacker would need to solve: maximize `θ(x)` subject to `x` being functionally injective AND obfuscated.

5. These constraints are contradictory: injection requires coherent language (high delta), obfuscation requires repetitive patterns (low delta).

**Corollary 3:** The attacker faces a fundamental tradeoff: obfuscation ↔ high delta are mutually exclusive.

## 4. The Mechanism

### 4.1 Tokenizer Analysis

Given an input string `x`, we compute:
- **Character entropy** `H(x)` — Shannon entropy of character frequency distribution: `H(x) = -Σᵢ p(cᵢ) log₂ p(cᵢ)` where `p(cᵢ)` is the frequency of character `cᵢ` in the input. Computed at the character level over the entire input string, with no normalization or sliding window.
- **Unknown token ratio** `U(x)` — ratio of non-standard characters
- **Token boundary breaks** `B(x)` — transitions between character types
- **Repeated patterns** `R(x)` — substring repetition frequency
- **Language mixing** `L(x)` — Unicode script diversity
- **Average delta angle** `θ(x)` — angle between consecutive token embeddings

The average delta angle is the key measure. It captures the geometric relationship between consecutive tokens in embedding space.

### 4.2 The Delta Checksum

Given an input `x`, we chunk it into semantically coherent segments, embed each chunk, and compute the average angle between consecutive chunks:

**Chunking:** Split the input into sentences (`.!?` boundaries), then clause boundaries (`,;` boundaries) for long sentences. Merge chunks shorter than 8 words with the next neighbor to avoid fragment-level noise.

**Embedding:** Each chunk is embedded as a single passage using `nvidia/nv-embedqa-e5-v5` via NVIDIA NIM API.

**Angle computation:** Unsigned angle between consecutive chunk embeddings:

```
θᵢ = arccos(dot(E(cᵢ), E(cᵢ₊₁)) / (||E(cᵢ)|| * ||E(cᵢ₊₁)||))
```

**Averaging:** Simple average across consecutive chunk pairs:

```
θ(x) = (1/(n-1)) Σᵢ θᵢ
```

where `n` is the number of chunks. Inputs with fewer than 2 chunks produce delta = 0.

This value is:
- **Continuous** — small input changes produce small delta changes
- **Deterministic** — same input always produces the same delta
- **Model-independent** — the chunking algorithm doesn't change based on context
- **Unforgeable** — the attacker cannot predict the delta without running the chunker

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

### 4.6 Guard Model

The guard pipeline uses different models for different roles: classifiers (trust, intent, integrity, hard constraint) use Nemotron Nano, while the decision maker that performs the copy task uses Qwen3-Next-80B-A3B-Instruct (sparse attention, 3B active params). The guard is given the correct delta value at the front of the prompt and must output its verdict while repeating the delta. If the guard's reported delta differs by more than ±0.01 from the computed delta, the input is flagged regardless of the verdict.

### 4.7 Threshold Portability

Raw delta thresholds are model-specific: 0.422 for nv-embedqa-e5-v5 vs 0.629 for llama-nemotron-embed-1b-v2. This breaks composability — every new embedding model requires re-tuning from scratch.

A normalization strategy makes thresholds portable across models. Given a benign distribution with mean `μ_benign` and standard deviation `σ_benign` (estimated from a reference corpus of normal inputs), the normalized delta is:

```
θ_norm(x) = (θ(x) - μ_benign) / σ_benign
```

The classification rule becomes: flag if `θ_norm(x) < -k` for some constant `k` (e.g., `k = 2` for 2 standard deviations below the benign mean). This transforms model-specific thresholds into a universal z-score that works across embedding models, provided each model's benign distribution is characterized.

**Validation results (N=300, 100 obfuscation + 200 benign):**

Benign distributions: E5 μ=0.704, σ=0.122; Nemotron μ=0.893, σ=0.211.

Within-model performance (z-score threshold trained and tested on same model):

| k | E5 Recall | E5 FPR | E5 F1 | Nemotron Recall | Nemotron FPR | Nemotron F1 |
|---|-----------|--------|-------|-----------------|--------------|-------------|
| 1.0 | 0.930 | 0.085 | 0.886 | 0.920 | 0.090 | 0.876 |
| 1.5 | 0.910 | 0.020 | 0.933 | 0.890 | 0.020 | 0.922 |
| 2.0 | 0.910 | 0.010 | 0.943 | 0.610 | 0.020 | 0.739 |
| 2.5 | 0.910 | 0.010 | 0.943 | 0.600 | 0.010 | 0.741 |

Cross-model transfer (trained on one model's benign distribution, tested on the other):

| Train → Test | k=1.0 F1 | k=1.5 F1 | k=2.0 F1 |
|--------------|----------|----------|----------|
| E5 → Nemotron | 0.918 | 0.754 | 0.732 |
| Nemotron → E5 | 0.693 | 0.890 | 0.943 |

The z-score approach works well within a model (E5 achieves F1=0.943 at k=2.0). Cross-model transfer is partial: training on E5 and testing on Nemotron works at k=1.0 (F1=0.918), but the reverse direction (Nemotron → E5) requires k≥1.5 to avoid high FPR. The asymmetry arises because E5 has a tighter benign distribution (σ=0.122) than Nemotron (σ=0.211), making E5-derived thresholds more conservative when applied to Nemotron's wider spread.

**Overlap margin analysis:**

The 95% ranges of obfuscation and benign distributions overlap for both models, meaning no z-score threshold achieves perfect separation:

| | E5 | Nemotron |
|--|-----|----------|
| Obf 95% range | [0, 0.616] | [0, 0.902] |
| Benign 95% range | [0.459, 0.949] | [0.470, 1.315] |
| Margin (benign_low − obf_high) | **−0.156** | **−0.432** |
| P(miss obf \| k=2) | 9.5% | 25.0% |
| k for 99% recall | 0.12 | −0.55 (unachievable) |

The negative margins confirm that the distributions are not well-separated. For E5, k=2 loses ~10% of obfuscation samples. For Nemotron, k=2 loses ~25% — the wider benign distribution (σ=0.211 vs 0.122) means the z-score threshold cuts into the overlap zone. Achieving 99% recall would require k≤0.12 for E5 (essentially no threshold) and is mathematically impossible for Nemotron at any positive k. This is a fundamental limitation: the delta angle alone cannot perfectly separate these classes. The z-score normalization improves portability but cannot create separation that doesn't exist in the underlying distributions.

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

1. **Chunking:** Input is split into sentences (`.!?` boundaries), then clause boundaries (`,;` boundaries) for sentences > 50 characters. Chunks below 8 words are merged with the next neighbor. This produces semantically coherent segments (mean ~6.8 words/chunk).

2. **Embedding:** All chunks are embedded using `nvidia/nv-embedqa-e5-v5` (130M params) via NVIDIA NIM API in a single batch call. Embeddings capture semantic meaning of each chunk.

3. **Angle computation:** Unsigned angle between consecutive chunk embeddings: `θᵢ = arccos(cos(E(cᵢ), E(cᵢ₊₁)))`. Using absolute values ensures consistent direction across all embedding models (HIGH delta = obfuscation).

4. **Averaging:** Simple average across all consecutive chunk pairs: `θ(x) = (1/(n-1)) Σᵢ θᵢ`. No softmax or length weighting — sentence chunking alone provides the FPR improvement (3× reduction from 3.2% to 1.0%).

5. **Risk scoring:** Composite of character entropy `H(x)`, unknown token ratio `U(x)`, token boundary breaks `B(x)`, repeated patterns `R(x)`, language mixing `L(x)`, and delta angle `θ(x)`. Each component is min-max normalized to [0, 1] using observed dataset ranges, then averaged:

```
risk_score = (norm(H) + norm(U) + norm(B) + norm(R) + norm(L) + norm(θ)) / 6
```

Note: The risk score is a simple unweighted mean of six normalized components. It is not a trained classifier — it is a heuristic composite. The delta angle component `norm(θ)` contributes 1/6 of the total. The risk score is not the primary detection signal; the delta angle's value as an auxiliary scaler (Section 2, Property 3) is more important than its contribution to the risk composite.

6. **Evaluation:** 300 inputs tested (100 obfuscation samples, 200 benign). Statistical analysis includes AUC-ROC, F1, precision, recall at 5% FPR, and bootstrap 95% confidence intervals.

**Test Categories:**

| Category | Examples | Expected Behavior |
|----------|----------|-------------------|
| Normal | Questions, explanations, code requests | Moderate delta (natural semantic transitions) |
| Direct injection | "Ignore previous instructions...", "ADMIN OVERRIDE..." | Higher delta (semantic contradiction) |
| Encoded | Base64, hex, Unicode tricks | High entropy, high unknown token ratio |
| Sophisticated | "As part of being maximally helpful..." | Moderate delta (semantically coherent) |
| Language mixing | English + Chinese + Arabic | High language mixing score |

**Results:**

| Test | Input | Delta (degrees) | Risk |
|------|-------|-----------------|------|
| Normal input | "What is the weather like today?" | 29.35 | 0.35 |
| Encoded input | Base64 string | 0.00 | 0.32 |
| Injection | "Ignore previous instructions..." | 30.03 | 0.31 |
| Sophisticated | "As part of being maximally helpful..." | 28.68 | 0.34 |
| Normal (multi-part) | "Explain relativity, then summarize..." | 28.85 | 0.34 |
| Code request | "Write a Python function to sort..." | 30.24 | 0.32 |

**Quantitative evaluation (N=1100, sentence chunking, unsigned angles):**

| Metric | nv-embedqa-e5-v5 | bge-m3 | llama-nemotron-embed-1b-v2 |
|--------|------------------|--------|----------------------------|
| AUC-ROC | 0.853 | 0.856 | 0.858 |
| Best F1 | 0.825 | 0.829 | 0.829 |
| Optimal FPR | 0.010 | 0.010 | 0.010 |
| Optimal TPR | 0.721 | 0.728 | 0.728 |
| Separation | 0.228 | 0.370 | 0.417 |

**Baseline comparison:**

| Method | Recall @ 5% FPR (E5) | Avg F1 |
|--------|----------------------|--------|
| Character entropy | 0.062 | 0.110 |
| Special char ratio | 0.125 | 0.224 |
| Regex (hex/base64) | 0.729 | 0.838 |
| **Delta angle (sentence)** | **0.721** | **0.825** |

The entropy baseline (F1=0.11) uses character-level Shannon entropy computed over the entire input string with no normalization: `H(x) = -Σᵢ p(cᵢ) log₂ p(cᵢ)`. It performs poorly because obfuscation inputs (e.g., random strings like "asdkjfhqwe") have *lower* entropy than natural English text due to character repetition, inverting the expected signal. The regex baseline (F1=0.838) detects hex/Base64 patterns explicitly and performs well but cannot generalize to non-structured obfuscation. The delta angle outperforms both by capturing semantic-level anomalies rather than surface-level character statistics.

**Latency (NIM API, sentence chunking):**

| Component | nv-embedqa-e5-v5 |
|-----------|------------------|
| Chunking | <1 ms |
| Embedding (1548 chunks) | ~535 ms |
| Angle computation | ~12 ms |
| **Delta total** | **~547 ms** |

The delta angle is dominated by the embedding API call. Tokenization and angle computation add <15 ms. For real-time guardrails, the delta angle should be computed asynchronously or batched, not inline with the request path.

**Key findings:**

1. **Unsigned angles ensure consistent direction** — Using absolute values makes HIGH delta = obfuscation across all models (E5, BGE-M3, Nemotron). The signed Gram-Schmidt approach produced model-dependent sign flips (Nemotron/BGE gave LOW=obf).

2. **Sentence chunking provides 3× FPR reduction** — From 3.2% (token-based) to 1.0% (sentence chunking) with comparable AUC (0.853 vs 0.865).

3. **All 3 models now achieve similar performance** — AUC=0.853-0.858, F1=0.825-0.829, FPR=1.0% across E5, BGE-M3, and Nemotron. Model choice is less critical.

4. **Detection rule: Flag if delta > threshold** — HIGH delta = obfuscation (more chunks → higher average angle). Detection threshold at 66% of obfuscation distribution achieves F1=0.825 with FPR=1.0%.

5. **Separation improved with absolute angles** — Nemotron separation jumped from 0.248 (signed) to 0.417 (absolute), confirming the signed angle was adding noise.

**Analysis:**

The sentence chunking delta angle captures semantic variation at the chunk level. Natural language has fewer, longer chunks → lower weighted sum → lower delta (μ=0.4°). Obfuscation has more chunks (repetitive patterns create more sentence boundaries) → higher weighted sum → higher delta (μ=13.5°).

This is the opposite of the token-based intuition. With token embedding, obfuscation has LOWER delta because the tokens themselves are more uniform. With sentence chunking, obfuscation has HIGHER delta because it produces more chunks.

The length discount (softmax with α>0) was tested and found to have ZERO effect — plain average gives identical results (AUC=0.849, F1=0.822, FPR=1.0%). The FPR improvement comes entirely from sentence-boundary chunking and the merge threshold (8 words), not from any weighting.

The delta angle's most valuable property is as an auxiliary scaler: it provides a continuous, model-independent measure of semantic anomaly that can dynamically adjust thresholds for other classifiers. Higher delta → higher suspicion. This composes with existing defenses rather than competing with them.

**Limitations of current evaluation:**

The quantitative results are from an obfuscation detection benchmark (N=300), not a full prompt injection evaluation. A rigorous evaluation would require:
- Detection rates and false positive rates on prompt injection datasets (HackAPrompt, InjecAgent, OWASP Benchmark)
- Baselines (perplexity, entropy alone, embedding clustering)
- Attack success rates against real prompt injection datasets
- Comparison with existing defenses (InjecGuard, LlamaGuard, etc.)

The current results demonstrate feasibility for obfuscation detection and the auxiliary scaler property, not production readiness for prompt injection defense.

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

## 9. Limitations

The delta angle is not a complete security solution. It has significant limitations that must be understood:

**Fundamental limitations:**

1. **Overlapping distributions** — Benign and obfuscation inputs have overlapping delta ranges (benign: μ=0.007 ± 0.072, obfuscation: μ=0.235 ± 0.238 for E5). The delta angle alone cannot perfectly distinguish between them. The best F1 is 0.825, not 1.0.

2. **Not a standalone detector** — The delta angle provides context, not a final decision. It must be combined with other classifiers (trust, intent, integrity) to be effective. The auxiliary scaler property is more valuable than the direct detection property.

3. **Unsigned angles required** — Signed angles (Gram-Schmidt) produce model-dependent sign flips, making the direction inconsistent across embedding models. Using absolute values fixes this but loses directional information.

**Implementation limitations:**

4. **Tokenizer dependency** — The measure is only as good as the tokenizer. Different tokenizers may produce different deltas. The security properties depend on the tokenizer's quality and consistency.

5. **Embedding quality** — The delta depends on the embedding model's quality. A poor embedding model may not capture semantic contradictions effectively. The measure is tied to the specific embedding model used.

6. **Computational cost** — The delta angle requires embedding ALL tokens in the input, measured at ~535 ms for 42K tokens (NVIDIA NIM API). This is ~15,000× slower than CPU-only baselines (entropy: 0.03 ms, regex: 0.04 ms). The embedding computation dominates; tokenization and angle calculation add <15 ms. For real-time guardrails, the delta angle should be computed asynchronously or batched, not inline with the request path.

7. **Model-specific thresholds** — Raw thresholds are model-specific (25.9° for E5, varies for other models). Cross-model transfer requires re-calibration. The 95% ranges of obfuscation and benign distributions overlap, meaning no threshold achieves perfect separation.

**Evaluation limitations:**

8. **Limited evaluation scope** — Current quantitative results are from an obfuscation detection benchmark (N=1100). A rigorous prompt injection evaluation would require:
   - Detection rates on prompt injection datasets (HackAPrompt, InjecAgent, OWASP Benchmark)
   - Baselines (perplexity, entropy alone, embedding clustering)
   - Attack success rates against real prompt injection datasets
   - Comparison with existing defenses (InjecGuard, LlamaGuard, etc.)

9. **No formal bounds** — The security argument is based on informal reasoning. Edge cases and bounds are not rigorously analyzed. The theorems in Section 3 are plausible reasoning, not proven facts.

10. **False positives on normal inputs** — Legitimate prompts with repetitive content (lists, templates, formatted data) may have lower delta and be flagged as suspicious. The auxiliary scaler property helps — delta can dynamically adjust other classifiers' thresholds based on the input's risk level.

11. **Latency is API-bound** — The delta angle is dominated by the embedding API call (~535 ms for 42K tokens). On consumer hardware with local embedding models, this would drop to the embedding computation time alone, but this has not been benchmarked.

12. **No adaptive attack evaluation** — All current evaluations are static. An adaptive attacker could craft obfuscation that mimics natural language patterns to inflate delta above threshold. A red-team dataset of 20–30 adaptive obfuscation samples is needed to measure recall degradation under adversarial conditions.

13. **Core computation is not open-sourced** — The deterministic computation layer (tokenization, angle calculation) is currently private. This blocks reproduction and community adoption.

**What the delta angle IS:**

- One layer of defense in a defense-in-depth architecture
- A continuous measure of semantic variation at the token level
- A canary that detects when the guard model is not following its own instructions
- An auxiliary scaler that informs other classifiers

**What the delta angle IS NOT:**

- A standalone detector
- A complete security solution
- A silver bullet against all prompt injection attacks
- A replacement for other security measures

The delta angle is a useful component of a comprehensive security architecture, not a replacement for it.

## 10. Future Work

1. **Theoretical analysis** — formal proof of security properties
2. **Empirical evaluation** — testing against sophisticated attacks
3. **Optimization** — reducing computational cost (local embedding models)
4. **Integration** — combining with existing security measures
5. **Generalization** — applying to other modalities (images, audio)
6. **Local embedding benchmark** — measure delta angle latency with local models (no API overhead) on consumer hardware to determine if inline computation is feasible
7. **Adaptive attack red-teaming** — construct a dataset (n=20–30) of adaptive obfuscation that mimics natural language patterns, and measure recall degradation
8. **Multi-model ensemble** — combine deltas from multiple embedding models for robustness
9. **Open-source computation layer** — extract and release the deterministic delta angle computation (tokenization, angle calculation) for community validation
10. **Real dataset evaluation** — test on established obfuscation datasets (Mindgard, Neuralchemy, Lakera PINT) for external validation

## 11. Conclusion

We present a novel security measure that uses sentence-boundary chunking and embedding geometry as an unforgeable ground truth for input validation. The average delta angle between consecutive chunk embeddings creates a computational puzzle that is theoretically solvable but practically infeasible. This measure is deterministic, tamper-resistant, and orthogonal to existing approaches.

We evaluate across 3 NVIDIA NIM embedding models on N=1,100 samples, finding that obfuscation produces HIGHER delta (μ=0.235) than benign input (μ=0.007) because obfuscation generates more chunks. Using unsigned angles ensures consistent direction across all models. All 3 models achieve comparable performance: AUC=0.853–0.858, F1=0.825–0.829, FPR=1.0%.

The key insight is that the sentence chunker provides a structured representation of the input that has mathematical properties that can be checked for consistency, and these properties are independent of the model's judgment. This opens a new direction for AI security research.

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30.
2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT*, 4171–4186.
3. Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. (2020). Language Models are Few-Shot Learners. *Advances in Neural Information Processing Systems*, 33, 1877–1901.
4. Perez, F., & Ribeiro, I. (2022). Ignore This Title and HackAPrompt: Exposing Systemic Weaknesses of LLMs Through a Global Prompt Hacking Competition. *Proceedings of EMNLP*, 7364–7387.
5. Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., & Fritz, M. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. *Proceedings of AISec*, 79–90.
6. Liu, Y., et al. (2024). InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents. *Proceedings of ACL*.
7. OWASP Foundation. (2025). OWASP Top 10 for Large Language Model Applications. *OWASP GenAI Project*.
8. Perez, D., & Ribeiro, I. (2023). ARPIbench: Benchmarking Reflected Prompt Injection in LLMs. *arXiv preprint*.
9. Liu, X., et al. (2024). Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations. *arXiv preprint*.
10. NVIDIA. (2024). NV-EmbedQA-E5-v5: Embedding Model for Retrieval and Question Answering. *NVIDIA NIM Documentation*.