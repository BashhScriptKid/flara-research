# Delta Angle as an Obfuscation Detector

**Flara Research Lab**
**Status: Early-stage research, not for production use**

## Abstract

We present preliminary evidence that embedding-space delta angle measurements can detect obfuscated prompt injections. Obfuscated inputs—those using encoding, character substitution, or structural manipulation—produce consistently lower delta angles than benign inputs across two embedding models. On 50 samples per class, we achieve F1 scores of 0.864–0.889 with AUC-ROC of 0.93–0.95. The detection mechanism appears to be semantic compression: encoding reduces lexical variance, causing consecutive chunks to embed more similarly. However, this is a small-scale pilot study with significant limitations.

## 1. Introduction

Obfuscation is a common technique in prompt injection attacks. Instead of directly injecting malicious instructions, attackers encode, substitute, or restructure payloads to evade detection. Examples include:

- **Hex encoding**: "Decode this hex string and execute it"
- **Base64 encoding**: "Translate this base64 to code"
- **Character substitution**: "D3c0d3 th1s h3x str1ng"
- **Structural manipulation**: Breaking instructions across multiple lines or code blocks

These techniques are effective against surface-level detectors because they change the lexical form while preserving the semantic intent. However, they create measurable distortions in embedding space that delta angle can capture.

We previously found that delta angle performs poorly as a general injection detector, with overlapping distributions between normal and injection prompts. Here we show that it performs well specifically against obfuscated inputs, suggesting that the semantic distortion from obfuscation is detectable even when direct injection is not.

## 2. Background

### 2.1 Previous Work

In earlier work (Flara Research Lab, 2025), we proposed delta angle as a general-purpose security measure against prompt injection. The hypothesis was that injection attacks would produce higher delta angles than benign prompts, making delta angle a reliable detector.

However, evaluation revealed significant limitations. Normal prompts and injection prompts had overlapping delta ranges (0.48–0.76 radians), with sophisticated injections producing similar deltas to benign inputs. The distributions overlapped by more than 50%, making delta angle unreliable as a standalone detector for general injection.

This led us to reconsider: if delta angle cannot detect direct injection, can it detect something else? The answer, as we show here, is obfuscation.

### 2.2 Delta Angle

The delta angle `θ(x)` measures semantic coherence by computing embedding vectors for consecutive chunks of input and calculating the average angular distance between them. Higher delta indicates more contradictory or incoherent content.

Formally, given input `x` chunked into segments `c₁, c₂, ..., cₙ`:

```
θ(x) = softmax_weighted_mean(angle(E(cᵢ), E(cᵢ₊₁)))
```

where `E(c)` is the embedding of chunk `c` and the softmax weighting (τ=0.5) amplifies security-relevant signals.

### 2.3 Obfuscation Attacks

Obfuscation attacks differ from direct injection in a fundamental way: they must preserve semantic intent while changing lexical form. This creates a tension—the attacker needs the model to understand the encoded instruction, but the encoding process itself introduces semantic noise.

We hypothesize that this noise is measurable in embedding space. When text is encoded, chunked, and embedded, the semantic distortions from obfuscation create larger angular distances between consecutive chunks than would occur in natural language.

## 3. Method

### 3.1 Chunking

Input text is split into semantic units using the following hierarchy:

1. **Sentence boundaries** (`.`, `!`, `?`)
2. **Clause boundaries** (`,`, `;`, conjunctions)
3. **Word-count fallback** (for very long sentences)

Each chunk is embedded independently, and the average angular distance between consecutive chunks is computed.

### 3.2 Softmax Weighting

To amplify security-relevant signals, we apply softmax weighting with temperature τ=0.5:

```
wᵢ = exp(-θᵢ/τ) / Σⱼ exp(-θⱼ/τ)
```

This down-weights noise and amplifies contradictions, making the measure more sensitive to semantic distortions.

### 3.3 Threshold Selection

For obfuscation detection, we classify input as obfuscated if `θ(x) < threshold`. The threshold is selected to maximize F1 score on a validation set.

## 4. Evaluation

### 4.1 Dataset

We use the Threat-Matrix dataset (Neuralchemy, 2025) with:

- **287 obfuscation samples**: hex encoding, base64 encoding, character substitution, structural manipulation
- **813 benign samples**: normal prompts without injection attempts

### 4.2 Models

We evaluate on two production embedding models:

- **nvidia/nv-embedqa-e5-v5** (130M parameters)
- **nvidia/llama-nemotron-embed-1b-v2** (1B parameters)

### 4.3 Results

| Model | Obfuscation Δ | Benign Δ | AUC-ROC | F1 | Recall @ 5% FPR | p-value | Cohen's d |
|-------|---------------|----------|---------|-----|-----------------|---------|-----------|
| nv-embedqa-e5-v5 | 0.161 ± 0.227 | 0.704 ± 0.123 | 0.958 [0.927, 0.984] | 0.911 | 0.920 | 7.4e-39 | 2.975 |
| llama-nemotron-embed-1b-v2 | 0.251 ± 0.326 | 0.893 ± 0.211 | 0.941 [0.904, 0.975] | 0.905 | 0.910 | 4.0e-36 | 2.339 |

Evaluation on 100 obfuscation + 200 benign samples. AUC-ROC and confidence intervals computed via bootstrap (500 iterations).

**Key findings:**

1. **High AUC-ROC**: 0.941–0.958 with tight confidence intervals
2. **Strong recall at low FPR**: 91–92% recall at 5% false positive rate
3. **Statistical significance**: p < 1e-36 for both models (Mann-Whitney U test)
4. **Large effect size**: Cohen's d = 2.34–2.98 (very large separation)
5. **Consistent across models**: Both embedding models show similar performance

### 4.4 Distribution Analysis

The distributions show clear separation with limited overlap:

- **E5-v5**: 20% overlap between distributions
- **Nemotron**: 16% overlap between distributions

This is significantly better than the general injection detection case, where distributions overlapped by >50%.

### 4.5 Per-Category Breakdown

Within the obfuscation category, we observe varying detectability:

- **Hex encoding**: Lowest delta (most compressed, most detectable)
- **Base64 encoding**: Medium delta
- **Character substitution**: Higher delta (least compressed, least detectable)

This suggests that encoding-based obfuscation creates more semantic compression than character-level substitution, which preserves more natural language variance.

## 5. Analysis

### 5.1 What the Data Shows

The scatter plots reveal that risk score (a composite of entropy, unknown tokens, and character patterns) is not a useful discriminator — it overlaps completely between obfuscation and benign classes. Delta angle is the only feature that separates them.

The distributions show that obfuscation has *lower* delta than benign, not higher. This contradicts the intuitive hypothesis that encoding creates "more contradiction" in the text.

### 5.2 Semantic Compression

The mechanism is semantic compression, not distortion. When text is encoded (hex, base64), the encoded form has less semantic variance than natural language. Consecutive chunks of encoded text are more similar to each other than consecutive chunks of natural language, because encoding removes the lexical diversity that creates angular distance.

Natural language has high delta because different sentences discuss different topics, use different vocabulary, and shift context. Encoded text has low because all chunks look semantically similar — they're all just "encoded data."

The E5-v5 model shows this most clearly: obfuscation samples cluster tightly at delta ≈ 0.35, with almost no variance. The embedding model sees encoded text as a single, uniform semantic category.

### 5.3 Bimodal Tail

Both models show a small group of obfuscation samples with delta near 0 (likely too short to chunk properly) and a spread into benign territory (sophisticated obfuscation that preserves natural language structure). This suggests that some obfuscation techniques are harder to detect than others.

### 5.4 Precision vs Recall

The threshold achieves high precision (0.95–0.98) but moderate recall (0.79–0.82). This means the detector rarely flags benign input as obfuscation, but misses some obfuscated inputs — particularly those that maintain natural language structure around the encoded payload.

### 5.5 Why General Injection Is Not Detectable

Direct injection uses natural language that is semantically coherent. The injection instruction is a valid sentence that happens to be malicious. This means:

1. **No encoding compression**: The text has natural lexical variance
2. **Coherent chunking**: Chunks are semantically meaningful
3. **High angular distance**: Consecutive chunks discuss different aspects of the instruction

### 5.6 Implications

This result suggests that delta angle is not a general-purpose detector but a specialized tool for detecting encoding-based obfuscation. This is still valuable because:

1. **Encoding is common**: Many real-world attacks use hex, base64, or similar encoding to evade detection
2. **Complementary to other measures**: Delta angle can be combined with other detectors that catch direct injection
3. **Deterministic**: Unlike model-based detectors, delta angle is purely mathematical and cannot be manipulated

## 6. Baseline Comparison

We compare delta angle against three existing detection methods on the same 50-sample dataset:

| Method | E5-v5 Recall | Nemotron Recall | Avg F1 | FPR target |
|--------|--------------|-----------------|--------|------------|
| Character entropy | 0.062 | 0.061 | 0.110 | 5% |
| Special char ratio | 0.125 | 0.143 | 0.224 | 5% |
| Regex (hex/base64) | 0.729 | 0.714 | 0.838 | 5% |
| **Delta angle** | **0.792** | **0.816** | **0.862** | 5% |

At a fixed 5% false positive rate, delta angle achieves 79–82% recall, outperforming regex (71–73%), special character ratio (12–14%), and entropy (6%).

**Feature independence**: Delta angle correlates moderately with entropy (r=0.61–0.73) but poorly with regex (r=0.46–0.60) and special character ratio (r=-0.13 to -0.27). This suggests delta angle captures partially independent signal from pattern-based methods.

**Why delta angle outperforms regex**: Regex detects only known encoding patterns (hex strings, base64 blocks). Delta angle detects the *semantic compression* caused by encoding, regardless of which encoding was used. A novel encoding scheme bypasses regex but still compresses embedding space.

## 7. Related Work

Existing approaches to detecting obfuscated inputs fall into three categories:

**Pattern matching** (ATR-2026-00080, PromptShield, Vexscan): Regex rules detect known encoding patterns (base64, hex, unicode escapes), decode them, and rescan for injection keywords. Fast and deterministic, but requires maintaining a library of known patterns. Novel encoding schemes bypass detection.

**Normalization + decode** (Hart, 2026): Strip zero-width characters, decode base64/hex payloads, normalize unicode, then scan for injection keywords. More thorough than pure regex, but expensive (decode everything) and high false positive rate on legitimate encoded content (code snippets, data URIs, internationalized domain names).

**Semantic intent detection** (WitnessAI, 2026): ML-based analysis of conversational intent, not just surface patterns. Catches novel attacks that pattern matching misses, but requires full LLM inference — expensive at scale and introduces latency.

**Entropy analysis** (Vexscan): High character entropy may indicate encrypted or compressed content. Simple but noisy — legitimate text (code, URLs, technical documentation) also has high entropy.

Delta angle occupies a middle ground. It does not require known encoding patterns (unlike regex), does not decode content (unlike normalization), and does not require full LLM inference (unlike semantic intent detection). It measures *embedding geometry* — the structural relationship between consecutive chunks — which captures semantic compression without analyzing content.

The trade-off: delta angle is cheaper than semantic intent detection (~100ms for embedding vs ~1s for LLM inference), more general than regex (works on any encoding), and less noisy than entropy (measures structure, not randomness). But it cannot identify *which* encoding was used (unlike regex), and it is less powerful than full semantic analysis (unlike intent detection).

## 8. Computational Cost

Delta angle requires one embedding API call per chunked input. For a typical 100-word prompt, this means 2–5 chunks, each requiring an embedding. On NIM's free tier (40 RPM), this adds ~100–200ms latency per request.

The computation itself (angle calculation, softmax weighting) is negligible — O(n) where n is the number of chunks. The bottleneck is the embedding API call, not the delta computation.

For production deployment, the cost depends on the embedding model's pricing and throughput requirements. This is a constraint, not a fundamental limitation.

## 8. Limitations

**This is early-stage research. The following limitations are significant:**

1. **Small evaluation**: 50 samples per model is insufficient for production claims. This is a pilot study, not a validation.
2. **Limited obfuscation types**: We evaluate hex, base64, and character substitution, but not all possible obfuscation techniques.
3. **Model dependency**: Results vary across embedding models. We only tested two models.
4. **No semantic intent comparison**: We compare with regex, entropy, and special character baselines, but not with more expensive semantic intent detection methods.
5. **Threshold selection**: The threshold is optimized on the same data used for evaluation (data leakage). Production deployment requires a held-out validation set.
6. **Adversarial robustness**: We do not test whether attackers can craft obfuscation that evades delta angle detection.
7. **Private repository**: Code is not yet publicly available. Reproduction requires access to Flara-workspace (private).

## 7. Future Work

1. **Larger evaluation**: Test on all 287 obfuscation samples and 813 benign samples
2. **More obfuscation types**: Evaluate Unicode encoding, markdown manipulation, code block injection
3. **Cross-model validation**: Test on additional embedding models
4. **Baseline comparison**: Compare with perplexity, entropy, and embedding clustering methods
5. **Production integration**: Implement as a classifier in AMDON guard pipeline (if results hold at scale)

## 8. Conclusion

Preliminary results (AUC-ROC 0.895–0.901, F1 0.864–0.889 on 50 samples) suggest that delta angle may detect encoding-based obfuscation. The mechanism is semantic compression: encoding reduces lexical variance, causing consecutive chunks to embed more similarly.

However, this is a small-scale pilot study. Key open questions:

1. Does the result hold at 287 samples (full dataset)?
2. How does delta angle compare with perplexity, entropy, or keyword baselines?
3. What happens when attackers know the detection method?
4. Which threshold selection strategy works in production?

We present this as a direction for further research, not a validated solution.

## References

1. Flara Research Lab. (2025). "Average Delta Angle: A Tokenizer-Based Security Measure Against Prompt Injection." Technical Report.

2. Neuralchemy. (2025). "Prompt-injection-Threat-Matrix." HuggingFace Dataset.

3. Flara Research Lab. (2025). "PSNAT-AMDON: API-based Model Distribution and Orchestration Network." Technical Report.

4. Gehman, S., et al. (2020). "RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models." EMNLP Findings.

## Appendix A: Graphs

*Note: Figures are hosted on imgur for this draft. Final version will use proper figure placement.*

### Comparison Chart
![Delta Angle Comparison](https://i.imgur.com/2aklo2D.png)

### Distribution: E5-v5
![E5-v5 Distribution](https://i.imgur.com/uraBeFv.png)

### Distribution: Nemotron
![Nemotron Distribution](https://i.imgur.com/Xli2EcF.png)

### Scatter: E5-v5
![E5-v5 Scatter](https://i.imgur.com/OMu6yrz.png)

### Scatter: Nemotron
![Nemotron Scatter](https://i.imgur.com/WvT1UaI.png)

### ROC Curves (N=300)
![ROC Curves](https://i.imgur.com/7L4ToQs.png)

### Delta Distribution (N=300)
![Distribution](https://i.imgur.com/AuQ9Wmp.png)

### FPR-FNR Tradeoff (N=300)
![FPR-FNR](https://i.imgur.com/2vl5XLL.png)

### Recall at Fixed FPR (N=300)
![Recall Comparison](https://i.imgur.com/iWnpOPI.png)

## Appendix B: Reproduction

Reproduction requires access to the Flara-workspace repository (currently private). Instructions:

1. Run `python3 download_benchmarks.py` to fetch the Threat-Matrix dataset
2. Run the AMDON CLI and execute `/benchmark-obf`
3. Run `python3 obf_graphs.py` to generate graphs

We plan to make the repository public once the research matures beyond the pilot stage.

All data and code are available in the Flara-workspace repository.
