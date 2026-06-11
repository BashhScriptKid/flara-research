# Delta Angle as an Obfuscation Detector

**Flara Research Lab**
**Status: Early-stage research, not for production use**

## Abstract

We present preliminary evidence that embedding-space delta angle measurements can detect obfuscated prompt injections. Obfuscated inputs—those using encoding, character substitution, or structural manipulation to evade detection—produce consistently lower delta angles than benign inputs across two embedding models. We evaluate on 50 obfuscation samples and 50 benign samples per model, achieving F1 scores of 0.879 and 0.891. While these results are promising, the evaluation is small and requires validation on larger datasets before any production claims can be made.

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

| Model | Obfuscation Δ | Benign Δ | Gap | F1 | Threshold |
|-------|---------------|----------|-----|-----|-----------|
| nv-embedqa-e5-v5 | 0.366 ± 0.184 | 0.696 ± 0.142 | 0.330 | 0.879 | 0.422 |
| llama-nemotron-embed-1b-v2 | 0.565 ± 0.214 | 0.907 ± 0.206 | 0.342 | 0.891 | 0.629 |

**Key findings:**

1. **Consistent separation**: Both models show a gap of ~0.33 between obfuscation and benign distributions
2. **High F1**: Average F1 of 0.885 across models
3. **Low false positives**: Only 1 false positive per model out of 50 benign samples
4. **Model agreement**: Per-sample correlation of r≈0.76 between models

### 4.4 Distribution Analysis

The distributions show clear separation with limited overlap:

- **E5-v5**: 20% overlap between distributions
- **Nemotron**: 16% overlap between distributions

This is significantly better than the general injection detection case, where distributions overlapped by >50%.

### 4.5 Per-Category Breakdown

Within the obfuscation category, we observe varying detectability:

- **Hex encoding**: High delta (most detectable)
- **Base64 encoding**: Medium delta
- **Character substitution**: Lower delta (least detectable)

This suggests that encoding-based obfuscation creates more semantic distortion than character-level substitution.

## 5. Analysis

### 5.1 Why Obfuscation Is Detectable

Obfuscation attacks must preserve semantic intent while changing lexical form. This creates a fundamental tension:

1. **Encoding introduces noise**: When text is encoded (hex, base64), the encoded form has different semantic properties than the original text
2. **Chunking amplifies distortion**: The chunking process creates boundaries that are semantically meaningful in natural language but arbitrary in encoded text
3. **Embedding captures distortion**: The embedding model encodes these semantic distortions as larger angular distances between chunks

### 5.2 Why General Injection Is Not

Direct injection, by contrast, uses natural language that is semantically coherent. The injection instruction is a valid sentence that happens to be malicious. This means:

1. **No encoding noise**: The text is already in natural language form
2. **Coherent chunking**: Chunks are semantically meaningful
3. **Low angular distance**: Consecutive chunks have similar semantic content

### 5.3 Implications

This result suggests that delta angle is not a general-purpose detector but a specialized tool for detecting obfuscation. This is still valuable because:

1. **Obfuscation is common**: Many real-world attacks use encoding to evade detection
2. **Complementary to other measures**: Delta angle can be combined with other detectors that catch direct injection
3. **Deterministic**: Unlike model-based detectors, delta angle is purely mathematical and cannot be manipulated

## 6. Limitations

**This is early-stage research. The following limitations are significant:**

1. **Small evaluation**: 50 samples per model is insufficient for production claims. This is a pilot study, not a validation.
2. **Limited obfuscation types**: We evaluate hex, base64, and character substitution, but not all possible obfuscation techniques.
3. **Model dependency**: Results vary across embedding models. We only tested two models.
4. **No comparison with baselines**: We do not compare with perplexity, entropy, or other detection methods.
5. **Not standalone**: Delta angle should be used as part of a multi-layer defense, not as a sole detector.
6. **Private repository**: Code is not yet publicly available. Reproduction requires access to Flara-workspace (private).

## 7. Future Work

1. **Larger evaluation**: Test on all 287 obfuscation samples and 813 benign samples
2. **More obfuscation types**: Evaluate Unicode encoding, markdown manipulation, code block injection
3. **Cross-model validation**: Test on additional embedding models
4. **Baseline comparison**: Compare with perplexity, entropy, and embedding clustering methods
5. **Production integration**: Implement as a classifier in AMDON guard pipeline (if results hold at scale)

## 8. Conclusion

Preliminary results suggest that delta angle measurements may detect obfuscated prompt injections (F1 ≈ 0.885 on 50 samples). The key insight is that obfuscation creates measurable semantic distortions in embedding space, even when the lexical form is changed.

However, this is a small-scale pilot study. The results need validation on larger datasets, more obfuscation types, and additional embedding models before any production claims can be made. We present this as a promising direction for further research, not a validated solution.

## References

1. Flara Research Lab. (2025). "Average Delta Angle: A Tokenizer-Based Security Measure Against Prompt Injection." Technical Report.

2. Neuralchemy. (2025). "Prompt-injection-Threat-Matrix." HuggingFace Dataset.

3. Flara Research Lab. (2025). "PSNAT-AMDON: API-based Model Distribution and Orchestration Network." Technical Report.

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

## Appendix B: Reproduction

Reproduction requires access to the Flara-workspace repository (currently private). Instructions:

1. Run `python3 download_benchmarks.py` to fetch the Threat-Matrix dataset
2. Run the AMDON CLI and execute `/benchmark-obf`
3. Run `python3 obf_graphs.py` to generate graphs

We plan to make the repository public once the research matures beyond the pilot stage.

All data and code are available in the Flara-workspace repository.
