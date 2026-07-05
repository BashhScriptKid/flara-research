# Obfuscation Detection Research

## Project Status
Active research — N=1100 dataset, ensemble model, enhanced keywords.

## What This Is
Research into using embedding-space delta angle measurements to detect obfuscated prompt injection attacks (hex, base64, character substitution).

## Key Finding
Obfuscation compresses semantic variance. Encoded text has lower delta angle than natural language because consecutive chunks embed more similarly. This is measurable and detectable.

## Results (N=1100)

| Metric | Full (9 regex) | Pruned (6 regex) |
|--------|---------------|-----------------|
| AUC-ROC | **0.9872** | 0.9855 |
| F1 | 0.971 | **0.972** |
| Det@1%FPR | 97.2% | 97.2% |
| Det@5%FPR | 100% | 100% |
| TP | **280** | 279 |
| FP | 10 | **8** |
| FN | **7** | 8 |

**Recommended:** Pruned (6 regex) — better FPR, simpler.
Single-chunk AUC: 0.9561

## Why Delta Beats Regex
Regex detects known encoding patterns. Delta angle detects semantic compression regardless of encoding type. Novel encodings bypass regex but still compress embedding space.

## Algorithm
1. **Sentence chunking** — Split text into semantic chunks using sentence boundaries
2. **Delta angle** — Compute unsigned cosine angle between consecutive chunk embeddings
3. **Paragraph delta** — Compute delta at paragraph granularity
4. **Regex v3** — 6-pattern coverage score with enhanced keywords (29 terms)
5. **Weird unicode** — Single-chunk bonus for unusual Unicode characters (math symbols, zero-width, fullwidth)
6. **Ensemble** — Weighted average: delta=0.042, regex=0.930, para=0.088, weird=0.210 (SC only)

## Enhanced Keywords (Session 24)
Added connective+action patterns to catch single-chunk obfuscation:
- `and (translate|decode|output|return|follow|respond|clean|provide|generate)`
- `but (if|when|unless|always|never)`
- `don't (fix|change|modify|translate|alter|edit|correct)`
- `only (translate|output|return|respond|provide|give)`
- `then (translate|decode|output|return|follow|respond)`
- `now (translate|decode|output|return|follow|respond|reverse)`

Result: +3 TP, 0 new FP, single-chunk AUC +1.9pp

## Weird Unicode Feature (Session 25)
Single-chunk-only bonus for unusual Unicode characters:
- Mathematical Alphanumeric (U+1D400–U+1D7FF): 𝖯𝖶𝖭𝖤𝖣
- Fullwidth Latin (uFF01–uFF5E)
- Enclosed characters (u2460–u24FF)
- Superscript/subscript (u00B2–u209F)
- Zero-width characters (u200B–u2064, uFEFF)
- Combining diacritics (u0300–u036F)
- Arrows/misc symbols (u2190–u21FF, u2B00–u2BFF)

Result: +2 TP, 0 new FP, SC AUC +1.25pp. Complementary to enhanced KW (zero overlap).

## Files

### Code
- `delta_sentence_chunk.py` — Clean CLI implementation
- `cache_sentence_chunks.py` — Cache builder
- `held_out_validation.py` — Stratified 70/30 validation
- `skipgram_delta_test.py` — Multi-hop skip-gram delta test
- `connective_split_test.py` — Connective-word splitting test

### Data
- `data/sentence_chunk_cache.json` — Cached deltas (1100 samples, 3 models)
- `data/para_deltas_full.npy` — Paragraph deltas (1100 samples)
- `data/obf_trigger.json` — 287 obfuscation samples
- `data/obf_benign.json` — 813 benign samples

### Graphs
- `graphs/final/` — 20 main paper figures
- `graphs/final/obf_analysis/` — 9 obfuscation analysis figures

### Paper
- `Delta_Angle_Obfuscation_Paper.md` — Research paper
- `../HANDOFF_DOCUMENT.md` — Research handoff for paper writer
- `../HANDOFF_VALIDATION.md` — Held-out validation results

## Dependencies
- Python venv at `../../.venv/` (scipy, seaborn, scikit-learn, matplotlib, pandas, requests)
- NIM API key for Nvidia NIM embedding models

## How to Run
```bash
cd research/obfuscation_detection
source ../../.venv/bin/activate
python3 delta_sentence_chunk.py  # Run detector
python3 cache_sentence_chunks.py  # Rebuild cache
```

## Rejected Approaches (Negative Results)
1. **Distribution distance** — Inverted signal (AUC=0.439)
2. **Token/word angle** — Inverted for encoding obfuscation (AUC=0.194)
3. **Signed angles** — Zero negatives across all 1100 samples
4. **Vector velocity** — Signal inverted (AUC=0.0000)
5. **BPE token density** — Inverted on single-chunk (AUC=0.1421)
6. **Sliding window deltas** — Inverted (AUC=0.3274)
7. **Semantic anchor** — Same failure mode as centroid distance
8. **Skip-gram delta** — Adjacent strictly better, 0 unique catches
9. **Connective splitting** — Signal inverted (benign higher than obfuscation)

## Related Work
- ATR-2026-00080, PromptShield, Vexscan — regex/pattern matching
- Hart (2026) — normalization + decode
- WitnessAI (2026) — semantic intent detection
- Vexscan — entropy analysis
- NAACL 2025 — Mixture of Encodings (defensive, not detection)
