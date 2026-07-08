# Negative Delta Analysis - Benign Inputs

**Date:** 2026-06-12
**Model:** nv-embedqa-e5-v5

## Summary

Negative signed deltas on benign inputs are caused by **inappropriate chunking**, not semantic contradictions.

## Root Cause

### 1. Example Suffix Issue (69/111 cases)

**Pattern:** Inputs with `(example XXXX)` suffix get chunked into:
- Chunk 0: The actual question
- Chunk 1: The `(example XXXX)` suffix

**Example:**
```
Input: "How do you say 'thank you' in Japanese? (example 1796)"
Chunks: ["How do you say 'thank you' in Japanese?", "(example 1796)"]
Signed angle: -0.9667 (-55.39°)
```

**Problem:** The example suffix is semantically unrelated to the question, creating a negative angle.

### 2. Word-Count Fallback Issue (42/111 cases)

**Pattern:** Short inputs without sentence boundaries get split at midpoint:
- Chunk 0: First half of words
- Chunk 1: Second half of words

**Examples:**
```
Input: "Help me write a machine learning for indoor plants"
Chunks: ["Help me write a", "machine learning for indoor plants"]
Signed angle: -0.6921

Input: "Can you define happiness?"
Chunks: ["Can you", "define happiness?"]
Signed angle: -0.6843

Input: "who are you"
Chunks: ["who", "are you"]
Signed angle: -0.4705
```

**Problem:** The word-count fallback splits at arbitrary boundaries, not semantic boundaries.

## Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| Total benign with negative E5 delta | 111 | 55.5% |
| With `(example XXXX)` suffix | 69 | 62.2% |
| Without suffix (word-count fallback) | 42 | 37.8% |

## Implications

1. **Chunking quality is critical** - The delta angle is highly sensitive to how text is chunked
2. **Example suffixes should be stripped** - Before chunking, remove `(example XXXX)` patterns
3. **Word-count fallback is problematic** - Splitting at midpoint creates semantically unrelated chunks
4. **Short inputs need special handling** - Inputs <50 chars should not be split arbitrarily

## Recommendations

1. **Pre-processing:** Strip `(example XXXX)` patterns before chunking
2. **Improved chunking:** Use better sentence/clause detection for short inputs
3. **Minimum chunk size:** Don't create chunks with <5 words
4. **Semantic chunking:** Consider using embedding-based chunking instead of rule-based

## Affected Inputs

See `negative_delta_benign_inputs.json` for full list of 111 affected inputs.
