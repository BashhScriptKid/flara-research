"""
Test three proposed approaches for single-chunk detection:

1. BPE Token Density / Character-to-Token Ratio (no API needed)
2. Sliding Window Token Deltas (N=4, N=8) — needs API
3. Semantic Anchor Trajectory — feasibility assessment
"""
import numpy as np
import json
import re
import os
import time
from collections import Counter, defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Try to import tiktoken for BPE tokenization
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("WARNING: tiktoken not available. Installing...")
    os.system("pip install tiktoken -q")
    import tiktoken
    TIKTOKEN_AVAILABLE = True


# ═══════════════════════════════════════════════════════════════════
# 1. BPE TOKEN DENSITY / CHARACTER-TO-TOKEN RATIO
# ═══════════════════════════════════════════════════════════════════
print("="*70)
print("TEST 1: BPE Token Density / Character-to-Token Ratio")
print("="*70)

# Load data
obf_trigger = json.load(open("data/obf_trigger.json"))
obf_benign = json.load(open("data/obf_benign.json"))
obf_samples = obf_trigger if isinstance(obf_trigger[0], str) else [s['text'] for s in obf_trigger]
ben_samples = obf_benign if isinstance(obf_benign[0], str) else [s['text'] for s in obf_benign]

# Type classification
def classify_encoding(text):
    t = text.lower()
    if 'decode this hex' in t or '\\x' in text: return 'hex'
    if 'base64' in t: return 'base64'
    if re.search(r'rot\d+', t): return 'rot'
    if text.count('\\') > 5: return 'backslash_visual'
    if re.search(r'[A-Z]\.[A-Z]\.[A-Z]', text): return 'dot_spacing'
    if '<|im_start|>' in text: return 'xml_tag'
    return 'unstructured'

def classify_unstructured(text):
    t = text.lower()
    if re.search(r'translat.*spanish.*english|translat.*to.*spanish|translat.*back', t): return 'translation_chain'
    if re.search(r'(he sido|hackeado|llave secreta|clave)', t): return 'translation_chain'
    if re.search(r'dictionary game|replace.*with|synonym|substitut', t): return 'substitution_game'
    if re.search(r'q to p|replace each letter', t): return 'cipher_substitution'
    if re.search(r'movie title|generate a title|form a title', t): return 'task_hijack'
    if re.search(r'f\(.*==.*\)|decode.*base', t): return 'encoding_puzzle'
    if re.search(r'previous letter|shift|back shift', t): return 'manual_cipher'
    if re.search(r'U\+[0-9A-Fa-f]{4}', text): return 'unicode_codepoint'
    if any(c in text for c in '🗣👍🎉🎯🔥💡⭐🚀✅❌💰🎭🎪🎨🎭🎵🎶🎤🎧🎸🎹🎺🎻🥁'): return 'emoji_encoding'
    if re.search(r'p[ ,]*w[ ,]*n[ ,]*e[ ,]*d', t): return 'pwned_spacing'
    if 'ihbpd' in t or 'ihb' in t or 'ihnbp' in t or 'ihnpe' in t: return 'pwned_abbreviation'
    if re.search(r'pwned means|pwned.*positive|pwned.*spanish|pwned stands for', t): return 'semantic_redef'
    if 'hola' in t or 'si' in t or 'me gusta' in t: return 'language_mixing'
    if re.search(r'reverse.*letter|reverse.*each|reverse.*sentence', t): return 'reverse_scramble'
    if re.search(r'(don.t tell|only respond|only output|do not print|do not use)', t): return 'constraint_inject'
    if re.search(r'gnitae|selppa|Jdfjbkhkg|P ohcl', t): return 'manual_cipher'
    if re.search(r'secret key|secret code|password|\$secret|\{secret', t): return 'extraction_trick'
    if re.search(r'script:|noop|print the sentence', t): return 'code_injection'
    return 'misc'

obf_types = []
for t in obf_samples:
    enc = classify_encoding(t)
    if enc == 'unstructured':
        obf_types.append(classify_unstructured(t))
    else:
        obf_types.append(enc)

# Tokenize with tiktoken (cl100k_base — GPT-4/Claude tokenizer)
enc = tiktoken.get_encoding("cl100k_base")

def bpe_metrics(text):
    """Compute BPE token density metrics."""
    tokens = enc.encode(text)
    n_tokens = len(tokens)
    n_chars = len(text)
    n_words = len(text.split())
    
    # Character-to-token ratio (higher = more fractured)
    char_token_ratio = n_chars / max(n_tokens, 1)
    
    # Word-to-token ratio (higher = more fractured per word)
    word_token_ratio = n_words / max(n_tokens, 1)
    
    # Token-to-word ratio (higher = more tokens per word = more fractured)
    token_word_ratio = n_tokens / max(n_words, 1)
    
    # Fraction of single-character tokens
    token_strs = [enc.decode([t]) for t in tokens]
    single_char_frac = sum(1 for t in token_strs if len(t) == 1) / max(n_tokens, 1)
    
    # Fraction of whitespace tokens (BPE splits on whitespace)
    whitespace_frac = sum(1 for t in token_strs if t.strip() == '') / max(n_tokens, 1)
    
    return {
        'n_tokens': n_tokens,
        'n_chars': n_chars,
        'n_words': n_words,
        'char_token_ratio': char_token_ratio,
        'word_token_ratio': word_token_ratio,
        'token_word_ratio': token_word_ratio,
        'single_char_frac': single_char_frac,
        'whitespace_frac': whitespace_frac,
    }

print(f"\nTokenizer: cl100k_base (GPT-4/Claude)")
print(f"Obfuscation samples: {len(obf_samples)}")
print(f"Benign samples: {len(ben_samples)}")

# Compute metrics for all samples
print("\nComputing BPE metrics...")
obf_metrics = [bpe_metrics(t) for t in obf_samples]
ben_metrics = [bpe_metrics(t) for t in ben_samples]

# Summary statistics
for metric_name in ['char_token_ratio', 'token_word_ratio', 'single_char_frac']:
    obf_vals = [m[metric_name] for m in obf_metrics]
    ben_vals = [m[metric_name] for m in ben_metrics]
    print(f"\n{metric_name}:")
    print(f"  Obfuscation: mean={np.mean(obf_vals):.4f} std={np.std(obf_vals):.4f} median={np.median(obf_vals):.4f}")
    print(f"  Benign:      mean={np.mean(ben_vals):.4f} std={np.std(ben_vals):.4f} median={np.median(ben_vals):.4f}")
    print(f"  Gap:         {abs(np.mean(obf_vals) - np.mean(ben_vals)):.4f}")

# AUC for each metric
from sklearn.metrics import roc_auc_score
print("\n--- AUC per metric ---")
for metric_name in ['char_token_ratio', 'token_word_ratio', 'single_char_frac']:
    obf_vals = np.array([m[metric_name] for m in obf_metrics])
    ben_vals = np.array([m[metric_name] for m in ben_metrics])
    all_vals = np.concatenate([obf_vals, ben_vals])
    labels = np.array([1]*len(obf_vals) + [0]*len(ben_vals))
    
    # Test both directions
    auc1 = roc_auc_score(labels, all_vals)
    auc2 = roc_auc_score(labels, -all_vals)
    auc = max(auc1, auc2)
    direction = "↑obf" if auc1 > auc2 else "↓obf"
    print(f"  {metric_name:25s}: AUC={auc:.4f} ({direction})")

# Per-type breakdown
print("\n--- Per-type char_token_ratio ---")
type_groups = defaultdict(list)
for i, t in enumerate(obf_types):
    type_groups[t].append(obf_metrics[i]['char_token_ratio'])

for t in sorted(type_groups.keys(), key=lambda x: -np.mean(type_groups[x])):
    vals = type_groups[t]
    print(f"  {t:25s}: mean={np.mean(vals):.4f} (n={len(vals)})")

# Best metric: combined BPE score
print("\n--- Combined BPE score ---")
# char_token_ratio seems most promising — let's check per-type AUC
for t in sorted(type_groups.keys()):
    if len(type_groups[t]) < 3:
        continue
    obf_vals = np.array(type_groups[t])
    ben_vals_arr = np.array([m['char_token_ratio'] for m in ben_metrics])
    all_vals = np.concatenate([obf_vals, ben_vals_arr])
    labels = np.array([1]*len(obf_vals) + [0]*len(ben_vals_arr))
    try:
        auc = roc_auc_score(labels, all_vals)
        print(f"  {t:25s}: AUC={auc:.4f} (n={len(obf_vals)})")
    except:
        print(f"  {t:25s}: AUC=N/A (n={len(obf_vals)})")

# ═══════════════════════════════════════════════════════════════════
# Combined BPE + delta test on single-chunk samples
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("BPE + DELTA ENSEMBLE ON SINGLE-CHUNK SAMPLES")
print("="*70)

cache = json.load(open("data/sentence_chunk_cache.json"))
e5_deltas = np.array(cache['models']['nvidia/nv-embedqa-e5-v5']['deltas'])
labels_cache = cache['labels']

# Single-chunk = delta == 0
single_chunk_mask = e5_deltas == 0.0
sc_obf = [i for i in range(len(labels_cache)) if single_chunk_mask[i] and labels_cache[i] == 'obfuscation']
sc_ben = [i for i in range(len(labels_cache)) if single_chunk_mask[i] and labels_cache[i] == 'benign']
print(f"Single-chunk: {len(sc_obf)} obf, {len(sc_ben)} ben")

# BPE char_token_ratio for single-chunk samples
# sc_obf/sc_ben are cache indices (0-1099). obf_metrics has 287 entries (0-286).
# Cache: first 287 = obfuscation, next 813 = benign
sc_obf_bpe = [bpe_metrics(cache['texts'][i])['char_token_ratio'] for i in sc_obf]
sc_ben_bpe = [bpe_metrics(cache['texts'][i])['char_token_ratio'] for i in sc_ben]

print(f"\nSingle-chunk BPE char_token_ratio:")
print(f"  Obf: mean={np.mean(sc_obf_bpe):.4f} std={np.std(sc_obf_bpe):.4f}")
print(f"  Ben: mean={np.mean(sc_ben_bpe):.4f} std={np.std(sc_ben_bpe):.4f}")

if len(sc_obf) > 1 and len(sc_ben) > 1:
    all_bpe = np.array(sc_obf_bpe + sc_ben_bpe)
    all_labels_sc = np.array([1]*len(sc_obf_bpe) + [0]*len(sc_ben_bpe))
    auc_bpe = roc_auc_score(all_labels_sc, all_bpe)
    print(f"  AUC (BPE alone on single-chunk): {auc_bpe:.4f}")
    
    # Combined: BPE + sentence delta (but delta=0 for all single-chunk)
    # So BPE is the ONLY signal for single-chunk — this is exactly the use case
    print(f"\n  BPE is the ONLY detection signal for single-chunk samples.")
    print(f"  This is the gap the ensemble currently cannot fill.")

# ═══════════════════════════════════════════════════════════════════
# 2. SLIDING WINDOW TOKEN DELTAS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 2: Sliding Window Token Deltas (N=4, N=8)")
print("="*70)

NIM_API_KEY = os.getenv('NIM_API_KEY', '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')
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nv-embedqa-e5-v5"

def nim_embed(texts, model=MODEL, batch_size=32):
    all_emb = [None] * len(texts)
    headers = {"Authorization": f"Bearer {NIM_API_KEY}", "Content-Type": "application/json"}
    def clean(t):
        t = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', t)
        t = re.sub(r'[\U0001D400-\U0001D7FF]', '', t)
        t = re.sub(r'\s{2,}', ' ', t)
        return t[:1600].strip() or "empty"
    for i in range(0, len(texts), batch_size):
        batch = [clean(t) for t in texts[i:i + batch_size]]
        try:
            r = requests.post(f"{NIM_BASE_URL}/embeddings", headers=headers,
                            json={"model": model, "input": batch, "input_type": "passage"}, timeout=60)
            r.raise_for_status()
            for j, item in enumerate(sorted(r.json()['data'], key=lambda x: x['index'])):
                all_emb[i + j] = np.array(item['embedding'], dtype=np.float32)
        except Exception as e:
            for j, t in enumerate(batch):
                idx = i + j
                if all_emb[idx] is not None: continue
                try:
                    r2 = requests.post(f"{NIM_BASE_URL}/embeddings", headers=headers,
                                     json={"model": model, "input": [t], "input_type": "passage"}, timeout=30)
                    r2.raise_for_status()
                    all_emb[idx] = np.array(r2.json()['data'][0]['embedding'], dtype=np.float32)
                except:
                    all_emb[idx] = np.zeros(4096, dtype=np.float32)
                time.sleep(0.1)
    return all_emb

def angle(e1, e2):
    n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
    if n1 < 1e-10 or n2 < 1e-10: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1))))

def sliding_window_tokens(text, window_size=4, step=2):
    """Split text into token windows, embed each, compute velocity."""
    tokens = enc.encode(text)
    if len(tokens) < window_size * 2:
        return None
    
    windows = []
    start = 0
    while start + window_size <= len(tokens):
        window_text = enc.decode(tokens[start:start + window_size])
        windows.append(window_text)
        start += step
    
    if len(windows) < 3:
        return None
    
    embs = nim_embed(windows)
    
    # Compute angles between consecutive windows
    angles_list = []
    for i in range(len(embs) - 1):
        a = angle(embs[i], embs[i+1])
        angles_list.append(a)
    
    angles_arr = np.array(angles_list)
    return {
        'velocity_mean': float(np.mean(angles_arr)),
        'velocity_max': float(np.max(angles_arr)),
        'velocity_std': float(np.std(angles_arr)),
        'accel_max': float(np.max(np.abs(np.diff(angles_arr)))) if len(angles_arr) > 1 else 0,
        'n_windows': len(windows),
    }

import requests

# Test on single-chunk obfuscation vs benign
print(f"\nTesting on single-chunk samples (window=4, step=2)...")
n_test_sw = min(15, len(sc_obf), len(sc_ben))
np.random.seed(42)
sw_obf_idx = np.random.choice(sc_obf, n_test_sw, replace=False)
sw_ben_idx = np.random.choice(sc_ben, n_test_sw, replace=False)

obf_sw_metrics = []
ben_sw_metrics = []

print("\nObfuscation:")
for i in sw_obf_idx:
    text = cache['texts'][i]
    words = text.split()
    m = sliding_window_tokens(text, window_size=4, step=2)
    if m:
        obf_sw_metrics.append(m)
        print(f"  [{i}] ({len(words)}w) vel={m['velocity_mean']:.1f}° max={m['velocity_max']:.1f}° accel={m['accel_max']:.1f}°")

print("\nBenign:")
for i in sw_ben_idx:
    text = cache['texts'][i]
    words = text.split()
    m = sliding_window_tokens(text, window_size=4, step=2)
    if m:
        ben_sw_metrics.append(m)
        print(f"  [{i}] ({len(words)}w) vel={m['velocity_mean']:.1f}° max={m['velocity_max']:.1f}° accel={m['accel_max']:.1f}°")

if obf_sw_metrics and ben_sw_metrics:
    print(f"\n--- Sliding Window Summary ---")
    obf_vel = [m['velocity_mean'] for m in obf_sw_metrics]
    ben_vel = [m['velocity_mean'] for m in ben_sw_metrics]
    obf_accel = [m['accel_max'] for m in obf_sw_metrics]
    ben_accel = [m['accel_max'] for m in ben_sw_metrics]
    
    print(f"Velocity mean: obf={np.mean(obf_vel):.1f}° ben={np.mean(ben_vel):.1f}°")
    print(f"Accel max:     obf={np.mean(obf_accel):.1f}° ben={np.mean(ben_accel):.1f}°")
    
    all_vel = obf_vel + ben_vel
    all_labels_sw = [1]*len(obf_vel) + [0]*len(ben_vel)
    try:
        auc_vel = roc_auc_score(all_labels_sw, all_vel)
        print(f"AUC (velocity): {auc_vel:.4f}")
    except:
        print("AUC (velocity): N/A")
    
    all_accel = obf_accel + ben_accel
    try:
        auc_accel = roc_auc_score(all_labels_sw, all_accel)
        print(f"AUC (accel_max): {auc_accel:.4f}")
    except:
        print("AUC (accel_max): N/A")

# ═══════════════════════════════════════════════════════════════════
# 3. SEMANTIC ANCHOR TRAJECTORY — FEASIBILITY ASSESSMENT
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 3: Semantic Anchor Trajectory — Feasibility Assessment")
print("="*70)

print("""
FEASIBILITY ANALYSIS:

The pitch suggests projecting input vectors onto a subspace defined by
"known system-prompt override directions." This requires:

1. A TRAINING SET of known adversarial directions:
   - Need hundreds of labeled "hijack direction" vectors
   - These would be mean embedding differences between paired
     (benign, adversarial) samples
   - Problem: The adversarial space is high-dimensional and diverse
     (hex, backslash, natural language, Unicode, etc.)

2. A HYPERPLANE separating benign from adversarial:
   - This is essentially a linear classifier in embedding space
   - But we already tested cosine distance from benign centroid (Session 17)
     and it FAILED (AUC=0.439, inverted)
   - A hyperplane is equivalent to a weighted combination of centroid distances
   - The fundamental problem: encoding obfuscation PRESERVES semantic content,
     so the embedding is near-benign regardless of projection direction

3. The "hijack sector" concept:
   - For a single sentence, the vector has ONE position in space
   - There's no trajectory to follow (no multi-chunk signal)
   - The "angle relative to adversarial hyperplane" is just a rotated version
     of cosine distance, which already failed

VERDICT: Not feasible without a fundamentally different embedding model
that produces adversarial-direction-sensitive representations. Current
models (E5, BGE-M3) encode MEANING, not INTENT. The "hijack sector"
doesn't exist in their representation space.

This is the same reason distance-from-centroid failed: the embedding
model doesn't distinguish between "I have been PWNED" and its hex
encoding because they MEAN the same thing.
""")

# ═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("="*70)
print("FINAL SUMMARY: ALL THREE APPROACHES")
print("="*70)

print("""
┌─────────────────────────────────┬──────────┬──────────────────────┐
│ Approach                        │ Status   │ Key Finding          │
├─────────────────────────────────┼──────────┼──────────────────────┤
│ 1. BPE Token Density            │ PROMISING│ AUC varies by type   │
│    (char-to-token ratio)        │          │ Best for hex/bsv     │
├─────────────────────────────────┼──────────┼──────────────────────┤
│ 2. Sliding Window Token Deltas  │ TESTED  │ See results above    │
│    (N=4, step=2)                │          │                      │
├─────────────────────────────────┼──────────┼──────────────────────┤
│ 3. Semantic Anchor Trajectory   │ NOT      │ Same failure mode as │
│    (adversarial hyperplane)     │ FEASIBLE │ centroid distance    │
└─────────────────────────────────┴──────────┴──────────────────────┘
""")
