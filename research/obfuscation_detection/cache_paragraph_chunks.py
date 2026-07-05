"""
Paragraph-level delta angle cache. Reconstructed: the original generator for
data/para_deltas_full.npy (RESEARCH_LOG.md Session 16/16b) was never committed --
only the output array and methodology notes survived. This rebuilds it from the
documented approach: paragraph-granularity chunking (no aggressive word-count
merging, unlike sentence chunking), embeddings via NIM, mean unsigned angle
between consecutive paragraph-chunk embeddings.

Session 16b reported ~2.47 paragraph chunks/sample vs ~1.41 sentence chunks/sample
(paragraph chunking produces MORE chunks than the aggressively-merged sentence
chunker, since it splits on newlines without re-merging short pieces) and a
sentence/paragraph correlation of 0.591 on the N=1100 dataset -- both used as a
sanity check against this reconstruction.
"""
import numpy as np
import json
import re
import os

from cache_sentence_chunks import nim_embed, compute_angle

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def chunk_paragraph(text):
    """Split on true paragraph breaks (blank lines) only -- single-newline text
    stays one chunk, same way single-sentence text stays one chunk for the
    sentence-level signal. This is what makes it a coarser, complementary
    signal rather than a near-duplicate of sentence chunking."""
    pieces = [p.strip() for p in re.split(r'\n\s*\n+', text) if p.strip()]
    return pieces if pieces else [text]


def main():
    obf_trigger = json.load(open("data/obf_trigger.json"))
    obf_benign = json.load(open("data/obf_benign.json"))
    obf_samples = obf_trigger if isinstance(obf_trigger[0], str) else [s['text'] for s in obf_trigger]
    ben_samples = obf_benign if isinstance(obf_benign[0], str) else [s['text'] for s in obf_benign]

    combined = {}
    for t in obf_samples: combined[t] = "obfuscation"
    for t in ben_samples: combined[t] = "benign"
    texts = list(combined.keys())
    labels = [combined[t] for t in texts]
    print(f"Total: {len(texts)} samples")

    all_chunks = []
    text_ranges = []
    for text in texts:
        chunks = chunk_paragraph(text)
        start = len(all_chunks)
        all_chunks.extend(chunks)
        text_ranges.append((start, len(all_chunks)))
    print(f"Total paragraph chunks: {len(all_chunks)} ({len(all_chunks)/len(texts):.2f}/sample)")

    model = "nvidia/nv-embedqa-e5-v5"
    print(f"\nEmbedding via {model}...")
    all_emb = nim_embed(all_chunks, model)

    deltas = []
    for si, ei in text_ranges:
        emb = all_emb[si:ei]
        if len(emb) < 2:
            deltas.append(0.0)
            continue
        angles = [compute_angle(emb[i], emb[i + 1]) for i in range(len(emb) - 1)]
        angles = [a for a in angles if a != 0.0]
        deltas.append(float(np.mean(angles)) if angles else 0.0)
    deltas = np.array(deltas)

    labels_arr = np.array(labels)
    ben = deltas[labels_arr == 'benign']
    obf = deltas[labels_arr == 'obfuscation']
    print(f"Benign: mu={np.mean(ben):.4f}  Obf: mu={np.mean(obf):.4f}")

    cache = json.load(open("data/sentence_chunk_cache.json"))
    sent_deltas = np.array(cache["models"][model]["deltas"])
    if len(sent_deltas) == len(deltas):
        corr = np.corrcoef(sent_deltas, deltas)[0, 1]
        print(f"Sentence/paragraph correlation: {corr:.3f} (Session 16b reference: 0.591)")

    np.save("data/para_deltas_full.npy", deltas)
    print(f"\nSaved data/para_deltas_full.npy ({len(deltas)} samples)")


if __name__ == "__main__":
    main()
