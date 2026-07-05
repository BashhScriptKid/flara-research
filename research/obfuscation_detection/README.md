# Obfuscation Detection Research

This folder contains research on using delta angle measurements to detect obfuscated prompt injection attacks.

## Key Findings

- Delta angle detects obfuscated inputs with F1 ≈ 0.885
- Consistent separation gap of 0.33–0.34 across embedding models
- Obfuscation creates measurable semantic distortions in embedding space
- Works well for encoding-based attacks (hex, base64)

## Files

- `Delta_Angle_Obfuscation_Paper.md` — Research paper
- `benchmark_standalone.py` — Standalone benchmark (no AMDON dependency)
- `obf_graphs.py` — Analysis and graph generation script
- `data/` — Benchmark results and statistics
- `graphs/` — Generated visualizations

## Reproduction

```bash
# Standalone (no AMDON required)
python3 benchmark_standalone.py

# Or with AMDON CLI
/benchmark-obf
python3 obf_graphs.py
```
