# Obfuscation Detection Research

## Project Status
Active research — early-stage, not for production.

## What This Is
Research into using embedding-space delta angle measurements to detect obfuscated prompt injection attacks (hex, base64, character substitution).

## Key Finding
Obfuscation compresses semantic variance. Encoded text has lower delta angle than natural language because consecutive chunks embed more similarly. This is measurable and detectable.

## Results (N=300)

| Metric | nv-embedqa-e5-v5 | llama-nemotron-embed-1b-v2 |
|--------|------------------|---------------------------|
| AUC-ROC | 0.958 [0.927, 0.984] | 0.941 [0.904, 0.975] |
| F1 | 0.911 | 0.905 |
| Recall @ 5% FPR | 0.920 | 0.910 |
| Precision | 0.902 | 0.901 |
| Cohen's d | 2.975 | 2.339 |
| p-value | 7.4e-39 | 4.0e-36 |

Baseline comparison (at 5% FPR):
- Delta angle: 91-92% recall
- Regex (hex/base64): 71-73% recall
- Special char ratio: 12-14% recall
- Character entropy: 6% recall

## Why Delta Beats Regex
Regex detects known encoding patterns. Delta angle detects semantic compression regardless of encoding type. Novel encodings bypass regex but still compress embedding space.

## Files

### Code
- `full_benchmark.py` — Main benchmark (N=300, batched API calls, bootstrap CI, statistical tests)
- `baseline_comparison.py` — FPR/FNR comparison against regex, entropy, special char baselines
- `comparison_graphs.py` — ROC, PR curves, threshold analysis (N=50)
- `benchmark_standalone.py` — Standalone delta angle computation (no AMDON dependency)
- `obf_graphs.py` — Earlier analysis script (N=50)

### Data
- `data/obf_trigger.json` — 287 obfuscation samples from Threat-Matrix
- `data/obf_benign.json` — 813 benign samples
- `data/full_dataset_results.csv` — Per-sample results (N=300, 600 rows)
- `data/full_benchmark_summary.csv` — Summary statistics
- `data/standalone_results.csv` — Earlier N=50 results
- `data/obf_combined_results.csv` — Earlier combined results

### Graphs
- `graphs/full_roc.png` — ROC curves (N=300)
- `graphs/full_distribution.png` — Delta distribution (N=300)
- `graphs/full_fpr_fnr.png` — FPR-FNR tradeoff (N=300)
- `graphs/full_recall_comparison.png` — Recall at fixed FPR (N=300)
- `graphs/obf_*.png` — Earlier N=50 graphs

### Paper
- `Delta_Angle_Obfuscation_Paper.md` — Research paper

## Dependencies
- Python venv at `../../.venv/` (scipy, seaborn, scikit-learn, matplotlib, pandas, requests)
- NIM API key in `full_benchmark.py` (Nvidia NIM free tier)

## How to Run
```bash
cd research/obfuscation_detection
source ../../.venv/bin/activate
python3 full_benchmark.py
```

## Open Questions
1. Does result hold on full 287+813 dataset?
2. How does it compare with semantic intent detection (expensive)?
3. What happens when attacker knows about delta angle?
4. Which threshold selection strategy works in production?

## Related Work
- ATR-2026-00080, PromptShield, Vexscan — regex/pattern matching
- Hart (2026) — normalization + decode
- WitnessAI (2026) — semantic intent detection
- Vexscan — entropy analysis
- NAACL 2025 — Mixture of Encodings (defensive, not detection)
