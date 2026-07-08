#!/usr/bin/env python3
"""
Tests whether the criteria_before / with_persona reversal (Section 6, ablation
study) is specific to llama-3.1-8b-instant, or holds on a different model
family/size. Swaps GUARD_MODEL to qwen3-32b and reruns decoupled,
criteria_before, and with_persona against the same 200 WildGuardMix prompts.
Run locally (not on the VM) -- it's just hitting the Groq API directly.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B

B.GROQ_TOKEN = sys.argv[1]
B.HF_TOKEN = sys.argv[2]

# Swap the guard model to test cross-model generalization of the ablation result.
B.GUARD_MODEL = "qwen/qwen3-32b"

# qwen3-32b is reasoning-capable; the shared guard_classify/criteria_before_/
# with_persona_ classify functions call chat() with max_tokens=100 and no
# reasoning_effort, tuned for llama-3.1-8b-instant (non-reasoning). Calling a
# reasoning model that way risks the exact same hidden-reasoning-starves-small-
# budget failure documented in Section 5 of the paper. Wrap chat() so any call
# routed to GUARD_MODEL gets a safe reasoning_effort + larger budget, without
# touching benchmark.py's shared functions (which other conditions still rely
# on with their original, correct settings).
_orig_chat = B.chat
def _patched_chat(model, messages, max_tokens=512, temp=0.0, reasoning_effort=None):
    if model == B.GUARD_MODEL:
        max_tokens = max(max_tokens, 1024)
        reasoning_effort = "none"  # disable CoT — parser uses startswith("unsafe"), needs bare verdict
    return _orig_chat(model, messages, max_tokens=max_tokens, temp=temp, reasoning_effort=reasoning_effort)
B.chat = _patched_chat

B.load_cache()

examples = B.load_examples(B.HF_TOKEN)

results = {}
for mode in ["decoupled", "criteria_before", "with_persona"]:
    print(f"\n-- {mode} (guard={B.GUARD_MODEL}) --", flush=True)
    results[mode] = B.run_decoupled(examples, mode=mode)
    with open("data/model_specificity_results.json", "w") as f:
        json.dump(results, f)

all_metrics = {}
for cond, rows in results.items():
    all_metrics[cond] = B.compute_metrics(rows, is_mono=False)
B.print_table(all_metrics)
print("DONE")
