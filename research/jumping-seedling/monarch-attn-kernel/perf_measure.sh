#!/usr/bin/env bash
# Hardware-counter measurement against src/bin/profile.rs, the actual
# empirical check on ROOFLINE_5500U.md's compute-bound claim. Split into
# two smaller perf-stat event groups (cycles+instructions, then
# cache-references+cache-misses) rather than one combined run, since this
# CPU's limited hardware PMC slots caused event multiplexing / dropped
# counters when requesting 6 events at once (confirmed empirically before
# writing this script, not assumed) -- and since we deliberately did NOT
# disable the NMI watchdog (an out-of-scope system change), staying within
# available counter slots is the correct fix, not a workaround.
set -euo pipefail

BIN=./target/release/profile
KERNELS=(causal sliding meta)
SEQ_LENS=(512 2048 8192)
ITERATIONS=3

for kernel in "${KERNELS[@]}"; do
  for seq in "${SEQ_LENS[@]}"; do
    echo "=== $kernel seq_len=$seq ==="
    echo "--- cycles/instructions (IPC -- compute-bound signal) ---"
    perf stat -e cycles,instructions "$BIN" "$kernel" "$seq" "$ITERATIONS" 2>&1 | grep -E "cycles|instructions|elapsed|kernel="
    echo "--- cache-references/cache-misses (memory-traffic signal) ---"
    perf stat -e cache-references,cache-misses "$BIN" "$kernel" "$seq" "$ITERATIONS" 2>&1 | grep -E "cache-references|cache-misses|elapsed|kernel="
    echo
  done
done
