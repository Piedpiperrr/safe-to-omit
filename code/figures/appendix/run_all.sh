#!/usr/bin/env bash
# Regenerate Figures 3-11 of the appendix into figures/ (CPU, about 45 s in total).
# Each script parses the numbers it plots, recomputes intervals and p-values from counts,
# and stops with an AssertionError if anything disagrees with the paper or the records.
set -euo pipefail
cd "$(dirname "$0")"
for f in fig03_null_paired fig04_null_sources fig05_boundaries fig06_baselines fig07_corrections \
         fig08_savings fig09_oos fig10_zero_u fig11_binormal; do
  echo "== $f"
  python3 "$f.py"
done
