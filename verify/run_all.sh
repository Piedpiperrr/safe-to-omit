#!/usr/bin/env bash
# Runs the three CPU checks and prints their RESULT lines.
# Usage, from the repository root:   bash verify/run_all.sh
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
fail=0
for s in verify_a_certification verify_b_e22_ratio verify_c_table2_row; do
    echo "--- $s"
    if ! "$PY" "verify/$s.py"; then fail=1; fi
    echo
done
if [ "$fail" -eq 0 ]; then
    echo "All three checks ran. Expected RESULT lines:"
    echo "  RESULT: PASS - all three deployed q reproduced from the released ledgers"
    echo "  RESULT: PASS - 0.547 reproduced from released records"
    echo "  RESULT: PASS - 95.0% skipped and 9/284 changed reproduced"
else
    echo "At least one check failed to run." >&2
fi
exit "$fail"
