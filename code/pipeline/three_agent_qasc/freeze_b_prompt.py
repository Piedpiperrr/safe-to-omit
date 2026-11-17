"""Run ONLY after the authors approve the B prompt: writes rebuttal/e1/FROZEN_B_PROMPT.json (path + sha256).
The generation job script (a scheduler script, not released) refuses to run without it, and checks that BPROMPT
is the frozen file.
  python freeze_b_prompt.py prompts/B_prompt_vN.json "<who approved, when>"
"""
import hashlib, json, sys, time
from pathlib import Path
E1 = Path(__file__).resolve().parents[1]
p, who = sys.argv[1], sys.argv[2]
out = E1 / 'FROZEN_B_PROMPT.json'
assert not out.exists(), 'already frozen'
out.write_text(json.dumps({'path': p, 'sha256': hashlib.sha256((E1 / p).read_bytes()).hexdigest(),
                           'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'approved_by': who}, indent=1) + '\n')
print(out.read_text())
