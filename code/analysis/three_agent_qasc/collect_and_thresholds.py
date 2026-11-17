"""CPU, after generation: merge shards of one split, check completeness, and (fit only) write the 20 candidate thresholds.

  python collect_and_thresholds.py collect  outputs/full/fit   fit
  python collect_and_thresholds.py thresholds outputs/full/fit
No gold is read. Writes <dir>/COLLECT_<split>.json and, for fit, <dir>/THRESHOLD_CANDIDATES.json.
"""
import glob
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
E1 = Path(__file__).resolve().parents[1]
STEPS_MAIN = ['A', 'probe', 'C_AC', 'B', 'C_ABC']
STEPS_CTRL = ['C_C', 'C_CF', 'B_noA', 'C_BC', 'B_nofact', 'C_ABnfC']


def load(d, split):
    rows = {}
    for f in sorted(glob.glob(str(Path(d) / f'records.{split}.shard*.jsonl'))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                assert r['id'] not in rows, f'duplicate record {r["id"]}'
                rows[r['id']] = r
    return rows


def collect(d, split):
    ids = json.loads((E1 / f'splits/{split}_ids.json').read_text())
    rows = load(d, split)
    missing = [i for i in ids if i not in rows]
    extra = [i for i in rows if i not in set(ids)]
    steps = STEPS_MAIN + (STEPS_CTRL if split == 'dev' else [])
    fails = {s: sum(1 for r in rows.values() if r['steps'].get(s) is None) for s in steps}
    invalid = {s: sum(1 for r in rows.values() if (r['steps'].get(s) or {}).get('invalid')) for s in steps if s.startswith('C_')}
    retried = sum(1 for r in rows.values() for f in r['failures'] if f['status'] == 'succeeded_on_retry')
    bp = sorted({r['b_prompt_sha256'] for r in rows.values()})
    rep = {'split': split, 'n_expected': len(ids), 'n_records': len(rows), 'missing': missing, 'extra': extra,
           'complete': not missing and not extra, 'step_failures_after_retry': fails, 'invalid_answers': invalid,
           'steps_succeeded_on_retry': retried, 'b_prompt_sha256': bp, 'gold_read': False,
           'records_sha256': {Path(f).name: hashlib.sha256(Path(f).read_bytes()).hexdigest()
                              for f in sorted(glob.glob(str(Path(d) / f'records.{split}.shard*.jsonl')))}}
    assert len(bp) <= 1, 'records were produced with more than one B prompt'
    Path(d, f'COLLECT_{split}.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rep.items() if k not in ['missing', 'extra', 'records_sha256']}, indent=1))
    return rep


def thresholds(d):
    import e1lib as L
    rep = collect(d, 'fit')
    assert rep['complete'], 'fit split incomplete: thresholds are not written'
    rows = load(d, 'fit')
    ids = json.loads((E1 / 'splits/fit_ids.json').read_text())
    # a missing probe (runtime failure after retry) is treated like a non-finite score: None, ranked last, never skips
    u = {i: (rows[i]['steps']['probe']['u'] if rows[i]['steps'].get('probe') else None) for i in ids}
    cand = L.candidate_thresholds([u[i] for i in ids])
    out = {'N_fit': len(ids), 'candidates': cand, 'rule': 'skip B when u <= tau; q = 1 always skips',
           'fit_scores_sha256': hashlib.sha256(json.dumps([(i, u[i]) for i in ids]).encode()).hexdigest(),
           'score': 'log non-top mass (PROTOCOL.md section 4)',
           'probe_missing_treated_as_nonfinite': sum(1 for i in ids if not rows[i]['steps'].get('probe')),
           'nonfinite': sum(1 for i in ids if (rows[i]['steps'].get('probe') or {}).get('nonfinite')),
           'b_prompt_sha256': rep['b_prompt_sha256'], 'gold_read': False}
    Path(d, 'THRESHOLD_CANDIDATES.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(cand, indent=0))


if __name__ == '__main__':
    if sys.argv[1] == 'collect':
        collect(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'thresholds':
        thresholds(sys.argv[2])
    else:
        raise SystemExit(__doc__)
