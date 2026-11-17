"""Version gate before the sealed-test evaluation (authors' conditions, 2026-11-10). Any failure -> exit 1, no evaluation.

  1. `git diff --stat c666d08 HEAD -- rebuttal/e1/code/e1lib.py rebuttal/e1/configs rebuttal/e1/prompts` is empty
     (run from the repository root; each pathspec must match files at c666d08, so an empty diff is not vacuous).
  2. The status file of every shard of fit (4), cal (6), dev (4), fit_controls (1) and test_sealed (4) is complete and
     records the same e1lib_sha256 (6417a0a7...) and B prompt sha256 (9a9653c6...); every record carries that B prompt sha.
  3. Sealed merge check: 2,000/2,000 records, no duplicates, no step failing after its retry.
  4. Consistency gate: every ClusterA job that started on one of these shards (JOBS.txt lines with cluster=clusterA) has
     outputs/clusterA/<job>/CHECK.json with gate_pass true. ClusterB is the reference platform (no gate).
Reads no gold. Writes outputs/sealed_eval/GATE.json.
"""
import glob
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

E1 = Path(__file__).resolve().parents[1]
ROOT = E1.parents[1]
FREEZE = 'c666d08'
PATHS = ['rebuttal/e1/code/e1lib.py', 'rebuttal/e1/configs', 'rebuttal/e1/prompts']
SHARDS = {'fit': ('fit', 4), 'cal': ('cal', 6), 'dev': ('dev', 4), 'fit_controls': ('fit', 1), 'test_sealed': ('test_sealed', 4)}
STEPS_MAIN = ['A', 'probe', 'C_AC', 'B', 'C_ABC']


def git(*a):
    return subprocess.run(['git', '-C', str(ROOT), *a], capture_output=True, text=True, check=True).stdout


def main():
    res, ok = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'head': git('rev-parse', 'HEAD').strip()}, True

    # 1. code / config / prompt unchanged since the protocol freeze
    diff = git('diff', '--stat', FREEZE, 'HEAD', '--', *PATHS)
    matched = {p: len(git('ls-tree', '-r', '--name-only', FREEZE, '--', p).split()) for p in PATHS}
    wt = git('status', '--porcelain', '--', *PATHS)
    g1 = diff.strip() == '' and all(matched.values()) and wt.strip() == ''
    res['1_git_diff'] = {'command': f'git diff --stat {FREEZE} HEAD -- ' + ' '.join(PATHS), 'output': diff,
                         'files_matched_at_freeze': matched, 'uncommitted_changes': wt, 'pass': g1}
    ok &= g1

    # 2. same e1lib and B prompt in every shard
    e1lib_freeze = hashlib.sha256(subprocess.run(['git', '-C', str(ROOT), 'show', f'{FREEZE}:rebuttal/e1/code/e1lib.py'],
                                                 capture_output=True, check=True).stdout).hexdigest()
    bp = json.loads((E1 / 'FROZEN_B_PROMPT.json').read_text())['sha256']
    assert e1lib_freeze.startswith('6417a0a7') and bp.startswith('9a9653c6')
    shard_rows, g2 = [], True
    for od, (split, n) in SHARDS.items():
        for k in range(n):
            st = E1 / f'outputs/full/{od}/status.{split}.shard{k}of{n}.json'
            s = json.loads(st.read_text()) if st.exists() else {}
            rec = E1 / f'outputs/full/{od}/records.{split}.shard{k}of{n}.jsonl'
            rec_bp = {json.loads(l)['b_prompt_sha256'] for l in open(rec) if l.strip()} if rec.exists() else set()
            row = {'shard': f'{od}/{split}.shard{k}of{n}', 'status_exists': st.exists(), 'complete': s.get('complete'),
                   'e1lib_sha256': s.get('e1lib_sha256'), 'b_prompt_sha256': (s.get('b_prompt') or {}).get('sha256'),
                   'records_b_prompt_sha256': sorted(rec_bp), 'host': (s.get('env') or {}).get('host'),
                   'job_id': (s.get('env') or {}).get('job_id')}
            row['pass'] = (row['complete'] is True and row['e1lib_sha256'] == e1lib_freeze and row['b_prompt_sha256'] == bp
                           and row['records_b_prompt_sha256'] == [bp])
            g2 &= row['pass']
            shard_rows.append(row)
    res['2_shard_versions'] = {'expected_e1lib_sha256': e1lib_freeze, 'expected_b_prompt_sha256': bp, 'shards': shard_rows,
                               'pass': g2}
    ok &= g2

    # 3. sealed merge check (no gold)
    ids = json.loads((E1 / 'splits/test_sealed_ids.json').read_text())
    rows, dup = {}, 0
    for f in sorted(glob.glob(str(E1 / 'outputs/full/test_sealed/records.test_sealed.shard*.jsonl'))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                dup += r['id'] in rows
                rows[r['id']] = r
    fails = {s: sum(1 for r in rows.values() if r['steps'].get(s) is None) for s in STEPS_MAIN}
    merge = {'n_expected': len(ids), 'n_records': len(rows), 'duplicates': dup, 'missing': sum(i not in rows for i in ids),
             'extra': sum(i not in set(ids) for i in rows), 'step_failures_after_retry': fails,
             'steps_succeeded_on_retry': sum(1 for r in rows.values() for x in r['failures'] if x['status'] == 'succeeded_on_retry'),
             'invalid_answers': {s: sum(1 for r in rows.values() if (r['steps'].get(s) or {}).get('invalid')) for s in ('C_AC', 'C_ABC')}}
    g3 = (merge['n_records'] == len(ids) == 2000 and not dup and not merge['missing'] and not merge['extra']
          and not any(fails.values()))
    merge['pass'] = g3
    res['3_sealed_merge'] = merge
    ok &= g3

    # 4. consistency gate of every ClusterA job that worked on these shards
    jobs = {}
    for od in SHARDS:
        p = E1 / f'outputs/full/{od}/JOBS.txt'
        for line in (p.read_text().splitlines() if p.exists() else []):
            m = re.match(r'job=(\S+) host=(\S+) .*split=(\S+) paths=\S+ shard=(\S+) .*', line)
            if m and 'cluster=clusterA' in line:
                jid = m.group(1).split('.')[0]
                jobs.setdefault(jid, set()).add(f'{od}/{m.group(4)}')
    clusterA = {}
    for jid, sh in sorted(jobs.items()):
        c = E1 / f'outputs/clusterA/{jid}/CHECK.json'
        gp = json.loads(c.read_text())['gate_pass'] if c.exists() else None
        clusterA[jid] = {'shards': sorted(sh), 'check_json': str(c.relative_to(E1)), 'gate_pass': gp}
    g4 = all(v['gate_pass'] is True for v in clusterA.values())
    res['4_consistency_gates'] = {'clusterA_jobs': clusterA, 'clusterB': 'reference platform, no gate', 'pass': g4}
    ok &= g4

    res['gate_pass'] = bool(ok)
    out = E1 / 'outputs/sealed_eval'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'GATE.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({'gate_pass': res['gate_pass'], '1_git_diff': {k: res['1_git_diff'][k] for k in ('output', 'files_matched_at_freeze', 'pass')},
                      '2_shard_versions': {'n_shards': len(shard_rows), 'pass': g2,
                                           'failing': [r['shard'] for r in shard_rows if not r['pass']]},
                      '3_sealed_merge': merge, '4_consistency_gates': res['4_consistency_gates']}, indent=1))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
