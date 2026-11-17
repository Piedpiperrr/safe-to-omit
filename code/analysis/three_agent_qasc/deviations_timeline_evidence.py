"""Evidence behind outputs/DEVIATIONS_TIMELINE.md (authors' request, 2026-11-10): milestone times, the commit history of
every DEVIATIONS item, and the item-10 checks. Reads no gold. Writes outputs/deviations_timeline/EVIDENCE.json.

Item 10 checks: (a) e1lib.py unchanged since c666d08 (git diff now, and the sealed-test gate record GATE.json);
(b) the C_ABC renderings saved before the freeze with B prompt v2 (outputs/tune_r2, outputs/smoke; git-ignored because
they quote question text) are compared with the post-freeze fit records by sha256, and checked for B's base sentence,
B's rules, the facts/note headers and clauses, and the fact texts (and in which turn a fact text occurs).
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1lib as L  # noqa: E402

E1 = L.E1


def git(*a):
    return subprocess.run(['git', *a], capture_output=True, text=True, check=True, cwd=E1).stdout


def mt(p):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(os.path.getmtime(E1 / p)))


def jobs_lines(d):
    return [l for l in (E1 / f'outputs/full/{d}/JOBS.txt').read_text().splitlines()]


def main():
    ev = {}
    # commit history of each DEVIATIONS item
    hist = {}
    prev = {}
    for line in filter(None, git('log', '--reverse', '--format=%h %cI', '--', 'DEVIATIONS.md').split('\n')):
        h, t = line.split()
        items, cur = {}, None
        for l in git('show', f'{h}:rebuttal/e1/DEVIATIONS.md').splitlines():
            m = re.match(r'^(\d+)\. \*\*', l)
            if m:
                cur = int(m.group(1))
                items[cur] = []
            if cur:
                items[cur].append(l)
        items = {k: '\n'.join(v).strip() for k, v in items.items()}
        for k, v in items.items():
            if k not in prev:
                hist.setdefault(k, []).append({'commit': h, 'utc': t, 'what': 'added'})
            elif prev[k] != v:
                hist.setdefault(k, []).append({'commit': h, 'utc': t, 'what': 'changed'})
        prev = items
    ev['item_history'] = hist

    # milestones
    rp = {}
    # replay job directories outputs/replay128/<job>; scheduler job ids are replaced by neutral labels in the release
    for j in ('job-46', 'job-47', 'job-48'):
        r = [json.loads(l) for l in open(E1 / f'outputs/replay128/{j}/requests.jsonl')]
        rp[j] = {'policy': json.loads((E1 / f'outputs/replay128/{j}/startup.json').read_text())['policy']['name'],
                 'first_timed_request': r[0]['utc'], 'last_timed_request': r[-1]['utc']}
    ev['milestones'] = {
        'protocol_freeze_c666d08': git('log', '-1', '--format=%cI', 'c666d08').strip(),
        'pre_freeze_runs': {r: json.loads((E1 / f'outputs/{r}/status.fit.shard0of1.json').read_text())['started_utc']
                            for r in ('tune_r1', 'tune_r2', 'smoke')},
        'jobs_lines': {d: [l for l in jobs_lines(d) if 'exit=' in l or 'start=' in l] for d in ('fit', 'cal', 'dev', 'fit_controls', 'test_sealed')},
        'calibration_first_read_COLLECT_cal_mtime': mt('outputs/full/cal/COLLECT_cal.json'),
        'calibration_tests_utc': json.loads((E1 / 'outputs/calibration/CALIBRATION.json').read_text())['utc'],
        'policy_freeze_commit_b762379': git('log', '-1', '--format=%cI', 'b762379').strip(),
        'development_first_read_COLLECT_dev_mtime': mt('outputs/full/dev/COLLECT_dev.json'),
        'fit_controls_first_read_mtime': mt('outputs/full/fit_controls/COLLECT_fit_controls.json'),
        'development_report_mtime': mt('outputs/dev_report/DEV_REPORT.json'),
        'sealed_first_read_GATE_utc': json.loads((E1 / 'outputs/sealed_eval/GATE.json').read_text())['utc'],
        'sealed_evaluation_utc_start': json.loads((E1 / 'outputs/sealed_eval/SEALED_EVAL.json').read_text())['utc_start'],
        'sealed_results_commit_7e9839c': git('log', '-1', '--format=%cI', '7e9839c').strip(),
        'timing_replays': rp,
        'timing_report_commit_1118aa1': git('log', '-1', '--format=%cI', '1118aa1').strip()}

    # item 10 (a)
    gate = json.loads((E1 / 'outputs/sealed_eval/GATE.json').read_text())
    ev['item10_code'] = {
        'git_diff_stat_c666d08_HEAD_e1lib_now': git('diff', '--stat', 'c666d08', 'HEAD', '--', 'code/e1lib.py'),
        'e1lib_sha256_now': L.sha_file(E1 / 'code/e1lib.py'),
        'e1lib_sha256_at_c666d08': hashlib.sha256(subprocess.run(['git', 'show', 'c666d08:rebuttal/e1/code/e1lib.py'], capture_output=True,
                                                                 check=True, cwd=E1).stdout).hexdigest(),
        'gate_utc': gate['utc'], 'gate_git_diff': {k: gate['1_git_diff'][k] for k in ('command', 'output', 'files_matched_at_freeze', 'pass')},
        'gate_shards': len(gate['2_shard_versions']['shards']),
        'gate_shard_e1lib_sha256': sorted({r['e1lib_sha256'] for r in gate['2_shard_versions']['shards']})}

    # item 10 (b)
    bp = L.load_b_prompt(E1 / 'prompts/B_prompt_v2.json')
    Q = L.load_queries()
    full = {}
    for f in glob.glob(str(E1 / 'outputs/full/fit/records.fit.shard*.jsonl')):
        for l in open(f):
            x = json.loads(l)
            full[x['id']] = x
    out = {}
    for run in ('tune_r2', 'smoke'):
        st = json.loads((E1 / f'outputs/{run}/status.fit.shard0of1.json').read_text())
        rec = {json.loads(l)['id']: json.loads(l) for l in open(E1 / f'outputs/{run}/records.fit.shard0of1.jsonl')}
        rows = []
        for l in open(E1 / f'outputs/{run}/rendered.fit.shard0of1.jsonl'):
            x = json.loads(l)
            abc = x['rendered'].get('C_ABC')
            if abc is None:
                continue
            q = Q[x['id']]
            r = {'id': x['id'], 'sha256_equals_post_freeze_fit_record': hashlib.sha256(abc.encode()).hexdigest()
                 == full[x['id']]['steps']['C_ABC']['rendered_sha256'],
                 'has_B_base_sentence': bp['base'] in abc, 'has_B_rules': bp['rules'] in abc,
                 'has_facts_header': bp['facts_header'] in abc, 'has_note_header': bp['note_header'] in abc,
                 'has_facts_clause': bp['facts_clause'].strip() in abc, 'has_note_clause': bp['note_clause'].strip() in abc}
            for fk in ('fact1', 'fact2'):
                ft = q[fk].strip()
                r[f'{fk}_text_in_rendering'] = ft in abc
                if ft in abc:
                    r[f'{fk}_text_in'] = {'B_message': ft in rec[x['id']]['steps']['B']['text'], 'A_message': ft in rec[x['id']]['steps']['A']['text'],
                                          'B_view_user_turn': ft in L.b_user_message(q, bp, facts=False, note=None),
                                          'A_user_turn': ft in L.a_user_message(q), 'final_receiver_prompt': ft in L.prompt.receiver_prompt(q)}
            rows.append(r)
        keys = [k for k in rows[0] if k != 'id' and isinstance(rows[0][k], bool)]
        out[run] = {'started_utc': st['started_utc'], 'b_prompt': st['b_prompt'], 'n_C_ABC_renderings': len(rows),
                    'counts': {k: sum(r[k] for r in rows) for k in keys},
                    'fact_text_locations': [{k: v for k, v in r.items() if k.endswith('_text_in') or k == 'id'} for r in rows
                                            if r['fact1_text_in_rendering'] or r['fact2_text_in_rendering']]}
    ev['item10_pre_freeze_renderings'] = out
    d = E1 / 'outputs/deviations_timeline'
    d.mkdir(parents=True, exist_ok=True)
    (d / 'EVIDENCE.json').write_text(json.dumps(ev, indent=1) + '\n')
    print(json.dumps({k: v for k, v in ev.items() if k != 'milestones'}, indent=1))
    print(json.dumps({k: v for k, v in ev['milestones'].items() if k != 'jobs_lines'}, indent=1))


if __name__ == '__main__':
    main()
