"""Smoke-test checks and the full-run cost estimate (CPU, login node; tokenizers only).

  python smoke_report.py outputs/smoke [outputs/tune_rX ...]
Checks on the smoke records: every path ran for all questions, no runtime failure, rendered inputs re-tokenize to the
recorded token counts, probe input = C(A) input + 4 prefix tokens, parser output for every raw answer, truncation.
Estimate: warm per-call means (smoke, plus main-path calls from tuning rounds if given) times the call counts of the
full run, plus measured per-question overhead and model load. Writes <smoke>/SMOKE_SUMMARY.json and SMOKE_TABLES.md.
"""
import glob
import json
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1lib as L  # noqa: E402

N = {'fit': 2000, 'cal': 3000, 'dev': 1000}
MAIN = ['A', 'probe', 'C_AC', 'B', 'C_ABC']
CTRL = ['C_C', 'C_CF', 'B_noA', 'C_BC', 'B_nofact', 'C_ABnfC']


def rows_of(d):
    out = []
    for f in sorted(glob.glob(str(Path(d) / 'records.*.jsonl'))):
        out += [json.loads(x) for x in open(f) if x.strip()]
    return out


def stats(v):
    v = list(v)
    return {'n': len(v), 'mean': st.mean(v), 'median': st.median(v), 'min': min(v), 'max': max(v)} if v else None


def per_question_wall(d):
    """(ended - started) / n for each status file; started_utc is written after model load, so this already
    excludes loading and includes record writing and Python overhead."""
    res = []
    for f in glob.glob(str(Path(d) / 'status.*.json')):
        s = json.loads(open(f).read())
        if not s.get('ended_utc') or not s.get('n_done_this_run'):
            continue
        t = (datetime.fromisoformat(s['ended_utc'].replace('Z', '+00:00')) -
             datetime.fromisoformat(s['started_utc'].replace('Z', '+00:00'))).total_seconds()
        res.append({'file': Path(f).name, 'per_q_s': t / s['n_done_this_run'], 'load_s': s['load_seconds_total'], 'n': s['n_done_this_run']})
    return res


def main(smoke, tunes):
    from transformers import AutoTokenizer
    rt = AutoTokenizer.from_pretrained(L.CONFIG['models']['receiver']['path'], local_files_only=True)
    R = rows_of(smoke)
    rep = {'smoke_dir': smoke, 'n_questions': len(R)}
    steps = MAIN + CTRL
    rep['all_steps_ran'] = {s: sum(1 for r in R if r['steps'].get(s)) for s in steps}
    rep['runtime_failures'] = [f | {'id': r['id']} for r in R for f in r['failures']]
    tab = {}
    for s in steps:
        vals = [r['steps'][s] for r in R if r['steps'].get(s)]
        tab[s] = {'n_in': stats(v['n_in'] for v in vals),
                  'n_out': stats(v['n_out'] for v in vals) if s != 'probe' else None,
                  'ms_cold_first_question': next((r['ms'][s] for r in R if r.get('first_after_load') and s in r['ms']), None),
                  'ms_warm': stats(r['ms'][s] for r in R if not r.get('first_after_load') and s in r['ms'])}
        if s.startswith('C_'):
            tab[s]['invalid'] = sum(v['invalid'] for v in vals)
            tab[s]['hit_64'] = sum(v['hit_max'] for v in vals)
        if s in ('A', 'B', 'B_noA', 'B_nofact'):
            tab[s]['truncated_256'] = sum(v['truncated'] for v in vals)
    rep['per_step'] = tab
    rep['probe_n_in_eq_C_AC_plus_4'] = all(r['steps']['probe']['n_in'] == r['steps']['C_AC']['n_in'] + 4 for r in R
                                           if r['steps'].get('probe') and r['steps'].get('C_AC'))
    # re-tokenize the saved rendered receiver inputs
    ret = []
    for f in sorted(glob.glob(str(Path(smoke) / 'rendered.*.jsonl'))):
        for line in open(f):
            x = json.loads(line)
            r = next(rr for rr in R if rr['id'] == x['id'])
            for s, text in x['rendered'].items():
                ret.append(len(rt(text)['input_ids']) == r['steps'][s]['n_in'])
    rep['rendered_retokenize_matches'] = f'{sum(ret)}/{len(ret)}'
    # parser on every raw answer
    rep['parse_examples'] = {s: sorted({(r['steps'][s]['raw'], r['steps'][s]['answer']) for r in R if r['steps'].get(s)}) for s in steps if s.startswith('C_')}
    rep['probe_u'] = [r['steps']['probe']['u'] for r in R if r['steps'].get('probe')]

    # ---------------- estimate ----------------
    warm = {s: [r['ms'][s] for r in R if not r.get('first_after_load') and s in r['ms']] for s in steps}
    src = {s: 'smoke' for s in steps}
    for d in tunes:
        for r in rows_of(d):
            if r.get('first_after_load'):
                continue
            for s in MAIN:
                if s in r['ms']:
                    warm[s].append(r['ms'][s])
                    src[s] = 'smoke+tuning'
    mean = {s: st.mean(v) / 1000 for s, v in warm.items() if v}
    main_s = sum(mean[s] for s in MAIN)
    ctrl_s = sum(mean[s] for s in CTRL)
    walls = per_question_wall(smoke) + sum((per_question_wall(d) for d in tunes), [])
    timed_sum = {}
    for d in [smoke] + tunes:
        for r in rows_of(d):
            if not r.get('first_after_load'):
                timed_sum.setdefault(d, []).append(sum(r['ms'].values()) / 1000)
    overhead = []
    for d in [smoke] + tunes:
        ws = per_question_wall(d)
        if ws and timed_sum.get(d):
            overhead.append(st.mean(x['per_q_s'] for x in ws) - st.mean(timed_sum[d]))
    over = max(0.0, st.mean(overhead)) if overhead else 0.0
    load = st.mean(w['load_s'] for w in walls) if walls else 120.0
    q_main = main_s + over
    q_dev = main_s + ctrl_s + over
    pair_s = {'fit': N['fit'] * q_main, 'cal': N['cal'] * q_main, 'dev': N['dev'] * q_dev}
    total_pair_s = sum(pair_s.values())
    pairs_per_node = 4
    est = {
        'warm_mean_s_per_call': mean, 'source_of_means': src, 'per_question_overhead_s': over, 'model_load_s': load,
        'per_question_s': {'main_paths': q_main, 'dev_main_plus_controls': q_dev},
        'pair_hours_by_split': {k: v / 3600 for k, v in pair_s.items()},
        'pair_hours_total': total_pair_s / 3600,
        'GPU_hours_total_2_GPUs_per_pair': 2 * total_pair_s / 3600,
        'layout': 'one 8-GPU node = 4 helper/receiver pairs, each pair one shard; 3 jobs (fit, cal, dev)',
        'node_wall_hours_by_split': {k: (v / pairs_per_node + load) / 3600 for k, v in pair_s.items()},
        'node_hours_total': sum((v / pairs_per_node + load) / 3600 for v in pair_s.values()),
    }
    est['GPU_hours_allocated_8_per_node'] = 8 * est['node_hours_total']
    est['suggested_walltime_h'] = {k: round(1.5 * v + 0.25, 2) for k, v in est['node_wall_hours_by_split'].items()}
    rep['estimate'] = est
    Path(smoke, 'SMOKE_SUMMARY.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    lines = ['| step | n_in mean (min-max) | n_out mean (max) | cold ms | warm ms mean / median / max | INVALID | trunc/hit |', '|---|---|---|---|---|---|---|']
    for s in steps:
        t = tab[s]
        ni, no, w = t['n_in'], t['n_out'], t['ms_warm']
        lines.append(f"| {s} | {ni['mean']:.0f} ({ni['min']}-{ni['max']}) | " + (f"{no['mean']:.1f} ({no['max']})" if no else '-') +
                     f" | {t['ms_cold_first_question']:.0f} | {w['mean']:.0f} / {w['median']:.0f} / {w['max']:.0f} | "
                     f"{t.get('invalid', '-')} | {t.get('truncated_256', t.get('hit_64', '-'))} |")
    Path(smoke, 'SMOKE_TABLES.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(json.dumps({k: v for k, v in rep.items() if k not in ['per_step', 'parse_examples']}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
