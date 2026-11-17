"""Export the medium/OBQA/Text lost-correction ledger (j = 1..20) and its label query log.

Written at release time (November 2026): the re-analysis computed these in memory (recompute.py, lost_correction()) and
saved only the summary rows (medium_text_lost_correction.csv). This script loads recompute.py from this folder, runs the
same steps with the same frozen risk.py and parser, records the per-candidate values and the queried IDs, and asserts that
every summary value equals the saved CSV. It produced results/cpu_reanalysis/results/medium_text_calibration_ledger.csv and
medium_text_label_queries.jsonl of the dataset.

    python export_medium_ledger.py <output dir> <path to medium_text_lost_correction.csv>

recompute.py resolves its inputs as described in this folder's README.
"""
import csv, importlib.util, json, os, sys
import numpy as np
from scipy.stats import binom
import pyarrow.parquet as pq

R0 = os.environ.get('R0_DIR', os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1]
SAVED = sys.argv[2]
os.makedirs(OUT, exist_ok=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


m = load('r0_recompute', R0 + '/recompute.py')
m.risk = load('r0_frozen_risk', str(m.RISK))
m.parser = load('r0_frozen_parser', str(m.PARSER))
risk = m.risk
t = pq.read_table(m.resolved(m.GOLD_OBQA), columns=['id', 'answerKey'], use_threads=False)
gold = dict(zip(t['id'].to_pylist(), t['answerKey'].to_pylist()))

ids, ans, af = m.medium('obqa', 'cal')
pp = m.unique(m.jl(f'records/{m.MED}/probes/obqa_cal.jsonl'))
u = np.array([pp[i]['ProbeMax'] for i in ids])
r, tt = np.asarray(ans['R']), np.asarray(ans['T'])
d = r != tt
cfg = m.js(f'results/{m.MED}/thresholds/obqa.json')
cuts = cfg['thresholds']
assert cuts == risk.thresholds(cfg['fit_scores']) and cfg['q_grid'] == risk.GRID
ledger, dep = risk.calibrate(u, d, cuts)
assert dep['q'] == 0
saved = m.cr(f'results/{m.MED}/calibration/obqa_T.csv')
for a, b in zip(ledger, saved):
    assert (a['routed'], a['changed']) == (int(b['routed']), int(b['changed']))
    assert abs(a['p_value'] - float(b['p_value'])) < 1e-12 and not a['accepted']

order = sorted(np.flatnonzero(d), key=lambda j: (u[j], ids[j]))
N = len(ids)
per_eps, queries = {}, {}
for eps in (.01, .02):
    queried, lost, at, chosen, first_rej, early = [], 0, 0, None, None, None
    allowable = max(k for k in range(N + 1) if binom.cdf(k, N, eps) <= .001)
    evaluated = {}
    for jj, (q, tau) in enumerate(zip(risk.GRID, cuts), start=1):
        mk = np.asarray(risk.mask(u, q, tau), bool)
        while at < len(order) and mk[order[at]]:
            j = order[at]; at += 1; queried.append(j)
            lc = bool(tt[j] == gold[ids[j]] and r[j] != gold[ids[j]])
            lost += int(lc)
            queries.setdefault(ids[j], {'id': ids[j], 'u': float(u[j]), 'query_candidate_j': jj, 'ans_R': str(r[j]), 'ans_Text': str(tt[j]),
                                        'gold': gold[ids[j]], 'lost_correction': lc, 'eps_requested': []})['eps_requested'].append(eps)
            if early is None and lost > allowable:
                early = len(queried)
        p = float(binom.cdf(lost, N, eps))
        evaluated[jj] = (lost, p, p <= .001)
        if p <= .001:
            chosen = jj
        else:
            first_rej = jj
            break
    per_eps[eps] = dict(evaluated=evaluated, queried=queried, allowable=allowable, early=early, first_rej=first_rej, chosen=chosen)

# compare with the saved summary rows
rows = {float(x['epsilon']): x for x in csv.DictReader(open(SAVED, newline=''))}
for eps, e in per_eps.items():
    x = rows[eps]
    assert len(e['queried']) == int(x['labels_needed_E21_Dseq']) == 36
    assert int(d.sum()) == int(x['D_all']) == 348
    assert e['allowable'] == int(x['cal_max_accepted_lost_count'])
    assert e['early'] == int(x['labels_early_stop_at_rejection'])
    assert e['chosen'] is None and float(x['max_acceptable_q']) == 0.0 and x['loss_test_passed'] == 'False'
    assert risk.GRID[e['first_rej'] - 1] == float(x['first_rejected_q'])
    assert e['evaluated'][e['first_rej']][0] == int(x['first_rejected_lost_count']) == 21
assert sorted(queries) == sorted(ids[j] for j in per_eps[.01]['queried'])

fields = ['j', 'q', 'tau', 'n', 'm', 'p_change', 'change_pass', 'l', 'p_loss_eps0.01', 'pass_eps0.01', 'p_loss_eps0.02', 'pass_eps0.02',
          'labels_queried_at_j', 'selected_label_free', 'selected_eps0.01', 'selected_eps0.02']
out = []
for jj, (row, q, tau) in enumerate(zip(ledger, risk.GRID, cuts), start=1):
    o = {'j': jj, 'q': q, 'tau': repr(float(tau)), 'n': row['routed'], 'm': row['changed'], 'p_change': repr(float(row['p_value'])),
         'change_pass': bool(row['accepted'])}
    for eps in (.01, .02):
        ev = per_eps[eps]['evaluated'].get(jj)
        o['l'] = ev[0] if ev else o.get('l', 'not reached')
        o[f'p_loss_eps{eps}'] = repr(ev[1]) if ev else 'not reached'
        o[f'pass_eps{eps}'] = ev[2] if ev else 'not reached'
    o['labels_queried_at_j'] = sum(1 for v in queries.values() if v['query_candidate_j'] == jj)
    o['selected_label_free'] = False
    o['selected_eps0.01'] = False
    o['selected_eps0.02'] = False
    out.append(o)
with open(os.path.join(OUT, 'medium_text_calibration_ledger.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
    w.writeheader()
    w.writerows(out)
with open(os.path.join(OUT, 'medium_text_label_queries.jsonl'), 'w') as f:
    for k, j in enumerate(per_eps[.01]['queried'], start=1):
        v = dict(queries[ids[j]])
        v['eps_requested'] = sorted(set(v['eps_requested']))
        f.write(json.dumps({'query_ordinal': k, **v}) + '\n')
print('ledger OK: 20 candidates; labels', len(per_eps[.01]['queried']), '; l at j=1', per_eps[.01]['evaluated'][1][0],
      '; p', per_eps[.01]['evaluated'][1][1], per_eps[.02]['evaluated'][1][1])
