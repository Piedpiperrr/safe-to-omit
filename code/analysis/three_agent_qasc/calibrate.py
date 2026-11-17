"""E1 calibration tests (PROTOCOL.md section 5) and the policy freeze.

  python calibrate.py --split cal --records-dir outputs/full/cal --out-dir outputs/calibration --freeze
(--split fit / --no --freeze is only for a dry run of the mechanics on fit data.)

For candidate tau_j (frozen fit thresholds, outputs/fit_diagnostics/THRESHOLD_CANDIDATES.json):
  S_j = calibration questions skipped (u <= tau_j; all questions for j = 20; a non-finite or missing score never skips
        otherwise), n = |S_j|, m = # of S_j where s (path AC) and b (path ABC) answer differently (two INVALID answers
        agree; a missing answer after a runtime failure counts as different), l = # of S_j where b is correct and s is not.
  change test: p_change = P[Bin(n, 0.05) <= m]; pass iff n > 0 and p_change <= 0.001  (no labels)
  loss test:   p_loss = P[Bin(N, eps) <= l], N = all calibration questions; pass iff p_loss <= 0.001
Labels: (1) change test for all 20 candidates without labels; (2) in increasing tau, query gold only for questions newly
skipped at that candidate on which s and b disagree (only those can be lost corrections); (3) a candidate's loss test is
decided after all its disagreements are labeled; (4) for a fixed eps stop at the first candidate failing the loss test;
(5) eps = 0.02 first, its labels cover eps = 0.01; never label beyond the largest candidate passing the change test;
(6) report the number of distinct labeled questions. Query order within a candidate: increasing u, then id.
P0 = largest candidate passing the change test; P1 / P2 = largest candidate passing both tests at eps 0.01 / 0.02;
a policy without a passing candidate always waits for B.
"""
import argparse
import csv
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1lib as L  # noqa: E402

E1 = L.E1


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


class Gold:
    """Answer keys, looked up one question at a time; every lookup is logged by the caller."""

    def __init__(self, allowed):
        self._g = {}
        allowed = set(allowed)
        for line in open(E1 / 'data/qasc/gold.jsonl'):
            g = json.loads(line)
            if g['id'] in allowed:
                self._g[g['id']] = g['answerKey']
        self.queried = []

    def query(self, i):
        self.queried.append(i)
        return self._g[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='cal', choices=['cal', 'fit'])
    ap.add_argument('--records-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--freeze', action='store_true', help='write rebuttal/e1/POLICY_FREEZE.json and outputs/policy/*.json')
    a = ap.parse_args()
    if a.freeze:
        assert a.split == 'cal', 'policies are frozen from the calibration split only'
        assert not (E1 / 'POLICY_FREEZE.json').exists(), 'policies already frozen'

    ids = L.split_ids(a.split)
    rows = {}
    files = sorted(glob.glob(str(Path(a.records_dir) / f'records.{a.split}.shard*.jsonl')))
    for f in files:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                assert r['id'] not in rows, 'duplicate record'
                rows[r['id']] = r
    assert set(rows) == set(ids), f'{a.split} incomplete or extra records'
    bp = {r['b_prompt_sha256'] for r in rows.values()}
    frozen_bp = json.loads((E1 / 'FROZEN_B_PROMPT.json').read_text())['sha256']
    assert bp == {frozen_bp}, 'records not produced with the frozen B prompt'
    N = len(ids)
    thr = json.loads((E1 / 'outputs/fit_diagnostics/THRESHOLD_CANDIDATES.json').read_text())
    cand = thr['candidates']
    assert [c['j'] for c in cand] == list(range(1, 21))

    def ans(i, step):
        s = rows[i]['steps'].get(step)
        return s['answer'] if s else None

    u = {i: (rows[i]['steps']['probe']['u'] if rows[i]['steps'].get('probe') else None) for i in ids}
    dis = {i: (ans(i, 'C_AC') is None or ans(i, 'C_ABC') is None or ans(i, 'C_AC') != ans(i, 'C_ABC')) for i in ids}
    S = [set(i for i in ids if L.skip(u[i], c['tau'])) for c in cand]
    for k in range(1, 20):
        assert S[k - 1] <= S[k], 'skip sets are not nested'

    # (1) change-rate tests, no labels
    table = []
    for c, Sj in zip(cand, S):
        n = len(Sj)
        m = sum(dis[i] for i in Sj)
        p = float(binom.cdf(m, n, 0.05)) if n else 1.0
        table.append({'j': c['j'], 'q': c['q'], 'tau': c['tau'], 'n': n, 'coverage': n / N, 'm': m,
                      'change_rate': (m / n if n else None), 'p_change': p, 'pass_change': n > 0 and p <= 0.001})
    passing = [t['j'] for t in table if t['pass_change']]
    j_max = max(passing) if passing else 0

    # (2)-(5) labels: eps = 0.02 first, increasing tau, never beyond j_max, stop at the first loss failure
    gold = Gold([i for i in ids if dis[i]])
    log = []
    l_by_j, lost_ids = {}, set()
    first_fail = {0.02: None, 0.01: None}
    order = 0
    for t in table:
        j = t['j']
        if j > j_max:
            break
        prev = S[j - 2] if j > 1 else set()
        new = sorted((i for i in S[j - 1] - prev if dis[i]), key=lambda i: (u[i] if u[i] is not None else float('inf'), i))
        for i in new:
            order += 1
            g = gold.query(i)
            lost = (ans(i, 'C_ABC') == g) and (ans(i, 'C_AC') != g)
            if lost:
                lost_ids.add(i)
            log.append({'order': order, 'id': i, 'candidate_j': j, 'eps_round': 0.02, 'u': u[i], 's_answer': ans(i, 'C_AC'),
                        'b_answer': ans(i, 'C_ABC'), 'gold': g, 'lost_correction': lost})
        l_by_j[j] = len(lost_ids)
        if float(binom.cdf(l_by_j[j], N, 0.02)) > 0.001:
            first_fail[0.02] = j
            break
    labeled_through = max(l_by_j) if l_by_j else 0
    for t in table:
        j = t['j']
        t['labels_complete'] = j in l_by_j
        t['l'] = l_by_j.get(j)
        for eps in (0.01, 0.02):
            key = f'eps{eps}'
            if j in l_by_j:
                p = float(binom.cdf(l_by_j[j], N, eps))
                t[f'p_loss_{key}'] = p
                t[f'pass_loss_{key}'] = p <= 0.001
            else:
                t[f'p_loss_{key}'] = None
                t[f'pass_loss_{key}'] = None
    for t in table:  # eps = 0.01 reuses the eps = 0.02 labels; its first failure cannot come later
        if t['labels_complete'] and not t['pass_loss_eps0.01']:
            first_fail[0.01] = t['j']
            break
    if first_fail[0.01] is None and first_fail[0.02] is not None:
        raise RuntimeError('eps 0.01 would need labels beyond the eps 0.02 stop; not expected (p_loss decreases in eps)')

    def best(eps):
        stop = first_fail[eps] if first_fail[eps] is not None else labeled_through + 1
        js = [t['j'] for t in table if t['j'] < stop and t['j'] <= j_max and t['pass_change'] and t['labels_complete']]
        return max(js) if js else None

    pol = {}
    for name, jsel in [('P0', j_max or None), ('P1', best(0.01)), ('P2', best(0.02))]:
        if jsel is None:
            pol[name] = {'policy': 'always wait for B', 'j': None, 'q': None, 'tau': None, 'cal_coverage': 0.0}
        else:
            t = table[jsel - 1]
            pol[name] = {'policy': 'skip B when u <= tau', 'j': jsel, 'q': t['q'], 'tau': t['tau'], 'cal_coverage': t['coverage'],
                         'n': t['n'], 'm': t['m'], 'p_change': t['p_change'], 'l': t['l']}
    assert len(set(gold.queried)) == len(gold.queried) == len(log)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec_sha = {Path(f).name: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
    res = {'utc': utc(), 'split': a.split, 'N': N, 'policies': pol, 'largest_change_pass_j': j_max or None,
           'first_loss_failure_j': {str(k): v for k, v in first_fail.items()}, 'labeled_through_j': labeled_through,
           'distinct_labeled_questions': len(log), 'lost_corrections_found': len(lost_ids), 'tests': table,
           'thresholds_sha256': hashlib.sha256((E1 / 'outputs/fit_diagnostics/THRESHOLD_CANDIDATES.json').read_bytes()).hexdigest(),
           'records_sha256': rec_sha, 'b_prompt_sha256': frozen_bp,
           'family_wise': '20 change tests + 2 x 20 loss tests at 0.001: Bonferroni 0.06 (PROTOCOL.md section 5)'}
    (out / 'CALIBRATION.json').write_text(json.dumps(res, indent=1) + '\n')
    with open(out / 'label_queries.jsonl', 'w') as f:
        for x in log:
            f.write(json.dumps(x) + '\n')
    with open(out / 'candidates.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    if a.freeze:
        freeze = {'frozen': True, 'utc': res['utc'], 'policies': pol, 'b_prompt_sha256': frozen_bp,
                  'calibration': 'outputs/calibration/CALIBRATION.json',
                  'calibration_sha256': hashlib.sha256((out / 'CALIBRATION.json').read_bytes()).hexdigest(),
                  'thresholds_sha256': res['thresholds_sha256'], 'distinct_labeled_questions': len(log)}
        (E1 / 'POLICY_FREEZE.json').write_text(json.dumps(freeze, indent=1) + '\n')
        pdir = E1 / 'outputs/policy'
        pdir.mkdir(parents=True, exist_ok=True)
        for name, p in pol.items():
            (pdir / f'{name}.json').write_text(json.dumps({'frozen': True, 'name': name, 'tau': p['tau'], 'q': p['q'], 'j': p['j'],
                                                           'policy': p['policy'], 'b_prompt_sha256': frozen_bp}, indent=1) + '\n')
    print(f"{'j':>2} {'q':>5} {'n':>5} {'m':>4} {'p_change':>9} chg {'l':>4} {'p_loss.01':>9} {'p_loss.02':>9}")
    for t in table:
        f = lambda x: '-' if x is None else f'{x:.3g}'
        print(f"{t['j']:>2} {t['q']:>5} {t['n']:>5} {t['m']:>4} {f(t['p_change']):>9} {'Y' if t['pass_change'] else 'n':>3} "
              f"{'-' if t['l'] is None else t['l']:>4} {f(t['p_loss_eps0.01']):>9} {f(t['p_loss_eps0.02']):>9}")
    print(json.dumps(pol, indent=1))
    print('distinct labeled:', len(log), 'first loss failure:', first_fail, 'largest change pass:', j_max)


if __name__ == '__main__':
    main()
