"""Fit-split accuracy (gold allowed on fit) and the label-based plug-in prediction of P1 / P2 (authors' step 2, 2026-11-10).

Reads outputs/full/fit (main paths) and, if present, outputs/full/fit_controls (paths C and CF); gold for fit ids only.
Correct = parsed answer equals gold; INVALID is incorrect (PROTOCOL.md section 3).
  accuracies: C alone, C reading the facts (if generated), A+C (s), A+B+C (b); G_B = acc(b) - acc(s) with a paired
  bootstrap 95% percentile interval (2,000 resamples over questions, seed 0).
P1/P2 prediction, same plug-in idea as P0 (E15-1): for candidate j, l_j = # fit questions skipped at j where b is correct
  and s is not (lost corrections); l_hat = round_half_up(N_cal * l_j / N_fit) with N_cal = 3,000;
  predicted loss pass iff P[Binomial(N_cal, eps) <= l_hat] <= 0.001. Predicted P1 (eps 0.01) / P2 (eps 0.02) = largest
  candidate predicted to pass both the change-rate test (P0_PREDICTION.json) and the loss test, scanning j upward and
  stopping at the first predicted loss failure (PROTOCOL.md section 5, rule 4); else "always wait for B".
No calibration output is read. Writes outputs/fit_diagnostics/{FIT_ACCURACY.json, P1_P2_PREDICTION.json, p1p2_candidates.csv}.
"""
import csv
import glob
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1lib as L  # noqa: E402

E1 = L.E1
OUT = E1 / 'outputs/fit_diagnostics'


def rhu(x):
    return int((Fraction(x) + Fraction(1, 2)).__floor__())


def load(d, split='fit'):
    rows = {}
    for f in sorted(glob.glob(str(Path(d) / f'records.{split}.shard*.jsonl'))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                rows[r['id']] = r
    return rows


def main():
    ids = L.split_ids('fit')
    n_cal = len(L.split_ids('cal'))
    R = load(E1 / 'outputs/full/fit')
    assert all(i in R for i in ids), 'fit incomplete'
    RC = load(E1 / 'outputs/full/fit_controls')
    fitset = set(ids)
    gold = {}
    for line in open(E1 / 'data/qasc/gold.jsonl'):
        g = json.loads(line)
        if g['id'] in fitset:
            gold[g['id']] = g['answerKey']
    assert set(gold) == fitset

    def correct(rows, i, step):
        s = rows.get(i, {}).get('steps', {}).get(step)
        return bool(s) and s['answer'] == gold[i]

    paths = {'A+C (s)': (R, 'C_AC'), 'A+B+C (b)': (R, 'C_ABC')}
    if RC and all(i in RC for i in ids):
        paths = {'C alone': (RC, 'C_C'), 'C reads facts': (RC, 'C_CF'), **paths}
    acc = {}
    for name, (rows, step) in paths.items():
        v = np.array([correct(rows, i, step) for i in ids], float)
        acc[name] = {'acc': float(v.mean()), 'correct': int(v.sum()), 'n': len(ids),
                     'invalid': sum(1 for i in ids if (rows[i]['steps'].get(step) or {}).get('invalid'))}
    s = np.array([correct(R, i, 'C_AC') for i in ids], float)
    b = np.array([correct(R, i, 'C_ABC') for i in ids], float)
    rng = np.random.default_rng(0)
    boots = [float((b[ix] - s[ix]).mean()) for ix in (rng.integers(0, len(ids), len(ids)) for _ in range(2000))]
    gb = {'G_B': float((b - s).mean()), 'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
          'b_right_s_wrong': int(((b == 1) & (s == 0)).sum()), 's_right_b_wrong': int(((s == 1) & (b == 0)).sum()),
          'bootstrap': '2,000 paired resamples over questions, seed 0, percentile interval'}

    cand = json.loads((E1 / 'outputs/fit_diagnostics/THRESHOLD_CANDIDATES.json').read_text())['candidates']
    p0 = {r['j']: r for r in json.loads((E1 / 'outputs/fit_diagnostics/P0_PREDICTION.json').read_text())['candidates']}
    u = {i: R[i]['steps']['probe']['u'] if R[i]['steps'].get('probe') else None for i in ids}
    rows = []
    for c in cand:
        sk = [i for i in ids if L.skip(u[i], c['tau'])]
        lost = sum(1 for i in sk if correct(R, i, 'C_ABC') and not correct(R, i, 'C_AC'))
        l_hat = rhu(Fraction(n_cal * lost, len(ids)))
        row = {'j': c['j'], 'q': c['q'], 'tau': c['tau'], 'fit_skipped': len(sk), 'fit_lost_corrections': lost, 'l_hat': l_hat,
               'predicted_change_pass': p0[c['j']]['predicted_pass']}
        for eps in (0.01, 0.02):
            p = float(binom.cdf(l_hat, n_cal, eps))
            row[f'p_loss_hat_eps{eps}'] = p
            row[f'predicted_loss_pass_eps{eps}'] = p <= 0.001
        rows.append(row)
    pred = {}
    for eps, name in ((0.01, 'P1'), (0.02, 'P2')):
        best = None
        for r in rows:  # increasing tau; stop at the first predicted loss failure
            if not r[f'predicted_loss_pass_eps{eps}']:
                break
            if r['predicted_change_pass']:
                best = r
        pred[name] = ({'j': best['j'], 'q': best['q'], 'tau': best['tau'], 'predicted_coverage': best['fit_skipped'] / len(ids),
                       'l_hat': best['l_hat'], 'p_loss_hat': best[f'p_loss_hat_eps{eps}']} if best else 'always wait for B')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'FIT_ACCURACY.json').write_text(json.dumps({'accuracy': acc, 'G_B': gb, 'gold_read': 'fit only',
                                                      'calibration_output_read': False,
                                                      'controls_generated_on_fit': bool(RC)}, indent=1) + '\n')
    (OUT / 'P1_P2_PREDICTION.json').write_text(json.dumps({'rule': __doc__.split('P1/P2 prediction, ')[1].split('No calibration')[0],
                                                          'N_fit': len(ids), 'N_cal': n_cal, 'predicted': pred,
                                                          'P0_predicted_from': 'P0_PREDICTION.json', 'candidates': rows,
                                                          'calibration_output_read': False}, indent=1) + '\n')
    with open(OUT / 'p1p2_candidates.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(acc, indent=1))
    print(json.dumps(gb, indent=1))
    for r in rows:
        print(r['j'], r['q'], r['fit_skipped'], r['fit_lost_corrections'], r['l_hat'], f"{r['p_loss_hat_eps0.01']:.3g}",
              f"{r['p_loss_hat_eps0.02']:.3g}", r['predicted_change_pass'])
    print(json.dumps(pred, indent=1))


if __name__ == '__main__':
    main()
