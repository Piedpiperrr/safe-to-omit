"""Fit-split score diagnostics and the label-free plug-in prediction of P0 (PROTOCOL.md sections 4 and 5).

Reads only outputs/full/fit/records.fit.*.jsonl (no gold, no calibration output).
Score u = log non-top mass (e1lib.Models.probe); u_1mp_diag = 1 - max p (the old score) for comparison.

Plug-in rule (the paper's E15-1 / Appendix D rule, unchanged): for candidate j with fit threshold tau_j,
  m_j = # fit questions skipped (u <= tau_j; all N_fit for j = 20), d_j = # of those where s (path AC) and b (path ABC)
  answer differently (two INVALID answers agree; a missing answer after a runtime failure counts as different),
  n_hat = round_half_up(N_cal * m_j / N_fit), k_hat = round_half_up(n_hat * d_j / m_j) (0 if m_j = 0),
  predicted pass iff P[Binomial(n_hat, 0.05) <= k_hat] <= 0.001 (n_hat = 0 -> p = 1, does not pass).
  Predicted P0 = largest predicted-passing candidate, else "always wait for B". Exact rational rounding.
Writes outputs/fit_diagnostics/{FIT_SCORE_DIAGNOSTICS.json, P0_PREDICTION.json, candidates.csv}.
"""
import csv
import glob
import hashlib
import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1lib as L  # noqa: E402

E1 = L.E1
FIT_DIR = E1 / 'outputs/full/fit'
OUT = E1 / 'outputs/fit_diagnostics'


def rhu(x):
    x = Fraction(x)
    return int((x + Fraction(1, 2)).__floor__())


def pval(k, n):
    return float(binom.cdf(k, n, 0.05)) if n else 1.0


def main():
    ids = L.split_ids('fit')
    n_cal = len(L.split_ids('cal'))
    rows = {}
    files = sorted(glob.glob(str(FIT_DIR / 'records.fit.shard*.jsonl')))
    for f in files:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                rows[r['id']] = r
    missing = [i for i in ids if i not in rows]
    assert not missing, f'fit incomplete: {len(missing)} missing'
    N = len(ids)
    pr = {i: rows[i]['steps'].get('probe') for i in ids}
    u = {i: (pr[i]['u'] if pr[i] else None) for i in ids}
    u1 = {i: (pr[i].get('u_1mp_diag') if pr[i] else None) for i in ids}
    assert all(pr[i] is None or pr[i].get('score') == 'log_nontop_mass' for i in ids), 'records were made with another score'

    def ans(i, path):
        s = rows[i]['steps'].get(path)
        return s['answer'] if s else None

    disagree = {i: (ans(i, 'C_AC') is None or ans(i, 'C_ABC') is None or ans(i, 'C_AC') != ans(i, 'C_ABC')) for i in ids}

    finite = [u[i] for i in ids if u[i] is not None and math.isfinite(u[i])]
    counts = {}
    for x in finite:
        counts[x] = counts.get(x, 0) + 1
    old = [u1[i] for i in ids if u1[i] is not None]
    oc = {}
    for x in old:
        oc[x] = oc.get(x, 0) + 1
    # ordering check: where 1 - max p > 0, sorting by u must give the same order (up to FP32 rounding of 1 - max p)
    pairs = sorted((u[i], u1[i]) for i in ids if u[i] is not None and u1[i] is not None and u1[i] > 0)
    inversions = sum(1 for a, b in zip(pairs, pairs[1:]) if b[1] < a[1] * (1 - 1e-6) - 1e-12)

    cand = L.candidate_thresholds([u[i] for i in ids])
    taus = [c['tau'] for c in cand[:19]]
    rows_out = []
    for c in cand:
        sk = [i for i in ids if L.skip(u[i], c['tau'])]
        m = len(sk)
        d = sum(disagree[i] for i in sk)
        n_hat = rhu(Fraction(n_cal * m, N))
        k_hat = rhu(Fraction(n_hat * d, m)) if m > 0 else 0
        p_hat = pval(k_hat, n_hat)
        rows_out.append({'j': c['j'], 'q': c['q'], 'rank': c['rank'], 'tau': c['tau'], 'fit_skipped_m': m,
                         'fit_coverage': m / N, 'fit_disagree_d': d, 'fit_change_rate': (d / m if m else None),
                         'n_hat': n_hat, 'k_hat': k_hat, 'p_hat': p_hat, 'predicted_pass': n_hat > 0 and p_hat <= 0.001})
    passing = [r for r in rows_out if r['predicted_pass']]
    p0 = max(passing, key=lambda r: r['j']) if passing else None

    diag = {
        'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'N_fit': N, 'score': 'log non-top mass (PROTOCOL.md section 4)',
        'nonfinite_or_missing': sum(1 for i in ids if u[i] is None), 'exact_zero': sum(1 for x in finite if x == 0.0),
        'distinct_finite_values': len(counts), 'questions_in_ties': sum(c for c in counts.values() if c > 1),
        'share_in_ties': sum(c for c in counts.values() if c > 1) / N, 'largest_tie': max(counts.values()) if counts else 0,
        'min_u': min(finite), 'median_u': sorted(finite)[len(finite) // 2], 'max_u': max(finite),
        'old_score_1_minus_max_p': {'exact_zero': sum(1 for x in old if x == 0.0), 'share_exact_zero': sum(1 for x in old if x == 0.0) / N,
                                    'questions_in_ties': sum(c for c in oc.values() if c > 1)},
        'ordering_vs_old_score_where_old_positive': {'pairs_checked': max(0, len(pairs) - 1), 'inversions': inversions},
        'candidates_distinct_j1_to_19': len(set(map(str, taus))) == 19, 'n_distinct_tau_j1_to_19': len(set(map(str, taus))),
        'candidates': [{k: r[k] for k in ['j', 'q', 'rank', 'tau', 'fit_skipped_m', 'fit_coverage']} for r in rows_out],
        'fit_disagreement_overall': sum(disagree.values()) / N,
        'records_sha256': {Path(f).name: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        'gold_read': False, 'calibration_output_read': False,
    }
    pred = {
        'utc': diag['utc'], 'rule': 'E15-1 / Appendix D label-free plug-in on fit counts (see module docstring)',
        'N_fit': N, 'N_cal': n_cal, 'alpha': 0.05, 'p_cutoff': 0.001,
        'predicted_P0': ({'j': p0['j'], 'q': p0['q'], 'tau': p0['tau'], 'predicted_coverage': p0['fit_coverage'],
                          'n_hat': p0['n_hat'], 'k_hat': p0['k_hat'], 'p_hat': p0['p_hat']} if p0 else 'always wait for B'),
        'candidates': rows_out, 'records_sha256': diag['records_sha256'], 'gold_read': False, 'calibration_output_read': False,
        'note': 'A wrong prediction does not change the system or the data (PROTOCOL.md section 5).',
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'FIT_SCORE_DIAGNOSTICS.json').write_text(json.dumps(diag, indent=1) + '\n')
    (OUT / 'P0_PREDICTION.json').write_text(json.dumps(pred, indent=1) + '\n')
    with open(OUT / 'candidates.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(json.dumps({k: v for k, v in diag.items() if k not in ['candidates', 'records_sha256']}, indent=1))
    for r in rows_out:
        print(r['j'], r['q'], r['tau'], r['fit_skipped_m'], round(r['fit_coverage'], 4), r['fit_disagree_d'], r['n_hat'], r['k_hat'],
              f"{r['p_hat']:.3g}", r['predicted_pass'])
    print('predicted P0:', pred['predicted_P0'])


if __name__ == '__main__':
    main()
