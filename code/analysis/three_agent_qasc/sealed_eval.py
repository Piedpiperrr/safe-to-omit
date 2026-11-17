"""Sealed-test evaluation, PROTOCOL.md Section 6 (2,000 questions; run once, after the authors released it on 2026-11-10).

Requires outputs/sealed_eval/GATE.json with gate_pass true (code/sealed_gate.py) and POLICY_FREEZE.json; refuses to run
if outputs/sealed_eval/SEALED_EVAL.json already exists (results are written only at the end, so a run that stops on an
error leaves nothing behind and the same computation can be rerun after a fix, recorded in DEVIATIONS.md).

Definitions (frozen; same as calibration and the development report):
  s = path A+C, b = path A+B+C; correct = parsed answer equals gold; INVALID is incorrect; two INVALID answers agree;
  a missing answer after a runtime failure counts as different (none occur: the gate requires 0 runtime failures).
  For a frozen policy (tau): S = sealed questions with finite u <= tau; n = |S|;
  m = # of S where s and b answer differently; l = # of S where b is correct and s is not (lost corrections);
  pi answers with s on S and with b elsewhere.
  R1 (change rate): two-sided 95% Clopper-Pearson upper bound of m / n below 0.05; n = 0 fails.
  R2 (lost corrections): two-sided 95% Clopper-Pearson upper bound of l / 2000 below epsilon
       (P1: 0.01, P2: 0.02; P0 has no epsilon of its own and is reported at both).
  Gains: G_B = acc(b) - acc(s), G_pi = acc(pi) - acc(s), G_pi - G_B / 2; paired bootstrap 95% percentile intervals,
       2,000 resamples over questions, numpy default_rng(seed 0); the same resample indices for every quantity/policy.
  Per policy, each criterion separately: R1, R2, lower(G_B) > 0, lower(G_pi) > 0; the timing criterion (Section 8) is
  reported separately from the replay jobs. "Keeps at least half of the gain" only if lower(G_pi - G_B / 2) > 0.
Writes outputs/sealed_eval/SEALED_EVAL.json and SEALED_EVAL.md.
Mechanics check only: --dry-run-split dev --out-dir <scratch> runs the same computation on the development split (whose
gold the development report already reads), skipping the gate and the run-once guard; it never touches the sealed split.
"""
import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1lib as L  # noqa: E402

E1 = L.E1
OUT = E1 / 'outputs/sealed_eval'
SEED, REPS = 0, 2000


def cp_upper(k, n):
    """Upper end of the two-sided 95% Clopper-Pearson interval (one-sided 97.5%)."""
    if n == 0:
        return None
    u = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    v = float(binomtest(k, n).proportion_ci(confidence_level=0.95, method='exact').high)
    assert abs(u - v) < 1e-12, (k, n, u, v)
    return u


def self_test():
    """The worked examples of PROTOCOL.md Section 6."""
    t = {'R1_n400_m11_pass': cp_upper(11, 400) < 0.05, 'R1_n400_m12_fail': not cp_upper(12, 400) < 0.05,
         'R2_eps0.01_l11_pass': cp_upper(11, 2000) < 0.01, 'R2_eps0.01_l12_fail': not cp_upper(12, 2000) < 0.01,
         'R2_eps0.02_l27_pass': cp_upper(27, 2000) < 0.02, 'R2_eps0.02_l28_fail': not cp_upper(28, 2000) < 0.02}
    assert all(t.values()), t
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run-split', choices=['dev'])
    ap.add_argument('--out-dir')
    a = ap.parse_args()
    split, out = 'test_sealed', OUT
    if a.dry_run_split:
        split, out = a.dry_run_split, Path(a.out_dir)
        assert out.resolve() != OUT.resolve()
    else:
        gate = json.loads((OUT / 'GATE.json').read_text())
        assert gate['gate_pass'] is True, 'version gate not passed'
        assert not (OUT / 'SEALED_EVAL.json').exists(), 'sealed evaluation already run'
    freeze = json.loads((E1 / 'POLICY_FREEZE.json').read_text())
    for name, p in freeze['policies'].items():
        pj = json.loads((E1 / f'outputs/policy/{name}.json').read_text())
        assert pj['frozen'] and pj['tau'] == p['tau'] and pj['j'] == p['j'] and pj['b_prompt_sha256'] == freeze['b_prompt_sha256']
    st = self_test()

    ids = L.split_ids(split)
    rows = {}
    files = sorted(glob.glob(str(E1 / f'outputs/full/{split}/records.{split}.shard*.jsonl')))
    for f in files:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                rows[r['id']] = r
    assert set(rows) == set(ids) and len(ids) == (2000 if split == 'test_sealed' else 1000)
    assert {r['b_prompt_sha256'] for r in rows.values()} == {freeze['b_prompt_sha256']}
    idset = set(ids)
    gold = {}
    for line in open(E1 / 'data/qasc/gold.jsonl'):
        g = json.loads(line)
        if g['id'] in idset:
            gold[g['id']] = g['answerKey']
    assert set(gold) == idset

    def ans(i, step):
        s = rows[i]['steps'].get(step)
        return s['answer'] if s else None

    N = len(ids)
    s = np.array([ans(i, 'C_AC') == gold[i] for i in ids], float)
    b = np.array([ans(i, 'C_ABC') == gold[i] for i in ids], float)
    dis = np.array([ans(i, 'C_AC') is None or ans(i, 'C_ABC') is None or ans(i, 'C_AC') != ans(i, 'C_ABC') for i in ids])
    u = [rows[i]['steps']['probe']['u'] if rows[i]['steps'].get('probe') else None for i in ids]
    rng = np.random.default_rng(SEED)
    IX = rng.integers(0, N, size=(REPS, N))

    def ci(x):
        bs = x[IX].mean(axis=1)
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    gb = b - s
    res = {'utc_start': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'N': N, 'self_test_protocol_examples': st,
           'bootstrap': {'resamples': REPS, 'seed': SEED, 'rng': 'numpy.random.default_rng', 'interval': 'percentile 2.5/97.5',
                         'paired': 'same resample indices for every quantity and policy'},
           'accuracy': {'s (A+C)': {'acc': float(s.mean()), 'correct': int(s.sum()),
                                    'invalid': sum(ans(i, 'C_AC') == 'INVALID' for i in ids)},
                        'b (A+B+C)': {'acc': float(b.mean()), 'correct': int(b.sum()),
                                      'invalid': sum(ans(i, 'C_ABC') == 'INVALID' for i in ids)}},
           'disagreements_all': int(dis.sum()),
           'G_B': {'G_B': float(gb.mean()), 'ci95': ci(gb), 'lower_gt_0': ci(gb)[0] > 0,
                   'b_right_s_wrong': int(((b == 1) & (s == 0)).sum()), 's_right_b_wrong': int(((s == 1) & (b == 0)).sum())},
           'policies': {}}
    eps_of = {'P0': (0.01, 0.02), 'P1': (0.01,), 'P2': (0.02,)}
    for name, p in freeze['policies'].items():
        sk = np.array([p['tau'] is not None and L.skip(u[k], p['tau']) for k in range(N)])
        n = int(sk.sum())
        m = int((sk & dis).sum())
        lost = int((sk & (b == 1) & (s == 0)).sum())
        pi = np.where(sk, s, b)
        gpi, half = pi - s, (pi - s) - gb / 2
        r1u = cp_upper(m, n)
        r2u = cp_upper(lost, N)
        pol = {'j': p['j'], 'q': p['q'], 'tau': p['tau'], 'skipped': n, 'coverage': n / N, 'n': n, 'm': m, 'l': lost,
               'change_rate': (m / n if n else None),
               'R1': {'cp_upper_95_two_sided': r1u, 'threshold': 0.05, 'pass': bool(n > 0 and r1u < 0.05)},
               'R2': {'cp_upper_95_two_sided': r2u, 'by_epsilon': {str(e): {'threshold': e, 'pass': bool(r2u < e)} for e in eps_of[name]}},
               'acc_pi': float(pi.mean()), 'correct_pi': int(pi.sum()),
               'G_pi': {'G_pi': float(gpi.mean()), 'ci95': ci(gpi)},
               'G_pi_minus_half_G_B': {'value': float(half.mean()), 'ci95': ci(half)}}
        crit = {'R1_pass': pol['R1']['pass'],
                **{f'R2_pass_eps{e}': pol['R2']['by_epsilon'][str(e)]['pass'] for e in eps_of[name]},
                'G_B_lower_gt_0': res['G_B']['lower_gt_0'], 'G_pi_lower_gt_0': pol['G_pi']['ci95'][0] > 0,
                'timing_saving_lower_gt_0': 'pending: Section 8 replay ' + {'P0': 'job-39', 'P1': 'job-38', 'P2': 'job-40'}[name]}
        pol['criteria'] = crit
        pol['keeps_at_least_half_of_gain'] = pol['G_pi_minus_half_G_B']['ci95'][0] > 0
        res['policies'][name] = pol
    res['records_sha256'] = {Path(f).name: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
    res['split'] = split
    res['gate_sha256'] = None if a.dry_run_split else hashlib.sha256((OUT / 'GATE.json').read_bytes()).hexdigest()
    res['policy_freeze_sha256'] = hashlib.sha256((E1 / 'POLICY_FREEZE.json').read_bytes()).hexdigest()
    res['utc_end'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

    f4 = lambda x: f'{x:.4f}'  # noqa: E731
    md = ['| | P0 | P1 | P2 |', '|---|---|---|---|']
    P = [res['policies'][k] for k in ('P0', 'P1', 'P2')]
    md.append('| j / q | ' + ' | '.join(f"{x['j']} / {x['q']}" for x in P) + ' |')
    md.append('| skipped n (coverage) | ' + ' | '.join(f"{x['n']} ({x['coverage']:.3f})" for x in P) + ' |')
    md.append('| m (changed among skipped) | ' + ' | '.join(str(x['m']) for x in P) + ' |')
    md.append('| R1 CP upper (< 0.05) | ' + ' | '.join(f"{f4(x['R1']['cp_upper_95_two_sided'])} {'pass' if x['R1']['pass'] else 'FAIL'}" for x in P) + ' |')
    md.append('| l (lost corrections) | ' + ' | '.join(str(x['l']) for x in P) + ' |')
    md.append('| R2 CP upper | ' + ' | '.join(f4(x['R2']['cp_upper_95_two_sided']) + ' ' + ', '.join(
        f"eps {e}: {'pass' if v['pass'] else 'FAIL'}" for e, v in x['R2']['by_epsilon'].items()) for x in P) + ' |')
    md.append('| acc(pi) | ' + ' | '.join(f4(x['acc_pi']) for x in P) + ' |')
    md.append('| G_pi [95% CI] | ' + ' | '.join(f"{f4(x['G_pi']['G_pi'])} [{f4(x['G_pi']['ci95'][0])}, {f4(x['G_pi']['ci95'][1])}]" for x in P) + ' |')
    md.append('| G_pi - G_B/2 [95% CI] | ' + ' | '.join(
        f"{f4(x['G_pi_minus_half_G_B']['value'])} [{f4(x['G_pi_minus_half_G_B']['ci95'][0])}, {f4(x['G_pi_minus_half_G_B']['ci95'][1])}]" for x in P) + ' |')
    md.append('')
    md.append(f"acc(s) {f4(s.mean())}, acc(b) {f4(b.mean())}, G_B {f4(gb.mean())} [{f4(res['G_B']['ci95'][0])}, {f4(res['G_B']['ci95'][1])}]")
    out.mkdir(parents=True, exist_ok=True)
    (out / 'SEALED_EVAL.md').write_text('\n'.join(md) + '\n')
    (out / 'SEALED_EVAL.json').write_text(json.dumps(res, indent=1) + '\n')
    print('\n'.join(md))
    print(json.dumps({k: v for k, v in res.items() if k not in ('records_sha256',)}, indent=1))


if __name__ == '__main__':
    main()
