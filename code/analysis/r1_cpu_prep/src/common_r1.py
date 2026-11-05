"""POST-HOC reviewer re-analysis (R1). Read-only access to existing P2_* folders; writes only under this stage."""
import json, csv, math
from pathlib import Path
from scipy.stats import binom, beta

P = Path(__file__).resolve().parents[1]
ROOT = P.parent
BND = ROOT / 'P2_CONFIDENCE_REFERENCE_BOUNDARIES_20260914T165012Z'
MED = ROOT / 'P2_MEDIUM_PAIR_BOUNDARY_EXTENSION_STAGE1_20260915T144407Z' / 'execution_retry1_20260915T164957Z'
MMLU = ROOT / 'P2_MMLU_PRO_BREADTH_STAGE1_20260916T002337Z'
ZG = ROOT / 'P2_ZERO_GOLD_CONTROLS_20260914T065145Z'
V2 = ROOT / 'P2_SCORING_V2_20260912T191445Z'
BIN = ROOT / 'P2_RISK_CALIBRATION_BINARY_20260914T043954Z'
E1 = ROOT / 'P2_E1_DIRECTION_20260913T053456Z'
LABEL = 'POST-HOC sensitivity analysis (reviewer R1); does not change any primary decision'

SETTINGS = [(p, t, r) for p in ['small', 'medium'] for t in ['OBQA', 'ARC'] for r in ['Text', 'C2C']] + \
           [('large', t, r) for t in ['OBQA', 'ARC'] for r in ['Text', 'C2C']] + \
           [('large', 'MMLU-Pro', r) for r in ['Text', 'C2C']]
EXPECTED_Q = {('small', t, r): 0.0 for t in ['OBQA', 'ARC'] for r in ['Text', 'C2C']}
EXPECTED_Q.update({('medium', 'OBQA', 'Text'): 0.0, ('medium', 'OBQA', 'C2C'): .55, ('medium', 'ARC', 'Text'): 0.0,
                   ('medium', 'ARC', 'C2C'): .60, ('large', 'OBQA', 'Text'): .80, ('large', 'OBQA', 'C2C'): .80,
                   ('large', 'ARC', 'Text'): .95, ('large', 'ARC', 'C2C'): .90, ('large', 'MMLU-Pro', 'Text'): .40,
                   ('large', 'MMLU-Pro', 'C2C'): .40})
TASK = {'obqa': 'OBQA', 'arc': 'ARC'}
REF = {'T': 'Text', 'C': 'C2C', 'Text': 'Text', 'C2C': 'C2C'}
GRID = [j / 20 for j in range(1, 21)]


def read(p): return json.loads(Path(p).read_text())


def jl(p):
    with Path(p).open() as f: return [json.loads(s) for s in f if s.strip()]


def csvread(p): return list(csv.DictReader(Path(p).open()))


def csvout(p, rows):
    rows = list(rows); Path(p).parent.mkdir(parents=True, exist_ok=True)
    with Path(p).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)) if rows else ['empty'])
        w.writeheader(); w.writerows(rows)


def tv(t): return float('inf') if t in ('Infinity', 'inf', float('inf')) else float(t)


def pval(k, n, alpha=.05): return float(binom.cdf(k, n, alpha)) if n else 1.


def cp999(k, n): return float(beta.ppf(.999, k + 1, n - k)) if n and k < n else 1.


def ledgers():
    """Unified 280-row calibration ledger: setting -> list of 20 rows (q, threshold, N, n, k, p, CP, accepted, source)."""
    out = {}
    src = BND / 'summary/calibration_all_160.csv'
    for r in csvread(src):
        s = (r['pair'], TASK[r['dataset']], REF[r['reference']])
        out.setdefault(s, []).append(dict(q=float(r['q']), threshold=r['threshold'], N=int(r['N']), n=int(r['n_R']), k=int(r['changed']),
                                          p=float(r['p_value']), CP=float(r['CP_upper_0_999']), accepted=r['accepted'] == 'True', source=str(src)))
    for ds in ['obqa', 'arc']:
        for b in ['T', 'C']:
            src = MED / f'calibration/{ds}_{b}.csv'
            for r in csvread(src):
                assert r['pair'] == 'medium' and r['task'] == ds and r['reference'] == b
                s = ('medium', TASK[ds], REF[b])
                out.setdefault(s, []).append(dict(q=float(r['q']), threshold=r['threshold'], N=int(r['N_cal']), n=int(r['routed']), k=int(r['changed']),
                                                  p=float(r['p_value']), CP=float(r['CP_upper_0_999']), accepted=r['accepted'] == 'True', source=str(src)))
    src = MMLU / 'calibration/40_test_ledger.csv'
    for r in csvread(src):
        s = ('large', 'MMLU-Pro', REF[r['reference']])
        assert r['stratum'] == f"large/MMLU-Pro/{s[2]}"
        out.setdefault(s, []).append(dict(q=float(r['q']), threshold=r['threshold'], N=int(r['N_cal']), n=int(r['routed']), k=int(r['changed']),
                                          p=float(r['p_value']), CP=float(r['CP_upper_0_999']), accepted=r['accepted'] == 'True', source=str(src)))
    assert sorted(out) == sorted(SETTINGS) and all(len(v) == 20 and [r['q'] for r in v] == GRID for v in out.values())
    return out


def deploy(rows, alpha=.05, delta=.001):
    """Primary rule: largest q with P[Bin(n,alpha)<=k]<=delta; 0 = fallback."""
    acc = [r['q'] for r in rows if pval(r['k'], r['n'], alpha) <= delta]
    return (acc[-1] if acc else 0.), acc


def deploy_fixed_sequence(rows, alpha=.05, level=.02):
    """Fixed-sequence: ascending q, each at `level`; stop at first non-acceptance; deploy last accepted."""
    last = 0.
    for r in rows:
        if pval(r['k'], r['n'], alpha) <= level: last = r['q']
        else: break
    return last


def thresholds(scores):
    """Fit-split order statistics ceil(j*N/20), j=1..19; q=1 -> Infinity (fixed R). Identical to risk.py / run_boundaries.py."""
    vals = sorted(scores); n = len(vals); assert n and all(math.isfinite(v) for v in vals)
    return [vals[(j * n + 19) // 20 - 1] if j < 20 else 'Infinity' for j in range(1, 21)]
