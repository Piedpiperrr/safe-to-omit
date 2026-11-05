"""P2 R2 (E5) CPU re-analysis. Read-only on all existing P2_* folders; writes only under this stage."""
import sys, json, csv, collections
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
ROOT = STAGE.parent
R1 = ROOT / 'P2_R1_CPU_20260919T045556Z'
sys.path.insert(0, str(R1 / 'src'))
sys.dont_write_bytecode = True

from common_r1 import (read, jl, csvread, csvout, tv, pval, cp999, thresholds, deploy,  # noqa: E402
                       GRID, SETTINGS, EXPECTED_Q, BND, MED, MMLU, ZG, V2, BIN, TASK, REF)  # noqa: E402
import data_r1  # noqa: E402
from data_r1 import load_pair_dataset, route_mask, MEDP  # noqa: E402

XFAM = ROOT / 'P2_R1_XFAM_20260919T095058Z'
X1 = XFAM / 'x1'
SCORESENS = ROOT / 'P2_SCORE_SENSITIVITY_20260912T055929Z'
LABEL = 'POST-HOC re-analysis (reviewer R2/E5); does not change any primary decision'
RES = STAGE / 'results'

# 21 settings: 14 main + 7 cross-family
MAIN = list(SETTINGS)
XSETTINGS = [('X1-Llama', 'OBQA', 'Text'), ('X1-Llama', 'OBQA', 'C2C'), ('X1-Llama', 'ARC', 'Text'), ('X1-Llama', 'ARC', 'C2C'),
             ('X2-OLMo', 'OBQA', 'Text'), ('X2-OLMo', 'ARC', 'Text'), ('X2-OLMo', 'MMLU-Pro', 'Text')]
ALL21 = MAIN + XSETTINGS
BENCH = {'OBQA': 'obqa', 'ARC': 'arc', 'MMLU-Pro': 'mmlu_pro'}


def sname(s): return f'{s[0]}/{s[1]}/{s[2]}'


def exposed_ids():
    """X: questions whose raw outputs were reviewed with gold during V2 parser rule development (P2_SCORE_SENSITIVITY
    case_review.jsonl), restricted to rows that later fell into a calibration split. Returns (records, per-benchmark set)."""
    cal = {ds: set(read(BND / f'splits/{ds}_cal_ids.json')) for ds in ['obqa', 'arc']}
    fit = {ds: set(read(BND / f'splits/{ds}_fit_ids.json')) for ds in ['obqa', 'arc']}
    dev = {ds: set(read(BND / f'splits/{ds}_dev_ids.json')) for ds in ['obqa', 'arc']}
    recs = collections.defaultdict(lambda: dict(pairs=set(), actions=set(), reviews=0, reasons=set()))
    allrev = collections.defaultdict(lambda: dict(pairs=set(), actions=set(), splits=set()))
    for r in jl(SCORESENS / 'case_review.jsonl'):
        ds, i = r['dataset'], str(r['id'])
        a = allrev[ds, i]; a['pairs'].add(r['pair']); a['actions'].add(r['action']); a['splits'].add(r['split'])
        if r['split'] == 'train' and i in cal[ds]:
            e = recs[ds, i]; e['pairs'].add(r['pair']); e['actions'].add(r['action']); e['reviews'] += 1
            e['reasons'].add(r['diagnostic_scoring']['reason'])
    X = {ds: {i for (d, i) in recs if d == ds} for ds in ['obqa', 'arc']}
    X['mmlu_pro'] = set()
    return recs, allrev, X, cal, fit, dev


def ledger_rows(u_cal, d_cal, cuts, keep=None):
    """20 tests with FROZEN thresholds `cuts` on the calibration split, optionally restricted to boolean mask `keep`."""
    if keep is not None:
        u_cal, d_cal = u_cal[keep], d_cal[keep]
    rows = []
    for q, t in zip(GRID, cuts):
        m = route_mask(u_cal, q, t); n = int(m.sum()); k = int(d_cal[m].sum())
        rows.append(dict(q=q, threshold=t, N=len(u_cal), n=n, k=k, p=pval(k, n), CP=cp999(k, n), accepted=pval(k, n) <= .001))
    return rows


INV = 'INVALID'


def load_main():
    """14 main settings -> dict with per-split ids, ProbeMax u, answers o_R/o_T/o_C, frozen thresholds."""
    data = {}
    for pair, task in [('small', 'OBQA'), ('small', 'ARC'), ('medium', 'OBQA'), ('medium', 'ARC'),
                       ('large', 'OBQA'), ('large', 'ARC'), ('large', 'MMLU-Pro')]:
        data[pair, task] = load_pair_dataset(pair, task)
    return data


def load_x2():
    """X2 (OLMo): per dataset -> ids/u/answers per split, frozen fit thresholds."""
    out = {}
    rows = {}
    for f in sorted((XFAM / 'results/runs').glob('chain_*.jsonl')):
        for r in jl(f):
            rows[r['dataset'], r['id']] = r
    for ds in ['obqa', 'arc', 'mmlu_pro']:
        D = {}
        for sp in ['fit', 'cal', 'dev']:
            ids = [r['id'] for r in jl(XFAM / f'records/populations/{ds}_{sp}.jsonl') if r['representative']]
            rr = [rows[ds, i] for i in ids]
            D[sp] = dict(ids=ids, u=[x['P']['ProbeMax'] for x in rr],
                         oR=[x['R']['parsed'] if x.get('runtime_error') is None else INV for x in rr],
                         oT=[x['T']['parsed'] if x.get('runtime_error') is None else INV for x in rr])
        D['cuts'] = thresholds(list(D['fit']['u']))
        out[ds] = D
    return out


def load_x1():
    """X1 (Llama helper -> Qwen3-0.6B receiver): R and ProbeMax from the small pair; Text/C2C from X1 runs."""
    ref = {'Text': {}, 'C2C': {}}
    for f in sorted((X1 / 'results/runs').glob('*.jsonl')):
        for r in jl(f):
            k = (r['dataset'], r['id'])
            if 'T' in r and r['T'] is not None and 'parsed' in (r['T'] or {}):
                ref['Text'][k] = r['T']['parsed'] if r['runtime_error'] is None else INV
            if 'C' in r and r['C'] is not None and 'parsed' in (r['C'] or {}):
                ref['C2C'][k] = r['C']['parsed'] if r['runtime_error'] is None else INV
    lab = {}
    for name in ['full_train', 'full_development']:
        for x in jl(V2 / f'labels/{name}_P2_SCORING_V2.jsonl'):
            if x['pair'] == 'small':
                lab[x['dataset'], x['id']] = x
    score = {}
    for ds in ['obqa', 'arc']:
        for sp in ['fit', 'cal', 'dev']:
            for r in jl(BND / f'records/small_{ds}_{sp}_probes.jsonl'):
                score[ds, r['id']] = r['ProbeMax']
    out = {}
    for ds in ['obqa', 'arc']:
        D = {}
        for sp in ['fit', 'cal', 'dev']:
            ids = [r['id'] for r in jl(X1 / f'records/populations/{ds}_{sp}.jsonl') if r['representative']]
            D[sp] = dict(ids=ids, u=[score[ds, i] for i in ids],
                         oR=[(lab[ds, i]['o_R'] if lab[ds, i]['valid_R'] else INV) for i in ids],
                         oT=[ref['Text'][ds, i] for i in ids], oC=[ref['C2C'][ds, i] for i in ids])
        D['cuts'] = thresholds(list(D['fit']['u']))
        out[ds] = D
    return out
