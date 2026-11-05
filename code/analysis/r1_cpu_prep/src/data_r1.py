"""POST-HOC: load saved probe vectors, R/reference machine answers and split units for all 14 settings (read-only)."""
import math
import numpy as np
from common_r1 import *

MEDP = ROOT / 'P2_MEDIUM_PAIR_BOUNDARY_EXTENSION_STAGE1_20260915T144407Z'  # parent: splits
CODE = {'Text': 'T', 'C2C': 'C'}


def entropy(p): return -sum(x * math.log(x) for x in p if x > 0)


def margin(p):
    s = sorted(p, reverse=True); return 1 - (s[0] - s[1])


def _scores(prow, probemax):
    p = list(prow['p_labels'].values())
    return dict(ProbeMax=probemax, Entropy=entropy(p), Margin=margin(p))


def load_pair_dataset(pair, task):
    """Returns dict split -> dict(ids, scores{family: np.array}, ans{R,T,C: list}) on the primary units, plus source paths.
    Units: fit/cal = frozen representatives (threshold/calibration unit); dev = the population used for the paper's dev table."""
    src = {}
    out = {}
    if pair in ('small', 'large') and task != 'MMLU-Pro':
        ds = task.lower()
        probes = {}
        if (pair, ds) == ('large', 'obqa'):
            f = ZG / 'records/probe_records.jsonl'; src['probes'] = str(f)
            for r in jl(f): probes[r['id']] = r
        else:
            for s in ['fit', 'cal', 'dev']:
                f = BND / f'records/{pair}_{ds}_{s}_probes.jsonl'; src[f'probes_{s}'] = str(f)
                for r in jl(f): probes[r['id']] = r
        labels = {}
        for s, name in [('train', 'full_train'), ('dev', 'full_development')]:
            f = V2 / f'labels/{name}_P2_SCORING_V2.jsonl'; src[f'labels_{s}'] = str(f)
            with f.open() as fh:
                for line in fh:
                    r = json.loads(line)
                    if r['pair'] == pair and r['dataset'] == ds: labels[s, r['id']] = r
        for s in ['fit', 'cal', 'dev']:
            ids = read(BND / f'splits/{ds}_{s}_representatives.json'); src[f'split_{s}'] = str(BND / f'splits/{ds}_{s}_representatives.json')
            lab = [labels['dev' if s == 'dev' else 'train', i] for i in ids]
            sc = [_scores(probes[i], probes[i]['ProbeMax']) for i in ids]
            out[s] = dict(ids=ids, scores={k: np.array([x[k] for x in sc]) for k in sc[0]},
                          ans={a: [r['o_' + a] for r in lab] for a in 'RTC'})
        out['stored_thresholds'] = read(BND / f'thresholds/{pair}_{ds}.json')['thresholds']
    elif pair == 'medium':
        ds = task.lower()
        for s in ['fit', 'cal', 'dev']:
            fp = MED / f'probes/{ds}_{s}.jsonl'; fa = MED / f'actions/{ds}_{s}.jsonl'
            src[f'probes_{s}'] = str(fp); src[f'actions_{s}'] = str(fa)
            probes = {r['id']: r for r in jl(fp)}
            A = {}
            with fa.open() as fh:
                for line in fh:
                    r = json.loads(line); A[r['id'], r['action']] = r['answer']
            ids = read(MEDP / f'splits/{ds}_{s}_representatives.json')
            if s == 'dev': assert ids == read(MEDP / f'splits/{ds}_dev_ids.json')
            src[f'split_{s}'] = str(MEDP / f'splits/{ds}_{s}_representatives.json')
            sc = [_scores(probes[i], probes[i]['ProbeMax']) for i in ids]
            out[s] = dict(ids=ids, scores={k: np.array([x[k] for x in sc]) for k in sc[0]},
                          ans={a: [A[i, a] for i in ids] for a in 'RTC'})
        out['stored_thresholds'] = read(MED / f'thresholds/{ds}.json')['thresholds']
    else:  # MMLU-Pro
        merged = {r['id']: r for r in jl(MMLU / 'summary/merged_numeric_rows.jsonl')}; src['answers'] = str(MMLU / 'summary/merged_numeric_rows.jsonl')
        probes = {}
        for sh in ['1', '2']:
            for f in sorted((MMLU / f'shards/{sh}/probes').glob('*.jsonl')):
                src[f'probes_shard{sh}_{f.stem}'] = str(f.resolve())
                with f.open() as fh:
                    for line in fh:
                        r = json.loads(line); probes[r['id']] = dict(p_labels=r['p_labels'], ProbeMax=r['ProbeMax'])
        for s in ['fit', 'cal', 'dev']:
            ids = [g['representative_id'] for g in read(MMLU / f'splits/{s}_groups.json')]; src[f'split_{s}'] = str(MMLU / f'splits/{s}_groups.json')
            for i in ids: assert merged[i]['u'] == probes[i]['ProbeMax']
            sc = [_scores(probes[i], merged[i]['u']) for i in ids]
            out[s] = dict(ids=ids, scores={k: np.array([x[k] for x in sc]) for k in sc[0]},
                          ans={a: [merged[i][a] for i in ids] for a in 'RTC'})
        out['stored_thresholds'] = [t['threshold'] if not t['fixed_R'] else 'Infinity' for t in read(MMLU / 'thresholds/fit_thresholds.json')['thresholds']]
    out['sources'] = src
    return out


def disagreement(split, ref):
    b = CODE[ref]; return np.array([x != y for x, y in zip(split['ans']['R'], split['ans'][b])], bool)


def load_all():
    data = {}
    for pair, task in [('small', 'OBQA'), ('small', 'ARC'), ('medium', 'OBQA'), ('medium', 'ARC'), ('large', 'OBQA'), ('large', 'ARC'), ('large', 'MMLU-Pro')]:
        data[pair, task] = load_pair_dataset(pair, task)
    return data


def route_mask(u, q, t):
    if q == 0: return np.zeros(len(u), bool)
    if q == 1: return np.ones(len(u), bool)
    return u <= tv(t)


def ledger_from_scores(fit_u, cal_u, cal_d):
    cuts = thresholds(list(fit_u))
    rows = []
    for q, t in zip(GRID, cuts):
        m = route_mask(cal_u, q, t); n = int(m.sum()); k = int(cal_d[m].sum())
        rows.append(dict(q=q, threshold=t, N=len(cal_u), n=n, k=k, p=pval(k, n), CP=cp999(k, n), accepted=pval(k, n) <= .001))
    return cuts, rows
