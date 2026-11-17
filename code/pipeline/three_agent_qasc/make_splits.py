"""E1 step 1b: de-duplicate QASC and cut fit / cal / dev / sealed-test ID lists.

Reads only data/qasc/queries.jsonl (no gold file is opened) and configs/split_config.json.
Writes splits/{fit,cal,dev,test_sealed,unused}_ids.json and splits/SPLIT_MANIFEST.json.
Run with: python -I make_splits.py
"""
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

E1 = Path(__file__).resolve().parents[1]
CFG = json.loads((E1 / 'configs/split_config.json').read_text())
QUERIES = E1 / 'data/qasc/queries.jsonl'
OUT = E1 / 'splits'


def norm(s):
    s = unicodedata.normalize('NFKC', str(s)).casefold()
    return re.sub(r'\s+', ' ', s).strip()


def key_sorted(q):
    return norm(q['question']) + ' || ' + ' | '.join(sorted(norm(c) for c in q['choice_text']))


def key_ordered(q):
    return norm(q['question']) + ' || ' + ' | '.join(norm(c) for c in q['choice_text'])


def sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def main():
    rows = [json.loads(x) for x in QUERIES.read_text().splitlines() if x.strip()]
    assert all('answerKey' not in r and 'combinedfact' not in r for r in rows)
    order = {'train': 0, 'validation': 1}
    rows.sort(key=lambda r: order[r['source_split']])  # stable: train rows first, file order kept
    n_raw = len(rows)

    # step 1: question id
    seen, step1, dup_id = set(), [], []
    for r in rows:
        if r['id'] in seen:
            dup_id.append(r['id'])
            continue
        seen.add(r['id'])
        step1.append(r)

    # step 2: normalized stem + choices (sorted choice texts)
    seen_k, step2, dup_text = {}, [], []
    for r in step1:
        k = key_sorted(r)
        if k in seen_k:
            dup_text.append({'dropped': r['id'], 'kept': seen_k[k], 'dropped_source': r['source_split']})
            continue
        seen_k[k] = r['id']
        step2.append(r)
    n_ordered_dups = len(step1) - len({key_ordered(r) for r in step1})

    # descriptive only: distinct questions that share a normalized stem
    stem_count = {}
    for r in step2:
        stem_count[norm(r['question'])] = stem_count.get(norm(r['question']), 0) + 1
    shared_stem_q = sum(c for c in stem_count.values() if c > 1)

    ids = sorted(r['id'] for r in step2)
    n = len(ids)
    sp = CFG['split']
    t = dict(sp['targets_in_order'])
    short = max(0, sum(t.values()) - n)
    cut_fit = min(short, t['fit'] - 1500)
    t['fit'] -= cut_fit
    short -= cut_fit
    cut_dev = min(short, t['dev'] - 800)
    t['dev'] -= cut_dev
    short -= cut_dev
    if short > 0:
        raise SystemExit(f'only {n} questions after de-duplication; rule cannot be met (short {short})')

    perm = np.random.default_rng(sp['seed']).permutation(n)
    shuffled = [ids[i] for i in perm]
    out, pos = {}, 0
    for name in sp['slice_order']:
        out[name] = shuffled[pos:pos + t[name]]
        pos += t[name]
    out['unused'] = shuffled[pos:]
    allsets = [set(v) for v in out.values()]
    assert sum(len(s) for s in allsets) == n and len(set().union(*allsets)) == n

    src = {r['id']: r['source_split'] for r in step2}
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        'queries_sha256': hashlib.sha256(QUERIES.read_bytes()).hexdigest(),
        'split_config_sha256': hashlib.sha256((E1 / 'configs/split_config.json').read_bytes()).hexdigest(),
        'gold_file_opened': False,
        'n_raw_train_plus_validation': n_raw,
        'n_after_id_dedup': len(step1),
        'n_dropped_repeated_id': len(dup_id),
        'n_after_text_dedup': n,
        'n_dropped_repeated_text_sorted_choices': len(dup_text),
        'n_repeated_text_if_choice_order_kept': n_ordered_dups,
        'dropped_text_duplicates': dup_text,
        'descriptive_shared_stems': {
            'distinct_stems': len(stem_count),
            'stems_shared_by_2plus_questions': sum(1 for c in stem_count.values() if c > 1),
            'questions_with_a_shared_stem': shared_stem_q,
            'max_questions_per_stem': max(stem_count.values()),
        },
        'shortfall_cuts': {'fit': cut_fit, 'dev': cut_dev},
        'sizes': {k: len(v) for k, v in out.items()},
        'source_split_mix': {k: {s: sum(1 for i in v if src[i] == s) for s in ['train', 'validation']} for k, v in out.items()},
        'seed': sp['seed'],
        'files': {},
    }
    for name, v in out.items():
        p = OUT / f'{name}_ids.json'
        p.write_text(json.dumps(v, indent=0) + '\n')
        manifest['files'][p.name] = {'n': len(v), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    (OUT / 'SPLIT_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ['dropped_text_duplicates', 'files']}, indent=1))


if __name__ == '__main__':
    sys.exit(main())
