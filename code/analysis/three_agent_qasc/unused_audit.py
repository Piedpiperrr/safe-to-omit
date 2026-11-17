"""Audit of the 1,051 QASC questions left out of the four splits (authors' question, 2026-11-10).

Checks (no gold is read):
  1. the five id lists in splits/ partition the 9,051 de-duplicated questions exactly (data/qasc/queries.jsonl holds all
     9,060 raw questions; the 9 text duplicates dropped are listed in the manifest) and match
     the sizes and sha256 in splits/SPLIT_MANIFEST.json; smoke/tune ids are subsets of fit, the timing panel of dev;
  2. every file under outputs/ (git-ignored generation records, rendered prompts, ClusterA check dirs, replay requests,
     smoke/tuning/debug runs included) and logs/ is scanned for 30-character QASC ids; any id from unused_ids.json is
     reported with the file;
  3. the id lists every generation was run on (status files' --ids-file / --split arguments) are listed.
Writes outputs/split_audit/UNUSED_AUDIT.json.
"""
import glob
import hashlib
import json
import re
from pathlib import Path

E1 = Path(__file__).resolve().parents[1]
ID = re.compile(rb'(?<![0-9A-Z])[0-9A-Z]{30}(?![0-9A-Z])')


def main():
    sp = {n: json.loads((E1 / f'splits/{n}_ids.json').read_text()) for n in ('fit', 'cal', 'dev', 'test_sealed', 'unused')}
    man = json.loads((E1 / 'splits/SPLIT_MANIFEST.json').read_text())
    q_ids = [json.loads(l)['id'] for l in open(E1 / 'data/qasc/queries.jsonl') if l.strip()]
    allsp = [i for v in sp.values() for i in v]
    unused = set(sp['unused'])
    sub = {n: json.loads((E1 / f'splits/{n}.json').read_text()) for n in ('smoke_ids_fit', 'tune_ids_fit', 'panel128_dev_ids')}
    dropped = {x['dropped'] for x in man['dropped_text_duplicates']}
    part = {'n_queries_raw': len(q_ids), 'n_dropped_text_duplicates': len(dropped), 'n_in_five_lists': len(allsp),
            'distinct_in_five_lists': len(set(allsp)),
            'lists_equal_queries_minus_dropped': (set(allsp) == set(q_ids) - dropped and len(allsp) == len(set(allsp))
                                                  == len(q_ids) - len(dropped) and not (dropped & set(allsp))),
            'sizes': {n: len(v) for n, v in sp.items()}, 'sizes_match_manifest': {n: len(v) for n, v in sp.items()} == man['sizes'],
            'sha256_match_manifest': all(hashlib.sha256((E1 / f'splits/{f}').read_bytes()).hexdigest() == v['sha256']
                                         for f, v in man['files'].items()),
            'smoke_tune_subset_of_fit': set(sub['smoke_ids_fit']) <= set(sp['fit']) and set(sub['tune_ids_fit']) <= set(sp['fit']),
            'panel128_subset_of_dev': set(sub['panel128_dev_ids']) <= set(sp['dev']),
            'unused_disjoint_from_other_lists': not (unused & (set(sp['fit']) | set(sp['cal']) | set(sp['dev']) | set(sp['test_sealed'])))}

    scanned, hits, nbytes, ids_seen = 0, {}, 0, set()
    files = [p for root in ('outputs', 'logs') for p in (E1 / root).rglob('*')
             if p.is_file() and 'split_audit' not in p.parts]
    for p in files:
        b = p.read_bytes()
        scanned += 1
        nbytes += len(b)
        found = {m.decode() for m in ID.findall(b)}
        ids_seen |= found
        h = found & (unused | dropped)
        if h:
            hits[str(p.relative_to(E1))] = sorted(h)
    runs = []
    for st in sorted(glob.glob(str(E1 / 'outputs/**/status.*.json'), recursive=True)):
        a = json.loads(Path(st).read_text()).get('args', {})
        runs.append({'status': str(Path(st).relative_to(E1)), 'split': a.get('split'), 'ids_file': a.get('ids_file'),
                     'shard': f"{a.get('shard')}/{a.get('nshards')}", 'paths': a.get('paths')})
    out = {'partition': part, 'scan': {'roots': ['outputs/', 'logs/'], 'files_scanned': scanned, 'bytes_scanned': nbytes,
                                       'distinct_qasc_ids_seen': len(ids_seen),
                                       'ids_seen_by_list': {n: len(ids_seen & set(v)) for n, v in sp.items()},
                                       'files_with_unused_ids': hits},
           'generation_runs': runs,
           'not_scanned': ['data/qasc/queries.jsonl and gold.jsonl (the dataset itself: all 9,051 ids by construction)',
                           'ClusterA login-node scratch (not on the shared filesystem)']}
    out['scan']['note'] = 'files_with_unused_ids also covers the 9 dropped duplicates'
    out['unused_never_used'] = (part['lists_equal_queries_minus_dropped'] and part['unused_disjoint_from_other_lists'] and not hits
                                and all(r['split'] in ('fit', 'cal', 'dev', 'test_sealed') for r in runs))
    d = E1 / 'outputs/split_audit'
    d.mkdir(parents=True, exist_ok=True)
    (d / 'UNUSED_AUDIT.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'generation_runs'}, indent=1))
    print('generation runs:', len(runs), 'splits used:', sorted({r['split'] for r in runs}),
          'ids_files:', sorted({r['ids_file'] for r in runs if r['ids_file']}))


if __name__ == '__main__':
    main()
