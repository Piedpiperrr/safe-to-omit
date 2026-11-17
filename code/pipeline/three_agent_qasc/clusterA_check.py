"""Cross-cluster consistency gate run at the start of the ClusterA job for the remaining shards (scheduler script, not released) (authors' request, 2026-11-10).

The 4 fit questions of the layout test (splits/smoke_ids_fit.json[:4]) are regenerated on each ClusterA GPU with the
generation layout (A, B and C on one A100, greedy, batch 1) and compared with the ClusterB fit records
(outputs/full/fit). Gate items, compared exactly: A message text, B message text, repr of the probe score u, and the
answers of paths AC and ABC. Any missing step or difference on any GPU fails the gate (exit 1) and the job runs no shard.
Also recorded, not gated: C raw outputs, generated token ids, u_1mp_diag, and the 8 label log-probabilities.
  python clusterA_check.py <run_dir>      (reads <run_dir>/check_gpu*/records.fit.shard0of1.jsonl)
Writes <run_dir>/CHECK.json (booleans and item names only, committed) and <run_dir>/CHECK_DIFFS.json (the differing
values, which quote model messages; not committed, like the records).
"""
import glob
import json
import sys
from pathlib import Path

E1 = Path(__file__).resolve().parents[1]


def get(r, *path):
    for p in path:
        if not isinstance(r, dict) or p not in r:
            return None
        r = r[p]
    return r


GATE = {'A_text': ('steps', 'A', 'text'), 'B_text': ('steps', 'B', 'text'), 'probe_u': ('steps', 'probe', 'u'),
        'AC_answer': ('steps', 'C_AC', 'answer'), 'ABC_answer': ('steps', 'C_ABC', 'answer')}
INFO = {'AC_raw': ('steps', 'C_AC', 'raw'), 'ABC_raw': ('steps', 'C_ABC', 'raw'), 'A_gen_ids': ('steps', 'A', 'gen_ids'),
        'B_gen_ids': ('steps', 'B', 'gen_ids'), 'u_1mp_diag': ('steps', 'probe', 'u_1mp_diag'),
        'logp_labels': ('steps', 'probe', 'logp_labels')}


def main(run):
    run = Path(run)
    ids = json.loads((E1 / 'splits/smoke_ids_fit.json').read_text())[:4]
    ref = {}
    for f in sorted(glob.glob(str(E1 / 'outputs/full/fit/records.fit.shard*.jsonl'))):
        for line in open(f):
            r = json.loads(line)
            if r['id'] in ids:
                ref[r['id']] = r
    assert set(ref) == set(ids), 'ClusterB fit records missing some check ids'
    gpus = sorted(p for p in glob.glob(str(run / 'check_gpu*')) if Path(p).is_dir())
    out = {'ids': ids, 'gate_items': list(GATE), 'n_gpus': len(gpus), 'per_gpu': {}, 'differences': []}
    ok = len(gpus) == 4
    diffs = []
    for gd in gpus:
        g = Path(gd).name
        new = {}
        for f in glob.glob(str(Path(gd) / 'records.fit.*.jsonl')):
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    new[r['id']] = r
        rows = []
        for i in ids:
            row = {'id': i, 'present': i in new}
            if i in new:
                for k, p in {**GATE, **INFO}.items():
                    a, b = get(new[i], *p), get(ref[i], *p)
                    eq = (a is not None) and (repr(a) == repr(b))
                    row[k] = eq
                    if not eq:
                        out['differences'].append({'gpu': g, 'id': i, 'item': k, 'gated': k in GATE})
                        diffs.append({'gpu': g, 'id': i, 'item': k, 'gated': k in GATE, 'clusterA': repr(a), 'clusterB': repr(b)})
                row['failures'] = new[i]['failures']
            rows.append(row)
        g_ok = all(r['present'] and all(r[k] for k in GATE) for r in rows)
        info_ok = all(r['present'] and all(r[k] for k in INFO) for r in rows)
        st = Path(gd) / 'status.fit.shard0of1.json'
        env = json.loads(st.read_text()).get('env') if st.exists() else None
        out['per_gpu'][g] = {'gate_pass': g_ok, 'info_items_all_equal': info_ok, 'env': env, 'rows': rows}
        ok = ok and g_ok
    out['gate_pass'] = ok
    (run / 'CHECK.json').write_text(json.dumps(out, indent=1) + '\n')
    (run / 'CHECK_DIFFS.json').write_text(json.dumps(diffs, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'gate_pass': ok, 'n_gpus': len(gpus),
                      'per_gpu': {g: {k: v[k] for k in ('gate_pass', 'info_items_all_equal')} for g, v in out['per_gpu'].items()},
                      'differences': out['differences'][:20]}, indent=1))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
