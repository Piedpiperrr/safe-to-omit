"""Two-GPU layout test before the timing panel (PROTOCOL.md section 8): 4 fit questions generated with A/B on cuda:0
and C on cuda:1 are compared with the full fit records (generated on one GPU). Differences are recorded, not fixed.
  python layout_check.py <layout_test_dir>
"""
import glob, json, sys
from pathlib import Path
E1 = Path(__file__).resolve().parents[1]
d = Path(sys.argv[1])
new = {json.loads(l)['id']: json.loads(l) for f in glob.glob(str(d / 'records.fit.*.jsonl')) for l in open(f) if l.strip()}
ref = {}
for f in glob.glob(str(E1 / 'outputs/full/fit/records.fit.*.jsonl')):
    for l in open(f):
        r = json.loads(l)
        if r['id'] in new:
            ref[r['id']] = r
res = []
for i, r in new.items():
    s, t = r['steps'], ref[i]['steps']
    res.append({'id': i, 'A_text_equal': s['A']['text'] == t['A']['text'], 'B_text_equal': s['B']['text'] == t['B']['text'],
                'u_equal': s['probe']['u'] == t['probe']['u'], 'u_two_gpu': s['probe']['u'], 'u_one_gpu': t['probe']['u'],
                'AC_equal': s['C_AC']['answer'] == t['C_AC']['answer'], 'ABC_equal': s['C_ABC']['answer'] == t['C_ABC']['answer'],
                'failures': r['failures']})
ok_run = len(new) == 4 and all(not x['failures'] for x in res)
out = {'n': len(new), 'ran_without_failure': ok_run, 'all_identical': all(all(v for k, v in x.items() if k.endswith('_equal')) for x in res),
       'per_question': res}
(d / 'LAYOUT_CHECK.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps({k: v for k, v in out.items() if k != 'per_question'}))
sys.exit(0 if ok_run else 1)
