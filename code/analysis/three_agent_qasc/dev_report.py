"""Development-split report items (PROTOCOL.md section 7; reporting only, nothing here selects eps or a deployment).

Run after POLICY_FREEZE.json exists. Reads outputs/full/dev (all paths) and dev gold. Correct = answer equals gold;
INVALID is incorrect.
  - accuracies: s = A+C, b = A+B+C, C alone, C reading fact1+fact2, B+C without A, A + B-without-facts + C;
  - G_B = acc(b) - acc(s) with a paired bootstrap 95% interval (2,000 resamples, seed 0); counting rule, descriptive:
    eps <= G_B / 2 for eps in {0.01, 0.02}, and each frozen policy's development coverage >= 20%;
  - frozen policies on dev (descriptive): coverage, changes among skipped, lost corrections, acc(pi), G_pi, G_pi - G_B/2;
  - mean per-question latency of each path from the generation timings (single A100, batch 1, components summed;
    not the timing panel of Section 8).
Writes outputs/dev_report/DEV_REPORT.json and DEV_REPORT.md.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1lib as L  # noqa: E402

E1 = L.E1


def boot_ci(x, seed=0, reps=2000):
    rng = np.random.default_rng(seed)
    n = len(x)
    b = [float(x[rng.integers(0, n, n)].mean()) for _ in range(reps)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    freeze = json.loads((E1 / 'POLICY_FREEZE.json').read_text())
    ids = L.split_ids('dev')
    rows = {}
    for f in sorted(glob.glob(str(E1 / 'outputs/full/dev/records.dev.shard*.jsonl'))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                rows[r['id']] = r
    assert set(rows) == set(ids), 'dev incomplete'
    dset = set(ids)
    gold = {}
    for line in open(E1 / 'data/qasc/gold.jsonl'):
        g = json.loads(line)
        if g['id'] in dset:
            gold[g['id']] = g['answerKey']

    def corr(step):
        return np.array([bool(rows[i]['steps'].get(step)) and rows[i]['steps'][step]['answer'] == gold[i] for i in ids], float)

    def ms(*steps):
        v = [sum(rows[i]['ms'].get(s, np.nan) for s in steps) for i in ids if not rows[i].get('first_after_load')]
        return float(np.nanmean(v))

    paths = {'s = A+C': ('C_AC', ('A', 'C_AC')), 'b = A+B+C': ('C_ABC', ('A', 'B', 'C_ABC')),
             'C alone': ('C_C', ('C_C',)), 'C reads fact1+fact2': ('C_CF', ('C_CF',)),
             'B+C without A': ('C_BC', ('B_noA', 'C_BC')), 'A + B without facts + C': ('C_ABnfC', ('A', 'B_nofact', 'C_ABnfC'))}
    acc = {}
    for name, (step, lat) in paths.items():
        c = corr(step)
        acc[name] = {'acc': float(c.mean()), 'ci95': boot_ci(c), 'correct': int(c.sum()),
                     'invalid': sum(1 for i in ids if (rows[i]['steps'].get(step) or {}).get('invalid')),
                     'runtime_failures': sum(1 for i in ids if rows[i]['steps'].get(step) is None),
                     'mean_latency_ms_generation': ms(*lat)}
    s, b = corr('C_AC'), corr('C_ABC')
    gb = float((b - s).mean())
    res = {'N': len(ids), 'accuracy': acc,
           'G_B': {'G_B': gb, 'ci95': boot_ci(b - s), 'half': gb / 2},
           'counting_rule': {'eps_le_half_G_B': {str(e): e <= gb / 2 for e in (0.01, 0.02)}}, 'policies_on_dev': {}}
    u = {i: (rows[i]['steps']['probe']['u'] if rows[i]['steps'].get('probe') else None) for i in ids}
    for name, p in freeze['policies'].items():
        if p['tau'] is None:
            sk = set()
        else:
            sk = {i for i in ids if L.skip(u[i], p['tau'])}
        pi = np.array([s[k] if i in sk else b[k] for k, i in enumerate(ids)], float)
        changed = sum(1 for i in sk if rows[i]['steps']['C_AC']['answer'] != rows[i]['steps']['C_ABC']['answer'])
        lost = sum(1 for k, i in enumerate(ids) if i in sk and b[k] == 1 and s[k] == 0)
        lat = [rows[i]['ms']['A'] + rows[i]['ms']['probe'] + (rows[i]['ms']['C_AC'] if i in sk else rows[i]['ms']['B'] + rows[i]['ms']['C_ABC'])
               for i in ids if not rows[i].get('first_after_load')]
        res['policies_on_dev'][name] = {
            'j': p['j'], 'q': p['q'], 'coverage': len(sk) / len(ids), 'coverage_ge_20pct': len(sk) / len(ids) >= 0.2,
            'skipped': len(sk), 'changed_among_skipped': changed, 'change_rate': (changed / len(sk) if sk else None),
            'lost_corrections': lost, 'acc_pi': float(pi.mean()), 'G_pi': float((pi - s).mean()), 'G_pi_ci95': boot_ci(pi - s),
            'G_pi_minus_half_G_B': float((pi - s).mean() - gb / 2),
            'G_pi_minus_half_G_B_ci95': boot_ci((pi - s) - (b - s) / 2),
            'mean_latency_ms_generation': float(np.mean(lat))}
    cf, bb = acc['C reads fact1+fact2'], acc['b = A+B+C']
    res['direct_facts_vs_pipeline'] = {'more_accurate': cf['acc'] > bb['acc'], 'faster': cf['mean_latency_ms_generation'] < bb['mean_latency_ms_generation'],
                                       'acc_diff_CF_minus_b': cf['acc'] - bb['acc'], 'ci95_CF_minus_b': boot_ci(corr('C_CF') - b)}
    out = E1 / 'outputs/dev_report'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'DEV_REPORT.json').write_text(json.dumps(res, indent=1) + '\n')
    lines = ['| path | accuracy | 95% CI | INVALID | mean latency (ms, generation) |', '|---|---|---|---|---|']
    for n, v in acc.items():
        lines.append(f"| {n} | {v['acc']:.3f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {v['invalid']} | {v['mean_latency_ms_generation']:.0f} |")
    (out / 'DEV_REPORT.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(json.dumps({k: v for k, v in res.items() if k != 'accuracy'}, indent=1))


if __name__ == '__main__':
    main()
