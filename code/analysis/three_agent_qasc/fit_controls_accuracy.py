"""Fit-split accuracies of the two controls, "C alone" (path C) and "C reading fact1 + fact2" (path CF) (DEVIATIONS 4, 7).

Same definition as code/fit_accuracy_and_p1p2_prediction.py: correct = parsed answer equals gold; INVALID is incorrect;
gold for fit ids only. Computed after the policy freeze (the controls finished later, in a ClusterA job), so it is
written to its own file and the committed FIT_ACCURACY.json / P1_P2_PREDICTION.json stay as they were. Descriptive only.
Writes outputs/fit_diagnostics/FIT_ACCURACY_CONTROLS.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1lib as L  # noqa: E402
from fit_accuracy_and_p1p2_prediction import load  # noqa: E402

E1 = L.E1


def main():
    ids = L.split_ids('fit')
    R = load(E1 / 'outputs/full/fit')
    RC = load(E1 / 'outputs/full/fit_controls')
    assert all(i in R for i in ids) and all(i in RC for i in ids), 'fit or fit controls incomplete'
    fitset = set(ids)
    gold = {}
    for line in open(E1 / 'data/qasc/gold.jsonl'):
        g = json.loads(line)
        if g['id'] in fitset:
            gold[g['id']] = g['answerKey']
    assert set(gold) == fitset

    def vec(rows, step):
        return np.array([bool(rows[i]['steps'].get(step)) and rows[i]['steps'][step]['answer'] == gold[i] for i in ids], float)

    def boot(x):
        rng = np.random.default_rng(0)
        b = [float(x[rng.integers(0, len(x), len(x))].mean()) for _ in range(2000)]
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

    out = {'n': len(ids), 'definition': 'correct = parsed answer equals gold; INVALID incorrect; 95% percentile bootstrap, '
                                        '2,000 resamples over questions, seed 0', 'accuracy': {},
           'computed_after_policy_freeze': True, 'enters_any_test': False}
    paths = {'C alone': (RC, 'C_C'), 'C reads facts': (RC, 'C_CF'), 'A+C (s)': (R, 'C_AC'), 'A+B+C (b)': (R, 'C_ABC')}
    vs = {}
    for name, (rows, step) in paths.items():
        v = vs[name] = vec(rows, step)
        out['accuracy'][name] = {'acc': float(v.mean()), 'ci95': boot(v), 'correct': int(v.sum()),
                                 'invalid': sum(1 for i in ids if (rows[i]['steps'].get(step) or {}).get('invalid')),
                                 'runtime_failures': sum(1 for i in ids if rows[i]['steps'].get(step) is None)}
    out['paired_differences'] = {f'{a} - {b}': {'diff': float((vs[a] - vs[b]).mean()), 'ci95': boot(vs[a] - vs[b])}
                                 for a, b in [('C reads facts', 'A+B+C (b)'), ('A+C (s)', 'C alone'), ('C reads facts', 'C alone')]}
    p = E1 / 'outputs/fit_diagnostics/FIT_ACCURACY_CONTROLS.json'
    p.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
