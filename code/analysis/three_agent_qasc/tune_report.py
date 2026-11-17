"""Summarise one B-prompt tuning round on fit questions (gold allowed: fit only; asserted).

Usage: python tune_report.py outputs/tune_r1 [outputs/tune_r2 ...]
Writes <dir>/TUNE_SUMMARY.json and <dir>/TUNE_EXAMPLES.md
"""
import glob
import json
import re
import sys
from pathlib import Path

E1 = Path(__file__).resolve().parents[1]
FIT = set(json.loads((E1 / 'splits/fit_ids.json').read_text()))
LABEL_PAT = re.compile(r'(?:\b(?:[Oo]ptions?|[Cc]hoices?|[Aa]nswers?)\s*(?:is\s*|:\s*)?\(?[A-H]\)?(?![A-Za-z]))'
                       r'|(?:\([A-H]\))|(?:(?<![A-Za-z])[A-H][.)](?=\s))|(?:\b[A-H]\s*(?:and|or|/)\s*[A-H]\b)')
OPTION_DISCUSS_PAT = re.compile(r'distractor|not (directly )?related|irrelevant|incorrect|not correct|correct (choice|option)|the (other )?(options|choices)'
                                r'|among the (options|choices)|listed|none of the|best (fits|answer)|(options?|choices?) (like|such as)'
                                r'|while (the )?other|aligns with|directly supports|consistent with the (option|choice)', re.I)
ANSWER_PAT = re.compile(r'\b(?:the\s+(?:correct\s+|right\s+|best\s+)?answer\s+(?:is|would be)|correct (?:answer|option|choice)'
                        r'|the answer to the question is)\b', re.I)


def load(d):
    rows = []
    for f in sorted(glob.glob(str(Path(d) / 'records.*.jsonl'))):
        rows += [json.loads(x) for x in open(f) if x.strip()]
    return rows


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def main(dirs):
    Q = {}
    for line in open(E1 / 'data/qasc/queries.jsonl'):
        r = json.loads(line)
        Q[r['id']] = r
    rows_all = {d: load(d) for d in dirs}
    need = set().union(*[{r['id'] for r in rows} for rows in rows_all.values()])
    assert need <= FIT, 'gold may be read for fit questions only'
    gold = {}
    for line in open(E1 / 'data/qasc/gold.jsonl'):
        g = json.loads(line)
        if g['id'] in need:
            gold[g['id']] = g['answerKey']
    for d, rows in rows_all.items():
        s = {'dir': d, 'n': len(rows)}
        ok = [r for r in rows if r['steps'].get('C_AC') and r['steps'].get('C_ABC')]
        s['n_complete_main'] = len(ok)
        s['runtime_failures'] = sum(1 for r in rows for f in r['failures'] if f['status'] != 'succeeded_on_retry')
        s['retries_succeeded'] = sum(1 for r in rows for f in r['failures'] if f['status'] == 'succeeded_on_retry')
        for p in ['C_AC', 'C_ABC']:
            s[p] = {'acc': sum(r['steps'][p]['answer'] == gold[r['id']] for r in ok) / len(ok),
                    'invalid': sum(r['steps'][p]['invalid'] for r in ok)}
        s['change_AC_vs_ABC'] = sum(r['steps']['C_AC']['answer'] != r['steps']['C_ABC']['answer'] for r in ok) / len(ok)
        s['AC_wrong_ABC_right'] = sum(r['steps']['C_AC']['answer'] != gold[r['id']] and r['steps']['C_ABC']['answer'] == gold[r['id']] for r in ok)
        s['AC_right_ABC_wrong'] = sum(r['steps']['C_AC']['answer'] == gold[r['id']] and r['steps']['C_ABC']['answer'] != gold[r['id']] for r in ok)
        A = [r['steps']['A'] for r in rows if r['steps'].get('A')]
        B = [r['steps']['B'] for r in rows if r['steps'].get('B')]
        s['A_tokens'] = {'mean': sum(x['n_out'] for x in A) / len(A), 'max': max(x['n_out'] for x in A), 'truncated': sum(x['truncated'] for x in A)}
        s['B_tokens'] = {'mean': sum(x['n_out'] for x in B) / len(B), 'max': max(x['n_out'] for x in B), 'truncated': sum(x['truncated'] for x in B)}
        s['B_label_mention_flags'] = [r['id'] for r in rows if r['steps'].get('B') and LABEL_PAT.search(r['steps']['B']['text'])]
        s['B_answer_statement_flags'] = [r['id'] for r in rows if r['steps'].get('B') and ANSWER_PAT.search(r['steps']['B']['text'])]
        s['B_option_discussion_flags'] = [r['id'] for r in rows if r['steps'].get('B') and OPTION_DISCUSS_PAT.search(r['steps']['B']['text'])]
        s['A_label_mention_flags'] = [r['id'] for r in rows if r['steps'].get('A') and LABEL_PAT.search(r['steps']['A']['text'])]
        s['B_contains_gold_option_text'] = sum(
            1 for r in rows if r['steps'].get('B') and Q[r['id']]['choice_text'][ord(gold[r['id']]) - 65].lower() in r['steps']['B']['text'].lower())
        s['A_contains_gold_option_text'] = sum(
            1 for r in rows if r['steps'].get('A') and Q[r['id']]['choice_text'][ord(gold[r['id']]) - 65].lower() in r['steps']['A']['text'].lower())
        u = [r['steps']['probe']['u'] if r['steps']['probe']['u'] is not None else float('inf') for r in ok if r['steps'].get('probe')]
        su = sorted(u)
        s['probe_u'] = {'n': len(u), 'min': su[0], 'median': su[len(su) // 2], 'max': su[-1],
                        'share_u_below_0.01': sum(x < 0.01 for x in u) / len(u),
                        'nonfinite': sum(r['steps']['probe']['nonfinite'] for r in ok if r['steps'].get('probe'))}
        chg = [r['steps']['C_AC']['answer'] != r['steps']['C_ABC']['answer'] for r in ok if r['steps'].get('probe')]
        s['auroc_u_vs_change'] = auroc(u, chg)
        s['probe_argmax_eq_AC_answer'] = sum(r['steps']['probe']['argmax'] == r['steps']['C_AC']['answer'] for r in ok) / len(ok)
        ms = {}
        for r in rows:
            if r.get('first_after_load'):
                continue
            for k, v in r['ms'].items():
                ms.setdefault(k, []).append(v)
        s['ms_mean_warm'] = {k: sum(v) / len(v) for k, v in ms.items()}
        Path(d, 'TUNE_SUMMARY.json').write_text(json.dumps(s, indent=1) + '\n')
        lines = [f'# Tuning examples: {d}', '']
        for r in rows[:12] + [r for r in rows if r['id'] in s['B_label_mention_flags'] + s['B_answer_statement_flags']][:8]:
            q = Q[r['id']]
            lines += [f"## {r['id']}  gold={gold[r['id']]}  AC={r['steps']['C_AC']['answer'] if r['steps'].get('C_AC') else None}"
                      f"  ABC={r['steps']['C_ABC']['answer'] if r['steps'].get('C_ABC') else None}"
                      f"  u={r['steps']['probe']['u']:.4f}" if r['steps'].get('probe') else f"## {r['id']}",
                      'Q: ' + q['question'] + '  | ' + '; '.join(f'{chr(65 + k)}. {t}' for k, t in enumerate(q['choice_text'])),
                      'fact1: ' + q['fact1'], 'fact2: ' + q['fact2'],
                      'A: ' + (r['steps']['A']['text'] if r['steps'].get('A') else '-'),
                      'B: ' + (r['steps']['B']['text'] if r['steps'].get('B') else '-'), '']
        Path(d, 'TUNE_EXAMPLES.md').write_text('\n'.join(lines) + '\n')
        print(json.dumps({k: v for k, v in s.items() if not k.endswith('_flags')}, indent=1))
        print('B label flags:', len(s['B_label_mention_flags']), ' B answer-statement flags:', len(s['B_answer_statement_flags']),
              ' B option-discussion flags:', len(s['B_option_discussion_flags']), ' A label flags:', len(s['A_label_mention_flags']))


if __name__ == '__main__':
    main(sys.argv[1:])
