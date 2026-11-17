"""Two design properties reported per split (author decision 3, 2026-11-09):
  1. share of A messages that name an option (option letter as a selection, e.g. "E. xerophytes", "(C)", "option D");
  2. share of B messages that contain the correct option's text (case-insensitive substring).
Definitions are frozen here; the 8.3% fit figure in the prep report came from the same regex on 96 tuning questions
(all 8 hits checked by hand). Gold is read only for the split asked (fit; dev for reporting; the sealed test only after
rebuttal/e1/POLICY_FREEZE.json exists).
  python leak_stats.py fit|dev|test_sealed
Writes outputs/leak_stats/<split>.json
"""
import glob
import json
import re
import sys
from pathlib import Path

E1 = Path(__file__).resolve().parents[1]
LABEL_PAT = re.compile(r'(?:\b(?:[Oo]ptions?|[Cc]hoices?|[Aa]nswers?)\s*(?:is\s*|:\s*)?\(?[A-H]\)?(?![A-Za-z]))'
                       r'|(?:\([A-H]\))|(?:(?<![A-Za-z])[A-H][.)](?=\s))|(?:\b[A-H]\s*(?:and|or|/)\s*[A-H]\b)')


def main(split):
    if split == 'test_sealed':
        assert (E1 / 'POLICY_FREEZE.json').exists(), 'sealed test: policies not frozen'
    ids = json.loads((E1 / f'splits/{split}_ids.json').read_text())
    rows = {}
    for f in sorted(glob.glob(str(E1 / f'outputs/full/{split}/records.{split}.shard*.jsonl'))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                rows[r['id']] = r
    assert all(i in rows for i in ids), 'split incomplete'
    need = set(ids)
    gold = {}
    for line in open(E1 / 'data/qasc/gold.jsonl'):
        g = json.loads(line)
        if g['id'] in need:
            gold[g['id']] = g['answerKey']
    Q = {}
    for line in open(E1 / 'data/qasc/queries.jsonl'):
        q = json.loads(line)
        if q['id'] in need:
            Q[q['id']] = q
    a_ok = [i for i in ids if rows[i]['steps'].get('A')]
    b_ok = [i for i in ids if rows[i]['steps'].get('B')]
    a_hits = [i for i in a_ok if LABEL_PAT.search(rows[i]['steps']['A']['text'])]
    b_hits = [i for i in b_ok if Q[i]['choice_text'][ord(gold[i]) - 65].lower() in rows[i]['steps']['B']['text'].lower()]
    a_gold = [i for i in a_ok if Q[i]['choice_text'][ord(gold[i]) - 65].lower() in rows[i]['steps']['A']['text'].lower()]
    out = {'split': split, 'N': len(ids),
           'A_names_an_option': {'n': len(a_hits), 'of': len(a_ok), 'share': len(a_hits) / len(a_ok), 'ids': a_hits},
           'B_contains_correct_option_text': {'n': len(b_hits), 'of': len(b_ok), 'share': len(b_hits) / len(b_ok)},
           'A_contains_correct_option_text': {'n': len(a_gold), 'of': len(a_ok), 'share': len(a_gold) / len(a_ok)},
           'label_regex': LABEL_PAT.pattern}
    d = E1 / 'outputs/leak_stats'
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{split}.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != 'ids'} if isinstance(v, dict) else v) for k, v in out.items()
                      if k != 'label_regex'}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
