"""E1 step 1a: check the downloaded QASC / SciQ parquet files and write the question-only and gold files.

Inputs : rebuttal/e1/data/raw/qasc_a34ba204/*.parquet, rebuttal/e1/data/raw/sciq_2c94ad3e/*.parquet
Outputs: rebuttal/e1/data/qasc/queries.jsonl   id, source_split, question, choice_labels, choice_text, fact1, fact2
         rebuttal/e1/data/qasc/gold.jsonl      id, answerKey   (never read by make_splits.py or by any prompt)
         rebuttal/e1/outputs/data_check/data_check.json

answerKey is copied, and only its format is checked (one label among the displayed labels); no label
statistics are computed here. combinedfact and formatted_question are not copied anywhere.
Run with: python -I data_prep.py   (OPENBLAS_NUM_THREADS=1 on the login node)
"""
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

E1 = Path(__file__).resolve().parents[1]
RAW = E1 / 'data/raw'
QASC = RAW / 'qasc_a34ba204'
SCIQ = RAW / 'sciq_2c94ad3e'
OUT = E1 / 'data/qasc'
CHECK = E1 / 'outputs/data_check'
EXPECTED_SHA = {
    'qasc/train': 'b9a297b5ab55f1605c7682ffbb7042c26d7ecb9ff1e1aa5a820d4e791c8302d1',
    'qasc/validation': 'd1ae34ae13c5fce2c55305372c203c6cdb789728d0d7e5ea2956d55bc33f40ae',
    'qasc/test': '495cfbd17abc1720b54785cdce68825104aaf60d7b8bdb2acac62157a31eb517',
    'sciq/train': '19644360954006d06e9ad3df07bddb34f8535c081b831d48f604603c713ac167',
    'sciq/validation': '455dd9f1d725cd3ecbce369799a2fbbdbbfecf51ab84a86d56ba3370dc847b8a',
    'sciq/test': '3a719356a29b127fc54ef3c7f51a034db4bd105d5717215e8c85d2aa58d60667',
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    report = {'qasc': {}, 'sciq': {}}
    for key, h in EXPECTED_SHA.items():
        ds, split = key.split('/')
        p = (QASC if ds == 'qasc' else SCIQ) / f'{split}-00000-of-00001.parquet'
        got = sha(p)
        assert got == h, (key, got)
        report[ds].setdefault('files', {})[split] = {'path': str(p.relative_to(E1)), 'sha256': got, 'bytes': p.stat().st_size}

    # ---------------- QASC ----------------
    queries, gold = [], []
    problems = []
    for split in ['train', 'validation', 'test']:
        t = pq.read_table(QASC / f'{split}-00000-of-00001.parquet')
        rows = t.to_pylist()
        info = {'rows': len(rows), 'schema': str(t.schema).split('\n')[:12]}
        n_choices = {}
        empty = {'fact1': 0, 'fact2': 0, 'answerKey': 0, 'question': 0}
        for r in rows:
            labels = list(r['choices']['label'])
            texts = list(r['choices']['text'])
            n_choices[len(labels)] = n_choices.get(len(labels), 0) + 1
            for f in empty:
                if not (r.get(f) or '').strip():
                    empty[f] += 1
            if split == 'test':
                continue
            disp = [chr(65 + i) for i in range(len(texts))]
            if labels != disp:
                problems.append({'id': r['id'], 'split': split, 'issue': 'labels_not_A_to_H_in_order', 'labels': labels})
            if r['answerKey'] not in labels:
                problems.append({'id': r['id'], 'split': split, 'issue': 'answerKey_not_a_displayed_label'})
            if any(not str(x).strip() for x in texts):
                problems.append({'id': r['id'], 'split': split, 'issue': 'empty_choice_text'})
            queries.append({'id': r['id'], 'source_split': split, 'question': r['question'],
                            'choice_labels': disp, 'choice_text': [str(x) for x in texts],
                            'fact1': r['fact1'], 'fact2': r['fact2']})
            gold.append({'id': r['id'], 'answerKey': r['answerKey']})
        info['n_choices_histogram'] = {str(k): v for k, v in sorted(n_choices.items())}
        info['empty_field_counts'] = empty
        info['distinct_ids'] = len({r['id'] for r in rows})
        report['qasc'][split] = info
    report['qasc']['problems'] = problems
    report['qasc']['queries_written'] = len(queries)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'queries.jsonl').open('w') as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
    with (OUT / 'gold.jsonl').open('w') as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + '\n')
    report['qasc']['queries_sha256'] = sha(OUT / 'queries.jsonl')
    report['qasc']['gold_sha256'] = sha(OUT / 'gold.jsonl')

    # ---------------- SciQ (fallback only: size and fields) ----------------
    for split in ['train', 'validation', 'test']:
        t = pq.read_table(SCIQ / f'{split}-00000-of-00001.parquet')
        cols = t.column_names
        rows = t.to_pylist()
        report['sciq'][split] = {
            'rows': len(rows), 'columns': cols,
            'empty_support': sum(1 for r in rows if not (r.get('support') or '').strip()),
        }
    report['sciq']['note'] = ('four options per question; the correct answer is a separate field '
                              '(correct_answer + distractor1-3), so option order must be built by the user; '
                              'support is one passage, not two facts')

    CHECK.mkdir(parents=True, exist_ok=True)
    (CHECK / 'data_check.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: {s: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'schema'})
                          for s, v in report[k].items() if s != 'files'} for k in report}, indent=1, ensure_ascii=False)[:6000])


if __name__ == '__main__':
    sys.exit(main())
