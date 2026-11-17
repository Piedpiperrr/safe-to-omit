"""Export the files the authors asked for to recheck the calibration tests (2026-11-10) into outputs/recheck/.

  fit.csv              id,u,gold,ans_s,ans_b,ans_C_alone,ans_C_facts   (fit split order; s = path AC, b = path ABC)
  cal.csv              id,u,ans_s,ans_b,gold,queried,query_candidate_j  (cal split order; gold only for the 40 queried
                       questions, taken from label_queries.jsonl; the calibration answer key is not opened here)
  thresholds.csv       j,tau   (frozen fit candidates; j = 20 is "Infinity" = always skip)
  policies.json        P0/P1/P2 as frozen in POLICY_FREEZE.json
  label_queries.jsonl  byte copy of outputs/calibration/label_queries.jsonl
  fit_label_logprobs.csv  id,argmax,lp_A..lp_H,u : the probe's 8 label log-probabilities (FP32 log_softmax over the
                       per-label logsumexp of the label token logits), as stored in the records
  source/              the source files that build the A, B, C(A), C(A,B) and probe inputs and compute u, with MANIFEST.txt
  rendered_example_fit.txt / .json   one fit question's full rendered inputs (skip, reference, probe; also A and B).
                       They quote question text, so they stay on the shared filesystem and are not committed (see .gitignore).
Floats are written with repr() (shortest string that round-trips the stored double; u and the log-probabilities are
FP32 values widened to double). INVALID answers are written as INVALID. Run from rebuttal/e1 with the c2c_official python.
"""
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1lib as L  # noqa: E402
from fit_accuracy_and_p1p2_prediction import load  # noqa: E402

E1 = L.E1
OUT = E1 / 'outputs/recheck'
LABELS = L.LABELS


def r(x):
    return '' if x is None else (x if isinstance(x, str) else repr(float(x)))


def ans(rows, i, step):
    s = rows.get(i, {}).get('steps', {}).get(step)
    return '' if s is None else s['answer']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fit_ids, cal_ids = L.split_ids('fit'), L.split_ids('cal')
    F, FC, C = load(E1 / 'outputs/full/fit'), load(E1 / 'outputs/full/fit_controls'), load(E1 / 'outputs/full/cal', 'cal')
    assert all(i in F and i in FC for i in fit_ids) and all(i in C for i in cal_ids)
    fitset = set(fit_ids)
    gold = {}
    for line in open(E1 / 'data/qasc/gold.jsonl'):
        g = json.loads(line)
        if g['id'] in fitset:
            gold[g['id']] = g['answerKey']

    with open(OUT / 'fit.csv', 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['id', 'u', 'gold', 'ans_s', 'ans_b', 'ans_C_alone', 'ans_C_facts'])
        for i in fit_ids:
            w.writerow([i, r(F[i]['steps']['probe']['u']), gold[i], ans(F, i, 'C_AC'), ans(F, i, 'C_ABC'),
                        ans(FC, i, 'C_C'), ans(FC, i, 'C_CF')])

    lq_src = E1 / 'outputs/calibration/label_queries.jsonl'
    LQ = {}
    for line in open(lq_src):
        x = json.loads(line)
        LQ[x['id']] = x
    with open(OUT / 'cal.csv', 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['id', 'u', 'ans_s', 'ans_b', 'gold', 'queried', 'query_candidate_j'])
        for i in cal_ids:
            q = LQ.get(i)
            assert q is None or (q['s_answer'] == ans(C, i, 'C_AC') and q['b_answer'] == ans(C, i, 'C_ABC')
                                 and q['u'] == C[i]['steps']['probe']['u'])
            w.writerow([i, r(C[i]['steps']['probe']['u']), ans(C, i, 'C_AC'), ans(C, i, 'C_ABC'),
                        q['gold'] if q else '', 1 if q else 0, q['candidate_j'] if q else ''])
    shutil.copyfile(lq_src, OUT / 'label_queries.jsonl')

    cand = json.loads((E1 / 'outputs/fit_diagnostics/THRESHOLD_CANDIDATES.json').read_text())['candidates']
    with open(OUT / 'thresholds.csv', 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['j', 'tau'])
        for c in cand:
            w.writerow([c['j'], r(c['tau'])])
    frz = json.loads((E1 / 'POLICY_FREEZE.json').read_text())
    (OUT / 'policies.json').write_text(json.dumps({'source': 'POLICY_FREEZE.json (tag e1-policy-freeze)',
                                                   'rule': 'skip B iff u <= tau (finite u); otherwise wait for B',
                                                   'policies': frz['policies'], 'b_prompt_sha256': frz['b_prompt_sha256'],
                                                   'thresholds_sha256': frz['thresholds_sha256']}, indent=1) + '\n')

    # 8 label log-probabilities on fit, and how exactly u can be recomputed from them
    d64, d32 = [], []
    with open(OUT / 'fit_label_logprobs.csv', 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['id', 'argmax'] + [f'lp_{a}' for a in LABELS] + ['u'])
        for i in fit_ids:
            p = F[i]['steps']['probe']
            lp = [p['logp_labels'][a] for a in LABELS]
            w.writerow([i, p['argmax']] + [r(x) for x in lp] + [r(p['u'])])
            top = int(np.argmax(lp))
            rest = np.array([x for k, x in enumerate(lp) if k != top])
            m = rest.max()
            u64 = float(m + math.log(np.exp(rest - m).sum()))
            r32 = rest.astype(np.float32)
            u32 = float(np.float32(r32.max()) + np.log(np.exp(r32 - r32.max()).sum(dtype=np.float32), dtype=np.float32))
            d64.append(abs(u64 - p['u']))
            d32.append(abs(u32 - p['u']))
    recompute = {'max_abs_diff_u_float64_recompute': max(d64), 'max_abs_diff_u_float32_recompute': max(d32),
                 'n_exact_float64': sum(x == 0 for x in d64), 'n_exact_float32': sum(x == 0 for x in d32), 'n': len(d64)}

    # source files that build the path inputs and compute u
    src = OUT / 'source'
    files = [(L.REPO / 'code/pipeline/native_action_runtime/legacy_methods.py',
              "frozen paper code: T2THelperBundle (A: BACKGROUND_PROMPT, chat template, greedy 256), "
              "T2TReceiverBundle.consume (C on the skip path A+C)"),
             (L.REPO / 'code/pipeline/native_action_runtime/protocol_min.py', 'frozen paper code: generation config, receiver call helpers'),
             (L.PROMPT_PY, 'frozen paper code: receiver template (receiver_prompt) and helper question body (build_prompt)'),
             (L.PARSER_PY, 'frozen paper code: answer parser v2 (ans_* columns)'),
             (E1 / 'code/e1lib.py', 'E1: a_user_message, b_user_message, c_messages (all paths), Models.run_A/run_B/run_C/probe, '
                                   'score_from_label_logits (u), candidate_thresholds, skip'),
             (E1 / 'code/e1_generate.py', 'E1: per-question order of the steps and which messages feed which path'),
             (E1 / 'configs/e1_config.json', 'E1: model paths/hashes, max_new_tokens, probe prefix ids and label token sets'),
             (E1 / 'prompts/B_prompt_v2.json', "E1: B's frozen prompt v2")]
    lines = ['# Source files that build the A, B, C(A) [skip], C(A,B) [reference] and probe inputs and compute u.',
             '# Paths are relative to the repository root; sha256 of each copied file.']
    for p, role in files:
        rel = p.relative_to(L.REPO)
        dst = src / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
        lines.append(f'{hashlib.sha256(dst.read_bytes()).hexdigest()}  {rel}  -- {role}')
    (src / 'MANIFEST.txt').write_text('\n'.join(lines) + '\n')

    # one fit question's full rendered inputs (the stored rendered C inputs of the run; probe and A/B rebuilt on CPU and
    # checked against the hashes and token counts stored in the record)
    rend = json.loads(open(E1 / 'outputs/full/fit/rendered.fit.shard0of4.jsonl').readline())
    i = rend['id']
    rec = F[i]
    Q = L.load_queries()[i]
    bp = L.load_b_prompt(E1 / 'prompts/B_prompt_v2.json')
    from transformers import AutoTokenizer
    rt = AutoTokenizer.from_pretrained(L.CONFIG['models']['receiver']['path'], local_files_only=True)
    ht = AutoTokenizer.from_pretrained(L.CONFIG['models']['helper']['path'], local_files_only=True)
    msgs = {'A': rec['steps']['A']['text'], 'B': rec['steps']['B']['text']}
    ac = rt.apply_chat_template(L.c_messages(Q, bp, 'AC', msgs), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    abc = rt.apply_chat_template(L.c_messages(Q, bp, 'ABC', msgs), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prefix = L.CONFIG['probe']['prefix_ids']
    probe_ids = rt(ac)['input_ids'] + prefix
    a_user, b_user = L.a_user_message(Q), L.b_user_message(Q, bp, facts=True, note=msgs['A'])
    a_rend = ht.apply_chat_template([{'role': 'user', 'content': a_user}], tokenize=False, add_generation_prompt=True)
    b_rend = ht.apply_chat_template([{'role': 'user', 'content': b_user}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    checks = {
        'C_AC_rendered_equals_stored_run_rendering': ac == rend['rendered']['C_AC'],
        'C_ABC_rendered_equals_stored_run_rendering': abc == rend['rendered']['C_ABC'],
        'C_AC_sha256_equals_record': L.sha_text(ac) == rec['steps']['C_AC']['rendered_sha256'],
        'C_ABC_sha256_equals_record': L.sha_text(abc) == rec['steps']['C_ABC']['rendered_sha256'],
        'probe_text_sha256_equals_record': L.sha_text(ac) == rec['steps']['probe']['rendered_sha256'],
        'probe_token_ids_sha256_equals_record': hashlib.sha256(np.array([probe_ids], dtype=np.int64).tobytes()).hexdigest()
                                                == rec['steps']['probe']['probe_ids_sha256'],
        'probe_n_in_equals_record': len(probe_ids) == rec['steps']['probe']['n_in'],
        'A_user_equals_stored': a_user == rend['A_user'], 'B_user_equals_stored': b_user == rend['B_user'],
        'A_n_in_equals_record': len(ht(a_rend)['input_ids']) == rec['steps']['A']['n_in'],
        'B_n_in_equals_record': len(ht(b_rend)['input_ids']) == rec['steps']['B']['n_in'],
        'B_user_sha256_equals_record': L.sha_text(b_user) == rec['steps']['B']['user_sha256'],
    }
    probe_text = ac + rt.decode(prefix, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    ex = {'id': i, 'split': 'fit', 'checks': checks,
          'skip_path_C_input (A+C)': ac, 'reference_path_C_input (A+B+C)': abc,
          'probe_input (A+C rendering + prefix token ids)': {'text': probe_text, 'prefix_ids': prefix, 'n_tokens': len(probe_ids)},
          'A_input': a_rend, 'B_input': b_rend,
          'outputs': {'A': msgs['A'], 'B': msgs['B'], 'C_AC_raw': rec['steps']['C_AC']['raw'], 'C_ABC_raw': rec['steps']['C_ABC']['raw'],
                      'u': rec['steps']['probe']['u'], 'logp_labels': rec['steps']['probe']['logp_labels']}}
    (OUT / 'rendered_example_fit.json').write_text(json.dumps(ex, indent=1, ensure_ascii=False) + '\n')
    sep = '\n' + '=' * 100 + '\n'
    txt = (f'fit question {i}; every check below must be True\n' + json.dumps(checks, indent=1) + sep
           + '[skip path] C input, path A+C (Qwen3-8B chat template, thinking off), generated with max 64 new tokens:\n' + sep + ac + sep
           + '[reference path] C input, path A+B+C:\n' + sep + abc + sep
           + f'[probe] the A+C input above followed by the prefix token ids {prefix} (decoded below); one forward pass, '
             f'label logits read at the last position ({len(probe_ids)} tokens):\n' + sep + probe_text + sep
           + '[A] helper input (Qwen2.5-7B-Instruct chat template):\n' + sep + a_rend + sep
           + '[B] B input (Qwen2.5-7B-Instruct chat template, facts + A note):\n' + sep + b_rend + sep
           + 'outputs of this question: ' + json.dumps(ex['outputs'], ensure_ascii=False, indent=1) + '\n')
    (OUT / 'rendered_example_fit.txt').write_text(txt)
    assert all(checks.values()), checks

    info = {'recompute_u_from_fit_label_logprobs': recompute, 'rendered_example_id': i, 'rendered_example_checks': checks}
    (OUT / 'EXPORT_INFO.json').write_text(json.dumps(info, indent=1) + '\n')
    sums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'SHA256SUMS'}
    (OUT / 'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n, h in sums.items()))
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
