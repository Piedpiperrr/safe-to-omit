"""E1 CPU preflight (login node, tokenizers only, no model weights loaded).

1. frozen code hashes (asserted by importing e1lib)
2. tokenizer chat-template hashes; A-H label token sets re-derived from the Qwen3-8B vocabulary with the paper's rule
   (non-special single tokens whose decoding, stripped, equals the label) and compared with configs/e1_config.json
3. the paper's OBQA rendering example (results/e13/PROMPTS_AND_PARSER.md) is rebuilt through e1lib and its token
   counts compared with the saved ones (helper 111, receiver Text 258, receiver-only 132, receiver-only probe 136)
4. parser v2 on synthetic A-H outputs; threshold candidates on a synthetic score list
5. prompt renderings and token counts for the 16 smoke fit questions with placeholder agent messages
Writes outputs/preflight/cpu_preflight.json and outputs/preflight/rendered_examples.txt
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1lib as L  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

OUT = L.E1 / 'outputs/preflight'


def main(bprompt_path):
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {'frozen_sha256_prefixes_ok': {str(k.relative_to(L.REPO)): v for k, v in L.FROZEN_SHA256.items()}}
    toks = {}
    for role in ['helper', 'receiver']:
        spec = L.CONFIG['models'][role]
        tok = AutoTokenizer.from_pretrained(spec['path'], local_files_only=True)
        assert L.sha_text(tok.chat_template) == spec['chat_template_sha256'], role
        toks[role] = tok
    rep['chat_templates_ok'] = True
    rt, ht = toks['receiver'], toks['helper']

    # label sets by the paper's vocabulary rule
    sets = {lab: [] for lab in L.LABELS}
    special = set(rt.all_special_ids)
    for v in sorted(set(rt.get_vocab().values())):
        if v in special:
            continue
        d = rt.decode([v], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if d.strip() in sets:
            sets[d.strip()].append(v)
    assert sets == L.CONFIG['probe']['label_token_sets'], (sets, L.CONFIG['probe']['label_token_sets'])
    rep['label_token_sets_rederived_equal'] = True
    rep['label_token_sets'] = {k: [(v, rt.convert_ids_to_tokens(v)) for v in ids] for k, ids in sets.items()}
    assert rt.encode(L.PREFIX, add_special_tokens=False) == L.CONFIG['probe']['prefix_ids']
    rep['prefix_ids_ok'] = L.CONFIG['probe']['prefix_ids']

    # paper OBQA example (4 options) through the same code paths
    ob = {'question_stem': 'Which relationship is true?', 'choice_labels': list('ABCD'),
          'choice_text': ['wind is renewable; metal is nonrenewable', 'wind is recyclable, metal is other',
                          'wind is happy, metal is other', 'wind is nonrenewable; metal is renewable']}
    msg = ('The essential background knowledge needed to answer the question is an understanding of the definitions of '
           'renewable and nonrenewable resources, as well as the properties of wind and metal.')
    h_ids = ht.apply_chat_template([{'role': 'user', 'content': L.a_user_message(ob)}], tokenize=True, add_generation_prompt=True,
                                   enable_thinking=False)
    t_msgs = [{'role': 'user', 'content': L.a_user_message(ob)}, {'role': 'assistant', 'content': msg},
              {'role': 'user', 'content': L.prompt.receiver_prompt(ob)}]
    t_ids = rt.apply_chat_template(t_msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    r_ids = rt.apply_chat_template([{'role': 'user', 'content': L.prompt.receiver_prompt(ob)}], tokenize=True,
                                   add_generation_prompt=True, enable_thinking=False)
    rep['paper_obqa_example_tokens'] = {'helper': [len(h_ids), 111], 'receiver_text': [len(t_ids), 258],
                                        'receiver_only': [len(r_ids), 132], 'receiver_only_probe': [len(r_ids) + 4, 136]}
    assert all(a == b for a, b in rep['paper_obqa_example_tokens'].values()), rep['paper_obqa_example_tokens']

    # parser v2 on A-H
    pc = []
    for lab in L.LABELS:
        p, a = L.parse('The correct answer is ' + lab)
        assert a == lab
        pc.append(lab)
    for bad in ['The correct answer is I', 'The correct answer is A or B', '', 'I cannot answer this question.']:
        assert L.parse(bad)[1] == 'INVALID', bad
    rep['parser_checks'] = {'valid_letters': pc, 'invalid_cases': 4}

    # thresholds
    th = L.candidate_thresholds([-i / 10 for i in range(1, 101)] )
    assert th[0]['rank'] == 5 and abs(th[0]['tau'] + 9.6) < 1e-12 and th[18]['rank'] == 95 and th[19]['tau'] == 'Infinity'
    assert L.skip(-9.6, th[0]['tau']) and not L.skip(-9.59, th[0]['tau']) and L.skip(0.0, 'Infinity') and L.skip(None, 'Infinity')
    assert not L.skip(None, th[18]['tau']) and not L.skip(float('nan'), th[18]['tau'])
    th2 = L.candidate_thresholds([-3.0] * 10 + [None] * 10)
    assert th2[9]['tau'] == -3.0 and th2[10]['tau'] == 'ALL_FINITE' and L.skip(-1.0, 'ALL_FINITE') and not L.skip(None, 'ALL_FINITE')
    rep['threshold_checks'] = 'ok'

    # smoke questions with placeholder messages
    bp = L.load_b_prompt(bprompt_path)
    Q = L.load_queries()
    fit = L.split_ids('fit')
    smoke = fit[:16]
    ph = {'A': '<A message placeholder>', 'B': '<B message placeholder>', 'B_noA': '<B (no A) placeholder>',
          'B_nofact': '<B (no facts) placeholder>'}
    counts = []
    lines = []
    for k, i in enumerate(smoke):
        q = Q[i]
        row = {'id': i,
               'A_in': len(ht.apply_chat_template([{'role': 'user', 'content': L.a_user_message(q)}], tokenize=True,
                                                  add_generation_prompt=True, enable_thinking=False)),
               'B_in_with_placeholder_note': len(ht.apply_chat_template(
                   [{'role': 'user', 'content': L.b_user_message(q, bp, facts=True, note=ph['A'])}], tokenize=True,
                   add_generation_prompt=True, enable_thinking=False))}
        for path in ['AC', 'ABC', 'C', 'CF', 'BC', 'ABnfC']:
            m = L.c_messages(q, bp, path, ph)
            row['C_' + path + '_in'] = len(rt.apply_chat_template(m, tokenize=True, add_generation_prompt=True, enable_thinking=False))
        counts.append(row)
        if k == 0:
            lines.append('=' * 30 + f' fit question {i} (placeholder agent messages) ' + '=' * 30)
            lines.append('\n----- A input (helper chat template) -----\n' + ht.apply_chat_template(
                [{'role': 'user', 'content': L.a_user_message(q)}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
            for tag, kw in [('B (facts + A note)', dict(facts=True, note=ph['A'])), ('B no A (B+C ablation)', dict(facts=True, note=None)),
                            ('B no facts (ablation)', dict(facts=False, note=ph['A'])), ("B turn as C sees it", dict(facts=False, note=None))]:
                lines.append(f'\n----- {tag}: user message -----\n' + L.b_user_message(q, bp, **kw))
            for path in ['AC', 'ABC', 'C', 'CF', 'BC', 'ABnfC']:
                m = L.c_messages(q, bp, path, ph)
                lines.append(f'\n----- C input, path {path} (receiver chat template) -----\n' + rt.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            lines.append('\n----- probe input = C input of path AC + prefix ids ' + str(L.CONFIG['probe']['prefix_ids']) + ' -----')
    rep['smoke_ids'] = smoke
    rep['smoke_prompt_token_counts_placeholder_messages'] = counts
    rep['b_prompt'] = {'path': bp['_path'], 'sha256': bp['_sha256'], 'version': bp['version']}
    (OUT / 'cpu_preflight.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    (OUT / 'rendered_examples.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in rep.items() if k not in ['smoke_prompt_token_counts_placeholder_messages', 'label_token_sets']}, indent=1))
    print('token counts (first 4):', json.dumps(counts[:4]))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else str(L.E1 / 'prompts/B_prompt_v1.json'))
