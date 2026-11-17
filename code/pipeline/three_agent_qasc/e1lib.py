"""E1 shared library: three-agent QASC chain (A -> [B] -> C) built on the paper's frozen Text-path code.

Reused unchanged (byte-identical to the paper's run copies; hashes asserted at import):
  code/pipeline/native_action_runtime/legacy_methods.py  BACKGROUND_PROMPT, T2THelperBundle (A), T2TReceiverBundle (A+C)
  code/pipeline/native_action_runtime/protocol_min.py    receiver_prompt_tensors, generate_receiver, apply_generation_config
  code/pipeline/confidence_boundaries/receiver_prompt.py receiver_prompt / build_prompt / choices_block / display_labels
  code/pipeline/answer_scoring_v2/scoring_v2.py          answer parser v2 (labels A-H passed per question)
The ARC/MMLU-Pro adaptation is repeated: format_openbook is pointed at receiver_prompt (use_template=True) and at the
plain body "question\n\nChoices:\n<A. ..>" (use_template=False), exactly as the MMLU-Pro stage did.

New in E1: agent B (same generate call as the paper's helper, new prompt), the five-turn A+B+C receiver
conversation, the B-only and facts-only receiver variants, and ProbeMax on the A+C prompt.
torch is imported lazily so that the prompt code can be used on the login node.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

E1 = Path(__file__).resolve().parents[1]
REPO = E1.parents[1]
FROZEN = REPO / 'code/pipeline'
NATIVE_DIR = FROZEN / 'native_action_runtime'
PROMPT_PY = FROZEN / 'confidence_boundaries/receiver_prompt.py'
PARSER_PY = FROZEN / 'answer_scoring_v2/scoring_v2.py'
FROZEN_SHA256 = {
    NATIVE_DIR / 'legacy_methods.py': '91b03c146f56',
    NATIVE_DIR / 'protocol_min.py': '23c702e16ff5',
    PROMPT_PY: '868a4760d92b',
    PARSER_PY: 'd05978f400cf69aa6bddc56168a260ac3ea9158abaa640e9331602dc9bfcc2f9',
}
CONFIG = json.loads((E1 / 'configs/e1_config.json').read_text())
LABELS = list('ABCDEFGH')
PREFIX = 'The correct answer is'


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 23), b''):
            h.update(b)
    return h.hexdigest()


def sha_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


for _p, _h in FROZEN_SHA256.items():
    assert sha_file(_p).startswith(_h), f'frozen file changed: {_p}'


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prompt = _load(PROMPT_PY, 'e1_frozen_receiver_prompt')
parser = _load(PARSER_PY, 'e1_frozen_parser_v2')
sys.path.insert(0, str(NATIVE_DIR))
import protocol_min  # noqa: E402
import legacy_methods as lm  # noqa: E402
sys.path.remove(str(NATIVE_DIR))


def helper_body(q):
    """The question text the paper's helper sees (format_openbook(use_template=False))."""
    return prompt.build_prompt(dataset='mmlu-redux', locale='', question=str(q['question_stem']),
                               choices=prompt.choices_block(q['choice_text']), use_cot=False, use_template=False)


def format_qasc(example, use_template=True):
    return prompt.receiver_prompt(example) if use_template else helper_body(example)


protocol_min.format_openbook = format_qasc
lm.format_openbook = format_qasc
# As in the paper's runtime.py: the legacy wrappers' internal timers and legacy parse are switched off;
# one outer synchronized timer per call is authoritative, and parser v2 is applied afterwards.
lm.cuda_wall_stage = lambda fn, device: (fn(), 0., 0.)
lm.wall_stage = lambda fn: (fn(), 0.)
lm.extract_prediction = lambda text: None
BACKGROUND_PROMPT = lm.BACKGROUND_PROMPT


# ----------------------------------------------------------------------------------------------
# Questions
# ----------------------------------------------------------------------------------------------
def load_queries():
    rows = {}
    with open(E1 / 'data/qasc/queries.jsonl') as f:
        for line in f:
            r = json.loads(line)
            assert 'answerKey' not in r and 'combinedfact' not in r
            rows[r['id']] = {'id': r['id'], 'question_stem': r['question'], 'choice_labels': list(r['choice_labels']),
                             'choice_text': list(r['choice_text']), 'fact1': r['fact1'], 'fact2': r['fact2']}
            assert rows[r['id']]['choice_labels'] == LABELS
    return rows


def split_ids(name):
    return json.loads((E1 / f'splits/{name}_ids.json').read_text())


# ----------------------------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------------------------
def load_b_prompt(path):
    path = Path(path)
    spec = json.loads(path.read_text())
    for k in ['version', 'base', 'note_clause', 'facts_clause', 'facts_header', 'note_header', 'rules']:
        assert k in spec, k
    spec['_sha256'] = sha_file(path)
    spec['_path'] = str(path)
    return spec


def b_user_message(q, bp, facts=True, note=None):
    """B's user message. facts=False drops the fact block and its clause; note=None drops A's note and its clause.
    With both off this is the fact-free, note-free instruction that C sees as B's turn (cf. Text+fact in the paper:
    the receiver's history carries the fact-free helper instruction)."""
    head = bp['base'] + (bp['note_clause'] if note is not None else '') + (bp['facts_clause'] if facts else '')
    s = head + '\n\n' + helper_body(q)
    if facts:
        s += '\n' + bp['facts_header'] + '\n1. ' + str(q['fact1']).strip() + '\n2. ' + str(q['fact2']).strip() + '\n'
    if note is not None:
        s += '\n' + bp['note_header'] + '\n' + str(note).strip() + '\n'
    return s + '\n\n' + bp['rules']


def a_user_message(q):
    return BACKGROUND_PROMPT.format(question=helper_body(q))


def c_messages(q, bp, path, msgs):
    """Receiver conversation for one path. msgs holds the agents' texts ('A', 'B', 'B_noA', 'B_nofact')."""
    final = prompt.receiver_prompt(q)
    a_turn = [{'role': 'user', 'content': a_user_message(q)}, {'role': 'assistant', 'content': msgs.get('A')}]
    b_view = b_user_message(q, bp, facts=False, note=None)
    if path == 'AC':
        return a_turn + [{'role': 'user', 'content': final}]
    if path == 'ABC':
        return a_turn + [{'role': 'user', 'content': b_view}, {'role': 'assistant', 'content': msgs['B']},
                         {'role': 'user', 'content': final}]
    if path == 'ABnfC':
        return a_turn + [{'role': 'user', 'content': b_view}, {'role': 'assistant', 'content': msgs['B_nofact']},
                         {'role': 'user', 'content': final}]
    if path == 'BC':
        return [{'role': 'user', 'content': b_view}, {'role': 'assistant', 'content': msgs['B_noA']},
                {'role': 'user', 'content': final}]
    if path == 'C':
        return [{'role': 'user', 'content': final}]
    if path == 'CF':
        qf = dict(q, question_stem='Useful fact: ' + str(q['fact1']).strip() + '\nUseful fact: ' + str(q['fact2']).strip()
                  + '\n' + str(q['question_stem']))
        return [{'role': 'user', 'content': prompt.receiver_prompt(qf)}]
    raise ValueError(path)


MAIN_PATHS = ['AC', 'ABC']
CONTROL_PATHS = ['C', 'CF', 'BC', 'ABnfC']
PATH_NEEDS = {'AC': ['A'], 'ABC': ['A', 'B'], 'ABnfC': ['A', 'B_nofact'], 'BC': ['B_noA'], 'C': [], 'CF': []}


def parse(raw):
    p = parser.parse_answer(raw, LABELS)
    return p, (p['answer'] if p['valid'] else 'INVALID')


# ----------------------------------------------------------------------------------------------
# Models (GPU side)
# ----------------------------------------------------------------------------------------------
class Models:
    def __init__(self, helper_device='cuda:0', receiver_device='cuda:1', verify_large_files=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.hdev, self.rdev = helper_device, receiver_device
        self.load_seconds, self.load_info, self.file_checks = {}, {}, {}
        for role, dev in [('helper', helper_device), ('receiver', receiver_device)]:
            spec = CONFIG['models'][role]
            self.file_checks[role] = check_model_files(spec, large=verify_large_files)
            t = time.perf_counter()
            tok = AutoTokenizer.from_pretrained(spec['path'], local_files_only=True)
            assert sha_text(tok.chat_template) == spec['chat_template_sha256']
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model, info = AutoModelForCausalLM.from_pretrained(spec['path'], local_files_only=True, torch_dtype=torch.bfloat16,
                                                               attn_implementation='sdpa', output_loading_info=True)
            assert all(not info.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']), info
            model = model.to(dev).eval().requires_grad_(False)
            protocol_min.apply_generation_config(model, {'do_sample': False, 'max_new_tokens': 64})
            assert model.config._attn_implementation == 'sdpa'
            setattr(self, role, model)
            setattr(self, role + '_tok', tok)
            torch.cuda.synchronize(dev)
            self.load_seconds[role] = time.perf_counter() - t
        self.th = lm.T2THelperBundle(self.helper, self.helper_tok)
        self.tr = lm.T2TReceiverBundle(self.receiver, self.receiver_tok)
        rt = self.receiver_tok
        self.prefix_ids = rt.encode(PREFIX, add_special_tokens=False)
        assert self.prefix_ids == CONFIG['probe']['prefix_ids']
        sets = CONFIG['probe']['label_token_sets']
        for lab, ids in sets.items():
            assert all(v not in rt.all_special_ids and rt.decode([v], skip_special_tokens=False,
                       clean_up_tokenization_spaces=False).strip() == lab for v in ids), lab
        self.ix = {lab: torch.tensor(sets[lab], device=receiver_device) for lab in LABELS}
        self.helper_eos = set(_as_list(self.helper.generation_config.eos_token_id))
        self.receiver_eos = set(_as_list(self.receiver.generation_config.eos_token_id))
        self.first = {'probe': True}

    def sync(self):
        for d in {self.hdev, self.rdev}:
            self.torch.cuda.synchronize(d)

    # ---------------- agents ----------------
    def run_A(self, q):
        out = self.th.run(q)  # the paper's helper: BACKGROUND_PROMPT, chat template, greedy, max_new_tokens=256
        ids = out['helper_generated_token_ids']
        return {'text': out['helper_message'], 'gen_ids': ids, 'n_in': out['helper_input_token_count'], 'n_out': len(ids),
                'truncated': len(ids) >= 256 and ids[-1] not in self.helper_eos, 'body_sha256': sha_text(out['helper_body'])}

    def run_B(self, q, bp, facts=True, note=None):
        torch = self.torch
        user = b_user_message(q, bp, facts=facts, note=note)
        tok = self.helper_tok
        inputs = tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=True, add_generation_prompt=True,
                                         return_tensors='pt', enable_thinking=False).to(self.hdev)
        with torch.inference_mode():
            outputs = self.helper.generate(inputs, max_new_tokens=CONFIG['max_new_tokens']['B'], do_sample=False)
        gen = outputs[:, inputs.shape[-1]:]
        text = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ids = gen[0].detach().cpu().tolist()
        return {'text': text, 'gen_ids': ids, 'n_in': int(inputs.shape[1]), 'n_out': len(ids),
                'truncated': len(ids) >= CONFIG['max_new_tokens']['B'] and ids[-1] not in self.helper_eos,
                'user_sha256': sha_text(user)}

    def run_C(self, q, bp, path, msgs):
        torch = self.torch
        tok = self.receiver_tok
        if path == 'AC':
            # the paper's Text receiver, unchanged
            res = self.tr.consume(q, helper_body(q), msgs['A'])
            raw, ids, n_in = res['generated_text'], res['generated_token_ids'], res['receiver_input_token_count']
            rendered = tok.apply_chat_template(c_messages(q, bp, 'AC', msgs), tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
            if self.first.get('AC', True):
                assert res['receiver_prompt_token_ids'] == tok(rendered)['input_ids']
                self.first['AC'] = False
        elif path in ('C', 'CF'):
            # the paper's receiver-only call (receiver_prompt_tensors + generate_receiver); CF only edits the question text
            msgs_c = c_messages(q, bp, path, msgs)
            rendered = tok.apply_chat_template(msgs_c, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            tensors = {k: v.to(self.rdev) for k, v in tok(rendered, return_tensors='pt').items()}
            if path == 'C' and self.first.get('C', True):
                _, r2, t2 = protocol_min.receiver_prompt_tensors(tok, q, self.rdev)
                assert r2 == rendered and t2['input_ids'].tolist() == tensors['input_ids'].tolist()
                self.first['C'] = False
            raw, gen = protocol_min.generate_receiver(self.receiver, tok, tensors)
            ids, n_in = gen.cpu().tolist(), int(tensors['input_ids'].shape[1])
        else:
            # multi-turn receiver: same calls as T2TReceiverBundle.consume, longer history
            msgs_c = c_messages(q, bp, path, msgs)
            rendered = tok.apply_chat_template(msgs_c, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tok.apply_chat_template(msgs_c, tokenize=True, add_generation_prompt=True, return_tensors='pt',
                                             enable_thinking=False).to(self.rdev)
            if self.first.get(path, True):
                assert inputs[0].tolist() == tok(rendered)['input_ids']
                self.first[path] = False
            with torch.inference_mode():
                outputs = self.receiver.generate(inputs, max_new_tokens=64, do_sample=False)
            gen = outputs[:, inputs.shape[-1]:]
            raw = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            ids, n_in = gen[0].detach().cpu().tolist(), int(inputs.shape[1])
        parsed, answer = parse(raw)
        return {'raw': raw, 'gen_ids': ids, 'n_in': n_in, 'n_out': len(ids),
                'hit_max': len(ids) >= 64 and ids[-1] not in self.receiver_eos,
                'parsed': parsed, 'answer': answer, 'invalid': answer == 'INVALID', 'rendered_sha256': sha_text(rendered),
                '_rendered': rendered}

    def probe(self, q, bp, msgs):
        """ProbeMax on the A+C prompt: rendered C-reads-A conversation + 'The correct answer is', one forward pass."""
        torch = self.torch
        tok, model = self.receiver_tok, self.receiver
        msgs_c = c_messages(q, bp, 'AC', msgs)
        rendered = tok.apply_chat_template(msgs_c, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        base = tok(rendered, return_tensors='pt')
        ids = torch.cat([base['input_ids'], torch.tensor([self.prefix_ids], dtype=base['input_ids'].dtype)], dim=1)
        mask = torch.cat([base['attention_mask'], torch.ones((1, len(self.prefix_ids)), dtype=base['attention_mask'].dtype)], dim=1)
        pos, n = int(torch.nonzero(mask[0], as_tuple=False)[-1, 0]), ids.shape[1]
        inp = {'input_ids': ids.to(self.rdev), 'attention_mask': mask.to(self.rdev)}
        calls = []
        hook = model.lm_head.register_forward_pre_hook(lambda m, a: calls.append(list(a[0].shape))) if self.first['probe'] else None
        with torch.inference_mode():
            out = model.model(**inp, use_cache=False, return_dict=True, output_hidden_states=False, output_attentions=False)
            assert out.past_key_values is None and pos == n - 1 and int(mask.sum()) == n
            last = out.last_hidden_state[:, pos:pos + 1, :]
            logits = model.lm_head(last)[0, 0, :].float()
            label_logits = torch.stack([torch.logsumexp(logits[self.ix[lab]], dim=0) for lab in LABELS])
            sc = score_from_label_logits(label_logits)
            lp, ps, top, u, u_1mp, finite = sc['logp'], sc['p'], sc['top'], sc['u'], sc['u_1mp'], sc['finite']
            union = torch.cat([self.ix[lab] for lab in LABELS])
            mass = float(torch.exp(torch.logsumexp(logits[union], 0) - torch.logsumexp(logits, 0)).item())
        if hook:
            hook.remove()
            assert calls == [[1, 1, model.config.hidden_size]], calls
            assert base['input_ids'][0].tolist() == tok.apply_chat_template(msgs_c, tokenize=True, add_generation_prompt=True,
                                                                            enable_thinking=False)
            self.first['probe'] = False
        return {'u': u, 'nonfinite': not finite, 'score': 'log_nontop_mass', 'u_1mp_diag': u_1mp,
                'logp_labels': dict(zip(LABELS, lp)), 'p_labels': dict(zip(LABELS, ps)), 'label_union_mass': mass,
                'argmax': LABELS[top],
                'n_in': int(n), 'probe_ids_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                'rendered_sha256': sha_text(rendered), 'first_probe_checks': bool(hook)}


def score_from_label_logits(label_logits):
    """PROTOCOL.md section 4. label_logits: FP32 tensor of per-label logsumexp over each label's token set (A-H).
    log_softmax over the labels; u = logsumexp of the seven non-top labels' log-probabilities (log non-top mass);
    non-finite -> u = None (always waits for B). 1 - max p is returned as u_1mp (diagnostic only)."""
    import torch
    logp = torch.log_softmax(label_logits.float(), dim=0)
    top = int(torch.argmax(logp).item())
    u_log = float(torch.logsumexp(torch.cat([logp[:top], logp[top + 1:]]), dim=0).item())
    lp = logp.cpu().tolist()
    ps = logp.exp().cpu().tolist()
    finite = all(math.isfinite(x) for x in lp) and math.isfinite(u_log)
    return {'logp': lp, 'p': ps, 'top': top, 'finite': finite, 'u': u_log if finite else None,
            'u_1mp': float(1.0 - max(ps)) if finite and all(math.isfinite(x) for x in ps) else None}


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def check_model_files(spec, large=False):
    """Small files: sha256 against the paper's MODEL_SOURCE_INDEX. Weight shards: size always, sha256 only if large=True."""
    res = {}
    for f in spec['files']:
        p = Path(spec['path']) / f['name']
        assert p.exists() and p.stat().st_size == f['bytes'], (p, 'size')
        if large or f['bytes'] < 50_000_000:
            got = sha_file(p)
            assert got == f['sha256'], (p, 'sha256')
            res[f['name']] = 'sha256_ok'
        else:
            res[f['name']] = 'size_ok'
    return res


# ----------------------------------------------------------------------------------------------
# Thresholds (CPU)
# ----------------------------------------------------------------------------------------------
def _rank_value(u):
    """Non-finite / missing scores (None) sort above every finite score: they never skip."""
    return math.inf if u is None or not math.isfinite(u) else float(u)


def candidate_thresholds(fit_scores):
    """PROTOCOL.md section 4. j = 1..19: tau_j = the ceil(j*N_fit/20)-th smallest fit score (ties kept, non-finite last);
    j = 20 (q = 1): always skip. If a rank lands on a non-finite score, tau is 'ALL_FINITE' (skip every finite score)."""
    vals = sorted(_rank_value(x) for x in fit_scores)
    n = len(vals)
    out = []
    for j in range(1, 20):
        r = (j * n + 19) // 20
        t = vals[r - 1]
        out.append({'j': j, 'q': j / 20, 'rank': r, 'tau': t if math.isfinite(t) else 'ALL_FINITE'})
    out.append({'j': 20, 'q': 1.0, 'rank': None, 'tau': 'Infinity'})
    return out


def skip(u, tau):
    """Skip B iff u <= tau. tau == 'Infinity' (j = 20) always skips (constant policy, no probe needed);
    otherwise a non-finite or missing score (None) never skips (PROTOCOL.md: routes to the reference)."""
    if tau == 'Infinity':
        return True
    if u is None or not math.isfinite(u):
        return False
    if tau == 'ALL_FINITE':
        return True
    return u <= tau
