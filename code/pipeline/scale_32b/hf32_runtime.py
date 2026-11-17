"""Paper Text/ProbeMax semantics, BF16 HF layer sharding across disjoint GPU pairs."""
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Runtime32:
    def __init__(self, cfg):
        self.cfg = cfg
        assert torch.cuda.device_count() == 4, 'ClusterA worker must expose exactly GPU 0,1,2,3'
        for package, version in cfg['versions'].items():
            assert importlib.metadata.version(package) == version, (package, version, importlib.metadata.version(package))
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        root = Path(cfg['source_root'])
        self.prompt = load_module('e2b_paper_prompt', root / 'code/pipeline/dev_e2e_replay/receiver_prompt.py')
        self.protocol = load_module('e2b_paper_protocol', root / 'code/pipeline/native_action_runtime/protocol_min.py')
        self.parser = load_module('e2b_parser', root / 'code/pipeline/answer_scoring_v2/scoring_v2.py')
        sys.path.insert(0, str(root / 'code/pipeline/native_action_runtime'))
        from legacy_methods import BACKGROUND_PROMPT
        sys.path.pop(0)
        self.background = BACKGROUND_PROMPT
        self.models, self.tokens, self.loads = {}, {}, {}
        for role in ['helper', 'receiver']:
            spec = cfg['models'][role]
            start = time.perf_counter()
            tok = AutoTokenizer.from_pretrained(spec['path'], local_files_only=True)
            assert hashlib.sha256(tok.chat_template.encode()).hexdigest() == spec['chat_template_sha256']
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = 'left'
            assert tok.pad_token_id == spec['pad_token_id']
            model = AutoModelForCausalLM.from_pretrained(
                spec['path'], local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
                low_cpu_mem_usage=True, device_map=spec['device_map']).eval().requires_grad_(False)
            self.protocol.apply_generation_config(model, {'do_sample': False, 'max_new_tokens': 64})
            assert model.hf_device_map == spec['device_map']
            assert {p.device.index for p in model.parameters()} == set(spec['gpus'])
            assert all(p.dtype == torch.bfloat16 and p.device.type == 'cuda' for p in model.parameters())
            self.models[role], self.tokens[role] = model, tok
            self.sync()
            self.loads[role] = time.perf_counter() - start
        assert self.tokens['receiver'].encode(cfg['prefix'], add_special_tokens=False) == cfg['prefix_ids']
        dev = self.models['receiver'].lm_head.weight.device
        self.label_ix = {l: torch.tensor(v, device=dev) for l, v in cfg['label_token_sets'].items()}

    def sync(self):
        for i in range(4):
            torch.cuda.synchronize(i)

    def reset_peak(self):
        self.sync()
        for i in range(4):
            torch.cuda.reset_peak_memory_stats(i)

    def memory(self):
        return {str(i): {'peak_allocated_bytes': torch.cuda.max_memory_allocated(i),
                         'peak_reserved_bytes': torch.cuda.max_memory_reserved(i),
                         'total_bytes': torch.cuda.get_device_properties(i).total_memory} for i in range(4)}

    def body(self, row):
        return self.prompt.build_prompt('mmlu-redux', '', row['question_stem'], self.prompt.choices_block(row['choice_text']), False, False)

    def tokens_for(self, role, messages):
        tok = self.tokens[role]
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        device = self.models[role].get_input_embeddings().weight.device
        tensor = torch.tensor([ids], dtype=torch.long, device=device)
        return {'input_ids': tensor, 'attention_mask': torch.ones_like(tensor)}

    @torch.inference_mode()
    def generate(self, role, messages, maximum):
        model, tok = self.models[role], self.tokens[role]
        inputs = self.tokens_for(role, messages)
        result = model.generate(**inputs, max_new_tokens=maximum, do_sample=False, pad_token_id=tok.pad_token_id)
        ids = result[0, inputs['input_ids'].shape[1]:].detach().cpu().tolist()
        return {'raw': tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                'generated_token_ids': ids, 'input_tokens': int(inputs['input_ids'].shape[1]), 'output_tokens': len(ids)}

    def answer(self, row, helper=None, role='receiver'):
        messages = [] if helper is None else [
            {'role': 'user', 'content': self.background.format(question=self.body(row))},
            {'role': 'assistant', 'content': helper['raw']}]
        messages.append({'role': 'user', 'content': self.prompt.receiver_prompt(row)})
        result = self.generate(role, messages, self.cfg['answer_max_new_tokens'])
        result['parsed'] = self.parser.parse_answer(result['raw'], row['choice_labels'])
        return result

    def helper(self, row):
        return self.generate('helper', [{'role': 'user', 'content': self.background.format(question=self.body(row))}], self.cfg['helper_max_new_tokens'])

    @torch.inference_mode()
    def probe(self, row):
        model = self.models['receiver']
        inputs = self.tokens_for('receiver', [{'role': 'user', 'content': self.prompt.receiver_prompt(row)}])
        prefix = torch.tensor([self.cfg['prefix_ids']], dtype=torch.long, device=inputs['input_ids'].device)
        ids = torch.cat([inputs['input_ids'], prefix], dim=1)
        mask = torch.ones_like(ids)
        hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True,
                             output_hidden_states=False, output_attentions=False)
        assert hidden.past_key_values is None
        # Explicit transfer is required because the layer-sharded backbone may return on its input device.
        last = hidden.last_hidden_state[:, -1:, :].to(model.lm_head.weight.device)
        logits = model.lm_head(last)[0, 0, :].float()
        labels = row['choice_labels']
        label_logits = torch.stack([torch.logsumexp(logits[self.label_ix[l]], dim=0) for l in labels])
        probs = torch.log_softmax(label_logits, dim=0).exp()
        return {'ProbeMax': float((1 - probs.max()).item()), 'p_labels': dict(zip(labels, probs.cpu().tolist())),
                'input_tokens': int(ids.shape[1]), 'probe_ids_sha256': hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                'cache_reused': False}

    def timed(self, fn):
        self.sync()
        start = time.perf_counter()
        result = fn()
        self.sync()
        return result, (time.perf_counter() - start) * 1000

    def complete_row(self, row, helper_baseline=False):
        self.reset_peak()
        self.sync()
        start = time.perf_counter()
        probe, pms = self.timed(lambda: self.probe(row))
        receiver, rms = self.timed(lambda: self.answer(row))
        helper, hms = self.timed(lambda: self.helper(row))
        text, tms = self.timed(lambda: self.answer(row, helper))
        standalone, sms = self.timed(lambda: self.answer(row, role='helper')) if helper_baseline else (None, 0.)
        self.sync()
        return {'id': row['id'], 'P': probe, 'R': receiver, 'H_message': helper, 'T': text, 'H_alone': standalone,
                'timing_ms': {'probe': pms, 'R': rms, 'helper': hms, 'T_receiver': tms, 'H_alone': sms,
                              'total': (time.perf_counter() - start) * 1000}, 'memory': self.memory()}

    def replay(self, row, deployment):
        def policy():
            mode = deployment['mode']
            probe = self.probe(row) if mode == 'selective' else None
            skip = mode == 'fixed_R' or (mode == 'selective' and probe['ProbeMax'] <= deployment['threshold'])
            helper = None if skip else self.helper(row)
            return {'skip': skip, 'P': probe, 'helper': helper, 'answer': self.answer(row, helper)}
        def reference():
            helper = self.helper(row)
            return {'skip': False, 'P': None, 'helper': helper, 'answer': self.answer(row, helper)}
        return policy, reference
