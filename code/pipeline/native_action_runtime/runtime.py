import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import torch
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import legacy_methods as lm
from protocol_min import apply_generation_config, receiver_prompt_tensors, generate_receiver
from scoring import score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'

def utc(): return datetime.now(timezone.utc).isoformat()
def save(path, obj): path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
def readrows(path): return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
def sync():
    for i in [0, 1]: torch.cuda.synchronize(i)

# Existing stage wrappers are diagnostic instrumentation. One outer wall timer is authoritative.
lm.cuda_wall_stage = lambda fn, device: (fn(), 0., 0.)
lm.wall_stage = lambda fn: (fn(), 0.)
# Discard the legacy wrappers' unused internal parse; authoritative frozen
# scoring is applied by run_p2_10 after the synchronized action timer.
lm.extract_prediction = lambda text: None

class Runner:
    def __init__(self, config):
        self.load_times = {}
        for name, device in [('helper', 'cuda:0'), ('receiver', 'cuda:1')]:
            start = time.perf_counter()
            spec = config['models'][name]
            tok = AutoTokenizer.from_pretrained(spec['path'], local_files_only=True)
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(spec['path'], local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval().requires_grad_(False)
            apply_generation_config(model, {'do_sample': False, 'max_new_tokens': 64})
            setattr(self, name, model)
            setattr(self, name+'_tok', tok)
            sync()
            self.load_times[name+'_seconds'] = time.perf_counter()-start
        assert self.receiver.generation_config.to_dict() == config['receiver_generation_config']
        lm.C2C_FUSERS = Path(config['fuser']['path'])
        lm.HELPER_CONFIG = self.helper.config
        lm.RECEIVER_CONFIG = self.receiver.config
        self.th = lm.T2THelperBundle(self.helper, self.helper_tok)
        self.tr = lm.T2TReceiverBundle(self.receiver, self.receiver_tok)
        self.ch = lm.C2CSharerHelperBundle(self.helper, self.receiver_tok)
        start = time.perf_counter()
        self.cr = lm.C2CReceiverSideFuserBundle(self.receiver, self.receiver_tok)
        for fuser in self.cr.projectors: fuser.eval().requires_grad_(False)
        sync()
        self.load_times['fusers_seconds'] = time.perf_counter()-start
        start = time.perf_counter()
        self.semantic_tok = AutoTokenizer.from_pretrained(config['models']['semantic_encoder']['path'], local_files_only=True, trust_remote_code=False)
        if self.semantic_tok.pad_token_id is None: self.semantic_tok.pad_token = self.semantic_tok.eos_token
        self.semantic = AutoModel.from_pretrained(config['models']['semantic_encoder']['path'], local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32).eval().to('cuda:0').requires_grad_(False)
        sync()
        self.load_times['semantic_encoder_seconds'] = time.perf_counter()-start

    @torch.inference_mode()
    def request(self, row, action):
        if action == 'receiver_only':
            _, _, tensors = receiver_prompt_tensors(self.receiver_tok, row, self.receiver.device)
            text, ids = generate_receiver(self.receiver, self.receiver_tok, tensors)
            return {'raw_answer': text, 'generated_token_ids': ids.cpu().tolist(), 'receiver_input_tokens': tensors['input_ids'].shape[1]}
        if action == 'text':
            message = self.th.run(row)
            result = self.tr.consume(row, message['helper_body'], message['helper_message'])
            return {'raw_answer': result['generated_text'], 'generated_token_ids': result['generated_token_ids'],
                    'receiver_input_tokens': result['receiver_input_token_count'], 'helper_message': message['helper_message'],
                    'helper_generated_token_ids': message['helper_generated_token_ids'], 'helper_input_tokens': message['helper_input_token_count']}
        source = self.ch.run(row)
        stacked = lm.cache_to_stacked(source['source_cache'], source['instruction_length'])
        # Actual local device transfer GPU0 -> GPU1, as in stage.
        cache = lm.stacked_to_cache(stacked, 'cuda:1')
        base = self.cr.prepare_base(row)
        assert base['instruction_length'] == source['instruction_length']
        result = self.cr.fuse_and_generate(base, cache)
        return {'raw_answer': result['generated_text'], 'generated_token_ids': result['generated_token_ids'],
                'receiver_input_tokens': result['receiver_input_token_count'], 'helper_input_tokens': source['instruction_length'],
                'transferred_kv_bytes': stacked.numel()*stacked.element_size(),
                '_projected': result['projected_tensor_for_validation']}
