"""Write configs/e1_config.json from the paper's frozen large-pair records (read-only). CPU, no model load.

Sources: P2_MMLU_PRO_BREADTH_STAGE1_20260916T002337Z/MODEL_SOURCE_INDEX.json (helper/receiver path, revision, file hashes,
chat-template hashes) and .../protocol/{label_token_sets,prefix_ids}.json (A-J sets of the same Qwen3-8B tokenizer; E1 keeps A-H).
cpu_preflight.py re-derives the A-H sets from the vocabulary with the paper's rule and checks they are equal.
"""
import json
import os
from pathlib import Path

E1 = Path(__file__).resolve().parents[1]
P2 = Path(os.environ.get('DATA_DIR', '$DATA_DIR'))  # research data root (a private absolute path in the run)
MMLU = P2 / 'P2_MMLU_PRO_BREADTH_STAGE1_20260916T002337Z'

idx = json.loads((MMLU / 'MODEL_SOURCE_INDEX.json').read_text())['models']
sets = json.loads((MMLU / 'protocol/label_token_sets.json').read_text())
prefix = json.loads((MMLU / 'protocol/prefix_ids.json').read_text())
models = {}
for role in ['helper', 'receiver']:
    s = idx[role]
    models[role] = {k: s[k] for k in ['repo_id', 'revision', 'path', 'aggregate_sha256', 'chat_template_sha256']}
    models[role]['files'] = [{'name': f['name'], 'bytes': f['bytes'], 'sha256': f['sha256']} for f in s['files']]
cfg = {
    'source': {'model_index': str(MMLU / 'MODEL_SOURCE_INDEX.json'), 'label_sets': str(MMLU / 'protocol/label_token_sets.json')},
    'agents': {
        'A': {'model': 'helper', 'prompt': 'paper Appendix L helper prompt (legacy_methods.BACKGROUND_PROMPT)', 'max_new_tokens': 256},
        'B': {'model': 'helper', 'prompt': 'prompts/B_prompt_<version>.json (draft; frozen after author approval)', 'max_new_tokens': 256},
        'C': {'model': 'receiver', 'prompt': 'paper receiver template, thinking off', 'max_new_tokens': 64},
    },
    'max_new_tokens': {'A': 256, 'B': 256, 'C': 64},
    'runtime': {'dtype': 'bfloat16', 'attention': 'sdpa', 'decoding': 'greedy (do_sample=False)', 'batch_size': 1,
                'thinking': False, 'tf32': False, 'seed': 0, 'helper_device': 'cuda:0', 'receiver_device': 'cuda:1'},
    'models': models,
    'probe': {
        'name': 'ProbeMax on the A+C prompt',
        'input': 'C reads A (paper Text-reading conversation), add_generation_prompt, thinking off, then the separately encoded prefix',
        'prefix': 'The correct answer is', 'prefix_ids': prefix['ids'],
        'label_token_sets': {k: sets[k] for k in 'ABCDEFGH'},
        'score': ('FP32 logits; per label logsumexp over its token set; log_softmax over A-H; u = logsumexp of the seven '
                  'non-top labels\' log-probabilities (log non-top mass, PROTOCOL.md section 4); non-finite -> u = None, '
                  'always waits for B; 1 - max p kept as u_1mp_diag (diagnostic only)'),
    },
    'parser': {'file': 'code/pipeline/answer_scoring_v2/scoring_v2.py',
               'sha256': 'd05978f400cf69aa6bddc56168a260ac3ea9158abaa640e9331602dc9bfcc2f9', 'labels': list('ABCDEFGH'),
               'INVALID': 'its own answer value; never equal to gold; INVALID vs a letter counts as a change'},
    'thresholds': {'candidates': 'j = 1..19: ceil(j*N_fit/20)-th smallest fit u (ties kept, non-finite ranked last); j = 20 (q = 1): always skip',
                   'rule': 'skip B when u <= tau; a non-finite score never skips (except the constant always-skip policy j = 20)'},
}
(E1 / 'configs/e1_config.json').write_text(json.dumps(cfg, indent=1) + '\n')
print('wrote', E1 / 'configs/e1_config.json')
