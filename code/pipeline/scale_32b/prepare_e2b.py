"""Prepare label-free paper inputs and the HF configuration; no model forward."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_ROOT = Path(os.environ.get('DATA_DIR', '$DATA_DIR'))  # shared-filesystem data root (a private absolute path in the run)
FIELDS = {'id', 'question_stem', 'choice_labels', 'choice_text'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    op = gzip.open if str(p).endswith('.gz') else open
    with op(p, 'rt') as f:
        values = [json.loads(s) for s in f]
    assert all(set(r) == FIELDS for r in values), 'Only label-free query sources are permitted'
    assert len({r['id'] for r in values}) == len(values)
    return values


def stable_write(p, text):
    if p.exists():
        assert p.read_text() == text, f'Prepared input/config differs: {p}'
    else:
        p.write_text(text)


def main():
    assert not (HERE / 'FREEZE.txt').exists(), 'Do not change prepared files after E2b freeze'
    from transformers import AutoTokenizer
    sources = []
    train_path = ROOT / 'records/boundaries_large_small/inputs/obqa_train_queries.jsonl.gz'
    dev_path = ROOT / 'records/boundaries_large_small/inputs/obqa_dev_queries.jsonl'
    held_path = ROOT / 'records/e09b_e09c/inputs/holdout_744_queries.jsonl'
    sources += [train_path, dev_path, held_path]
    train = {r['id']: r for r in rows(train_path)}
    inputs = HERE / 'inputs'
    inputs.mkdir(exist_ok=True)
    pops = {}
    for split, expected_n in [('fit', 2100), ('cal', 1366), ('dev', 742), ('heldout', 744)]:
        ids_path = ROOT / (f'splits/development_and_calibration/obqa_{split}_ids.json' if split != 'heldout'
                           else 'results/e09b_e09c/inputs/holdout_744_ids.json')
        ids = read(ids_path)
        data = train if split in ['fit', 'cal'] else {r['id']: r for r in rows(dev_path if split == 'dev' else held_path)}
        selected = [data[i] for i in ids]
        assert len(ids) == len(set(ids)) == expected_n
        p = inputs / f'obqa_{split}_queries.jsonl'
        stable_write(p, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
        pops[split] = {'path': str(p), 'N': expected_n, 'sha256': sha(p), 'ids_source': str(ids_path)}
        sources.append(ids_path)
    ids_by_split = {sp: {r['id'] for r in rows(v['path'])} for sp, v in pops.items()}
    assert len(set.union(*ids_by_split.values())) == 4952
    cfg = {'status': 'PREPARED_INPUTS_NOT_EXECUTION_AUTHORIZATION', 'source_root': str(ROOT),
           'python': str(DATA_ROOT / 'software/envs/c2c_official/bin/python'),
           'versions': {'torch': '2.6.0', 'transformers': '4.52.4', 'accelerate': '1.9.0'},
           'backend': 'HF transformers only; no vLLM; no quantization or CPU/disk weight offload',
           'dtype': 'bfloat16', 'attention': 'sdpa', 'do_sample': False, 'batch_size': 1,
           'helper_max_new_tokens': 256, 'answer_max_new_tokens': 64, 'prefix': 'The correct answer is',
           'seed': 0, 'alpha': .05, 'delta_per_candidate': .001, 'q_grid': [i/20 for i in range(1,21)],
           'epsilons': [.01, .02], 'bootstrap_replicates': 2000, 'bootstrap_seed': 0,
           'pilot_size': 300, 'pilot_min_gain_pp': 2.0, 'smoke_size': 16,
           'full_stage_shards': 4, 'inputs': pops, 'models': {}}
    specs = [('helper', 'Qwen/Qwen2.5-32B-Instruct', '5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd', [0,1]),
             ('receiver', 'Qwen/Qwen3-32B', '9216db5781bf21249d130ec9da846c4624c16137', [2,3])]
    for role, repo, rev, gpus in specs:
        p = DATA_ROOT / 'cache/huggingface/hub' / ('models--' + repo.replace('/', '--')) / 'snapshots' / rev
        conf = read(p / 'config.json')
        assert conf['num_hidden_layers'] == 64 and not conf.get('tie_word_embeddings', False)
        tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        device_map = {'model.embed_tokens': gpus[0], 'model.norm': gpus[1], 'lm_head': gpus[1]}
        device_map.update({f'model.layers.{i}': gpus[0] if i < 32 else gpus[1] for i in range(64)})
        cfg['models'][role] = {'repo': repo, 'revision': rev, 'path': str(p), 'gpus': gpus, 'device_map': device_map,
                               'pad_token_id': tok.pad_token_id, 'chat_template_sha256': hashlib.sha256(tok.chat_template.encode()).hexdigest(),
                               'config_sha256': sha(p / 'config.json'), 'index_sha256': sha(p / 'model.safetensors.index.json')}
        if role == 'receiver':
            cfg['prefix_ids'] = tok.encode(cfg['prefix'], add_special_tokens=False)
            label_sets = {l: [] for l in ['A','B','C','D']}
            special = set(tok.all_special_ids)
            for i in sorted(set(tok.get_vocab().values())):
                if i not in special:
                    label = tok.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()
                    if label in label_sets:
                        label_sets[label].append(i)
            assert all(label_sets.values())
            cfg['label_token_sets'] = label_sets
    paper_files = ['code/pipeline/dev_e2e_replay/receiver_prompt.py', 'code/pipeline/native_action_runtime/protocol_min.py',
                   'code/pipeline/native_action_runtime/legacy_methods.py', 'code/pipeline/answer_scoring_v2/scoring_v2.py',
                   'code/pipeline/confidence_boundaries/run_boundaries.py', 'code/analysis/e15/scripts/e15_1.py',
                   'code/analysis/e21/scripts/e21_1_cert.py', 'code/analysis/e16/replay_llama/src/execute.py',
                   'splits/timing_panels/obqa_dev_panel_ids_e2e_replay.json']
    sources += [ROOT / x for x in paper_files]
    cfg['source_sha256'] = {str(p): sha(p) for p in sources}
    cfg['panel_ids'] = str(ROOT / paper_files[-1])
    # Draft configuration can be refreshed while editing; the entry guard forbids this after FREEZE.
    (HERE / 'e2b_config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    print(json.dumps({'status': 'PREPARATION_ONLY_PASS', 'N': {k: v['N'] for k,v in pops.items()},
                      'smoke_ids': [r['id'] for r in rows(pops['fit']['path'])[:16]], 'config': str(HERE / 'e2b_config.json')}))


if __name__ == '__main__':
    main()
