"""Full-development replay of synchronous, early-return and reference arms."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run(args):
    assert os.environ.get('PBS_JOBID'), 'GPU execution requires PBS allocation'
    from e2a_v2_gate import start,freeze_path
    from e2_debug_ops import read as read_json,finish,plan
    frozen,u,out,allocation=start(args.manifest,args.rank)
    args.pair=u['pair'];deadline=time.monotonic()+3000
    cfg=read(HERE/'e2a_v2_config.json');op=plan('e2a')
    root=Path(cfg['source_root']);spec=cfg['model_pairs'][args.pair];job_id=os.environ['PBS_JOBID']
    records=out/'batches.jsonl'
    assert not records.exists()
    for path, expected in cfg['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    paths = {'models': {r: s['path'] for r, s in spec['models'].items()}, 'datasets': cfg['panels'],
             'output': str(out), 'checkpoint': None, 'log': str(HERE / 'logs'), 'temporary': os.environ.get('TMPDIR'),
             'caches': {k: v for k, v in os.environ.items() if k in ['HF_HOME', 'HF_HUB_CACHE', 'HF_DATASETS_CACHE', 'XDG_CACHE_HOME', 'TORCH_HOME', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH']}}
    assert str(Path.home()) not in json.dumps(paths), paths  # as run: the literal home directory of the run account
    print('RESOLVED_PATHS', json.dumps(paths), flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ngpu = 2 if args.layout == 'two-gpu' else 1
    assert torch.cuda.device_count() == ngpu, (torch.cuda.device_count(), ngpu)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prompt = module('e2_paper_prompt', root / 'code/pipeline/dev_e2e_replay/receiver_prompt.py')
    protocol = module('e2_paper_protocol', root / 'code/pipeline/native_action_runtime/protocol_min.py')
    parser = module('e2_v2_parser', root / 'code/pipeline/answer_scoring_v2/scoring_v2.py')
    # Import the original constant directly; this module loads no models.
    sys.path.insert(0, str(root / 'code/pipeline/native_action_runtime'))
    from legacy_methods import BACKGROUND_PROMPT
    sys.path.pop(0)

    def sync():
        for i in range(ngpu):
            torch.cuda.synchronize(i)

    models, toks, load_seconds = {}, {}, {}
    for role, device in [('helper', 'cuda:0'), ('receiver', 'cuda:1' if ngpu == 2 else 'cuda:0')]:
        t = time.perf_counter()
        s = spec['models'][role]
        tok = AutoTokenizer.from_pretrained(s['path'], local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = 'left'
        assert tok.pad_token_id == s['generation_pad_token_id']
        model = AutoModelForCausalLM.from_pretrained(s['path'], local_files_only=True, torch_dtype=torch.bfloat16,
                                                    attn_implementation='sdpa').to(device).eval().requires_grad_(False)
        protocol.apply_generation_config(model, {'do_sample': False, 'max_new_tokens': 64})
        if role == 'receiver':
            assert model.generation_config.to_dict() == spec['receiver_generation_config']
        toks[role], models[role] = tok, model
        sync()
        load_seconds[role] = time.perf_counter() - t
    meta = {'utc': utc(), 'job_id': job_id, 'host': socket.gethostname(), 'pair': args.pair, 'layout': args.layout,
            'gpus': [str(torch.cuda.get_device_properties(i)) for i in range(ngpu)],
            'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'numpy']},
            'load_seconds': load_seconds, 'timing': cfg['timing'], 'paths': paths, 'prefix_cache': False,
            'freeze_commit': frozen['commit'], 'freeze_sha256':hashlib.sha256(freeze_path().read_bytes()).hexdigest(), 'exclusive': True,
            'unit':u,'allocation':allocation,'warmup':{'performed':False,'events':[]},'manifest':args.manifest,
            'config_sha256': hashlib.sha256((HERE / 'e2a_v2_config.json').read_bytes()).hexdigest(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()}
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    label_ix = {l: torch.tensor(ids, device=models['receiver'].device) for l, ids in spec['label_token_sets'].items()}
    prefix_ids = toks['receiver'].encode(cfg['prefix'], add_special_tokens=False)

    def chat_ids(tok, messages):
        return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)

    def pad(tok, sequences, device):
        return tok.pad({'input_ids': sequences, 'attention_mask': [[1] * len(x) for x in sequences]},
                       padding=True, return_tensors='pt').to(device)

    def body(q):
        return prompt.build_prompt('mmlu-redux', '', q['question_stem'], prompt.choices_block(q['choice_text']), False, False)

    @torch.inference_mode()
    def probe(rows):
        tok, model = toks['receiver'], models['receiver']
        seq = [chat_ids(tok, [{'role': 'user', 'content': prompt.receiver_prompt(q)}]) + prefix_ids for q in rows]
        inputs = pad(tok, seq, model.device)
        position_ids = inputs['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(inputs['attention_mask'] == 0, 1)
        hidden = model.model(**inputs, position_ids=position_ids, use_cache=False, return_dict=True,
                             output_hidden_states=False, output_attentions=False)
        assert hidden.past_key_values is None
        logits = model.lm_head(hidden.last_hidden_state[:, -1:, :])[:, 0, :].float()
        result = []
        for j, q in enumerate(rows):
            labels = q['choice_labels']
            label_logits = torch.stack([torch.logsumexp(logits[j, label_ix[l]], dim=0) for l in labels])
            probabilities = torch.log_softmax(label_logits, dim=0).exp()
            result.append({'score': float((1 - probabilities.max()).item()),
                           'p_labels': dict(zip(labels, probabilities.cpu().tolist())), 'input_tokens': len(seq[j])})
        return result

    @torch.inference_mode()
    def generate(role, conversations, max_new):
        if not conversations:
            return []
        tok, model = toks[role], models[role]
        seq = [chat_ids(tok, c) for c in conversations]
        inputs = pad(tok, seq, model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        new = outputs[:, inputs['input_ids'].shape[1]:].cpu().tolist()
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        trimmed = []
        for ids in new:
            end = next((k + 1 for k, i in enumerate(ids) if i in eos), len(ids))
            trimmed.append(ids[:end])
        texts = tok.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [{'text': text, 'token_ids': ids, 'input_tokens': len(s)} for text, ids, s in zip(texts, trimmed, seq)]

    from e2a_v2_batch import execute_batch, arm_order, ARMS
    assert cfg['arms']==list(ARMS) and cfg['repetitions']==1

    def helpers(rows):
        chats = [[{'role': 'user', 'content': BACKGROUND_PROMPT.format(question=body(q))}] for q in rows]
        return generate('helper', chats, cfg['helper_max_new_tokens'])

    def answers(rows, messages):
        chats = []
        for q, h in zip(rows, messages):
            chat = [] if h is None else [
                {'role': 'user', 'content': BACKGROUND_PROMPT.format(question=body(q))},
                {'role': 'assistant', 'content': h['text']}]
            chat.append({'role': 'user', 'content': prompt.receiver_prompt(q)})
            chats.append(chat)
        return generate('receiver', chats, cfg['receiver_max_new_tokens'])

    ds=u['dataset'];dataset_index=cfg['datasets'].index(ds)
    rows=[json.loads(line) for line in Path(cfg['panels'][ds]['queries']).read_text().splitlines()]
    threshold=spec['deployments'][ds]['threshold']
    assert len(rows)==u['population_N'] and u['start']%max(u['batch_sizes'])==0
    meta['id_order']=[r['id'] for r in rows[u['start']:u['stop']]];meta['N']=len(meta['id_order'])
    if u['warmup']:
        info=op['warmup_inputs'][ds];path=Path(info['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
        warm=[json.loads(line) for line in path.read_text().splitlines()]
        assert not {r['id'] for r in warm}.intersection(r['id'] for r in rows)
        events=[]
        for size in [1,8,32]:
            for k in [0,1]:
                batch=warm[k*size:(k+1)*size]
                for arm in ARMS:
                    execute_batch(batch,arm,threshold,probe,helpers,answers,parser.parse_answer,sync,clock=lambda:0.)
                    events.append({'B':size,'batch':k,'arm':arm,'ids':[r['id'] for r in batch]})
                answers(batch,[None]*len(batch));sync()
                events.append({'B':size,'batch':k,'arm':'plain_receiver_kernel_coverage','ids':[r['id'] for r in batch]})
        meta['warmup']={'performed':True,'timed':False,'used_in_statistics':False,'events':events}
    meta['first_measured_id']=rows[u['start']]['id']
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    with records.open('x') as stream:
        count=0
        for batch_size in u['batch_sizes']:
            batch_size_index=cfg['batch_sizes'].index(batch_size)
            for start_index in range(u['start'],u['stop'],batch_size):
                assert time.monotonic()<deadline-120,'INSUFFICIENT_FINISH_RESERVE'
                batch_id=start_index//batch_size;batch=rows[start_index:min(start_index+batch_size,u['stop'])]
                modes=arm_order(cfg['pair_order'].index(args.pair),dataset_index,batch_size_index,batch_id)
                for order_position,mode in enumerate(modes):
                    completion_events=[]
                    def deliver(ready):
                        completion_events.append({'ids':[r['id'] for r in ready],'latency_ms':ready[0]['latency_ms']})
                    result,elapsed,phases=execute_batch(batch,mode,threshold,probe,helpers,answers,parser.parse_answer,sync,on_complete=deliver)
                    record={'pair':args.pair,'dataset':ds,'batch_size':batch_size,'batch_id':batch_id,'panel_start':start_index,
                            'mode':mode,'order_position':order_position,'repetition':0,'N':len(batch),'latency_ms':elapsed,
                            'phase_end_ms':phases,'completion_events':completion_events,'requests':result,'utc':utc(),
                            'first_batch_of_segment':count==0,'unit_key':u['key'],'segment_first_id':meta['first_measured_id'],
                            'job_id':job_id,'host':socket.gethostname(),'layout':args.layout}
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush();count+=1
                if batch_id%16==0:print('E2A_V2_PROGRESS',u['key'],batch_size,start_index+len(batch),flush=True)
    assert count==3*sum((u['stop']-u['start']+b-1)//b for b in u['batch_sizes'])
    dest=finish('e2a',u,out,{'N':meta['N'],'ids':meta['id_order'],'batches':count,'records_file':'batches.jsonl',
                'records_sha256':hashlib.sha256(records.read_bytes()).hexdigest(),'freeze_sha256':meta['freeze_sha256'],
                'max_memory_allocated':[torch.cuda.max_memory_allocated(i) for i in range(ngpu)]})
    print('E2A_V2_COMPLETE',str(dest),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--rank',type=int,default=0)
    ap.add_argument('--layout',choices=['two-gpu'],default='two-gpu');args=ap.parse_args()
    def timeout(signum,frame):raise TimeoutError('OPERATIONAL_50_MINUTE_LIMIT')
    signal.signal(signal.SIGTERM,timeout)
    try:run(args)
    except BaseException:
        import traceback
        from e2_debug_ops import partial,put
        u=read(args.manifest)['units'][args.rank];out=partial('e2a',u['key'])
        if out.exists() and not (out/'FAILED.json').exists():
            put(out/'FAILED.json',{'utc':utc(),'traceback':traceback.format_exc(),'retry_allowed':u['attempt']==0,
                                  'included_in_analysis':False,'partial_outputs_preserved':True})
        raise


if __name__ == '__main__':
    main()
