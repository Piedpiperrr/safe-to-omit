"""E1 generation: one shard of one split, A/B/C messages + ProbeMax + receiver answers, batch size 1.

Per question (each A and B message generated once and shared by every path that reads it):
  main     : A -> probe(C reads A) -> C(A)            [path AC, the cheap path]
             B(facts + A note) -> C(A, B)             [path ABC, the reference system]
  controls : C alone [C]; C reads fact1+fact2 [CF]; B(facts, no A) -> C(B) [BC]; B(A note, no facts) -> C(A, B') [ABnfC]
Controls run only on dev (or with --smoke on fit). No gold is read. The sealed test split is refused unless
rebuttal/e1/POLICY_FREEZE.json exists. A failed step is recorded and retried once; if the retry also fails the
question stays in the records with the failure (no post-hoc removal) and steps that need its output are marked.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1lib as L  # noqa: E402


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['fit', 'cal', 'dev', 'test_sealed'])
    ap.add_argument('--ids-file', help='optional subset of the split (JSON list)')
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--paths', choices=['main', 'all', 'controls_c'], default='main',
                    help="controls_c = only C alone and C reading the facts (added 2026-11-10 for fit accuracies, DEVIATIONS.md item 4)")
    ap.add_argument('--smoke', action='store_true', help='allow controls on fit (smoke test only)')
    ap.add_argument('--b-prompt', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--helper-gpu', type=int, default=0)
    ap.add_argument('--receiver-gpu', type=int, default=1)
    ap.add_argument('--save-rendered', type=int, default=0, help='store full rendered inputs for the first N questions')
    ap.add_argument('--verify-large-files', action='store_true', help='sha256 of every weight shard before loading')
    ap.add_argument('--deadline-epoch', type=float, default=float(os.environ.get('E1_DEADLINE_EPOCH', 'inf')))
    a = ap.parse_args()

    if a.split == 'test_sealed' and not (L.E1 / 'POLICY_FREEZE.json').exists():
        raise SystemExit('REFUSED: sealed test split before POLICY_FREEZE.json exists')
    if a.paths == 'controls_c' and a.split not in ('fit', 'dev'):
        raise SystemExit('REFUSED: controls_c runs on fit or dev only')
    if a.paths == 'all' and a.split != 'dev' and not a.smoke:
        raise SystemExit('REFUSED: controls/ablations run on dev only (use --smoke for the fit smoke test)')

    split_ids = L.split_ids(a.split)
    ids = split_ids
    if a.ids_file:
        ids = json.loads(Path(a.ids_file).read_text())
        assert set(ids) <= set(split_ids), 'ids-file is not a subset of the split'
    ids = ids[a.offset:]
    if a.limit is not None:
        ids = ids[:a.limit]
    ids = [x for k, x in enumerate(ids) if k % a.nshards == a.shard]

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{a.split}.shard{a.shard}of{a.nshards}'
    rec_path = out / f'records.{tag}.jsonl'
    fail_path = out / f'failures.{tag}.jsonl'
    rend_path = out / f'rendered.{tag}.jsonl'
    status_path = out / f'status.{tag}.json'
    # As run, a cross-host shard lock was taken here (scheduling only, 2026-11-10: one writer per shard across ClusterB
    # and ClusterA; exit 73 if busy). The lock helper is not released.
    done = set()
    if rec_path.exists():
        for line in rec_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)['id'])
    todo = [i for i in ids if i not in done]
    if not todo:  # scheduling only (2026-11-10): a finished shard exits before loading models; status file kept as is
        print(utc(), tag, f'nothing to do: all {len(ids)} assigned ids already have records; models not loaded, '
              'status file left unchanged', flush=True)
        return

    bp = L.load_b_prompt(a.b_prompt)
    Q = L.load_queries()
    t_load = time.perf_counter()
    M = L.Models(f'cuda:{a.helper_gpu}', f'cuda:{a.receiver_gpu}', verify_large_files=a.verify_large_files)
    load_s = time.perf_counter() - t_load
    import torch
    env = {'job_id': os.environ.get('PBS_JOBID'), 'host': os.uname().nodename,
           'gpus': [torch.cuda.get_device_name(d) for d in [a.helper_gpu, a.receiver_gpu]],
           'gpu_mem_bytes': [torch.cuda.get_device_properties(d).total_memory for d in [a.helper_gpu, a.receiver_gpu]],
           'torch': torch.__version__, 'transformers': __import__('transformers').__version__}
    status = {'tag': tag, 'started_utc': utc(), 'args': vars(a), 'env': env, 'load_seconds_total': load_s,
              'load_seconds': M.load_seconds, 'file_checks': M.file_checks, 'b_prompt': {'version': bp['version'], 'sha256': bp['_sha256']},
              'e1lib_sha256': L.sha_file(Path(L.__file__)), 'generate_sha256': L.sha_file(Path(__file__)),
              'n_assigned': len(ids), 'n_already_done': len(done), 'gold_read': False}
    status_path.write_text(json.dumps(status, indent=1) + '\n')

    def timed(fn):
        M.sync()
        t0 = time.perf_counter()
        v = fn()
        M.sync()
        return v, (time.perf_counter() - t0) * 1000

    n_rendered = 0
    n_done = 0
    stopped = None
    with rec_path.open('a') as fr, fail_path.open('a') as ff, rend_path.open('a') as fx:
        for qn, i in enumerate(todo):
            if time.time() > a.deadline_epoch - 300:
                stopped = 'deadline'
                break
            q = Q[i]
            rec = {'id': i, 'split': a.split, 'utc': utc(), 'b_prompt_version': bp['version'], 'b_prompt_sha256': bp['_sha256'],
                   'first_after_load': qn == 0 and not done, 'steps': {}, 'ms': {}, 'failures': [], 'gold_read': False}
            msgs = {}
            rendered = {}

            def step(name, fn, needs=()):
                missing = [n for n in needs if n not in rec['steps'] or rec['steps'][n] is None]
                if missing:
                    rec['steps'][name] = None
                    rec['failures'].append({'step': name, 'status': 'not_run_dependency_failed', 'missing': missing})
                    return None
                for attempt in (1, 2):
                    try:
                        v, ms = timed(fn)
                        if isinstance(v, dict) and '_rendered' in v:
                            rendered[name] = v.pop('_rendered')
                        rec['steps'][name] = v
                        rec['ms'][name] = ms
                        if attempt == 2:
                            rec['failures'].append({'step': name, 'status': 'succeeded_on_retry'})
                        return v
                    except Exception as e:  # noqa: BLE001
                        err = {'id': i, 'step': name, 'attempt': attempt, 'error': repr(e), 'traceback': traceback.format_exc(), 'utc': utc()}
                        ff.write(json.dumps(err) + '\n')
                        ff.flush()
                        torch.cuda.empty_cache()
                rec['steps'][name] = None
                rec['failures'].append({'step': name, 'status': 'runtime_failure_after_retry'})
                return None

            if a.paths == 'controls_c':
                step('C_C', lambda: M.run_C(q, bp, 'C', msgs))
                step('C_CF', lambda: M.run_C(q, bp, 'CF', msgs))
            else:
                A = step('A', lambda: M.run_A(q))
                if A:
                    msgs['A'] = A['text']
                step('probe', lambda: M.probe(q, bp, msgs), needs=['A'])
                step('C_AC', lambda: M.run_C(q, bp, 'AC', msgs), needs=['A'])
                B = step('B', lambda: M.run_B(q, bp, facts=True, note=msgs['A']), needs=['A'])
                if B:
                    msgs['B'] = B['text']
                step('C_ABC', lambda: M.run_C(q, bp, 'ABC', msgs), needs=['A', 'B'])
            if a.paths == 'all':
                step('C_C', lambda: M.run_C(q, bp, 'C', msgs))
                step('C_CF', lambda: M.run_C(q, bp, 'CF', msgs))
                Bn = step('B_noA', lambda: M.run_B(q, bp, facts=True, note=None))
                if Bn:
                    msgs['B_noA'] = Bn['text']
                step('C_BC', lambda: M.run_C(q, bp, 'BC', msgs), needs=['B_noA'])
                Bf = step('B_nofact', lambda: M.run_B(q, bp, facts=False, note=msgs['A']), needs=['A'])
                if Bf:
                    msgs['B_nofact'] = Bf['text']
                step('C_ABnfC', lambda: M.run_C(q, bp, 'ABnfC', msgs), needs=['A', 'B_nofact'])
            fr.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fr.flush()
            if n_rendered < a.save_rendered:
                fx.write(json.dumps({'id': i, 'rendered': rendered,
                                     'A_user': L.a_user_message(q),
                                     'B_user': L.b_user_message(q, bp, facts=True, note=msgs.get('A')),
                                     'B_noA_user': L.b_user_message(q, bp, facts=True, note=None),
                                     'B_nofact_user': L.b_user_message(q, bp, facts=False, note=msgs.get('A'))}, ensure_ascii=False) + '\n')
                fx.flush()
                n_rendered += 1
            n_done += 1
            if n_done % 25 == 0:
                print(utc(), tag, 'done', n_done, '/', len(todo), flush=True)
    status.update({'ended_utc': utc(), 'n_done_this_run': n_done, 'stopped': stopped,
                   'complete': (len(done) + n_done) == len(ids)})
    status_path.write_text(json.dumps(status, indent=1) + '\n')
    print(utc(), tag, 'finished', n_done, 'complete' if status['complete'] else 'INCOMPLETE', flush=True)


if __name__ == '__main__':
    main()
