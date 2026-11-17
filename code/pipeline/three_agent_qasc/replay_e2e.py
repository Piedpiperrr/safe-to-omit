"""E1 end-to-end latency replay on the 128-question dev panel (batch size 1, no warm-up, cold requests kept).

Topology: A and B (Qwen2.5-7B-Instruct) on cuda:0, C (Qwen3-8B) and the probe on cuda:1, one node.
Arms, interleaved per question; the order rotates left by panel ordinal modulo 2 (as in the paper's replay):
  reference : A -> B -> C(A, B)
  policy    : A -> probe(C reads A) -> skip if u <= tau -> C(A)        else -> B -> C(A, B)
A and B are generated live in every request (nothing is reused across requests or arms).
Per request: T_A, T_probe, T_B, T_C (each between device synchronisations on both GPUs) and the end-to-end time
from question fields to the parsed answer. A failed request is recorded, retried once and flagged; nothing is dropped.
Requires the frozen policy file (tau from calibration); refuses to run without it.
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
    ap.add_argument('--policy', required=True, help='frozen policy JSON: {"tau": float or "Infinity", "q": .., "b_prompt_sha256": ..}')
    ap.add_argument('--b-prompt', required=True)
    ap.add_argument('--panel', default=str(L.E1 / 'splits/panel128_dev_ids.json'))
    ap.add_argument('--n', type=int, default=128)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--offline-dir', help='dev generation records, for live-vs-offline identity checks (recorded, not enforced)')
    a = ap.parse_args()

    pol = json.loads(Path(a.policy).read_text())
    bp = L.load_b_prompt(a.b_prompt)
    assert pol.get('frozen') is True, 'policy file is not marked frozen'
    assert pol['b_prompt_sha256'] == bp['_sha256'], 'policy was frozen with a different B prompt'
    tau = pol['tau']
    panel = json.loads(Path(a.panel).read_text())[:a.n]
    assert set(panel) <= set(L.split_ids('dev')) and len(set(panel)) == len(panel)
    Q = L.load_queries()
    offline = {}
    if a.offline_dir:
        for f in sorted(Path(a.offline_dir).glob('records.dev.*.jsonl')):
            for line in open(f):
                r = json.loads(line)
                offline[r['id']] = r

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'requests.jsonl').exists(), 'no automatic resume: start a new out-dir'
    t0 = time.perf_counter()
    M = L.Models('cuda:0', 'cuda:1')
    M.sync()
    import torch
    (out / 'startup.json').write_text(json.dumps({
        'utc': utc(), 'job_id': os.environ.get('PBS_JOBID'), 'host': os.uname().nodename, 'startup_wall_s': time.perf_counter() - t0,
        'load_seconds': M.load_seconds, 'gpus': [torch.cuda.get_device_name(i) for i in [0, 1]],
        'policy': pol, 'b_prompt': {'version': bp['version'], 'sha256': bp['_sha256']}, 'panel_n': len(panel),
        'e1lib_sha256': L.sha_file(Path(L.__file__)), 'replay_sha256': L.sha_file(Path(__file__)), 'warmup_requests': 0}, indent=1) + '\n')

    def timed(fn):
        M.sync()
        s = time.perf_counter()
        v = fn()
        M.sync()
        return v, (time.perf_counter() - s) * 1000

    def request(q, arm):
        T = {'A': 0.0, 'probe': 0.0, 'B': 0.0, 'C': 0.0}
        M.sync()
        start = time.perf_counter()
        qq = {k: q[k] for k in ['id', 'question_stem', 'choice_labels', 'choice_text', 'fact1', 'fact2']}
        A, T['A'] = timed(lambda: M.run_A(qq))
        msgs = {'A': A['text']}
        pr, route, B = None, 'ABC', None
        if arm == 'policy':
            pr, T['probe'] = timed(lambda: M.probe(qq, bp, msgs))
            route = 'AC' if L.skip(pr['u'], tau) else 'ABC'
        if route == 'ABC':
            B, T['B'] = timed(lambda: M.run_B(qq, bp, facts=True, note=msgs['A']))
            msgs['B'] = B['text']
        C, T['C'] = timed(lambda: M.run_C(qq, bp, route, msgs))
        M.sync()
        total = (time.perf_counter() - start) * 1000
        C.pop('_rendered', None)
        return {'route': route, 'T_ms': T, 'e2e_ms': total, 'other_ms': total - sum(T.values()),
                'A': {k: A[k] for k in ['text', 'n_in', 'n_out', 'truncated']},
                'B': ({k: B[k] for k in ['text', 'n_in', 'n_out', 'truncated']} if B else None),
                'probe': ({k: pr[k] for k in ['u', 'nonfinite', 'argmax', 'n_in', 'probe_ids_sha256']} if pr else None),
                'C': {k: C[k] for k in ['raw', 'answer', 'invalid', 'n_in', 'n_out']}}

    first = {'policy': True, 'reference': True}
    first_any = True
    with open(out / 'requests.jsonl', 'a') as fr, open(out / 'failures.jsonl', 'a') as ff:
        for ordinal, i in enumerate(panel):
            arms = ['policy', 'reference']
            k = ordinal % 2
            for pos, arm in enumerate(arms[k:] + arms[:k]):
                for attempt in (1, 2):
                    try:
                        r = request(Q[i], arm)
                        break
                    except Exception as e:  # noqa: BLE001
                        ff.write(json.dumps({'id': i, 'arm': arm, 'attempt': attempt, 'error': repr(e),
                                             'traceback': traceback.format_exc(), 'utc': utc()}) + '\n')
                        ff.flush()
                        torch.cuda.empty_cache()
                        r = None
                rec = {'id': i, 'panel_ordinal': ordinal, 'order_position': pos, 'arm': arm, 'utc': utc(),
                       'cold_first_request_overall': first_any, 'cold_first_request_of_arm': first[arm],
                       'retried': attempt == 2, 'runtime_failure': r is None, 'tau': tau}
                if r:
                    rec.update(r)
                    off = offline.get(i)
                    if off:
                        s = off['steps']
                        rec['offline_identity'] = {
                            'A_text_equal': s.get('A') is not None and s['A']['text'] == r['A']['text'],
                            'B_text_equal': None if not r['B'] else (s.get('B') is not None and s['B']['text'] == r['B']['text']),
                            'u_equal': None if not r['probe'] else (s.get('probe') is not None and s['probe']['u'] == r['probe']['u']),
                            'answer_equal': (s.get('C_' + r['route']) or {}).get('answer') == r['C']['answer']}
                fr.write(json.dumps(rec, ensure_ascii=False) + '\n')
                fr.flush()
                first[arm] = False
                first_any = False
            if ordinal % 16 == 15:
                print(utc(), 'panel', ordinal + 1, '/', len(panel), flush=True)
    (out / 'REPLAY_DONE.json').write_text(json.dumps({'utc': utc(), 'requests': 2 * len(panel)}) + '\n')


if __name__ == '__main__':
    main()
