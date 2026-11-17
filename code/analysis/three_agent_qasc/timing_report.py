"""Development timing panel report (PROTOCOL.md Section 8), one replay job (policy vs reference, 128 dev questions).

Written 2026-11-10 before any replay output existed. Reads outputs/replay128/<job>/ (requests.jsonl, startup.json,
node.txt, nvidia-smi.txt, layout_test/LAYOUT_CHECK.json, REPLAY_DONE.json). Cold requests are kept.
  saving_i = e2e_reference_i - e2e_policy_i (ms) per panel question (paired by question);
  mean saving and relative saving 1 - mean(policy) / mean(reference), each with a paired percentile bootstrap 95%
  interval (2,000 resamples over questions, numpy default_rng(0)); the timing criterion of Section 6 is
  "lower end of the mean-saving interval > 0".
Also reported: per-arm mean e2e and components (T_A, T_probe, T_B, T_C, other), the policy's skip share on the panel and
the mean e2e by route, runtime failures / retries, live-vs-offline identity, the layout test, node and GPU model.
A question whose policy or reference request failed after its retry has no saving; such pairs are counted and left out
of the means (none expected).
  python timing_report.py outputs/replay128/<job>   -> <job dir>/TIMING_REPORT.json and TIMING_REPORT.md
"""
import json
import sys
from pathlib import Path

import numpy as np

SEED, REPS = 0, 2000


def main(d):
    d = Path(d)
    assert (d / 'REPLAY_DONE.json').exists(), 'replay not finished'
    st = json.loads((d / 'startup.json').read_text())
    R = [json.loads(l) for l in open(d / 'requests.jsonl') if l.strip()]
    by = {}
    for r in R:
        by.setdefault(r['id'], {})[r['arm']] = r
    panel = sorted(by, key=lambda i: by[i][next(iter(by[i]))]['panel_ordinal'])
    ok = [i for i in panel if all(a in by[i] and not by[i][a]['runtime_failure'] for a in ('policy', 'reference'))]
    pol = np.array([by[i]['policy']['e2e_ms'] for i in ok])
    ref = np.array([by[i]['reference']['e2e_ms'] for i in ok])
    sav = ref - pol
    rng = np.random.default_rng(SEED)
    IX = rng.integers(0, len(ok), size=(REPS, len(ok)))
    bs_sav = sav[IX].mean(axis=1)
    bs_rel = 1 - pol[IX].mean(axis=1) / ref[IX].mean(axis=1)

    def comp(arm):
        rs = [by[i][arm] for i in ok]
        out = {'mean_e2e_ms': float(np.mean([r['e2e_ms'] for r in rs]))}
        for k in ('A', 'probe', 'B', 'C'):
            out[f'mean_T_{k}_ms'] = float(np.mean([r['T_ms'][k] for r in rs]))
        out['mean_other_ms'] = float(np.mean([r['other_ms'] for r in rs]))
        return out

    routes = [by[i]['policy']['route'] for i in ok]
    ident = [by[i][a].get('offline_identity') for i in ok for a in ('policy', 'reference') if by[i][a].get('offline_identity')]
    lay = d / 'layout_test/LAYOUT_CHECK.json'
    res = {'job_dir': str(d), 'node': (d / 'node.txt').read_text().strip() if (d / 'node.txt').exists() else None,
           'host': st.get('host'), 'gpus': st.get('gpus'), 'policy': st.get('policy'), 'panel_n': len(panel),
           'pairs_timed': len(ok), 'pairs_without_saving': len(panel) - len(ok),
           'requests': len(R), 'runtime_failures': sum(r['runtime_failure'] for r in R), 'retried': sum(r['retried'] for r in R),
           'warmup_requests': st.get('warmup_requests'),
           'cold_first_request': {a: next((r['e2e_ms'] for r in R if r['arm'] == a and r['cold_first_request_of_arm']), None)
                                  for a in ('policy', 'reference')},
           'saving_ms': {'mean': float(sav.mean()), 'median': float(np.median(sav)),
                         'ci95': [float(np.percentile(bs_sav, 2.5)), float(np.percentile(bs_sav, 97.5))]},
           'relative_saving': {'value': float(1 - pol.mean() / ref.mean()),
                               'ci95': [float(np.percentile(bs_rel, 2.5)), float(np.percentile(bs_rel, 97.5))]},
           'bootstrap': {'resamples': REPS, 'seed': SEED, 'rng': 'numpy.random.default_rng', 'interval': 'percentile', 'paired': 'by question'},
           'timing_criterion_saving_lower_gt_0': float(np.percentile(bs_sav, 2.5)) > 0,
           'policy_arm': comp('policy'), 'reference_arm': comp('reference'),
           'policy_skip_share_on_panel': routes.count('AC') / len(routes),
           'policy_mean_e2e_by_route_ms': {rt: float(np.mean([by[i]['policy']['e2e_ms'] for i in ok if by[i]['policy']['route'] == rt]))
                                           for rt in sorted(set(routes))},
           'offline_identity': {k: f"{sum(1 for x in ident if x.get(k) is True)}/{sum(1 for x in ident if x.get(k) is not None)}"
                                for k in ('A_text_equal', 'B_text_equal', 'u_equal', 'answer_equal')},
           'layout_check': json.loads(lay.read_text()) if lay.exists() else None}
    if res['layout_check']:
        res['layout_check'] = {k: v for k, v in res['layout_check'].items() if k != 'per_question'}
    (d / 'TIMING_REPORT.json').write_text(json.dumps(res, indent=1) + '\n')
    s, rs = res['saving_ms'], res['relative_saving']
    md = [f"policy {res['policy'].get('name')} (j {res['policy'].get('j')}, q {res['policy'].get('q')}); host {res['host']}; GPUs {res['gpus']}",
          f"pairs timed {res['pairs_timed']}/{res['panel_n']}; failures {res['runtime_failures']}; retried {res['retried']}; skip share on panel {res['policy_skip_share_on_panel']:.3f}",
          f"mean e2e: policy {res['policy_arm']['mean_e2e_ms']:.0f} ms, reference {res['reference_arm']['mean_e2e_ms']:.0f} ms",
          f"saving {s['mean']:.0f} ms [{s['ci95'][0]:.0f}, {s['ci95'][1]:.0f}]; relative {rs['value']:.3f} [{rs['ci95'][0]:.3f}, {rs['ci95'][1]:.3f}]; "
          f"criterion (lower > 0): {res['timing_criterion_saving_lower_gt_0']}",
          f"layout check: {res['layout_check']}", f"offline identity: {res['offline_identity']}"]
    (d / 'TIMING_REPORT.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main(sys.argv[1])
