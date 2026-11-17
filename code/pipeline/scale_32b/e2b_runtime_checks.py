"""Non-statistical consistency preflight and explicit path warmups."""
import json
from e2_debug_ops import units,completed
from e2b_gate import freeze_path,sha


def signature(row):
    return {'helper':row['H_message']['raw'],'R_raw':row['R']['raw'],'R_parsed':row['R']['parsed'],
            'Text_raw':row['T']['raw'],'Text_parsed':row['T']['parsed'],
            'u_repr':repr(row['P']['score']),'score_finite':row['P']['score_finite']}


def pilot_reference():
    rows=[];sources=[]
    for u in units('e2b','pilot'):
        d,c,m=completed('e2b',u,sha(freeze_path()));rr=[json.loads(x) for x in (d/c['records_file']).read_text().splitlines()]
        assert [r['ordinal'] for r in rr]==list(range(u['start'],u['stop']))
        rows+=rr;sources.append({'records':str(d/c['records_file']),'sha256':c['records_sha256']})
    assert len(rows)==300 and [r['ordinal'] for r in rows]==list(range(300))
    return rows[:2],sources


def check(rt,fit_rows):
    ref,sources=pilot_reference();pairs=[]
    for q,expected in zip(fit_rows[:2],ref):
        assert q['id']==expected['id']
        observed=rt.complete_row(q,helper_baseline=False)
        a,b=signature(expected),signature(observed)
        pairs.append({'id':q['id'],'match':a==b,'different_fields':[k for k in a if a[k]!=b[k]],'expected':a,'observed':b})
    return {'status':'PASS' if all(r['match'] for r in pairs) else 'CONSISTENCY_MISMATCH',
            'pairs':pairs,'sources':sources,'used_in_statistics':False,'gold_read':False}


def warmup(rt,rows,deployments,cfg):
    events=[]
    for q in rows:
        for name in cfg['timing_policy_order']:
            calls=rt.replay(q,deployments[name])
            for arm in cfg['timing_arms']:
                calls[arm]();rt.sync();events.append({'id':q['id'],'policy':name,'path':arm})
            # Exercise both selective computational paths regardless of the
            # natural route on these warmup-only questions. No threshold is changed.
            rt.probe(q);rt.answer(q,None);rt.sync()
            events.append({'id':q['id'],'policy':name,'path':'probe_then_R_kernel_coverage'})
            rt.probe(q);h=rt.helper(q);rt.answer(q,h);rt.sync()
            events.append({'id':q['id'],'policy':name,'path':'probe_then_helper_then_Text_kernel_coverage'})
    return {'performed':True,'ids':[q['id'] for q in rows],'events':events,'timed':False,'used_in_statistics':False}
