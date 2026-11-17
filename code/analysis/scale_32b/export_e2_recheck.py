"""CPU-only compatibility exports; never change native results or frozen analyses.

No model imports, new labels, inference, statistical selection, or use-release.
Missing raw diagnostics remain empty, following the authors' explicit decision.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import tempfile

from e2b_gate import sha
from e2b_analyze import Analysis, answer, read, score, score_text, write, csvwrite

HERE=Path(__file__).resolve().parent
# Internal research documents copied into each bundle; they are not part of the release (the released deviations are
# the English DEVIATIONS.md files of the dataset).
DOCS=['E2_DEVIATIONS_ZH.md','DEBUG_HANDOFF_ZH.md','RECHECK_SCHEMA_ZH.md']


def csvread(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def native(a,stage):
    p=a.recheck/(stage+'_export.json');receipt=read(a.source(p))
    assert receipt['freeze_sha256']==sha(HERE/'FREEZE_DEBUG.txt')
    assert receipt['analysis_commit']==a.frozen['commit']
    assert receipt['analysis_sha256']==a.frozen['analysis_sha256']
    for name,h in receipt['files'].items():
        assert Path(name).name==name
        assert sha(a.source(a.recheck/name))==h
    return receipt


def calibration(a,out):
    """Rename existing fields, retaining the native query order and exact scores."""
    a.prerequisite('cal_PASS');native(a,'cal')
    fit=a.load('pilot')+a.load('fit_rest');cal=a.load('cal')
    ledger=read(a.source(a.out/'calibration.json'))
    base_fit=csvread(a.recheck/'fit.csv');base_cal=csvread(a.recheck/'cal.csv')
    assert [r['id'] for r in base_fit]==[r['id'] for r in fit]
    assert [r['id'] for r in base_cal]==[r['id'] for r in cal]
    queries=ledger['label_queries'];queried={q['id']:q for q in queries}
    assert len(queried)==len(queries)
    by_id={r['id']:r for r in cal}
    for rows,raw in [(base_fit,fit),(base_cal,cal)]:
        for r,v in zip(rows,raw):
            assert (r['u'],r['ans_s'],r['ans_b'])==(score_text(score(v)),answer(v['R']),answer(v['T']))
    for r in base_cal:
        q=queried.get(r['id'])
        assert r['gold']==(q['gold'] if q else '')
        assert r['queried']==str(int(q is not None))
        assert r['query_candidate_j']==(str(q['query_candidate_j']) if q else '')
    csvwrite(out/'fit.csv',['id','u','gold','ans_s','ans_b'],base_fit)
    csvwrite(out/'cal.csv',['id','u','gold','ans_s','ans_b','queried','query_candidate_j'],base_cal)
    write(out/'thresholds.csv',(a.recheck/'thresholds.csv').read_text())
    mapped=[]
    for i,q in enumerate(queries,1):
        assert q['query_ordinal']==i
        r=by_id[q['id']];s,b=answer(r['R']),answer(r['T'])
        assert s!=b and q['u']==score_text(score(r))
        assert q['eps_requested'] and q['gold'] in 'ABCD'
        mapped.append({'order':i,'id':q['id'],'candidate_j':q['query_candidate_j'],
                       'eps_round':min(q['eps_requested']),'u':q['u'],'s_answer':s,'b_answer':b,
                       'gold':q['gold'],'lost_correction':int(b==q['gold'] and s!=q['gold']),
                       'eps_requested':q['eps_requested']})
    write(out/'label_queries.jsonl',''.join(json.dumps(q,ensure_ascii=False,allow_nan=False)+'\n' for q in mapped))
    candidates={c['j']:c for c in ledger['candidates']};policies={}
    for name,p in ledger['policies'].items():
        if p['j'] is None:
            assert p['mode']=='fixed_Text'
            n=m=l=0;pc=1.;tau=None;known=0;complete=True
        else:
            c=candidates[p['j']];tau=c['tau'];assert tau==p['threshold']
            selected=[r for r in cal if math.isfinite(score(r)) and score(r)<=float(tau)]
            disagreements=[r for r in selected if answer(r['R'])!=answer(r['T'])]
            n,m=len(selected),len(disagreements);assert (n,m)==(c['n_skip'],c['k_changed'])
            pc=c['p_change'];known=sum(r['id'] in queried for r in disagreements);complete=known==m
            l=sum(answer(r['T'])==queried[r['id']]['gold'] and answer(r['R'])!=queried[r['id']]['gold'] for r in disagreements) if complete else None
        policies[name]={'j':p['j'],'tau':tau,'n':n,'m':m,'l':l,'p_change':pc,
                        'mode':p['mode'],'epsilon':p['epsilon'],'q':p['q'],
                        'calibration_fallback':p['calibration_fallback'],'N_cal':len(cal),
                        'l_complete':complete,'queried_disagreements':known,
                        'native_deployment':p}
    write(out/'policies.json',policies)
    # The frozen probe never serialized logp or argmax; no reconstruction from u.
    lp=[{'id':r['id'],'lp_A':'','lp_B':'','lp_C':'','lp_D':'','u':score_text(score(r)),'argmax':''} for r in fit]
    csvwrite(out/'fit_label_logprobs.csv',['id','lp_A','lp_B','lp_C','lp_D','u','argmax'],lp)
    gaps={'fit_label_logprobs':{'status':'NOT_RECORDED','rows':len(fit),'blank_columns':['lp_A','lp_B','lp_C','lp_D','argmax'],
                              'reason':'Frozen probe saved u, not label log probabilities. No re-inference or reconstruction; authors approved keeping the run.'},
          'policy_l_unobserved':[name for name,p in policies.items() if not p['l_complete']]}
    write(out/'EXPORT_GAPS.json',gaps)
    return gaps


def heldout(a,out):
    a.prerequisite('heldout_EVAL');native(a,'heldout')
    for name in ['heldout.csv','heldout_evaluation.json']:write(out/name,(a.recheck/name).read_text())
    rows=csvread(out/'heldout.csv');raw=a.load('heldout')
    assert len(rows)==744 and [r['id'] for r in rows]==[r['id'] for r in raw]
    for r,v in zip(rows,raw):assert (r['u'],r['ans_s'],r['ans_b'])==(score_text(score(v)),answer(v['R']),answer(v['T']))
    # Extra standalone helper predictions use already evaluated held-out gold.
    csvwrite(out/'heldout_single_models.csv',['id','gold','ans_R','ans_Text','ans_helper'],
             [{'id':r['id'],'gold':r['gold'],'ans_R':r['ans_s'],'ans_Text':r['ans_b'],'ans_helper':answer(v['H_alone'])} for r,v in zip(rows,raw)])


def development(a,out):
    """Export evaluated development rows; copy all statistical artifacts verbatim."""
    release=a.prerequisite('dev_EVAL')
    evaluation=a.source(a.out/'dev_evaluation.json')
    assert str(evaluation) in release['evidence_sha256']
    assert read(evaluation)['N']==742
    rows=a.load('dev');ids=[r['id'] for r in rows]
    assert len(rows)==len(set(ids))==742
    gold=a.gold(ids)
    csvwrite(out/'dev.csv',['id','u','ans_s','ans_b','gold'],
             [{'id':r['id'],'u':score_text(score(r)),'ans_s':answer(r['R']),
               'ans_b':answer(r['T']),'gold':gold[r['id']]} for r in rows])
    helper_present=[r.get('H_alone') is not None for r in rows]
    assert all(helper_present) or not any(helper_present),'Partial helper baseline: do not silently omit rows'
    if all(helper_present):
        csvwrite(out/'dev_single_models.csv',['id','gold','ans_R','ans_Text','ans_helper'],
                 [{'id':r['id'],'gold':gold[r['id']],'ans_R':answer(r['R']),
                   'ans_Text':answer(r['T']),'ans_helper':answer(r['H_alone'])} for r in rows])
    (out/'dev_evaluation.json').write_bytes(evaluation.read_bytes())
    # Reuse the already exported policy/count mapping; do not recompute calibration.
    bundle=HERE/'transfer/heldout';verify_bundle(bundle)
    a.source(bundle/'SHA256SUMS')
    for name in ['thresholds.csv','policies.json']:
        source=a.source(bundle/'e2b_recheck'/name)
        (out/name).write_bytes(source.read_bytes())
    manifest(out)


def timing(a,out):
    a.prerequisite('timing_REPORT');native(a,'timing');raw=a.load('replay')
    units={m['unit']['key']:m['unit'] for m in a.segment_metadata['replay']}
    per=csvread(a.recheck/'timing_requests.csv');by={(r['id'],r['policy'],r['arm']):r for r in per}
    result=[]
    for r in raw:
        for policy,arms in r['policies'].items():
            for arm in ['policy','reference','receiver_alone']:
                v=arms[arm];old=by[r['id'],policy,arm]
                assert float(old['latency_ms'])==arms[arm+'_latency_ms']
                u=units[r['unit_key']]
                result.append(dict(old,route='R_alone' if arm=='receiver_alone' else 'skip' if v['skip'] else 'wait',
                                   segment_index=u['segment_index'],segment_start=r['segment_start'],
                                   is_segment_first=int(r['ordinal']==r['segment_start']),
                                   probe_ms='',helper_ms='',receiver_ms='',component_timing_status='NOT_RECORDED',
                                   latency_source='measured_synchronized_outer_arm_wall_time'))
    assert len(result)==len(per)==128*3*3
    csvwrite(out/'timing_requests.csv',list(result[0]),result)
    for name in ['timing_results.json','timing_records.jsonl']:write(out/name,(a.recheck/name).read_text())
    return {'status':'NOT_RECORDED','rows':len(result),'blank_columns':['probe_ms','helper_ms','receiver_ms'],
            'reason':'Frozen replay saved only measured whole-arm latency. Component timings from generation are not substituted.'}


def e2a(out,evidence):
    from e2_debug_ops import units,completed
    from e2a_v2_gate import verify,freeze_path
    frozen=verify();cfg=read(HERE/'e2a_v2_config.json');rows=[];segments=[]
    def source(p):evidence[str(p)]=sha(p);return p
    for u in units('e2a'):
        d,c,m=completed('e2a',u,sha(freeze_path()));segments.append(m)
        for name in ['COMPLETE.json','metadata.json',c['records_file']]:source(d/name)
        rr=[json.loads(s) for s in (d/c['records_file']).read_text().splitlines()]
        for b in rr:
            for r in b['requests']:
                rows.append({'pair':b['pair'],'dataset':b['dataset'],'arm':b['mode'],'batch_size':b['batch_size'],
                             'batch_id':b['batch_id'],'id':r['id'],'route':'skip' if r['skip'] else 'wait','skip':r['skip'],
                             'completion_ms':repr(float(r['latency_ms'])),'batch_wall_ms':repr(float(b['latency_ms'])),
                             'completion_origin':'measured_from_own_arm_batch_start',
                             'answer':r['parsed']['answer'] if r['parsed']['valid'] else 'INVALID',
                             'raw_answer':r['answer']['text'],'helper_message':r['helper']['text'] if r['helper'] else '',
                             'u':score_text(r['probe']['score']) if r['probe'] else '',
                             'unit_key':u['key'],'is_segment_first':int(r['id']==b['segment_first_id']),
                             'execution_position':b['order_position'],'job_id':b['job_id'],'host':b['host'],
                             'phase_end_ms':json.dumps(b['phase_end_ms'],sort_keys=True), 'source':str(d/c['records_file'])})
    counts=[]
    for pair in cfg['pair_order']:
        for ds,info in cfg['panels'].items():
            ids=read(info['ids'])
            for B in cfg['batch_sizes']:
                for arm in cfg['arms']:
                    selected=[r for r in rows if (r['pair'],r['dataset'],r['batch_size'],r['arm'])==(pair,ds,B,arm)]
                    assert [r['id'] for r in selected]==ids
                    counts.append({'pair':pair,'dataset':ds,'batch_size':B,'arm':arm,'N':len(selected),'status':'COMPLETE'})
    assert len(counts)==36
    csvwrite(out/'per_request.csv',list(rows[0]),rows);csvwrite(out/'cells.csv',list(counts[0]),counts)
    write(out/'segments.json',segments)
    # Include the committed analyzer's original tables, without recomputing statistics here.
    summaries=csvread(HERE/'e2a_v2_arms.csv')
    assert len(summaries)==36 and all(r['status']=='COMPLETE' for r in summaries)
    native_rows=csvread(HERE/'e2a_v2_per_request.csv')
    key=lambda r:(r['pair'],r['dataset'],int(r.get('B',r.get('batch_size'))),r['arm'],r['id'])
    lookup={key(r):r for r in native_rows}
    assert len(lookup)==len(native_rows)==len(rows)
    for r in rows:
        old=lookup[key(r)]
        assert float(old['latency_ms'])==float(r['completion_ms']) and old['answer']==r['answer']
    for name in ['E2A_V2_REPORT.md','e2a_v2_arms.csv','e2a_v2_latency.csv','e2a_v2_consistency.csv',
                 'e2a_v2_sensitivity.csv','e2a_v2_batches.csv','e2a_v2_per_request.csv','e2a_v2_segments.json']:
        write(out/name,source(HERE/name).read_text())
    return {'freeze_commit':frozen['commit'],'freeze_sha256':sha(freeze_path()),'analysis_sha256':frozen['analysis_sha256']}


def manifest(root):
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    write(root/'SHA256SUMS',''.join(sha(p)+'  '+str(p.relative_to(root))+'\n' for p in files))


def verify_bundle(root):
    manifest_path=root/'SHA256SUMS';expected={}
    for line in manifest_path.read_text().splitlines():
        digest,name=line.split('  ',1);p=root/name
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert sha(p)==digest,name;expected[name]=digest
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS'}
    assert actual==set(expected)
    return expected


def export(stage,here=HERE):
    assert here==HERE
    dest=HERE/'transfer'/stage
    if dest.exists():
        verify_bundle(dest);print('EXISTING_IMMUTABLE_EXPORT_VERIFIED',dest);return dest
    script=Path(__file__).resolve();repo=HERE.parents[1];relative=str(script.relative_to(repo))
    # The operator may run only committed export code.
    committed=subprocess.check_output(['git','show','HEAD:'+relative],cwd=repo)
    assert committed==script.read_bytes(),'Commit export code before using it on real outputs'
    export_commit=subprocess.check_output(['git','log','-1','--format=%H','--',relative],cwd=repo,text=True).strip()
    for name in DOCS:assert subprocess.check_output(['git','show','HEAD:'+str((HERE/name).relative_to(repo))],cwd=repo)==(HERE/name).read_bytes(),name
    if stage=='dev':
        # Use the completed p45 units while keeping the frozen analysis untouched.
        from e2b_plan45 import activate
        activate()
    tmp=HERE/'tmp';tmp.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='recheck_export_',dir=tmp) as td:
        package=Path(td)/stage;package.mkdir();out=package/('e2a_v2_recheck' if stage=='e2a_v2' else 'e2b_recheck');out.mkdir()
        evidence={};gaps={}
        if stage=='e2a_v2':provenance=e2a(out,evidence)
        else:
            a=Analysis()
            if stage=='dev':development(a,out)
            else:gaps=calibration(a,out)
            if stage=='heldout':heldout(a,out)
            if stage=='timing':
                gaps['timing_components']=timing(a,out)
                # calibration writes its own gaps before timing adds the second known gap.
                write(out/'TIMING_GAPS.json',gaps['timing_components'])
            evidence.update(a.evidence)
            provenance={'freeze_commit':a.frozen['commit'],'freeze_sha256':sha(HERE/'FREEZE_DEBUG.txt'),'analysis_sha256':a.frozen['analysis_sha256']}
            if stage=='dev':
                provenance.update(gold_source_sha256=dict(a.gold_evidence),statistical_recomputation=False,
                                  N=742,helper_baseline_exported=(out/'dev_single_models.csv').exists())
        for name in DOCS:write(package/name,(HERE/name).read_text());evidence[str(HERE/name)]=sha(HERE/name)
        for name in ['E2_DEVIATIONS_RUNTIME_ZH.jsonl','FREEZE.txt','FREEZE_DEBUG.txt','E2A_V2_FREEZE.txt','E2A_V2_FREEZE_DEBUG.txt']:
            if (HERE/name).exists():write(package/name,(HERE/name).read_text());evidence[str(HERE/name)]=sha(HERE/name)
        write(package/'EXPORT_PROVENANCE.json',dict(provenance,stage=stage,created_utc=datetime.now(timezone.utc).isoformat(),
              exporter_commit=export_commit,exporter_sha256={str(script):sha(script)},source_sha256=evidence,
              external_use='PENDING_RECHECK',diagnostic_gaps=gaps,
              note='Compatibility formatting only, added after pilot submission. Frozen science and native exports are preserved. No recheck success is implied.'))
        manifest(package);verify_bundle(package);dest.parent.mkdir(exist_ok=True);package.rename(dest)
    print('TRANSFER_READY_PENDING_RECHECK',dest);return dest


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['cal','dev','heldout','timing','e2a_v2','verify']);p.add_argument('--bundle',type=Path)
    args=p.parse_args()
    if args.stage=='verify':
        assert args.bundle;verify_bundle(args.bundle);print('HASHES_VERIFIED_NOT_SCIENTIFIC_RECHECK',args.bundle)
    else:
        assert args.bundle is None;export(args.stage)

if __name__=='__main__':main()
