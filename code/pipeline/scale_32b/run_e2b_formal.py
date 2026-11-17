"""Debug operational worker; scientific Runtime32 and frozen configuration unchanged."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import traceback

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent


def atomic_json(path,value):
    # Copied unchanged from the as-run claim helper module (not released): atomic JSON write.
    path=Path(path)
    tmp=path.with_name(path.name+f'.tmp-{os.getpid()}')
    with tmp.open('x') as f:
        json.dump(value,f,indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--rank',type=int,required=True);a=ap.parse_args()
    from e2_debug_ops import read,put,unit_by_key,partial,root,finish,preflight_barrier,digest
    from e2b_formal_gate import require_formal_stage
    from e2b_gate import stage_rows,sha,freeze_path
    manifest=read(a.manifest);u=manifest['units'][a.rank];assert manifest['kind']=='e2b' and u==unit_by_key('e2b',u['key'])
    frozen=require_formal_stage(u['stage']);cfg=read(HERE/'e2b_formal_config.json');op=read(HERE/'e2b_debug_plan.json')
    job=os.environ['PBS_JOBID'];host=socket.gethostname()  # as run, also asserted a ClusterA compute node (check removed)
    jr=root('e2b')/'jobs'/job;evidence=jr/f'allocation/node_{a.rank}/allocation.json';allocation=read(evidence)
    assert allocation['exclusive'] and allocation['host']==host and len(allocation['nodes'])==manifest['nodes']
    population=stage_rows(cfg,u['stage']);selected=population[u['start']:u['stop']]
    assert len(population)==u['population_N']
    # As run, a persistent per-unit claim on the shared filesystem was taken here (scheduling only; helper not released).
    out=partial('e2b',u['key']);out.mkdir(parents=True,exist_ok=False)
    put(out/'STARTED.json',{'job_id':job,'unit':u,'host':host,'utc':datetime.now(timezone.utc).isoformat()})
    deadline=time.monotonic()+op['execution_budget_seconds']
    def timeout(signum,frame):raise TimeoutError('OPERATIONAL_50_MINUTE_OR_DEADLINE_LIMIT')
    signal.signal(signal.SIGTERM,timeout)
    meta={'unit':u,'job_id':job,'host':host,'N':len(selected),'id_order':[r['id'] for r in selected],
          'created_utc':datetime.now(timezone.utc).isoformat(),'analysis_sha256':frozen['analysis_sha256'],
          'freeze_sha256':sha(freeze_path()),'freeze_commit':frozen['commit'],'gold_read':False,
          'allocation':str(evidence),'manifest':str(a.manifest),'manifest_sha256':sha(a.manifest),
          'consistency':{'status':'NOT_APPLICABLE_PILOT'},'warmup':{'performed':False,'timed':False,'events':[]}}
    atomic_json(out/'metadata.json',meta)
    print('E2B_FORMAL_PATHS',json.dumps({'models':{r:s['path'] for r,s in cfg['models'].items()},'inputs':cfg['inputs'],
          'incomplete_output':str(out),'temporary':os.environ.get('TMPDIR'),'logs':str(HERE/'logs'),'checkpoint':None}),flush=True)
    try:
        from hf32_formal_runtime import Runtime32,torch
        start=time.perf_counter();rt=Runtime32(cfg)
        meta.update(load_seconds=rt.loads,startup_seconds=time.perf_counter()-start,load_memory=rt.memory(),
                    gpus=[str(torch.cuda.get_device_properties(i)) for i in range(4)],device_maps={r:m.hf_device_map for r,m in rt.models.items()},versions=cfg['versions'])
        fit_rows=stage_rows(cfg,'pilot')
        if u['stage']!='pilot':
            from e2b_runtime_checks import check
            meta['consistency']=check(rt,fit_rows)
            atomic_json(out/'metadata.json',meta)
            if meta['consistency']['status']!='PASS':
                try:put(jr/'CONSISTENCY_MISMATCH.json',{'unit':u['key'],'metadata':str(out/'metadata.json'),'retry_allowed':False})
                except FileExistsError:pass
                raise RuntimeError('CONSISTENCY_MISMATCH: no retry')
        preflight_barrier(jr,u['key'],manifest['nodes'],{'status':'PASS','unit':u['key']},deadline)
        deployments=read(HERE/'e2b_results/deployment.json') if u['stage']=='replay' else None
        if u['warmup']:
            from e2b_runtime_checks import warmup
            warm=[fit_rows[i] for i in op['replay_warmup_fit_ordinals']]
            assert not {r['id'] for r in warm}.intersection(r['id'] for r in population)
            meta['warmup']=warmup(rt,warm,deployments,cfg)
        if u['stage']=='replay':meta['cold_start_caveat']=op['cold_start_caveat']
        atomic_json(out/'metadata.json',meta)
        with (out/'records.jsonl').open('x') as f:
            for ordinal,row in enumerate(selected,start=u['start']):
                assert time.monotonic()<deadline-120,'INSUFFICIENT_FINISH_RESERVE'
                assert datetime.now(timezone.utc)<datetime.fromisoformat(cfg['deadline_utc']),'E2B_DEADLINE_REACHED'
                if u['stage']=='replay':
                    result={'id':row['id'],'ordinal':ordinal,'policies':{}}
                    names=cfg['timing_policy_order'];offset=ordinal%3;names=names[offset:]+names[:offset]
                    for name in names:
                        calls=rt.replay(row,deployments[name]);rr={};order=cfg['timing_arms']
                        shift=(ordinal+cfg['timing_policy_order'].index(name))%3;order=order[shift:]+order[:shift];rr['execution_order']=order
                        for mode in order:rr[mode],rr[mode+'_latency_ms']=rt.timed(calls[mode])
                        result['policies'][name]=rr
                else:result=rt.complete_row(row,helper_baseline=u['stage'] in ['dev','heldout'])
                result.update(ordinal=ordinal,unit_key=u['key'],segment_start=u['start'],host=host,job_id=job)
                f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
                if ordinal%10==0 or ordinal+1==u['stop']:print('E2B_PROGRESS',u['key'],ordinal+1,u['stop'],flush=True)
        dest=finish('e2b',u,out,{'N':len(selected),'ids':[r['id'] for r in selected],
                              'freeze_sha256':sha(freeze_path()),'records_file':'records.jsonl','records_sha256':sha(out/'records.jsonl')})
        print('E2B_COMPLETE',str(dest),flush=True)
    except BaseException:
        bad=(jr/'CONSISTENCY_MISMATCH.json').exists()
        atomic_json(out/'FAILED.json',{'utc':datetime.now(timezone.utc).isoformat(),'traceback':traceback.format_exc(),
                                      'retry_allowed':not bad and u['attempt']==0,'partial_outputs_preserved':True,'included_in_analysis':False})
        pf=jr/'preflight'/(u['key']+'.json')
        if not pf.exists():put(pf,{'status':'FAILED','unit':u['key']})
        raise


if __name__=='__main__':main()
