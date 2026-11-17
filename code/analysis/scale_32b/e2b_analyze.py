"""Frozen CPU analysis/export entry for E2b. No model imports or GPU execution.

All subcommands use the same pre-generation committed file. Results are internal
until a separate recheck receipt matches the complete export manifest.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import beta, binom, rankdata
from e2b_gate import verify_freeze, stage_rows, sha, freeze_path
from e2b_formal_gate import validate_release

HERE=Path(__file__).resolve().parent
POLICIES=['joint_eps001','behavior','joint_eps002']
EPS=[.01,.02]


def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def finite(u):return u is not None and math.isfinite(float(u))
def score(r):return float(r['P']['score']) if r['P']['score_finite'] and finite(r['P']['score']) else math.inf
def score_text(u):return repr(float(u)) if finite(u) else 'inf'
def answer(a):return a['parsed']['answer'] if a['parsed']['valid'] else 'INVALID'
def accept(k,n,risk):return n>0 and float(binom.cdf(k,n,risk))<=.001

def cp(k,n,level=.95):
    if n==0:return [None,None]
    tail=(1-level)/2
    return [0. if k==0 else float(beta.ppf(tail,k,n-k+1)),1. if k==n else float(beta.ppf(1-tail,k+1,n-k))]

def upper(k,n,quantile=.999):
    return 1. if n==0 or k==n else float(beta.ppf(quantile,k+1,n-k))

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=data if isinstance(data,str) else json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if path.exists():assert path.read_text()==text,'Preserve existing artifact: '+str(path)
    else:
        with path.open('x') as f:f.write(text)
    return path

def csvwrite(path,fields,rows):
    f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    return write(path,f.getvalue())

def arrays(rows):
    return np.array([score(r) for r in rows]),np.array([answer(r['R']) for r in rows]),np.array([answer(r['T']) for r in rows])

def grid(u):
    ordered=np.sort(np.where(np.isfinite(u),u,np.inf));n=len(u)
    return [{'j':j,'q':j/20,'tau':score_text(ordered[(j*n+19)//20-1]) if j<20 else 'inf'} for j in range(1,21)]

def mask(u,tau):return np.isfinite(u)&(u<=float(tau))
def route(u,p):return np.zeros(len(u),bool) if p['mode']=='fixed_Text' else mask(u,p['threshold'])

def paired_means(values,reps=2000):
    v=np.asarray(values,dtype=float)
    if v.ndim==1:v=v[:,None]
    rng=np.random.default_rng(0);boot=[]
    for start in range(0,reps,100):
        ix=rng.integers(0,len(v),size=(min(100,reps-start),len(v)))
        boot.append(v[ix].mean(axis=1))
    lo,hi=np.quantile(np.concatenate(boot),[.025,.975],axis=0)
    return [{'mean':float(v[:,i].mean()),'ci95':[float(lo[i]),float(hi[i])]} for i in range(v.shape[1])]

def auroc(u,d):
    d=np.asarray(d,bool);pos=int(d.sum());neg=len(d)-pos
    if not pos or not neg:return None
    ranks=rankdata(u,method='average')
    return float((ranks[d].sum()-pos*(pos+1)/2)/(pos*neg))

def pilot_stats(rows,gold,reps=2000):
    u,s,b=arrays(rows);y=np.array([gold[r['id']] for r in rows]);d=s!=b
    G=paired_means((b==y).astype(int)-(s==y).astype(int),reps)[0]
    led=[]
    for cut in grid(u):
        msk=mask(u,cut['tau']);m=int(msk.sum());k=int(d[msk].sum())
        n_hat=(2*1366*m+len(u))//(2*len(u))
        k_hat=(2*n_hat*k+m)//(2*m) if m else 0
        p=float(binom.cdf(k_hat,n_hat,.05)) if n_hat else 1.
        led.append(dict(cut,m=m,d=k,n_hat=n_hat,k_hat=k_hat,p=p,accepted=n_hat>0 and p<=.001))
    accepted=[c for c in led if c['accepted']]
    net=int((b==y).sum()-(s==y).sum())
    return {'N':len(rows),'R_correct':int((s==y).sum()),'Text_correct':int((b==y).sum()),
            'R_accuracy':float((s==y).mean()),'Text_accuracy':float((b==y).mean()),'net_correct':net,'G':G,
            'gain_retention_priority':net>=6,'disagreements':int(d.sum()),'disagreement_rate':float(d.mean()),
            'AUROC':auroc(u,d),'nonfinite_n':int((~np.isfinite(u)).sum()),
            'plugin_candidates':led,'plugin_q':accepted[-1]['q'] if accepted else None}

def calibrate(rows,cuts,query_gold):
    """Query only finite-score covered disagreements, in (u,id) order; no all-gold map."""
    u,s,b=arrays(rows);ids=[r['id'] for r in rows];d=s!=b;n_all=len(rows)
    known={};queries=[];candidates=[];active={e:True for e in EPS};loss_state={e:[] for e in EPS}
    for cut in cuts:
        m=mask(u,cut['tau']);n=int(m.sum());k=int((m&d).sum())
        pc=float(binom.cdf(k,n,.05)) if n else 1.
        c=dict(cut,n_skip=n,k_changed=k,p_change=pc,accepted_change=n>0 and pc<=.001,
               change_upper999=upper(k,n),loss={})
        needed=sorted([i for i in np.flatnonzero(m&d) if ids[i] not in known],key=lambda i:(u[i],ids[i])) if any(active.values()) else []
        if needed:
            fetched=query_gold([ids[i] for i in needed])
            assert set(fetched)=={ids[i] for i in needed}
            for i in needed:
                g=fetched[ids[i]];assert g in 'ABCD' and len(g)==1
                known[ids[i]]=g
                queries.append({'query_ordinal':len(queries)+1,'id':ids[i],'u':score_text(u[i]),'gold':g,
                                'query_candidate_j':cut['j'],'eps_requested':[e for e in EPS if active[e]]})
        for e in EPS:
            key=str(e)
            if active[e]:
                kL=sum(b[i]==known[ids[i]] and s[i]!=known[ids[i]] for i in np.flatnonzero(m&d))
                kW=sum(s[i]==known[ids[i]] and b[i]!=known[ids[i]] for i in np.flatnonzero(m&d))
                p=float(binom.cdf(kL,n_all,e));ok=p<=.001
                state={'evaluated':True,'k_lost':int(kL),'k_gained':int(kW),'N_denominator':n_all,
                       'p':p,'upper999':upper(kL,n_all),'accepted':ok,'accepted_joint':ok and c['accepted_change']}
                if not ok:active[e]=False
            else:
                state={'evaluated':False,'accepted':False,'accepted_joint':False,'reason':'monotone rejection after earlier candidate'}
            c['loss'][key]=state;loss_state[e].append((cut,k,state))
        candidates.append(c)
    def deploy(selected,epsilon=None):
        if not selected:return {'mode':'fixed_Text','q':0.,'j':None,'threshold':None,'epsilon':epsilon,'calibration_fallback':True,'D_dep':0}
        c=selected[-1]
        return {'mode':'selective','q':c['q'],'j':c['j'],'threshold':c['tau'],'epsilon':epsilon,
                'calibration_fallback':False,'D_dep':c['k_changed']}
    policies={'behavior':deploy([c for c in candidates if c['accepted_change']])}
    summaries={}
    for e,name in zip(EPS,['joint_eps001','joint_eps002']):
        accepted=[c for c in candidates if c['loss'][str(e)]['accepted']]
        joint=[c for c in candidates if c['loss'][str(e)]['accepted_joint']]
        rejected=[(cut,k,state) for cut,k,state in loss_state[e] if state['evaluated'] and not state['accepted']]
        policies[name]=deploy(joint,e)
        summaries[str(e)]={'loss_only':deploy(accepted,e),'joint':policies[name],
                           'first_rejected_j':rejected[0][0]['j'] if rejected else None,
                           'D_seq':rejected[0][1] if rejected else int((np.isfinite(u)&d).sum())}
    return {'N':n_all,'D_all':int(d.sum()),'D_query_union':len(known),'candidates':candidates,'loss_summaries':summaries,
            'policies':policies,'label_queries':queries,'familywise_bonferroni_bound':.06}

def evaluation(rows,gold,policies,reps=2000):
    u,s,b=arrays(rows);y=np.array([gold[r['id']] for r in rows]);N=len(rows)
    h=np.array([answer(r['H_alone']) for r in rows]);R=(s==y).astype(int);T=(b==y).astype(int);H=(h==y).astype(int)
    masks={name:route(u,policies[name]) for name in POLICIES}
    pis={name:np.where(masks[name],s,b) for name in POLICIES}
    correct={name:(pis[name]==y).astype(int) for name in POLICIES}
    names=['R','Text','helper']+POLICIES;values=[R,T,H]+[correct[n] for n in POLICIES]
    for name in POLICIES:
        names += [name+':G',name+':G_pi',name+':loss',name+':delta_acc',name+':half_gain_margin']
        P=correct[name];values += [T-R,P-R,T-P,P-T,(P-R)-(T-R)/2]
    stats=dict(zip(names,paired_means(np.column_stack(values),reps)))
    system={n:{'correct':int(v.sum()),'accuracy':stats[n]} for n,v in zip(['R','Text','helper'],[R,T,H])}
    result={}
    for name in POLICIES:
        m=masks[name];ns=int(m.sum());kc=int((m&(s!=b)).sum());kl=int((m&(T==1)&(R==0)).sum())
        G=stats[name+':G'];GP=stats[name+':G_pi'];interval=cp(kc,ns)
        result[name]={'deployment':policies[name],'N':N,'n_skip':ns,'coverage':ns/N,'kappa_alpha':ns/N*.05,
                      'k_change':kc,'conditional_change':kc/ns if ns else None,'change_ci95':interval,
                      'marginal_change':kc/N,'k_lost':kl,'lost_denominator':N,'lost_ci95':cp(kl,N),
                      'correct':int(correct[name].sum()),'accuracy':stats[name],
                      **{k:stats[name+':'+k] for k in ['G','G_pi','loss','delta_acc','half_gain_margin']},
                      'retention_ratio':GP['mean']/G['mean'] if G['mean']!=0 else None,
                      'retention_unstable':abs(G['mean'])<.01 or G['ci95'][0]<=0<=G['ci95'][1],
                      'R1':ns>0 and interval[1]<.05,'R2':{str(e):cp(kl,N)[1]<e for e in EPS},
                      'positive_gains':G['ci95'][0]>0 and GP['ci95'][0]>0,
                      'at_least_half_gain':stats[name+':half_gain_margin']['ci95'][0]>0,
                      'INVALID_policy':int((pis[name]=='INVALID').sum())}
    return {'N':N,'systems':system,'policies':result,'nonfinite_n':int((~np.isfinite(u)).sum()),
            'INVALID_R':int((s=='INVALID').sum()),'INVALID_Text':int((b=='INVALID').sum()),
            'INVALID_helper':int((h=='INVALID').sum()),'scope':'model-scale extension on previously exposed benchmarks'}


class Analysis:
    def __init__(self):
        self.frozen=verify_freeze();self.cfg=read(HERE/'e2b_formal_config.json')
        self.out=HERE/'e2b_results';self.recheck=HERE/'e2b_recheck';self.evidence={};self.gold_evidence={}
        assert self.frozen['analysis_sha256'][str(Path(__file__).resolve())]==sha(__file__)
        self.out.mkdir(exist_ok=True)

    def source(self,path):
        p=Path(path);self.evidence[str(p)]=sha(p);return p

    def load(self,stage):
        from e2_debug_ops import units,completed,root
        expected=stage_rows(self.cfg,stage);rows=[];metadata=[]
        for u in units('e2b',stage):
            d,c,meta=completed('e2b',u,sha(freeze_path()))
            for name in ['COMPLETE.json','metadata.json',c['records_file']]:self.source(d/name)
            if u.get('parent'):self.source(root('e2b')/'recovery'/(u['parent']+'.json'))
            assert meta['freeze_commit']==self.frozen['commit'] and meta['analysis_sha256']==self.frozen['analysis_sha256']
            assert datetime.fromisoformat(meta['created_utc'])>=datetime.fromisoformat(self.frozen['utc'])
            rr=[json.loads(s) for s in (d/c['records_file']).read_text().splitlines()]
            ids=[r['id'] for r in expected[u['start']:u['stop']]]
            assert [r['id'] for r in rr]==c['ids']==meta['id_order']==ids
            assert [r['ordinal'] for r in rr]==list(range(u['start'],u['stop']))
            assert c['N']==meta['N']==len(rr);rows+=rr;metadata.append(meta)
        assert [r['id'] for r in rows]==[r['id'] for r in expected]
        self.segment_metadata=getattr(self,'segment_metadata',{});self.segment_metadata[stage]=metadata
        return rows

    def gold(self,ids):
        if not ids:return {}
        import pyarrow as pa
        import pyarrow.parquet as pq
        pa.set_cpu_count(1);pa.set_io_thread_count(1)
        spec=self.cfg['gold_source'];p=Path(spec['path']);assert sha(p)==spec['sha256']
        self.gold_evidence[str(p)]=spec['sha256']
        # Return labels only for requested IDs; never create a whole-calibration gold map.
        table=pq.read_table(p,columns=['id','answerKey'],filters=[('id','in',list(ids))],use_threads=False)
        result=dict(zip(table.column('id').to_pylist(),table.column('answerKey').to_pylist()))
        assert set(result)==set(ids) and all(v in ['A','B','C','D'] for v in result.values())
        return result

    def prerequisite(self,name):
        r=validate_release(name,self.frozen);self.source(self.out/(name+'.json'));return r

    def release(self,name,artifacts,extra=None):
        for p in artifacts:self.source(p)
        r={'status':'PASS','utc':utc(),'analysis_commit':self.frozen['commit'],
           'analysis_sha256':self.frozen['analysis_sha256'],'freeze_sha256':sha(freeze_path()),
           'evidence_sha256':dict(self.evidence),'gold_source_sha256':dict(self.gold_evidence),
           'external_use':'PENDING_RECHECK',**(extra or {})}
        if (self.out/(name+'.json')).exists():
            prior=validate_release(name,self.frozen)
            assert {k:v for k,v in prior.items() if k!='utc'}=={k:v for k,v in r.items() if k!='utc'}
            return prior
        write(self.out/(name+'.json'),r);return r

    def pilot(self,args):
        rows=self.load('pilot');stats=pilot_stats(rows,self.gold([r['id'] for r in rows]),self.cfg['bootstrap_replicates'])
        p=write(self.out/'pilot_metrics.json',stats)
        priority=stats['gain_retention_priority'];feasible=args.resource_feasible and bool(args.resource_note)
        if not priority and not feasible:
            print('PILOT_METRICS_READY: below gain gate; record a feasible resource plan before releasing fit_rest');return
        if not priority:assert datetime.now(timezone.utc)<datetime.fromisoformat(self.cfg['below_gate_resource_decision_cutoff_utc'])
        self.release('pilot_REVIEW',[p],{'decision':'CONTINUE','gain_retention_priority':priority,
                                      'resource_feasible':feasible,'resource_note':args.resource_note})

    def fit(self):
        self.prerequisite('pilot_REVIEW');rows=self.load('pilot')+self.load('fit_rest');u,_,_=arrays(rows)
        assert len(rows)==2100
        p=write(self.out/'thresholds.json',{'N':len(rows),'thresholds':grid(u),'nonfinite_n':int((~np.isfinite(u)).sum())})
        self.release('fit_PASS',[p])

    def cal(self):
        self.prerequisite('fit_PASS');cuts=read(self.source(self.out/'thresholds.json'))['thresholds']
        rows=self.load('cal');result=calibrate(rows,cuts,self.gold)
        p=write(self.out/'calibration.json',result)
        deployment=write(self.out/'deployment.json',result['policies'])
        queries=write(self.out/'label_queries.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result['label_queries']))
        self.release('cal_PASS',[p,deployment,queries])
        # Release first: export/recheck does not gate subsequent generation.
        self.export_cal()

    def manifest(self):
        self.recheck.mkdir(exist_ok=True)
        files=sorted(p for p in self.recheck.iterdir() if p.is_file() and p.name not in ['SHA256SUMS','RECHECK_COMPLETE.json'])
        text=''.join(sha(p)+'  '+p.name+'\n' for p in files)
        # This manifest grows as held-out and timing exports are added. Earlier per-stage receipts remain immutable.
        (self.recheck/'SHA256SUMS').write_text(text)

    def export_receipt(self,stage,files):
        p=write(self.recheck/(stage+'_export.json'),{'stage':stage,'freeze_sha256':sha(freeze_path()),
                'analysis_commit':self.frozen['commit'],'analysis_sha256':self.frozen['analysis_sha256'],
                'files':{p.name:sha(p) for p in files},'external_use':'PENDING_RECHECK'})
        self.manifest();return p

    def export_cal(self):
        self.prerequisite('cal_PASS');fit=self.load('pilot')+self.load('fit_rest');cal=self.load('cal')
        gold=self.gold([r['id'] for r in fit]);ledger=read(self.source(self.out/'calibration.json'))
        queried={r['id']:r for r in ledger['label_queries']}
        def row(r):return {'id':r['id'],'u':score_text(score(r)),'ans_s':answer(r['R']),'ans_b':answer(r['T'])}
        files=[csvwrite(self.recheck/'fit.csv',['id','u','gold','ans_s','ans_b'],[dict(row(r),gold=gold[r['id']]) for r in fit]),
               csvwrite(self.recheck/'cal.csv',['id','u','ans_s','ans_b','gold','queried','query_candidate_j'],
                        [dict(row(r),gold=queried.get(r['id'],{}).get('gold',''),queried=int(r['id'] in queried),
                              query_candidate_j=queried.get(r['id'],{}).get('query_candidate_j','')) for r in cal]),
               csvwrite(self.recheck/'thresholds.csv',['j','tau'],[{'j':c['j'],'tau':score_text(float(c['tau']))} for c in ledger['candidates']]),
               write(self.recheck/'policies.json',(self.out/'deployment.json').read_text()),
               write(self.recheck/'label_queries.jsonl',(self.out/'label_queries.jsonl').read_text())]
        self.export_receipt('cal',files)

    def evaluate(self,stage):
        self.prerequisite('cal_PASS');policies=read(self.source(self.out/'deployment.json'));rows=self.load(stage)
        gold=self.gold([r['id'] for r in rows]);result=evaluation(rows,gold,policies,self.cfg['bootstrap_replicates'])
        result.update(external_use='PENDING_RECHECK',analysis_sha256=self.frozen['analysis_sha256'])
        p=write(self.out/(stage+'_evaluation.json'),result)
        self.release(stage+'_EVAL',[p],{'status':'EVALUATED'})
        if stage=='heldout':
            files=[csvwrite(self.recheck/'heldout.csv',['id','u','ans_s','ans_b','gold'],
                            [{'id':r['id'],'u':score_text(score(r)),'ans_s':answer(r['R']),'ans_b':answer(r['T']),'gold':gold[r['id']]} for r in rows]),
                   write(self.recheck/'heldout_evaluation.json',p.read_text())]
            self.export_receipt('heldout',files)

    def timing(self):
        self.prerequisite('cal_PASS');rows=self.load('replay');gold=self.gold([r['id'] for r in rows])
        dev={r['id']:r for r in self.load('dev')};policies=read(self.source(self.out/'deployment.json'))
        per=[];results={}
        for name in POLICIES:
            cols={m:[] for m in ['policy','reference','receiver_alone']};correct={m:0 for m in cols};diff={m:0 for m in cols}
            for ordinal,r in enumerate(rows):
                assert r['ordinal']==ordinal
                rr=r['policies'][name]
                for mode in cols:
                    t=rr[mode+'_latency_ms'];assert math.isfinite(t) and t>0
                    ans=answer(rr[mode]['answer']);cols[mode].append(t);correct[mode]+=ans==gold[r['id']]
                    saved=dev[r['id']]
                    saved_role='T' if mode=='reference' else 'R' if mode=='receiver_alone' else ('R' if route(np.array([score(saved)]),policies[name])[0] else 'T')
                    diff[mode]+=ans!=answer(saved[saved_role])
                    pp=rr[mode]['P'];u=pp['score'] if pp and pp['score_finite'] else None
                    per.append({'id':r['id'],'ordinal':ordinal,'policy':name,'arm':mode,'execution_position':rr['execution_order'].index(mode),
                                'latency_ms':repr(float(t)),'answer':ans,'gold':gold[r['id']],'skip':rr[mode]['skip'],'u':score_text(u),'unit_key':r['unit_key'],'host':r['host'],'job_id':r['job_id'],
                                'excluded_in_first_request_sensitivity':ordinal==r['segment_start']})
            systems={m:{'N':len(rows),'correct':int(correct[m]),'accuracy':correct[m]/len(rows),'mean_ms':float(np.mean(v)),
                        'median_ms':float(np.median(v)),'p99_ms':float(np.quantile(v,.99)),
                        'throughput_qps':len(rows)*1000/sum(v),'answer_diff_saved_dev':int(diff[m])} for m,v in cols.items()}
            saving=paired_means(np.array(cols['reference'])-np.array(cols['policy']),self.cfg['bootstrap_replicates'])[0]
            keep=np.array([r['ordinal']!=r['segment_start'] for r in rows])
            sensitivity=paired_means((np.array(cols['reference'])-np.array(cols['policy']))[keep],self.cfg['bootstrap_replicates'])[0]
            results[name]={'exclude_first_per_segment':{'N':int(keep.sum()),'excluded_ids':[r['id'] for r,k in zip(rows,keep) if not k],'saving_ms':sensitivity},'systems':systems,'saving_ms':saving,'positive_savings':saving['ci95'][0]>0,
                           'conclusion':'positive benefit established' if saving['ci95'][0]>0 else 'positive benefit not established',
                           'negative_point_saving':saving['mean']<0}
        p=write(self.out/'timing_results.json',{'N':len(rows),'policies':results,'external_use':'PENDING_RECHECK',
                        'analysis_sha256':self.frozen['analysis_sha256'],'segments':self.segment_metadata['replay'],'source':'actual synchronized per-request timing, not component sums; mandatory preflight prevents a strict cold-model claim'})
        self.release('timing_REPORT',[p],{'status':'EVALUATED'})
        files=[csvwrite(self.recheck/'timing_requests.csv',list(per[0]),per),
               write(self.recheck/'timing_records.jsonl',''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows)),
               write(self.recheck/'timing_results.json',p.read_text())]
        self.export_receipt('timing',files)

    def report(self):
        lines=['# E2b internal results: not for external use before the recheck is complete','','Scope: model-scale extension on previously exposed benchmarks. Development and held-out accuracy and the 128-question measured timing panel are reported separately.']
        tables={};assessment={}
        for stage in ['dev','heldout']:
            p=self.out/(stage+'_evaluation.json')
            if not p.exists():lines+=['',stage+': not complete; no pass decision can be made.'];continue
            self.prerequisite(stage+'_EVAL');r=read(self.source(p));tables[stage]=r
            lines+=['',f"## {stage}, N={r['N']}",'','|policy|q|skipped|changed/skipped|R / Text / policy accuracy %|G / Gπ points|','|---|---:|---:|---|---|---|']
            for name,x in r['policies'].items():
                lines.append(f"|{name}|{x['deployment']['q']}|{x['n_skip']}|{x['k_change']}/{x['n_skip']}|{r['systems']['R']['accuracy']['mean']*100:.2f} / {r['systems']['Text']['accuracy']['mean']*100:.2f} / {x['accuracy']['mean']*100:.2f}|{x['G']['mean']*100:.2f} / {x['G_pi']['mean']*100:.2f}|")
            lines+=['',f"Helper-alone accuracy: {r['systems']['helper']['accuracy']['mean']*100:.2f}%. Paired intervals and all criteria are in {stage}_evaluation.json."]
        timing=None
        if (self.out/'timing_results.json').exists():
            self.prerequisite('timing_REPORT');timing=read(self.source(self.out/'timing_results.json'))
            lines+=['','## 128-question development timing panel','','|policy|R / Text / policy mean latency ms|saving ms [95% CI]|conclusion|','|---|---|---|---|']
            for name,x in timing['policies'].items():
                m=x['systems'];sv=x['saving_ms'];lines.append(f"|{name}|{m['receiver_alone']['mean_ms']:.2f} / {m['reference']['mean_ms']:.2f} / {m['policy']['mean_ms']:.2f}|{sv['mean']:.2f} [{sv['ci95'][0]:.2f}, {sv['ci95'][1]:.2f}]|{x['conclusion']}{'; negative benefit' if x['negative_point_saving'] else ''}|")
            lines+=['','Per-segment nodes, consistency checks and warmup records are in timing_results.json. The first segment has no extra warmup, but the mandatory consistency replay ran before it, so it is not called a strict cold start or equivalent to a continuous replay.', '', '|policy|N without the first panel question of each segment|saving ms [95% CI]|','|---|---:|---|']
            for name,x in timing['policies'].items():
                q=x['exclude_first_per_segment'];v=q['saving_ms'];lines.append(f"|{name}|{q['N']}|{v['mean']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]|")
        else:lines+=['','Timing not complete; complete success is not established.']
        if 'heldout' in tables:
            for name,x in tables['heldout']['policies'].items():
                assessment[name]={}
                for e in ([.01,.02] if name=='behavior' else [x['deployment']['epsilon']]):
                    complete=x['R1'] and x['R2'][str(e)] and x['positive_gains'] and bool(timing and timing['policies'][name]['positive_savings'])
                    assessment[name][str(e)]={'complete_success':bool(complete),'R1':x['R1'],'R2':x['R2'][str(e)],
                         'positive_gains':x['positive_gains'],'positive_savings':timing['policies'][name]['positive_savings'] if timing else None,
                         'at_least_half_gain':x['at_least_half_gain']}
            lines+=['','Complete-success assessment (conjunction of R1, R2, the lower ends of the G and G_pi intervals and the lower end of the measured-saving interval):','', '```json',json.dumps(assessment,ensure_ascii=False,indent=2),'```']
        lines+=['','All reports are internal material pending recheck; cal_PASS releases generation only, not external use. Recheck exports are in e2b_recheck/; SHA256SUMS is updated as stages are added.']
        # Reports may be rebuilt as new stages complete; immutable release artifacts above remain unchanged.
        (HERE/'E2B_REPORT.md').write_text('\n'.join(lines)+'\n')
        (self.out/'report_PROVENANCE.json').write_text(json.dumps({'analysis_commit':self.frozen['commit'],
            'analysis_sha256':self.frozen['analysis_sha256'],'freeze_sha256':sha(freeze_path()),
            'evidence_sha256':dict(self.evidence),'report_sha256':sha(HERE/'E2B_REPORT.md'),
            'external_use':'PENDING_RECHECK'},indent=2)+'\n')
        return assessment


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['pilot','fit','cal','export-cal','dev','heldout','timing','report'])
    ap.add_argument('--resource-feasible',action='store_true');ap.add_argument('--resource-note',default='');args=ap.parse_args()
    a=Analysis()
    if args.stage=='pilot':a.pilot(args)
    elif args.stage in ['dev','heldout']:a.evaluate(args.stage)
    elif args.stage=='export-cal':a.export_cal()
    else:getattr(a,args.stage)()
    print('E2B_CPU_COMPLETE',args.stage,'INTERNAL_PENDING_RECHECK')


if __name__=='__main__':main()
