"""Full-development latency and consistency, paired bootstrap by original batch."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def parsed(r):
    return r['parsed']['answer'] if r['parsed']['valid'] else 'INVALID'


def save(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def metrics(p,r,replicates):
    assert [x['N'] for x in p]==[x['N'] for x in r]
    n=np.array([x['N'] for x in p]);N=int(n.sum())
    pl=np.array([z['latency_ms'] for x in p for z in x['requests']])
    rl=np.array([z['latency_ms'] for x in r for z in x['requests']])
    ps=np.array([sum(z['latency_ms'] for z in x['requests']) for x in p])
    rs=np.array([sum(z['latency_ms'] for z in x['requests']) for x in r])
    pw=np.array([x['latency_ms'] for x in p]);rw=np.array([x['latency_ms'] for x in r])
    m={'N':N,'paired_batches':len(p),'policy_mean_ms':pl.mean(),'reference_mean_ms':rl.mean(),
       'policy_median_ms':np.median(pl),'reference_median_ms':np.median(rl),
       'policy_p99_ms':np.quantile(pl,.99),'reference_p99_ms':np.quantile(rl,.99),
       'policy_batch_mean_ms':pw.mean(),'reference_batch_mean_ms':rw.mean(),
       'policy_batch_total_s':pw.sum()/1000,'reference_batch_total_s':rw.sum()/1000,
       'policy_throughput_qps':N*1000/pw.sum(),'reference_throughput_qps':N*1000/rw.sum(),
       'request_mean_saving_ms':rl.mean()-pl.mean(),'request_relative_saving':1-ps.sum()/rs.sum(),
       'batch_wall_relative_saving':1-pw.sum()/rw.sum(),
       'throughput_gain_qps':N*1000/pw.sum()-N*1000/rw.sum()}
    rng=np.random.default_rng(0);boot={k:[] for k in ['request_mean_saving_ms','request_relative_saving','batch_wall_relative_saving','throughput_gain_qps']}
    for start in range(0,replicates,200):
        ix=rng.integers(0,len(p),size=(min(200,replicates-start),len(p)))
        bn=n[ix].sum(1);bp=ps[ix].sum(1);br=rs[ix].sum(1);wp=pw[ix].sum(1);wr=rw[ix].sum(1)
        for k,v in [('request_mean_saving_ms',(br-bp)/bn),('request_relative_saving',1-bp/br),
                    ('batch_wall_relative_saving',1-wp/wr),('throughput_gain_qps',bn*1000/wp-bn*1000/wr)]:boot[k].extend(v)
    for k,v in boot.items():m[k+'_ci95_low'],m[k+'_ci95_high']=np.quantile(v,[.025,.975])
    return {k:float(v) if isinstance(v,np.floating) else v for k,v in m.items()}


def conclusion(m):
    low=m['request_mean_saving_ms_ci95_low'];high=m['request_mean_saving_ms_ci95_high']
    return {'point_direction':'negative benefit' if m['request_mean_saving_ms']<0 else 'non-negative point estimate',
            'interval_conclusion':'positive benefit established' if low>0 else ('negative benefit established' if high<0 else 'positive benefit not established')}


def differences(a,b):
    overlap=a['helper'] is not None and b['helper'] is not None
    pa,pb=a['probe'],b['probe']
    both=pa is not None and pb is not None
    finite=both and math.isfinite(pa['score']) and math.isfinite(pb['score'])
    return {'answer_diff_n':int(parsed(a)!=parsed(b)),
            'raw_answer_diff_n':int(a['answer']['text']!=b['answer']['text']),
            'helper_overlap_n':int(overlap),
            'helper_diff_overlap_n':int(overlap and a['helper']['text']!=b['helper']['text']),
            'helper_presence_diff_n':int((a['helper'] is None)!=(b['helper'] is None)),
            'probe_compared_n':int(both),'probe_exact_diff_n':int(both and pa['score']!=pb['score']),
            'probe_nonfinite_pair_n':int(both and not finite),
            'route_diff_n':int(a['skip']!=b['skip']),
            'max_finite_probe_abs_diff':abs(pa['score']-pb['score']) if finite else None}


def sensitivity(p,r,replicates):
    excluded={x['segment_first_id'] for x in p+r}
    def keep(blocks):
        result=[]
        for x in blocks:
            rr=[z for z in x['requests'] if z['id'] not in excluded]
            if rr:result.append(dict(x,N=len(rr),requests=rr))
        return result
    pp,rr=keep(p),keep(r);m=metrics(pp,rr,replicates)
    return {'N':m['N'],'excluded_ids':'|'.join(sorted(excluded)),**{k:v for k,v in m.items() if k.startswith('request_mean_saving_ms')},**conclusion(m)}


def main():
    ap=argparse.ArgumentParser();ap.parse_args()
    from e2_debug_ops import units,completed,root
    from e2a_v2_gate import verify,freeze_path
    from e2b_gate import sha
    frozen=verify()
    cfg=json.loads((HERE/'e2a_v2_config.json').read_text());run=root('e2a')
    assert run.is_dir(),'No execution directory; do not manufacture a results report'
    latency=[];summaries=[];consistency=[];requests=[];batch_rows=[];warnings=[];segments=[];sensitivity_rows=[]
    for pair in cfg['pair_order']:
        rows=[];sources=[]
        for u in [x for x in units('e2a') if x['pair']==pair]:
            try:d,c,m=completed('e2a',u,sha(freeze_path()))
            except FileNotFoundError:
                warnings.append(u['key']+': missing COMPLETE; partial records excluded in their entirety');continue
            assert m['freeze_commit']==frozen['commit'] and m['warmup']['performed']==u['warmup']
            path=d/c['records_file'];rr=[json.loads(line) for line in path.read_text().splitlines()]
            assert len(rr)==c['batches']
            for r in rr:r['_source']=str(path)
            rows+=rr;sources.append(str(path));segments.append(m)
        source=';'.join(sources)
        assert len({(x['dataset'],x['batch_size'],x['batch_id'],x['mode']) for x in rows})==len(rows)
        for ds,info in cfg['panels'].items():
            ids=json.loads(Path(info['ids']).read_text());cells={}
            for b in cfg['batch_sizes']:
                for mode in cfg['arms']:
                    blocks=sorted([x for x in rows if (x['dataset'],x['batch_size'],x['mode'])==(ds,b,mode)],key=lambda x:x['batch_id'])
                    flat=[r for x in blocks for r in x['requests']]
                    complete=[r['id'] for r in flat]==ids
                    for x in blocks:
                        expected=ids[x['batch_id']*b:(x['batch_id']+1)*b]
                        assert [r['id'] for r in x['requests']]==expected and x['N']==len(expected)
                        assert all(0<=r['latency_ms']<=x['latency_ms'] for r in x['requests'])
                        batch_rows.append({'pair':pair,'dataset':ds,'B':b,'arm':mode,'batch_id':x['batch_id'],
                                           'order_position':x['order_position'],'N':x['N'],'batch_wall_ms':x['latency_ms'],'unit_key':x['unit_key'],'host':x['host'],'job_id':x['job_id'],'source':x['_source']})
                        for r in x['requests']:
                            requests.append({'pair':pair,'dataset':ds,'B':b,'arm':mode,'id':r['id'],'batch_id':x['batch_id'],
                                             'answer':parsed(r),'raw_answer':r['answer']['text'],
                                             'helper_message':None if r['helper'] is None else r['helper']['text'],
                                             'probe_score':None if r['probe'] is None else r['probe']['score'],
                                             'skip':r['skip'],'latency_ms':r['latency_ms'],'unit_key':x['unit_key'],'host':x['host'],'job_id':x['job_id'],'source':x['_source']})
                    cells[b,mode]=(blocks,flat,complete)
                    summary={'pair':pair,'dataset':ds,'B':b,'arm':mode,'status':'COMPLETE' if complete else 'INCOMPLETE',
                             'expected_N':len(ids),'observed_N':len(flat),'source':str(source)}
                    if complete:
                        ll=np.array([r['latency_ms'] for r in flat]);wall=np.array([x['latency_ms'] for x in blocks])
                        summary.update(mean_ms=float(ll.mean()),median_ms=float(np.median(ll)),p99_ms=float(np.quantile(ll,.99)),
                                       batch_mean_ms=float(wall.mean()),batch_total_s=float(wall.sum()/1000),
                                       throughput_qps=float(len(ids)*1000/wall.sum()),invalid_n=sum(not r['parsed']['valid'] for r in flat))
                    summaries.append(summary)
                for candidate,comparator in cfg['comparisons']:
                    p,r=cells[b,candidate],cells[b,comparator]
                    row={'pair':pair,'dataset':ds,'B':b,'candidate':candidate,'comparator':comparator,
                         'status':'COMPLETE' if p[2] and r[2] else 'INCOMPLETE','source':str(source)}
                    if p[2] and r[2]:
                        m=metrics(p[0],r[0],cfg['bootstrap_replicates'])
                        row.update({k.replace('policy_','candidate_').replace('reference_','comparator_'):v for k,v in m.items()})
                        row.update(conclusion(m))
                    latency.append(row)
                    sr={k:row[k] for k in ['pair','dataset','B','candidate','comparator','status','source']}
                    if p[2] and r[2]:sr.update(sensitivity(p[0],r[0],cfg['bootstrap_replicates']))
                    sensitivity_rows.append(sr)
            checks=[(b,mode,1,mode,'versus_own_B1') for b in cfg['batch_sizes'] for mode in cfg['arms']]
            checks += [(b,'early_return',b,'synchronous','schedule_difference') for b in cfg['batch_sizes']]
            checks += [(b,mode,b,'reference','policy_reference_change') for b in cfg['batch_sizes'] for mode in ['synchronous','early_return']]
            for b,mode,bb,other,kind in checks:
                pa,pb=cells[b,mode],cells[bb,other]
                row={'pair':pair,'dataset':ds,'B':b,'arm':mode,'comparison':kind,'other_B':bb,'other_arm':other,
                     'status':'COMPLETE' if pa[2] and pb[2] else 'INCOMPLETE','expected_N':len(ids),'source':str(source)}
                if pa[2] and pb[2]:
                    counts={};maxdiff=None
                    for x,y in zip(pa[1],pb[1]):
                        assert x['id']==y['id']
                        flags=differences(x,y);diff=flags.pop('max_finite_probe_abs_diff')
                        if diff is not None:maxdiff=max(maxdiff or 0.,diff)
                        for k,v in flags.items():counts[k]=counts.get(k,0)+v
                    row.update(counts,N=len(ids),max_finite_probe_abs_diff=maxdiff,
                               skip_n=sum(x['skip'] for x in pa[1]),invalid_n=sum(not x['parsed']['valid'] for x in pa[1]))
                    if kind=='policy_reference_change':
                        row.update(skip_changed_n=sum(x['skip'] and parsed(x)!=parsed(y) for x,y in zip(pa[1],pb[1])),
                                   wait_changed_n=sum(not x['skip'] and parsed(x)!=parsed(y) for x,y in zip(pa[1],pb[1])),
                                   change_rate=counts['answer_diff_n']/len(ids))
                consistency.append(row)
    assert len(summaries)==len(latency)==36
    for name,rr in [('sensitivity',sensitivity_rows),('arms',summaries),('latency',latency),('consistency',consistency),('per_request',requests),('batches',batch_rows)]:
        save('e2a_v2_'+name+'.csv',rr)
    lines=['# E2a v2: three-arm scheduling replay on the full development sets','','Four independent debug-queue jobs (on failure, only complete smaller rerun pieces are merged). OBQA 742, ARC 299; each of the four strategies at B=1/8/32, one execution per arm per batch.',
           '', 'For each strategy, the synchronous, early-return and reference arms run on the same node with the same grouping, independently, in a fixed cyclic order by global batch index. All latencies come from actual completion times; jobs of different strategies may run on different nodes (see the segment metadata).',
           '', 'Exclusive ClusterA node, helper on GPU 0 / receiver on GPU 1. allocation/ keeps the scheduler job record and nvidia-smi taken at job start. No prefix or cross-request cache; the KV cache within one generation is kept.',
           '', 'The development sets were exposed before. 20,000 bootstrap replicates paired by original batch, seed=0, the last batch weighted by its actual number of requests; p99 is descriptive only. All 36 arm cells and 36 comparisons are listed; incomplete cells are not extrapolated.',
           '', 'An interval that crosses or touches 0 is written "positive benefit not established"; a negative point estimate is reported as "negative benefit". INVALID answers and differences in messages, scores, routes and answers are kept; where outputs change, the certificate is not claimed to transfer directly.',
           '', '|strategy|B|arm|status|per-request mean/median/p99 ms|batch mean ms|throughput questions/s|','|---|---:|---|---|---|---:|---:|']
    for x in summaries:
        values=f"{x['mean_ms']:.2f}/{x['median_ms']:.2f}/{x['p99_ms']:.2f}|{x['batch_mean_ms']:.2f}|{x['throughput_qps']:.3f}" if x['status']=='COMPLETE' else f"incomplete {x['observed_N']}/{x['expected_N']}|—|—"
        lines.append(f"|{x['pair']}/{x['dataset']}|{x['B']}|{x['arm']}|{x['status']}|{values}|")
    lines+=['','|strategy|B|candidate / comparator|mean saving per request ms [95% CI]|conclusion|','|---|---:|---|---|---|']
    for x in latency:
        values=f"{x['request_mean_saving_ms']:.2f} [{x['request_mean_saving_ms_ci95_low']:.2f}, {x['request_mean_saving_ms_ci95_high']:.2f}]|{x['point_direction']}; {x['interval_conclusion']}" if x['status']=='COMPLETE' else 'incomplete|not evaluated'
        lines.append(f"|{x['pair']}/{x['dataset']}|{x['B']}|{x['candidate']} / {x['comparator']}|{values}|")
    lines+=['','|strategy|B|arm / comparison|status|answer differences|message differences/jointly waiting|score differences/probed pairs|route differences|','|---|---:|---|---|---:|---|---|---:|']
    for x in consistency:
        values=f"{x['answer_diff_n']}|{x['helper_diff_overlap_n']}/{x['helper_overlap_n']}|{x['probe_exact_diff_n']}/{x['probe_compared_n']}|{x['route_diff_n']}" if x['status']=='COMPLETE' else '—|—|—|—'
        lines.append(f"|{x['pair']}/{x['dataset']}|{x['B']}|{x['arm']} / {x['comparison']}|{x['status']}|{values}|")
    lines+=['','|strategy|B|candidate / comparator|N without the first question of each segment|sensitivity saving ms [95% CI]|conclusion|','|---|---:|---|---:|---|---|']
    for x in sensitivity_rows:
        v=f"{x['N']}|{x['request_mean_saving_ms']:.2f} [{x['request_mean_saving_ms_ci95_low']:.2f}, {x['request_mean_saving_ms_ci95_high']:.2f}]|{x['point_direction']}; {x['interval_conclusion']}" if x['status']=='COMPLETE' else '—|incomplete|not evaluated'
        lines.append(f"|{x['pair']}/{x['dataset']}|{x['B']}|{x['candidate']} / {x['comparator']}|{v}|")
    lines+=['','The sensitivity analysis removes the first question of each segment (paired by original order); the main results keep all questions. Node and warmup logs are in e2a_v2_segments.json; the first question of a segment is not evidence of an identical hardware state.','']
    for m in segments:
        lines.append(f"- {m['unit']['key']}: node {m['host']}, job {m['job_id']}, warmup {m['warmup']['performed']}, events {len(m['warmup'].get('events',[]))}.")
    (HERE/'e2a_v2_segments.json').write_text(json.dumps(segments,indent=2,ensure_ascii=False)+'\n')
    lines+=['',f'Source `{run}`; all statistics and per-question messages/scores/routes/answers are in e2a_v2_*.csv; the actual execution order is in batches.jsonl.',
            '', 'The earlier E2a run followed the E2A_FREEZE of 11/9 18:25 CT; it is kept as the 128-question historical variant "skipped requests wait for the whole batch" and is not a same-hardware scheduling control for this run.']
    lines+=['','Execution gaps:']+warnings if warnings else []
    (HERE/'E2A_V2_REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
