"""Operational-only units, completion selection, queue checks and one smaller retry.

Release note: the scheduler-facing functions of the run (terminal_state, prepare_retry, parse_queue, count_jobs,
submit: job-record checks, queue checks and job submission) are not included. The unit plan, completion checks,
the one-smaller-retry layout (children) and the preflight barrier are unchanged.
"""
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from datetime import datetime,timezone
from e2b_gate import sha

HERE=Path(__file__).resolve().parent

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def digest(obj):
    import hashlib
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def put(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n')

def root(kind):return HERE/('e2b_debug_runs' if kind=='e2b' else 'e2a_v2_debug_runs')
def plan(kind):return read(HERE/('e2b_debug_plan.json' if kind=='e2b' else 'e2a_v2_debug_plan.json'))
def partial(kind,key):return root(kind)/'incomplete'/key
def complete(kind,key):return root(kind)/'complete'/key

def children(unit):
    assert unit['attempt']==0,'Only one smaller retry is authorized'
    start,stop=unit['start'],unit['stop']
    if unit['kind']=='e2a':
        step=max(unit['batch_sizes']);mid=((start+stop)//2//step)*step
        if mid<=start:mid=start+step
    else:mid=(start+stop)//2
    assert start<mid<stop,'Cannot shrink further without changing original grouping'
    out=[]
    for tag,lo,hi in [('a',start,mid),('b',mid,stop)]:
        u=dict(unit,key=unit['key']+'_r1'+tag,start=lo,stop=hi,attempt=1,parent=unit['key'])
        u['warmup']=unit['warmup'] or (unit['stage']=='replay' and lo>0) or (unit['kind']=='e2a' and lo>0)
        out.append(u)
    return out

def units(kind,stage=None):
    result=[]
    for u in plan(kind)['units']:
        if stage and u['stage']!=stage and not (stage=='dev_heldout' and u['stage'] in ['dev','heldout']):continue
        recovery=root(kind)/'recovery'/(u['key']+'.json')
        if recovery.exists():
            r=read(recovery);assert r['parent_spec_sha256']==digest(u)
            expected=children(u);assert r['children']==expected
            result.extend(expected)
        else:result.append(u)
    return result

def unit_by_key(kind,key):
    found=[u for u in units(kind) if u['key']==key];assert len(found)==1,key
    return found[0]

def completed(kind,u,freeze_sha):
    d=complete(kind,u['key']);c=read(d/'COMPLETE.json');m=read(d/'metadata.json')
    assert c['status']=='COMPLETE' and c['freeze_sha256']==m['freeze_sha256']==freeze_sha
    assert c['unit_spec_sha256']==digest(u) and m['unit']==u
    assert c['metadata_sha256']==sha(d/'metadata.json') and c['records_sha256']==sha(d/c['records_file'])
    if kind=='e2b' and u['stage']!='pilot':assert m['consistency']['status']=='PASS'
    return d,c,m

def finish(kind,u,out,receipt):
    receipt.update(status='COMPLETE',utc=now(),unit_spec_sha256=digest(u),metadata_sha256=sha(out/'metadata.json'))
    put(out/'COMPLETE.json',receipt)
    dest=complete(kind,u['key']);dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists();out.rename(dest)
    return dest

def preflight_barrier(job_root,key,nodes,status,deadline):
    p=job_root/'preflight';put(p/(key+'.json'),status)
    while True:
        rows=[read(f) for f in p.glob('*.json')]
        if any(r['status']!='PASS' for r in rows):raise RuntimeError('JOB_PREFLIGHT_FAILED')
        if len(rows)==nodes:return
        assert time.monotonic()<deadline,'PREFLIGHT_TIMEOUT'
        time.sleep(1)
