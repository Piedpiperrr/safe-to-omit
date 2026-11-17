"""Revised pre-v2-output freeze and isolated per-unit claims; original freeze retained."""
import json
import os
from pathlib import Path
import socket
from e2b_gate import sha

HERE=Path(__file__).resolve().parent

def freeze_path():return HERE/'E2A_V2_FREEZE_DEBUG.txt'

def verify():
    p=freeze_path();assert p.is_file(),'E2A_V2_DEBUG_NOT_FROZEN'
    r=json.loads(p.read_text());assert r['status']=='FROZEN_OPERATIONAL_REVISION'
    assert sha(HERE/'E2A_V2_FREEZE.txt')==r['original_freeze_sha256']
    for name,h in r['files'].items():assert sha(name)==h,name
    return r

def start(manifest_path,rank):
    from e2_debug_ops import read,unit_by_key,partial,root,put
    r=verify();job=os.environ['PBS_JOBID'];manifest=read(manifest_path)
    # As run, also asserted a ClusterA compute node (host name and job id pattern); check removed in the release.
    assert manifest['kind']=='e2a' and manifest['nodes']==1 and rank==0
    u=manifest['units'][rank];assert u==unit_by_key('e2a',u['key'])
    a=read(root('e2a')/'jobs'/job/'allocation/node_0/allocation.json')
    assert a['exclusive'] and a['job_id']==job and a['host']==socket.gethostname()
    # As run, a persistent per-unit claim on the shared filesystem was taken here (scheduling only; helper not released).
    out=partial('e2a',u['key']);out.mkdir(parents=True,exist_ok=False)
    put(out/'STARTED.json',{'job_id':job,'unit':u,'host':socket.gethostname()})
    return r,u,out,a
