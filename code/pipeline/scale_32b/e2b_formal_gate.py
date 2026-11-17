"""Formal release checks for the revised protocol; no gain-only stop rule."""
import json
from pathlib import Path
from datetime import datetime,timezone
from e2b_gate import verify_freeze,sha,freeze_path

HERE=Path(__file__).resolve().parent


def validate_release(name,frozen=None):
    frozen=frozen or verify_freeze()
    p=HERE/'e2b_results'/f'{name}.json';r=json.loads(p.read_text())
    assert r['freeze_sha256']==sha(freeze_path())
    assert r['analysis_commit']==frozen['commit']
    assert r['analysis_sha256']==frozen['analysis_sha256']
    cfg=json.loads((HERE/'e2b_formal_config.json').read_text())
    for source,h in r.get('gold_source_sha256',{}).items():
        # Check pinned identity without opening the gold dataset in a GPU worker.
        assert source==cfg['gold_source']['path'] and h==cfg['gold_source']['sha256']
    assert r['evidence_sha256']
    for source,h in r['evidence_sha256'].items():assert sha(source)==h,source
    return r


def require_formal_stage(stage):
    frozen=verify_freeze()
    cfg=json.loads((HERE/'e2b_formal_config.json').read_text())
    assert datetime.now(timezone.utc)<datetime.fromisoformat(cfg['deadline_utc']),'E2B_DEADLINE_REACHED'
    needed={'pilot':None,'fit_rest':'pilot_REVIEW','cal':'fit_PASS','dev':'cal_PASS','heldout':'cal_PASS','replay':'cal_PASS'}
    assert stage in needed
    name=needed[stage]
    if name:
        r=validate_release(name,frozen)
        if name=='pilot_REVIEW':
            assert r['decision']=='CONTINUE'
            if not r['gain_retention_priority']:
                assert r['resource_feasible']
                assert datetime.fromisoformat(r['utc'])<datetime.fromisoformat(cfg['below_gate_resource_decision_cutoff_utc'])
        else:assert r['status']=='PASS'
        if name=='cal_PASS':assert str(HERE/'e2b_results/deployment.json') in r['evidence_sha256']
    return frozen
