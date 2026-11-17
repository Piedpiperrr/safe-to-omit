"""Freeze full development IDs while retaining the paper's E2a policies."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    assert not (HERE/'E2A_V2_FREEZE.txt').exists()
    for name in ['e2a_v2_runs','locks/e2a_v2','submissions/e2a_v2.json']:
        assert not (HERE/name).exists(), 'Pre-output revision required: '+name
    cfg=json.loads((HERE/'e2a_config.json').read_text())
    cfg['version']='E2a_v2_three_arm_pre_output_revision'
    cfg['arms']=['synchronous','early_return','reference']
    cfg['repetitions']=1
    cfg['pair_order']=['large','llama']
    cfg['arm_order']='rotate arms left by (pair_index+dataset_index+batch_size_index+batch_index) modulo 3'
    cfg['grouping']='original development ID order; consecutive B requests; retain the shorter last batch; same grouping for all three arms'
    cfg['comparisons']=[['early_return','reference'],['synchronous','reference'],['early_return','synchronous']]
    cfg['report_scope']='all 36 arm cells and all 36 matched comparisons; retain nonpositive, incomplete and INVALID results; no result-dependent reruns'
    cfg['inference']='20000 paired-batch percentile bootstrap replicates, seed 0, request-weighted unequal tails; two-sided 95%; p99 descriptive only'
    cfg['benefit_rule']='CI lower >0 establishes positive savings; CI crossing/touching zero: positive benefit not established; negative point savings reported as negative benefit'
    cfg['panels']={}
    cfg['exclusive_nodes']=True
    cfg['prefix_cache']=False
    cfg['timing']='direct synchronized wall-clock batch start to request completion callback; synchronous/reference complete together, early_return completes by sub-batch; never reconstruct from components'
    cfg['bootstrap_unit']='paired original-order batch, request-weighted for unequal final batch'
    cfg['p99']='descriptive only, no inferential interval or gate'
    for ds,n in [('obqa',742),('arc',299)]:
        ids=ROOT/f'splits/development_and_calibration/{ds}_dev_ids.json'
        src=ROOT/f'records/boundaries_large_small/inputs/{ds}_dev_queries.jsonl'
        order=json.loads(ids.read_text()); rows=[json.loads(s) for s in src.read_text().splitlines()]
        byid={r['id']:r for r in rows}
        assert len(order)==len(set(order))==len(byid)==n
        selected=[byid[i] for i in order]
        assert all(set(r)=={'id','question_stem','choice_labels','choice_text'} for r in selected)
        target=HERE/f'inputs/e2a_v2_{ds}_dev_queries.jsonl'
        text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected)
        if target.exists():assert target.read_text()==text
        else:target.write_text(text)
        cfg['panels'][ds]={'queries':str(target),'ids':str(ids),'N':n}
        for p in [src,ids,target]:cfg['source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (HERE/'e2a_v2_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print('E2A_V2_PREPARED', {k:v['N'] for k,v in cfg['panels'].items()})


if __name__=='__main__':main()
