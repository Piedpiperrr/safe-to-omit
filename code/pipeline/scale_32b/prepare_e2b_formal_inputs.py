"""Restore two path-anonymization substitutions in separate formal input copies.
Original paper inputs and bab40ab smoke inputs are read-only. No gold is read.
"""
import hashlib
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
# Original paper inputs (a private absolute path in the run).
ORIGINAL=Path(os.environ.get('DATA_DIR','$DATA_DIR'))/'P2_CONFIDENCE_REFERENCE_BOUNDARIES_20260914T165012Z/inputs'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not (HERE/'FREEZE.txt').exists() and not (HERE/'e2b_runs').exists()
    p=HERE/'e2b_formal_config.json';cfg=json.loads(p.read_text());repairs=[]
    for split,name,target_id,index,text in [('fit','obqa_train_queries.jsonl','14-1424',3,'eagle'),('dev','obqa_dev_queries.jsonl','12-577',2,'Eagle')]:
        old=HERE/f'inputs/obqa_{split}_queries.jsonl'
        rows=[json.loads(s) for s in old.read_text().splitlines()]
        source=ORIGINAL/name;reference={r['id']:r for r in map(json.loads,source.read_text().splitlines())}
        changes=[]
        for ordinal,r in enumerate(rows):
            original=reference[r['id']]
            assert set(original)==set(r) and r['question_stem']==original['question_stem'] and r['choice_labels']==original['choice_labels']
            if r['choice_text']!=original['choice_text']:
                assert r['id']==target_id and r['choice_text'][index]=='sharedfs' and original['choice_text'][index]==text
                assert all(a==b for j,(a,b) in enumerate(zip(r['choice_text'],original['choice_text'])) if j!=index)
                changes.append({'id':target_id,'ordinal_zero_based':ordinal,'label':r['choice_labels'][index],'before':'sharedfs','after':text})
                r['choice_text']=original['choice_text']
        assert len(changes)==1
        out=HERE/f'inputs/obqa_{split}_queries_formal.jsonl'
        data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
        if out.exists():assert out.read_text()==data
        else:out.write_text(data)
        cfg['inputs'][split]['path']=str(out);cfg['inputs'][split]['sha256']=sha(out)
        repairs.append({'split':split,'reviewed_input':str(old),'reviewed_sha256':sha(old),
                        'original_paper_input':str(source),'original_paper_sha256':sha(source),'changes':changes})
    cfg['formal_input_repairs']={'reason':'restore original paper option text altered by release path anonymization before any formal E2b output',
                                  'smoke_inputs_unchanged':True,'pilot_300_unchanged':True,'repairs':repairs}
    p.write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps(cfg['formal_input_repairs'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
